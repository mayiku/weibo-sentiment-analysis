"""DeepSeek-backed batch sentiment classification for short Weibo comments."""

from __future__ import annotations

import json
import re
import time
from typing import Any, List

from config import (
    DEEPSEEK_SENTIMENT_BATCH_RETRIES,
    DEEPSEEK_SENTIMENT_BATCH_SIZE,
    DEEPSEEK_SENTIMENT_MAX_TEXT_LENGTH,
)
from src.ai_agent.deepseek_client import DeepSeekClient
from src.logger import get_logger

from .base import ModelInfo, SentimentAnalyzer, SentimentResult


log = get_logger(__name__)
VALID_LABELS = {"积极", "中性", "消极"}
LABEL_ALIASES = {
    "正面": "积极", "positive": "积极", "pos": "积极", "积极": "积极",
    "中立": "中性", "neutral": "中性", "neu": "中性", "中性": "中性",
    "负面": "消极", "negative": "消极", "neg": "消极", "消极": "消极",
}


class DeepSeekSentimentAnalyzer(SentimentAnalyzer):
    """Classify comments in validated JSON batches using DeepSeek Chat."""

    def __init__(self, client: Any = None, batch_size: int = None,
                 max_text_length: int = None, batch_retries: int = None):
        self.client = client or DeepSeekClient()
        self.batch_size = max(1, int(batch_size or DEEPSEEK_SENTIMENT_BATCH_SIZE))
        self.batch_retries = max(
            1, int(batch_retries or DEEPSEEK_SENTIMENT_BATCH_RETRIES)
        )
        self.max_text_length = max(
            40, int(max_text_length or DEEPSEEK_SENTIMENT_MAX_TEXT_LENGTH)
        )
        self.partial_fallback_count = 0
        self.partial_fallback_reasons: list[str] = []
        self._snow_fallback = None
        super().__init__()

    def _get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="DeepSeek Weibo Sentiment",
            version="1.0.0",
            provider="DeepSeek API",
            supports_gpu=False,
            batch_size=self.batch_size,
            description="面向微博短文本的三分类语义模型，支持反讽、否定与网络语言",
        )

    def analyze(self, text: str, **kwargs) -> SentimentResult:
        return self.analyze_batch([text], **kwargs)[0]

    def analyze_batch(self, texts: List[str], **kwargs) -> List[SentimentResult]:
        if not texts:
            return []
        self.partial_fallback_count = 0
        self.partial_fallback_reasons = []
        results: list[SentimentResult] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            results.extend(self._analyze_resilient_batch(batch))
        return results

    def _analyze_resilient_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Retry malformed model output, then split only the failed batch."""
        last_error = None
        for attempt in range(1, self.batch_retries + 1):
            try:
                return self._analyze_api_batch(texts)
            except Exception as exc:
                last_error = exc
                if not self._is_recoverable_batch_error(exc):
                    raise
                log.warning(
                    "DeepSeek 情绪批次格式异常 (%d 条, %d/%d): %s",
                    len(texts), attempt, self.batch_retries, exc,
                )

        if len(texts) > 1:
            midpoint = len(texts) // 2
            log.warning(
                "DeepSeek 情绪批次连续失败，拆分为 %d + %d 条重试",
                midpoint, len(texts) - midpoint,
            )
            return (
                self._analyze_resilient_batch(texts[:midpoint])
                + self._analyze_resilient_batch(texts[midpoint:])
            )

        # A single malformed item must not discard hundreds of successful
        # DeepSeek classifications. Degrade this item only and record scope.
        from .snow_analyzer import SnowNLPAnalyzer

        if self._snow_fallback is None:
            self._snow_fallback = SnowNLPAnalyzer()
        fallback = self._snow_fallback.analyze_batch(texts)
        for result in fallback:
            result.analysis = f"DeepSeek局部降级: {result.analysis}"
        self.partial_fallback_count += len(texts)
        self.partial_fallback_reasons.append(str(last_error))
        log.warning("DeepSeek 单条格式仍异常，仅该条降级到 SnowNLP: %s", last_error)
        return fallback

    @staticmethod
    def _is_recoverable_batch_error(exc: Exception) -> bool:
        """Only split/retry malformed structured output, not auth/network failures."""
        return str(exc).startswith("DeepSeek 返回")

    def _analyze_api_batch(self, texts: list[str]) -> list[SentimentResult]:
        started = time.perf_counter()
        items = [
            {"id": index, "text": str(text)[:self.max_text_length]}
            for index, text in enumerate(texts)
        ]
        prompt = (
            "请对以下微博评论进行情绪三分类。必须结合否定、反问、讽刺、emoji、"
            "饭圈黑话和上下文语气判断。积极=明确支持/喜爱；消极=批评/嘲讽/不满；"
            "中性=事实陈述、信息不足或无明显态度。\n"
            "只返回 JSON 数组，不要 Markdown。每项严格且仅包含 id、label、confidence；"
            "label 只能是积极、中性、消极，confidence 是 0 到 1。\n"
            f"输入：{json.dumps(items, ensure_ascii=False)}"
        )
        response = self.client.chat(
            prompt,
            system_prompt="你是严格、可复核的中文社交媒体情绪分类器。",
            temperature=0.0,
            max_tokens=max(400, len(texts) * 35),
        )
        if not response:
            error = getattr(self.client, "last_error", None) or "API 未返回内容"
            raise RuntimeError(f"DeepSeek 情绪分析失败: {error}")

        parsed = self._parse_response(response, len(texts))
        elapsed_per_item = (time.perf_counter() - started) / len(texts)
        output = []
        for index in range(len(texts)):
            item = parsed[index]
            label = item["label"]
            confidence = item["confidence"]
            score = self._score_for(label, confidence)
            reason = item.get('reason', '')
            output.append(SentimentResult(
                label=label,
                score=score,
                confidence=confidence,
                model_time=elapsed_per_item,
                analysis=f"DeepSeek语义判断: {reason}" if reason else "DeepSeek语义判断",
                enhanced=True,
            ))
        log.info("DeepSeek 情绪批次完成: %d 条", len(output))
        return output

    @staticmethod
    def _parse_response(response: str, expected: int) -> dict[int, dict]:
        content = response.strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
        match = re.search(r"\[.*\]", content, flags=re.S)
        if not match:
            raise RuntimeError("DeepSeek 返回内容不包含 JSON 数组")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"DeepSeek 返回 JSON 无法解析: {exc}") from exc
        if not isinstance(data, list) or len(data) != expected:
            raise RuntimeError(
                f"DeepSeek 返回 {len(data) if isinstance(data, list) else 0} 条，预期 {expected} 条"
            )

        normalized = {}
        for raw in data:
            if not isinstance(raw, dict):
                raise RuntimeError("DeepSeek 返回项不是对象")
            try:
                item_id = int(raw.get("id"))
                confidence = max(0.01, min(1.0, float(raw.get("confidence", 0))))
            except (TypeError, ValueError) as exc:
                raise RuntimeError("DeepSeek 返回的 id 或 confidence 无效") from exc
            label = LABEL_ALIASES.get(str(raw.get("label", "")).strip().lower())
            if item_id in normalized or not 0 <= item_id < expected or label not in VALID_LABELS:
                raise RuntimeError("DeepSeek 返回的 id 或 label 无效")
            normalized[item_id] = {
                "label": label,
                "confidence": confidence,
                "reason": str(raw.get("reason", ""))[:40],
            }
        if set(normalized) != set(range(expected)):
            raise RuntimeError("DeepSeek 返回的 id 不完整")
        return normalized

    @staticmethod
    def _score_for(label: str, confidence: float) -> float:
        if label == "积极":
            return min(1.0, 0.6 + 0.4 * confidence)
        if label == "消极":
            return max(0.0, 0.4 - 0.4 * confidence)
        return 0.5
