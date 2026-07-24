"""DataFrame-oriented compatibility API for the sentiment package.

This module is the canonical home of the functions consumed by ``app.py``.
Keeping them inside the package avoids the ambiguous ``src/sentiment.py`` vs
``src/sentiment/`` import that previously made these functions unreachable.
"""

from collections import Counter
from functools import lru_cache
import importlib.util
import re
from typing import Optional

import pandas as pd

from config import DEFAULT_SENTIMENT_MODEL, DEEPSEEK_API_KEY, STOPWORDS
from src.logger import get_logger

from .base import AnalyzerFactory
from .evaluator import ModelEvaluator


log = get_logger(__name__)


@lru_cache(maxsize=8)
def _get_cached_analyzer(model_type: str, use_gpu: bool = False):
    """Reuse heavyweight model instances across health checks and comparisons."""
    kwargs = {"use_gpu": use_gpu} if model_type in {"paddle", "bert"} else {}
    return AnalyzerFactory.create_analyzer(model_type, **kwargs)


def _validate_results(texts: list[str], results: list, model_type: str) -> None:
    """Reject silent model failures before they contaminate statistics."""
    if len(results) != len(texts):
        raise RuntimeError(
            f"模型 {model_type} 返回 {len(results)} 条结果，但输入包含 {len(texts)} 条评论"
        )
    valid_labels = {"积极", "消极", "中性"}
    for index, (text, result) in enumerate(zip(texts, results)):
        if not text.strip():
            continue
        if result.label not in valid_labels:
            raise RuntimeError(f"模型 {model_type} 第 {index + 1} 条返回未知标签: {result.label}")
        if not 0.0 <= float(result.score) <= 1.0:
            raise RuntimeError(f"模型 {model_type} 第 {index + 1} 条返回非法分数: {result.score}")
        if result.confidence <= 0 or str(result.analysis).startswith("分析失败"):
            raise RuntimeError(
                f"模型 {model_type} 第 {index + 1} 条推理失败: {result.analysis}"
            )


def preprocess_text(text: str) -> str:
    """Remove URLs/punctuation, tokenize Chinese text, and drop stopwords."""
    import jieba

    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text), flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(word for word in jieba.cut(text) if word not in STOPWORDS and len(word) > 1)


def analyze_sentiment(
    df: pd.DataFrame, model_type: Optional[str] = None, **kwargs
) -> pd.DataFrame:
    """Analyze a comment DataFrame using the requested model.

    Any unavailable optional model falls back to SnowNLP. The returned frame
    always contains ``nlp_result``, ``nlp_score``, and ``clean_text`` columns.
    """
    if "评论内容" not in df.columns:
        raise ValueError(f"DataFrame 缺少 '评论内容' 列。现有列: {list(df.columns)}")

    requested_model = model_type or DEFAULT_SENTIMENT_MODEL
    if requested_model == "auto":
        requested_model = DEFAULT_SENTIMENT_MODEL

    texts = df["评论内容"].astype(str).tolist()
    effective_model = requested_model
    fallback_reason = None
    try:
        analyzer = (
            AnalyzerFactory.create_analyzer(requested_model, **kwargs)
            if kwargs else _get_cached_analyzer(requested_model, False)
        )
        results = analyzer.analyze_batch(texts)
        _validate_results(texts, results, requested_model)
        partial_fallback_count = int(getattr(analyzer, "partial_fallback_count", 0) or 0)
        if partial_fallback_count:
            effective_model = f"{requested_model}+snownlp"
            fallback_reason = (
                f"DeepSeek 结构化输出异常，{partial_fallback_count} 条评论局部降级到 SnowNLP"
            )
    except Exception as exc:
        if requested_model == "snownlp":
            raise
        log.warning("模型 %s 不可用，回退到 SnowNLP: %s", requested_model, exc)
        fallback_reason = str(exc)
        analyzer = _get_cached_analyzer("snownlp", False)
        results = analyzer.analyze_batch(texts)
        _validate_results(texts, results, "snownlp")
        effective_model = "snownlp"
        partial_fallback_count = 0

    output = df.copy()
    output["nlp_result"] = [result.label for result in results]
    output["nlp_score"] = [round(float(result.score), 2) for result in results]
    output["nlp_confidence"] = [round(float(result.confidence), 4) for result in results]
    output["clean_text"] = [preprocess_text(text) for text in texts]

    model_info = analyzer.get_model_info().to_dict()
    output.attrs["analysis_metadata"] = {
        "requested_model": requested_model,
        "effective_model": effective_model,
        "model_name": model_info.get("name", effective_model),
        "model_version": model_info.get("version", ""),
        "model_provider": model_info.get("provider", ""),
        "fallback_used": effective_model != requested_model,
        "fallback_reason": fallback_reason,
        "partial_fallback_count": partial_fallback_count,
    }

    counts = output["nlp_result"].value_counts().to_dict()
    weighted = get_sentiment_stats(output)
    log.info(
        "情感分析完成 — 模型: %s | 唯一文本 %d 条: 积极%d 消极%d 中性%d | 原始声量 %d 条: 积极%d 消极%d 中性%d",
        effective_model,
        len(output),
        counts.get("积极", 0),
        counts.get("消极", 0),
        counts.get("中性", 0),
        weighted["total"], weighted["positive"], weighted["negative"], weighted["neutral"],
    )
    return output


@lru_cache(maxsize=4)
def check_model_health(model_type: str) -> dict:
    """Run real single and batch inference and return a serializable health report."""
    samples = ["这个产品非常好，我很喜欢！", "太差了，非常失望。", "今天发布了新的公告。"]
    started = __import__('time').perf_counter()
    try:
        analyzer = _get_cached_analyzer(model_type, False)
        single = analyzer.analyze(samples[0])
        batch = analyzer.analyze_batch(samples)
        _validate_results([samples[0]], [single], model_type)
        _validate_results(samples, batch, model_type)
        if batch[0].label != "积极" or batch[1].label != "消极":
            raise RuntimeError(
                "基础极性校验未通过: "
                f"正面样本={batch[0].label}, 负面样本={batch[1].label}"
            )
        degraded = False
        detail = "真实单条与批量推理通过"
        if model_type == "hybrid" and hasattr(analyzer, "get_config"):
            config = analyzer.get_config()
            degraded = not bool(config.get("ai_corrector_available"))
            if degraded:
                detail = "本地 SnowNLP + 微博规则可用；AI 语义校正未配置"
        return {
            "model": model_type,
            "available": True,
            "degraded": degraded,
            "detail": detail,
            "elapsed_seconds": round(__import__('time').perf_counter() - started, 3),
            "single": single.to_dict(),
            "batch": [result.to_dict() for result in batch],
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model_type,
            "available": False,
            "degraded": False,
            "detail": str(exc),
            "elapsed_seconds": round(__import__('time').perf_counter() - started, 3),
            "single": None,
            "batch": [],
            "error": str(exc),
        }


def get_sentiment_stats(df: pd.DataFrame) -> dict:
    """Return volume-weighted and unique-text sentiment statistics."""
    if "nlp_result" not in df.columns:
        raise ValueError("DataFrame 缺少 'nlp_result' 列，请先执行情感分析")

    if "duplicate_count" in df.columns:
        weights = pd.to_numeric(df["duplicate_count"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    else:
        weights = pd.Series(1, index=df.index, dtype=int)

    weighted_counts = weights.groupby(df["nlp_result"]).sum().to_dict()
    unique_counts = df["nlp_result"].value_counts().to_dict()
    total = int(weights.sum())
    unique_total = len(df)
    positive = int(weighted_counts.get("积极", 0))
    negative = int(weighted_counts.get("消极", 0))
    neutral = int(weighted_counts.get("中性", 0))
    return {
        "total": total,
        "unique_total": unique_total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "pos_pct": round(positive / total * 100, 1) if total else 0,
        "neg_pct": round(negative / total * 100, 1) if total else 0,
        "neu_pct": round(neutral / total * 100, 1) if total else 0,
        "unique_positive": int(unique_counts.get("积极", 0)),
        "unique_negative": int(unique_counts.get("消极", 0)),
        "unique_neutral": int(unique_counts.get("中性", 0)),
        "unique_pos_pct": round(unique_counts.get("积极", 0) / unique_total * 100, 1) if unique_total else 0,
        "unique_neg_pct": round(unique_counts.get("消极", 0) / unique_total * 100, 1) if unique_total else 0,
        "unique_neu_pct": round(unique_counts.get("中性", 0) / unique_total * 100, 1) if unique_total else 0,
    }


def extract_top_keywords(df: pd.DataFrame, top_n: int = 20) -> list:
    """Extract the most frequent tokens from ``clean_text``."""
    if "clean_text" not in df.columns:
        raise ValueError("DataFrame 缺少 'clean_text' 列，请先执行情感分析")

    words = Counter()
    for index, text in df["clean_text"].dropna().items():
        weight = 1
        if "duplicate_count" in df.columns:
            try:
                weight = max(int(df.at[index, "duplicate_count"]), 1)
            except (TypeError, ValueError):
                weight = 1
        token_counts = Counter(str(text).split())
        words.update({word: count * weight for word, count in token_counts.items()})
    return words.most_common(top_n)


def get_available_models() -> list:
    """Return only models that pass real inference and polarity checks."""
    return [
        model_type
        for model_type in AnalyzerFactory.get_supported_analyzers()
        if check_model_health(model_type)["available"]
    ]


def get_configured_models() -> list:
    """Return models whose required runtime is configured, without inference."""
    models = ["hybrid"]
    if DEEPSEEK_API_KEY:
        models.append("deepseek")
    models.append("snownlp")
    if importlib.util.find_spec("paddle") and importlib.util.find_spec("paddlenlp"):
        models.append("paddle")
    if importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"):
        models.append("bert")
    return models


def get_model_health_report() -> dict:
    """Return health details for every configured sentiment model."""
    return {
        model_type: check_model_health(model_type)
        for model_type in AnalyzerFactory.get_supported_analyzers()
    }


def get_model_info(model_type: str) -> dict:
    """Return serializable metadata for a supported model."""
    try:
        info = _get_cached_analyzer(model_type, False).get_model_info()
        return info.to_dict()
    except Exception as exc:
        return {"error": str(exc)}


def compare_models_on_dataframe(df: pd.DataFrame, sample_size: int = 100) -> dict:
    """Compare available models on a reproducible sample of a DataFrame."""
    if "评论内容" not in df.columns:
        return {"error": "DataFrame 缺少 '评论内容' 列"}

    sample_df = (
        df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    )
    texts = sample_df["评论内容"].astype(str).tolist()
    evaluator = ModelEvaluator()
    evaluator.sample_size = sample_size
    try:
        model_types = get_available_models()
        analyzers = {
            model_type: _get_cached_analyzer(model_type, False)
            for model_type in model_types
        }
        comparison = evaluator.compare_models(texts, model_types, analyzers=analyzers)
        return {
            "sample_size": len(texts),
            "comparison": comparison,
            "agreement": evaluator.calculate_agreement(comparison),
        }
    except Exception as exc:
        return {"error": str(exc)}
