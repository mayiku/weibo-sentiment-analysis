"""
SnowNLP情感分析适配器
兼容现有实现，提供统一的接口
"""

import re
import time
from typing import List, Dict, Any
from snownlp import SnowNLP

from .base import SentimentAnalyzer, SentimentResult, ModelInfo
from .weibo_enhancer import WeiboEnhancer
from .constants import WeiboConstants
from src.logger import get_logger


log = get_logger(__name__)


class SnowNLPAnalyzer(SentimentAnalyzer):
    """SnowNLP情感分析适配器"""

    def __init__(self, use_enhancement: bool = True):
        self.use_enhancement = use_enhancement
        self.enhancer = WeiboEnhancer() if use_enhancement else None
        self.constants = WeiboConstants()

        # 继承现有的反问句检测模式（与原有实现保持一致）
        self.rhetorical_patterns = [
            r'难道.*不', r'怎么.*会', r'谁.*会', r'你觉得.*吗\?', r'这也.*吧\?',
            r'哪里(?!.*(有|在|是|可以|能)).*\?',
            r'.*在哪\??', r'.*什么(好|值得|应该).*\?',
        ]

        self.question_patterns = [
            r'哪里有', r'哪里可以', r'哪里能', r'哪里是',
            r'什么时间', r'什么地点', r'怎么联系', r'如何(操作|使用|联系)',
        ]

        super().__init__()

    def _get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        return ModelInfo(
            name="SnowNLP",
            version="0.12.3",
            provider="SnowNLP Library",
            supports_gpu=False,
            batch_size=1,  # SnowNLP不支持真正的批量处理
            description="基于传统统计方法的中文情感分析模型，支持基本情感识别"
        )

    def analyze(self, text: str, **kwargs) -> SentimentResult:
        """
        使用SnowNLP进行情感分析

        Args:
            text: 待分析文本
            **kwargs: 额外参数

        Returns:
            SentimentResult: 情感分析结果
        """
        if not text or not text.strip():
            return self._create_empty_result("中性")

        start_time = time.time()

        try:
            # 使用SnowNLP进行基础分析
            s = SnowNLP(text)
            base_score = s.sentiments

            # 反问句检测
            base_score = self._detect_rhetorical_question(text, base_score)

            # 自定义词典调整
            adjusted_score = self._apply_custom_dictionary(text, base_score)

            # 分类情感
            sentiment_label = self.classify_sentiment(adjusted_score)

            # 创建基础结果
            result = SentimentResult(
                label=sentiment_label,
                score=round(adjusted_score, 4),
                confidence=0.7,  # SnowNLP的置信度中等
                model_time=0.0,
                analysis=f"SnowNLP分析: {sentiment_label} (得分: {adjusted_score:.3f})"
            )

            # 应用微博场景增强
            if self.use_enhancement and self.enhancer:
                result = self.enhancer.enhance_sentiment(text, result)

            # 计算处理时间
            result.model_time = time.time() - start_time

            return result

        except Exception as e:
            log.error("SnowNLP 分析失败: %s", e)
            raise RuntimeError(f"SnowNLP 分析失败: {e}") from e

    def _detect_rhetorical_question(self, text: str, base_score: float) -> float:
        """
        检测反问句，如果检测到反问句则反转情感得分
        （与原有实现保持一致）

        Args:
            text: 待分析文本
            base_score: 基础情感得分

        Returns:
            float: 调整后的情感得分
        """
        is_rhetorical = False

        # 检测反问模式
        for pattern in self.rhetorical_patterns:
            if re.search(pattern, text):
                # 检查是否为真正的反问句（排除一般疑问句）
                is_question = any(re.search(q, text) for q in self.question_patterns)
                if not is_question:
                    is_rhetorical = True
                    break

        # 如果是反问句，反转情感得分
        return 1 - base_score if is_rhetorical else base_score

    def _apply_custom_dictionary(self, text: str, base_score: float) -> float:
        """
        应用自定义词典调整情感得分
        （与原有实现保持一致，但使用新版词典）

        Args:
            text: 待分析文本
            base_score: 基础情感得分

        Returns:
            float: 调整后的情感得分
        """
        adjusted_score = base_score

        # 负向词汇调整
        for word in self.constants.NEGATIVE_WORDS:
            if word in text:
                adjusted_score = max(0, adjusted_score - 0.2)

        # 正向词汇调整
        for word in self.constants.POSITIVE_WORDS:
            if word in text:
                adjusted_score = min(1, adjusted_score + 0.2)

        return adjusted_score

    def _create_empty_result(self, label: str = "中性", error_msg: str = "") -> SentimentResult:
        """创建空的结果对象"""
        analysis = "分析失败" if error_msg else "文本为空"
        if error_msg:
            analysis += f": {error_msg}"

        return SentimentResult(
            label=label,
            score=0.5,
            confidence=0.0,
            model_time=0.0,
            analysis=analysis
        )

    def analyze_batch(self, texts: List[str], **kwargs) -> List[SentimentResult]:
        """
        批量情感分析（SnowNLP不支持真正的批量处理，逐条处理）

        Args:
            texts: 待分析文本列表
            **kwargs: 额外参数

        Returns:
            List[SentimentResult]: 情感分析结果列表
        """
        results = []
        batch_start_time = time.time()

        for text in texts:
            result = self.analyze(text, **kwargs)
            results.append(result)

        # 计算平均处理时间
        batch_time = time.time() - batch_start_time
        if results:
            avg_time = batch_time / len(results)
            for result in results:
                result.model_time = avg_time

        return results

    def get_config(self) -> Dict[str, Any]:
        """获取分析器配置"""
        return {
            'use_enhancement': self.use_enhancement,
            'model_info': self.model_info.to_dict()
        }


# 兼容性函数 - 保持与原有实现一致
def enhance_sentiment_analysis(text: str) -> float:
    """
    兼容原有函数的增强情感分析
    （保持与原有实现完全一致）

    Args:
        text: 待分析文本

    Returns:
        float: 情感得分(0-1)
    """
    analyzer = SnowNLPAnalyzer(use_enhancement=False)
    result = analyzer.analyze(text)
    return result.score


def classify_sentiment(score: float,
                      pos_threshold: float = 0.6,
                      neg_threshold: float = 0.4) -> str:
    """
    兼容原有函数的情感分类

    Args:
        score: 情感得分
        pos_threshold: 积极阈值
        neg_threshold: 消极阈值

    Returns:
        str: 情感标签
    """
    if score > pos_threshold:
        return '积极'
    elif score < neg_threshold:
        return '消极'
    else:
        return '中性'
