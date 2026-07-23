"""
情感分析模块的抽象基类和统一接口定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class SentimentLabel(Enum):
    """情感标签枚举"""
    POSITIVE = "积极"
    NEGATIVE = "消极"
    NEUTRAL = "中性"
    UNKNOWN = "未知"


@dataclass
class SentimentResult:
    """情感分析结果"""
    label: str = "中性"
    score: float = 0.5
    confidence: float = 0.5
    model_time: float = 0.0
    analysis: str = ""
    enhanced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'label': self.label,
            'score': round(self.score, 4),
            'confidence': round(self.confidence, 4),
            'model_time': round(self.model_time, 4),
            'analysis': self.analysis,
            'enhanced': self.enhanced
        }


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    version: str = "1.0.0"
    provider: str = ""
    supports_gpu: bool = False
    batch_size: int = 32
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return model metadata in the format expected by the web UI."""
        return {
            "name": self.name,
            "version": self.version,
            "provider": self.provider,
            "supports_gpu": self.supports_gpu,
            "batch_size": self.batch_size,
            "description": self.description,
            "parameters": self.parameters,
        }


class SentimentAnalyzer(ABC):
    """情感分析器抽象基类"""

    def __init__(self):
        self.model_info = self._get_model_info()

    @abstractmethod
    def _get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        pass

    @abstractmethod
    def analyze(self, text: str, **kwargs) -> SentimentResult:
        """
        单文本情感分析

        Args:
            text: 待分析文本
            **kwargs: 额外参数

        Returns:
            SentimentResult: 情感分析结果
        """
        pass

    def analyze_batch(self, texts: List[str], **kwargs) -> List[SentimentResult]:
        """
        批量情感分析

        Args:
            texts: 待分析文本列表
            **kwargs: 额外参数

        Returns:
            List[SentimentResult]: 情感分析结果列表
        """
        results = []
        total_start = time.time()

        try:
            # 如果模型支持批量处理
            batch_size = kwargs.get('batch_size', self.model_info.batch_size)

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_start = time.time()

                # 逐个分析（子类可以重写实现真正的批量处理）
                for text in batch:
                    result = self.analyze(text, **kwargs)
                    results.append(result)

                batch_time = time.time() - batch_start

                # 记录批量分析时间
                for j in range(len(batch)):
                    results[i + j].model_time = batch_time / len(batch)

        except Exception as e:
            # 批量处理失败，回退到逐条处理
            print(f"批量处理失败，回退到逐条处理: {e}")
            results = [self.analyze(text, **kwargs) for text in texts]

        return results

    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        return self.model_info

    def classify_sentiment(self, score: float,
                          pos_threshold: float = 0.6,
                          neg_threshold: float = 0.4) -> str:
        """
        根据得分分类情感

        Args:
            score: 情感得分（0-1）
            pos_threshold: 积极阈值
            neg_threshold: 消极阈值

        Returns:
            str: 情感标签
        """
        if score >= pos_threshold:
            return SentimentLabel.POSITIVE.value
        elif score <= neg_threshold:
            return SentimentLabel.NEGATIVE.value
        else:
            return SentimentLabel.NEUTRAL.value


class AnalyzerFactory:
    """分析器工厂类"""

    @staticmethod
    def create_analyzer(analyzer_type: str = "hybrid", **kwargs) -> SentimentAnalyzer:
        """
        创建情感分析器实例

        Args:
            analyzer_type: 分析器类型
                - "snownlp": SnowNLP分析器
                - "paddle": PaddleNLP分析器（默认）
                - "bert": BERT分析器
                - "hybrid": 混合分析器
            **kwargs: 分析器参数

        Returns:
            SentimentAnalyzer: 分析器实例

        Raises:
            ValueError: 不支持的模型类型
        """
        analyzer_type = analyzer_type.lower()

        if analyzer_type == "snownlp":
            from .snow_analyzer import SnowNLPAnalyzer
            return SnowNLPAnalyzer(**kwargs)

        elif analyzer_type == "paddle":
            from .paddle_analyzer import PaddleNLPAnalyzer
            return PaddleNLPAnalyzer(**kwargs)

        elif analyzer_type == "bert":
            from .bert_analyzer import BertAnalyzer
            return BertAnalyzer(**kwargs)

        elif analyzer_type == "hybrid":
            from .hybrid_analyzer import HybridAnalyzer
            return HybridAnalyzer(**kwargs)

        elif analyzer_type == "deepseek":
            from .deepseek_analyzer import DeepSeekSentimentAnalyzer
            return DeepSeekSentimentAnalyzer(**kwargs)

        else:
            raise ValueError(f"不支持的模型类型: {analyzer_type}")

    @staticmethod
    def get_supported_analyzers() -> List[str]:
        """获取支持的分析器类型列表"""
        return ["hybrid", "deepseek", "snownlp", "paddle", "bert"]

    @staticmethod
    def get_analyzer_info(analyzer_type: str) -> Dict[str, Any]:
        """获取分析器信息"""
        try:
            analyzer = AnalyzerFactory.create_analyzer(analyzer_type)
            info = analyzer.get_model_info()
            return {
                'type': analyzer_type,
                'name': info.name,
                'version': info.version,
                'provider': info.provider,
                'supports_gpu': info.supports_gpu,
                'batch_size': info.batch_size,
                'description': info.description
            }
        except Exception as e:
            return {
                'type': analyzer_type,
                'error': str(e)
            }
