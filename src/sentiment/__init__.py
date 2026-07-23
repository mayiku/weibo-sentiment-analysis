"""
新版情感分析模块 - 多模型支持的统一接口
支持 SnowNLP、PaddleNLP、BERT、Hybrid 四种分析模式
"""

from .base import SentimentAnalyzer, SentimentResult, ModelInfo, AnalyzerFactory
from .evaluator import ModelEvaluator
from .weibo_enhancer import WeiboEnhancer
from .constants import WeiboConstants

__all__ = [
    'SentimentAnalyzer',
    'SentimentResult',
    'ModelInfo',
    'AnalyzerFactory',
    'SnowNLPAnalyzer',
    'PaddleNLPAnalyzer',
    'BertAnalyzer',
    'HybridAnalyzer',
    'ModelEvaluator',
    'WeiboEnhancer',
    'WeiboConstants'
]

from .compatibility import (
    analyze_sentiment,
    check_model_health,
    compare_models_on_dataframe,
    extract_top_keywords,
    get_available_models,
    get_configured_models,
    get_model_health_report,
    get_model_info,
    get_sentiment_stats,
    preprocess_text,
)

__all__.extend([
    'analyze_sentiment',
    'check_model_health',
    'compare_models_on_dataframe',
    'extract_top_keywords',
    'get_available_models',
    'get_configured_models',
    'get_model_health_report',
    'get_model_info',
    'get_sentiment_stats',
    'preprocess_text',
])


def __getattr__(name):
    """Load framework-specific analyzers only when explicitly requested."""
    if name == 'SnowNLPAnalyzer':
        from .snow_analyzer import SnowNLPAnalyzer
        return SnowNLPAnalyzer
    if name == 'PaddleNLPAnalyzer':
        from .paddle_analyzer import PaddleNLPAnalyzer
        return PaddleNLPAnalyzer
    if name == 'BertAnalyzer':
        from .bert_analyzer import BertAnalyzer
        return BertAnalyzer
    if name == 'HybridAnalyzer':
        from .hybrid_analyzer import HybridAnalyzer
        return HybridAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
