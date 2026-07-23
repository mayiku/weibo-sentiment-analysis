"""
情感分析模块 — 兼容新版多模型架构的接口
支持 SnowNLP、PaddleNLP、BERT、Hybrid 四种分析模式
保持与原有实现完全兼容
"""

import re
import pandas as pd
from typing import List, Dict, Any, Optional

from config import SENTIMENT_POS_THRESHOLD, SENTIMENT_NEG_THRESHOLD, STOPWORDS
from config import DEFAULT_SENTIMENT_MODEL
from src.logger import get_logger

# 导入新版分析器（如果可用）
try:
    from sentiment import (
        SentimentAnalyzer, AnalyzerFactory, ModelEvaluator,
        SentimentResult, WeiboEnhancer
    )
    NEW_ANALYZERS_AVAILABLE = True
except ImportError:
    NEW_ANALYZERS_AVAILABLE = False
    print("新版情感分析模块不可用，使用原生SnowNLP实现")

log = get_logger(__name__)

# ── 原有实现（保持完全兼容）─────────────────────────────────────────

# 情感词典（保留原有实现）
NEGATIVE_WORDS = [
    '屎', '垃圾', '丑', '差', '烂', '恶心', '无语', '坑爹', '离谱', '失望',
    '垃圾', '恶心', '辣鸡', '吐了', '恶心心', '下头', '无语子',
]

POSITIVE_WORDS = [
    '棒', '好', '美', '强', '赞', '厉害', '优秀', '牛', '绝了', '完美', '好看',
    '绝绝子', 'yyds', 'YYDS', '封神', '神仙', '牛逼', '给力', '爱了',
]

# 反问句检测模式（保留原有实现）
RHETORICAL_PATTERNS = [
    r'难道.*不', r'怎么.*会', r'谁.*会', r'你觉得.*吗\?', r'这也.*吧\?',
    r'哪里(?!.*(有|在|是|可以|能)).*\?',
    r'.*在哪\??', r'.*什么(好|值得|应该).*\?',
]

QUESTION_PATTERNS = [
    r'哪里有', r'哪里可以', r'哪里能', r'哪里是',
    r'什么时间', r'什么地点', r'怎么联系', r'如何(操作|使用|联系)',
]


def enhance_sentiment_analysis(text: str) -> float:
    """
    增强版情感分析 — SnowNLP + 反问检测 + 自定义词典
    （保持与原有实现完全一致）

    返回: 0.0 (消极) ~ 1.0 (积极) 的情感得分
    """
    from snownlp import SnowNLP

    s = SnowNLP(text)
    base_score = s.sentiments

    # 反问句检测
    is_rhetorical = False
    for pattern in RHETORICAL_PATTERNS:
        if re.search(pattern, text):
            is_question = any(re.search(q, text) for q in QUESTION_PATTERNS)
            if not is_question:
                is_rhetorical = True
                break

    adjusted_score = 1 - base_score if is_rhetorical else base_score

    # 自定义词典调整
    for word in NEGATIVE_WORDS:
        if word in text:
            adjusted_score = max(0, adjusted_score - 0.2)
    for word in POSITIVE_WORDS:
        if word in text:
            adjusted_score = min(1, adjusted_score + 0.2)

    return adjusted_score


def classify_sentiment(score: float) -> str:
    """将情感得分映射为标签（保持原有实现）"""
    if score > SENTIMENT_POS_THRESHOLD:
        return '积极'
    elif score < SENTIMENT_NEG_THRESHOLD:
        return '消极'
    else:
        return '中性'


def preprocess_text(text: str) -> str:
    """
    预处理文本 — URL移除 + jieba分词 + 去停用词
    返回分词后的空格分隔字符串
    """
    import jieba

    # 移除 URL 和特殊字符
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)

    # jieba 分词 + 去停用词
    words = jieba.cut(text)
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]

    return " ".join(words)


# ── 新版多模型支持功能──────────────────────────────────────────────

def analyze_sentiment(df: pd.DataFrame, model_type: str = None, **kwargs) -> pd.DataFrame:
    """
    新版情感分析函数 — 支持多模型选择

    输入: 包含 '评论内容' 列的 DataFrame
    输出: 新增 nlp_result, nlp_score, clean_text 列

    Args:
        df: 包含评论内容的DataFrame
        model_type: 模型类型，可选值:
            "snownlp" - 原有SnowNLP实现（默认）
            "paddle" - PaddleNLP模型
            "bert" - BERT模型
            "hybrid" - 混合模型
            "auto" - 根据配置自动选择
        **kwargs: 分析器参数

    Returns:
        pd.DataFrame: 包含分析结果的DataFrame
    """
    if '评论内容' not in df.columns:
        raise ValueError(f"DataFrame 缺少 '评论内容' 列。现有列: {list(df.columns)}")

    # 确定模型类型
    if model_type is None:
        model_type = DEFAULT_SENTIMENT_MODEL

    if model_type == "auto":
        model_type = DEFAULT_SENTIMENT_MODEL

    log.info(f"开始情感分析 — 模型: {model_type}, 共 %d 条评论", len(df))

    # 优先使用新版分析器
    if NEW_ANALYZERS_AVAILABLE and model_type != "snownlp":
        try:
            return _analyze_with_new_models(df, model_type, **kwargs)
        except Exception as e:
            log.warning(f"新版模型 {model_type} 分析失败，回退到SnowNLP: {e}")
            model_type = "snownlp"

    # 使用原有的SnowNLP实现
    return _analyze_with_snownlp(df)


def _analyze_with_new_models(df: pd.DataFrame, model_type: str, **kwargs) -> pd.DataFrame:
    """使用新版模型进行情感分析"""
    df = df.copy()
    df['nlp_result'] = ''
    df['nlp_score'] = 0.0
    df['clean_text'] = ''

    # 创建分析器
    analyzer = AnalyzerFactory.create_analyzer(model_type, **kwargs)

    # 提取评论内容
    texts = df['评论内容'].astype(str).tolist()

    # 批量分析
    results = analyzer.analyze_batch(texts)

    pos = neg = neu = fail = 0
    for idx, (text, result) in enumerate(zip(texts, results)):
        try:
            df.loc[idx, 'nlp_score'] = round(result.score, 2)
            df.loc[idx, 'nlp_result'] = result.label
            df.loc[idx, 'clean_text'] = preprocess_text(text)

            if result.label == '积极':
                pos += 1
            elif result.label == '消极':
                neg += 1
            else:
                neu += 1
        except Exception as e:
            log.warning("分析第 %d 行出错: %s", idx, e)
            df.loc[idx, 'nlp_result'] = '分析失败'
            df.loc[idx, 'nlp_score'] = 0.0
            df.loc[idx, 'clean_text'] = ''
            fail += 1

    log.info("情感分析完成 — 积极: %d, 消极: %d, 中性: %d, 失败: %d", pos, neg, neu, fail)
    return df


def _analyze_with_snownlp(df: pd.DataFrame) -> pd.DataFrame:
    """使用原有SnowNLP实现进行情感分析"""
    df = df.copy()
    df['nlp_result'] = ''
    df['nlp_score'] = 0.0
    df['clean_text'] = ''

    pos = neg = neu = 0
    for idx, row in df.iterrows():
        try:
            comment = str(row['评论内容'])
            score = enhance_sentiment_analysis(comment)
            label = classify_sentiment(score)

            df.loc[idx, 'nlp_score'] = round(score, 2)
            df.loc[idx, 'nlp_result'] = label
            df.loc[idx, 'clean_text'] = preprocess_text(comment)

            if label == '积极':
                pos += 1
            elif label == '消极':
                neg += 1
            else:
                neu += 1
        except Exception as e:
            log.warning("分析第 %d 行出错: %s", idx, e)
            df.loc[idx, 'nlp_result'] = '分析失败'
            df.loc[idx, 'nlp_score'] = 0.0
            df.loc[idx, 'clean_text'] = ''

    log.info("情感分析完成 — 积极: %d, 消极: %d, 中性: %d", pos, neg, neu)
    return df


def get_sentiment_stats(df: pd.DataFrame) -> dict:
    """从分析结果中提取统计数据（保持原有接口）"""
    total = len(df)
    counts = df['nlp_result'].value_counts().to_dict()
    return {
        'total': total,
        'positive': counts.get('积极', 0),
        'negative': counts.get('消极', 0),
        'neutral': counts.get('中性', 0),
        'pos_pct': round(counts.get('积极', 0) / total * 100, 1) if total else 0,
        'neg_pct': round(counts.get('消极', 0) / total * 100, 1) if total else 0,
        'neu_pct': round(counts.get('中性', 0) / total * 100, 1) if total else 0,
    }


def extract_top_keywords(df: pd.DataFrame, top_n: int = 20) -> list:
    """从 clean_text 列提取高频关键词（保持原有接口）"""
    from collections import Counter

    all_words = []
    for text in df['clean_text']:
        if text:
            all_words.extend(str(text).split())

    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)
    log.info("提取 Top %d 关键词: %s", top_n,
             [(w, c) for w, c in top_words[:5]] + ['...'])
    return top_words


# ── 新增功能：模型评估和对比───────────────────────────────────────

def compare_models_on_dataframe(df: pd.DataFrame, sample_size: int = 100) -> dict:
    """
    在DataFrame数据上对比不同模型的表现

    Args:
        df: 包含评论内容的DataFrame
        sample_size: 抽样数量

    Returns:
        dict: 模型对比结果
    """
    if not NEW_ANALYZERS_AVAILABLE:
        return {"error": "新版分析模块不可用"}

    try:
        # 抽样数据
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df

        texts = sample_df['评论内容'].astype(str).tolist()

        # 创建评估器
        evaluator = ModelEvaluator()
        evaluator.sample_size = sample_size

        # 对比模型
        comparison = evaluator.compare_models(texts)

        return {
            "sample_size": len(texts),
            "comparison": comparison,
            "agreement": evaluator.calculate_agreement(comparison)
        }

    except Exception as e:
        return {"error": str(e)}


def get_available_models() -> list:
    """获取可用的情感分析模型列表"""
    if not NEW_ANALYZERS_AVAILABLE:
        return ["snownlp"]

    try:
        available_models = []
        for model_type in AnalyzerFactory.get_supported_analyzers():
            try:
                analyzer = AnalyzerFactory.create_analyzer(model_type)
                # 简单测试模型是否可用
                analyzer.analyze("测试")
                available_models.append(model_type)
            except:
                pass
        return available_models
    except:
        return ["snownlp"]


def get_model_info(model_type: str) -> dict:
    """获取模型详细信息"""
    if not NEW_ANALYZERS_AVAILABLE or model_type == "snownlp":
        return {
            "name": "SnowNLP",
            "description": "基于传统统计方法的情感分析",
            "provider": "SnowNLP Library",
            "supports_gpu": False
        }

    try:
        analyzer = AnalyzerFactory.create_analyzer(model_type)
        info = analyzer.get_model_info()
        return info.to_dict()
    except Exception as e:
        return {"error": str(e)}


# 保持原有的导入兼容性
if __name__ == "__main__":
    # 简单的功能测试
    test_df = pd.DataFrame({
        '评论内容': ['这个电影很好看', '太糟糕了', '一般般']
    })

    result_df = analyze_sentiment(test_df, model_type="snownlp")
    print("测试结果:")
    print(result_df[['评论内容', 'nlp_result', 'nlp_score']])