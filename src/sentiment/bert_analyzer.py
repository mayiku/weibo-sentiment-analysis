"""
BERT情感分析器
使用HuggingFace的Chinese-RoBERTa模型进行情感分析
"""

import time
import warnings
from typing import List, Dict, Any

from .base import SentimentAnalyzer, SentimentResult, ModelInfo
from .weibo_enhancer import WeiboEnhancer
from config import SENTIMENT_GPU_ENABLED, SENTIMENT_MAX_TEXT_LENGTH
from src.logger import get_logger


log = get_logger(__name__)


TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                             pipeline, Pipeline)
    import numpy as np

    # 检查 PyTorch 是否正常加载
    _ = torch.__version__
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"[WARNING] Transformers/BERT 不可用: {e}")
    print("[INFO] BERT模型将自动降级使用SnowNLP")


class BertAnalyzer(SentimentAnalyzer):
    """BERT情感分析器"""

    def __init__(self, model_name: str = None, use_gpu: bool = None, use_enhancement: bool = True):
        self.model_name = model_name or "uer/roberta-base-finetuned-jd-binary-chinese"
        self.use_gpu = use_gpu if use_gpu is not None else SENTIMENT_GPU_ENABLED
        self.use_enhancement = use_enhancement
        self.enhancer = WeiboEnhancer() if use_enhancement else None
        self.max_length = SENTIMENT_MAX_TEXT_LENGTH

        # 延迟加载模型
        self._tokenizer = None
        self._model = None
        self._pipeline = None
        self._is_loaded = False

        super().__init__()

    def _get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        return ModelInfo(
            name="BERT-Chinese-Sentiment",
            version="4.35.0",
            provider="HuggingFace",
            supports_gpu=True,
            batch_size=16,  # BERT模型批量大小较小
            description="基于Chinese-RoBERTa的细粒度情感分析模型，准确率高但资源消耗较大",
            parameters={
                'model_name': self.model_name,
                'max_length': self.max_length
            }
        )

    def _load_model(self):
        """加载BERT模型"""
        if self._is_loaded:
            return

        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers未安装，请运行: pip install transformers torch")

            # 设置设备
            device = 0 if self.use_gpu and torch.cuda.is_available() else -1
            if device == -1:
                self.use_gpu = False

            # 创建情感分析pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=device,
                max_length=self.max_length,
                truncation=True
            )

            self._is_loaded = True
            print(f"BERT模型加载成功，模型: {self.model_name}, 设备: {'GPU' if self.use_gpu else 'CPU'}")

        except Exception as e:
            raise RuntimeError(f"BERT模型加载失败: {e}")

    def analyze(self, text: str, **kwargs) -> SentimentResult:
        """
        使用BERT进行情感分析

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
            # 确保模型已加载
            self._load_model()

            # 缩短过长的文本（保持语义完整性）
            if len(text) > self.max_length:
                text = self._truncate_text(text)

            # 执行情感分析
            results = self._pipeline([text])

            if not results:
                return self._create_empty_result("中性", "分析结果为空")

            result_data = results[0]

            # 解析BERT结果
            sentiment_label = result_data.get('label', '').lower()
            confidence = result_data.get('score', 0.5)

            # 转换标签格式（不同模型标签可能不同）
            label, score = self._convert_sentiment_result(sentiment_label, confidence)

            # 计算处理时间
            model_time = time.time() - start_time

            # 创建基础结果
            result = SentimentResult(
                label=label,
                score=round(score, 4),
                confidence=round(confidence, 4),
                model_time=model_time,
                analysis=f"BERT分析: {label} (置信度: {confidence:.3f})"
            )

            # 应用微博场景增强
            if self.use_enhancement and self.enhancer:
                result = self.enhancer.enhance_sentiment(text, result)

            return result

        except Exception as e:
            log.error("BERT 分析失败: %s", e)
            raise RuntimeError(f"BERT 分析失败: {e}") from e

    def analyze_batch(self, texts: List[str], **kwargs) -> List[SentimentResult]:
        """
        批量情感分析 - 使用BERT的批量处理能力

        Args:
            texts: 待分析文本列表
            **kwargs: 额外参数

        Returns:
            List[SentimentResult]: 情感分析结果列表
        """
        if not texts:
            return []

        batch_start_time = time.time()

        try:
            # 确保模型已加载
            self._load_model()

            # 过滤空文本并截断长文本
            valid_texts = []
            valid_indices = []

            for i, text in enumerate(texts):
                if text and text.strip():
                    if len(text) > self.max_length:
                        text = self._truncate_text(text)
                    valid_texts.append(text)
                    valid_indices.append(i)

            if not valid_texts:
                return [self._create_empty_result("中性") for _ in texts]

            # 执行批量分析
            batch_results = self._pipeline(valid_texts, batch_size=kwargs.get('batch_size', 16))

            # 构建完整的结果列表
            results = [self._create_empty_result("中性") for _ in texts]

            for idx, result_data in zip(valid_indices, batch_results):
                sentiment_label = result_data.get('label', '').lower()
                confidence = result_data.get('score', 0.5)

                label, score = self._convert_sentiment_result(sentiment_label, confidence)

                results[idx] = SentimentResult(
                    label=label,
                    score=round(score, 4),
                    confidence=round(confidence, 4),
                    model_time=0.0,  # 批量处理时间在最后统一计算
                    analysis=f"BERT分析: {label} (置信度: {confidence:.3f})"
                )

            # 计算平均处理时间
            batch_time = time.time() - batch_start_time
            if valid_texts:
                avg_time = batch_time / len(valid_texts)
                for idx in valid_indices:
                    results[idx].model_time = avg_time

            # 应用微博场景增强
            if self.use_enhancement and self.enhancer:
                results = self.enhancer.batch_enhance(texts, results)

            return results

        except Exception as e:
            log.error("BERT 批量分析失败: %s", e)
            raise RuntimeError(f"BERT 批量分析失败: {e}") from e

    def _convert_sentiment_result(self, label: str, confidence: float) -> tuple[str, float]:
        """
        转换BERT模型输出为统一格式

        Args:
            label: 原始标签
            confidence: 置信度

        Returns:
            tuple[str, float]: (情感标签, 得分)
        """
        # 不同BERT模型可能有不同的标签命名
        positive_keywords = ['positive', 'pos', '正向', '积极', '好评']
        negative_keywords = ['negative', 'neg', '负向', '消极', '差评']

        label_lower = label.lower()

        if any(keyword in label_lower for keyword in positive_keywords):
            return "积极", confidence
        elif any(keyword in label_lower for keyword in negative_keywords):
            return "消极", 1 - confidence
        else:
            return "中性", 0.5

    def _truncate_text(self, text: str) -> str:
        """智能截断长文本，尽量保持语义完整性"""
        if len(text) <= self.max_length:
            return text

        # 尝试在句子边界处截断
        sentences = text.split('。')
        truncated = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 添加句号
            sentence_with_dot = sentence + '。'
            sentence_length = len(sentence_with_dot)

            if current_length + sentence_length <= self.max_length:
                truncated.append(sentence)
                current_length += sentence_length
            else:
                # 如果当前句子已经过长，进行单词级截断
                if current_length == 0:
                    words = sentence.split()
                    truncated_words = []
                    word_length = 0

                    for word in words:
                        if word_length + len(word) + 1 <= self.max_length:
                            truncated_words.append(word)
                            word_length += len(word) + 1
                        else:
                            break

                    truncated.append(' '.join(truncated_words))
                break

        result = '。'.join(truncated)
        if result and not result.endswith('。'):
            result += '。'

        return result[:self.max_length]

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

    def get_config(self) -> Dict[str, Any]:
        """获取分析器配置"""
        return {
            'model_name': self.model_name,
            'use_gpu': self.use_gpu,
            'use_enhancement': self.use_enhancement,
            'model_loaded': self._is_loaded,
            'model_info': self.model_info.to_dict()
        }

    def __del__(self):
        """清理GPU内存"""
        if self._model and self.use_gpu:
            try:
                self._model.cpu()
                torch.cuda.empty_cache()
            except:
                pass
