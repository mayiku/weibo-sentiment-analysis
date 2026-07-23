"""
PaddleNLP情感分析器
使用PaddleNLP UIE-Sentiment模型，专门针对中文情感分析优化
"""

import os
import time
import warnings
from typing import List, Dict, Any

from .base import SentimentAnalyzer, SentimentResult, ModelInfo
from .weibo_enhancer import WeiboEnhancer
from config import SENTIMENT_GPU_ENABLED
from src.logger import get_logger


log = get_logger(__name__)

# Paddle 3.x defaults to the new PIR format (``inference.json``), while the
# installed PaddleNLP 2.6 Taskflow loader expects ``inference.pdmodel``.
# Force the legacy static format before Paddle is imported.
os.environ.setdefault("FLAGS_enable_pir_api", "0")


try:
    import numpy as np
    import paddle
    from paddlenlp import Taskflow
    from paddlenlp.data import JiebaTokenizer, Vocab
    from paddlenlp.taskflow.models import LSTMModel
    from paddlenlp.utils.env import PPNLP_HOME
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    warnings.warn("PaddleNLP不可用，请安装paddlenlp和paddlepaddle")


class PaddleNLPAnalyzer(SentimentAnalyzer):
    """PaddleNLP情感分析器"""

    def __init__(self, use_gpu: bool = None, use_enhancement: bool = True):
        self.use_gpu = use_gpu if use_gpu is not None else SENTIMENT_GPU_ENABLED
        self.use_enhancement = use_enhancement
        self.enhancer = WeiboEnhancer() if use_enhancement else None

        # 延迟加载模型
        self._taskflow = None
        self._model = None
        self._tokenizer = None
        self._pad_token_id = 0
        self._is_loaded = False

        super().__init__()

    def _get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        return ModelInfo(
            name="PaddleNLP-BiLSTM-Sentiment",
            version="2.6.1",
            provider="PaddlePaddle",
            supports_gpu=True,
            batch_size=32,
            description="PaddleNLP 官方中文二分类情感模型，支持批量推理"
        )

    def _load_model(self):
        """加载PaddleNLP模型"""
        if self._is_loaded:
            return

        try:
            if not PADDLE_AVAILABLE:
                raise ImportError("PaddleNLP未安装，请运行: pip install paddlenlp paddlepaddle")

            # 设置GPU设备
            if self.use_gpu and paddle.device.is_compiled_with_cuda():
                paddle.set_device('gpu')
            else:
                paddle.set_device('cpu')
                self.use_gpu = False

            model_dir = os.path.join(
                PPNLP_HOME, "taskflow", "sentiment_analysis", "bilstm"
            )
            model_path = os.path.join(model_dir, "model_state.pdparams")
            vocab_path = os.path.join(model_dir, "vocab.txt")

            # Let Taskflow fetch the official assets on a fresh installation.
            # Its PaddleNLP 2.6 static wrapper is incompatible with Paddle 3.x,
            # so inference below deliberately uses the same weights in dynamic mode.
            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                try:
                    Taskflow("sentiment_analysis", device_id=-1)
                except Exception:
                    pass
            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                raise FileNotFoundError("官方 BiLSTM 模型资源下载失败")

            vocab = Vocab.load_vocabulary(
                vocab_path, unk_token="[UNK]", pad_token="[PAD]"
            )
            self._pad_token_id = vocab.to_indices("[PAD]")
            self._tokenizer = JiebaTokenizer(vocab)
            self._model = LSTMModel(
                len(vocab),
                2,
                direction="bidirect",
                padding_idx=self._pad_token_id,
                pooling_type="max",
            )
            self._model.set_dict(paddle.load(model_path))
            self._model.eval()

            self._is_loaded = True
            print(f"PaddleNLP模型加载成功，设备: {'GPU' if self.use_gpu else 'CPU'}")

        except Exception as e:
            raise RuntimeError(f"PaddleNLP模型加载失败: {e}")

    def _predict(self, texts: List[str]) -> List[dict]:
        """Run the official BiLSTM weights through Paddle dynamic mode."""
        sequences = [self._tokenizer.encode(text) for text in texts]
        lengths = np.asarray([len(sequence) for sequence in sequences], dtype="int64")
        token_ids = np.full(
            (len(sequences), int(lengths.max())), self._pad_token_id, dtype="int64"
        )
        for index, sequence in enumerate(sequences):
            token_ids[index, : len(sequence)] = sequence

        with paddle.no_grad():
            indices, probabilities = self._model(
                paddle.to_tensor(token_ids), paddle.to_tensor(lengths)
            )
        indices = indices.tolist()
        probabilities = probabilities.numpy().tolist()
        label_map = {0: "negative", 1: "positive"}
        return [
            {
                "text": text,
                "label": label_map[index],
                "score": float(max(probability)),
            }
            for text, index, probability in zip(texts, indices, probabilities)
        ]

    def analyze(self, text: str, **kwargs) -> SentimentResult:
        """
        使用PaddleNLP进行情感分析

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

            # 执行情感分析
            results = self._predict([text])

            if not results:
                return self._create_empty_result("中性", "分析结果为空")

            result_data = results[0]

            # 解析PaddleNLP结果
            sentiment_label = str(result_data.get('label', '')).lower()
            confidence = float(result_data.get('score', result_data.get('confidence', 0.0)))

            # 转换标签格式
            if sentiment_label == 'positive':
                label = "积极"
                score = confidence
            elif sentiment_label == 'negative':
                label = "消极"
                score = 1 - confidence  # PaddleNLP的negative置信度需要转换
            else:
                label = "中性"
                score = 0.5

            # 确保分数在合理范围内
            score = max(0.0, min(1.0, score))

            # 计算处理时间
            model_time = time.time() - start_time

            # 创建基础结果
            result = SentimentResult(
                label=label,
                score=round(score, 4),
                confidence=round(confidence, 4),
                model_time=model_time,
                analysis=f"PaddleNLP分析: {label} (置信度: {confidence:.3f})"
            )

            # 应用微博场景增强
            if self.use_enhancement and self.enhancer:
                result = self.enhancer.enhance_sentiment(text, result)

            return result

        except Exception as e:
            log.error("PaddleNLP 分析失败: %s", e)
            raise RuntimeError(f"PaddleNLP 分析失败: {e}") from e

    def analyze_batch(self, texts: List[str], **kwargs) -> List[SentimentResult]:
        """
        批量情感分析 - 使用PaddleNLP的批量处理能力

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

            # 过滤空文本
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return [self._create_empty_result("中性") for _ in texts]

            # 执行批量分析
            batch_results = self._predict(valid_texts)

            # 构建完整的结果列表
            results = []
            valid_idx = 0

            for text in texts:
                if not text or not text.strip():
                    # 空文本结果
                    results.append(self._create_empty_result("中性"))
                else:
                    # 处理有效文本结果
                    if valid_idx < len(batch_results):
                        result_data = batch_results[valid_idx]
                        valid_idx += 1

                        # 解析结果
                        sentiment_label = str(result_data.get('label', '')).lower()
                        confidence = float(result_data.get('score', result_data.get('confidence', 0.0)))

                        # 转换标签格式
                        if sentiment_label == 'positive':
                            label = "积极"
                            score = confidence
                        elif sentiment_label == 'negative':
                            label = "消极"
                            score = 1 - confidence
                        else:
                            label = "中性"
                            score = 0.5

                        score = max(0.0, min(1.0, score))

                        results.append(SentimentResult(
                            label=label,
                            score=round(score, 4),
                            confidence=round(confidence, 4),
                            model_time=0.0,  # 批量处理时间在最后统一计算
                            analysis=f"PaddleNLP分析: {label} (置信度: {confidence:.3f})"
                        ))
                    else:
                        results.append(self._create_empty_result("中性", "分析结果缺失"))

            # 计算平均处理时间
            batch_time = time.time() - batch_start_time
            if results:
                avg_time = batch_time / len(valid_texts)
                for result in results:
                    if result.score != 0.5:  # 仅对有效结果设置时间
                        result.model_time = avg_time

            # 应用微博场景增强
            if self.use_enhancement and self.enhancer:
                results = self.enhancer.batch_enhance(texts, results)

            return results

        except Exception as e:
            log.error("PaddleNLP 批量分析失败: %s", e)
            raise RuntimeError(f"PaddleNLP 批量分析失败: {e}") from e

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
            'use_gpu': self.use_gpu,
            'use_enhancement': self.use_enhancement,
            'model_loaded': self._is_loaded,
            'model_info': self.model_info.to_dict()
        }

    def __del__(self):
        """清理资源"""
        if self._taskflow or self._model:
            try:
                # PaddleNLP Taskflow会自动管理资源
                pass
            except:
                pass
