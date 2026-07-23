"""
混合模型情感分析器
结合SnowNLP初步分析和AI语义校正，提高微博语言识别准确率
"""

import time
from typing import List, Dict, Any

from .base import SentimentAnalyzer, SentimentResult, ModelInfo, AnalyzerFactory
from .weibo_enhancer import WeiboEnhancer
from src.logger import get_logger


log = get_logger(__name__)


try:
    # 尝试导入AI校正模块（如果可用）
    from src.ai_corrector import AICorrector
    AI_CORRECTOR_AVAILABLE = True
except ImportError:
    AI_CORRECTOR_AVAILABLE = False
    log.info("Hybrid 本地规则模式可用；AI 语义校正模块未配置")


class HybridAnalyzer(SentimentAnalyzer):
    """混合模型情感分析器"""

    def __init__(self, ai_correction_threshold: float = 0.3, use_enhancement: bool = True):
        self.ai_correction_threshold = ai_correction_threshold  # 需要AI校正的置信度阈值
        self.use_enhancement = use_enhancement
        self.enhancer = WeiboEnhancer() if use_enhancement else None

        # 初始化基础分析器（SnowNLP作为基准）
        self.base_analyzer = AnalyzerFactory.create_analyzer("snownlp", use_enhancement=False)

        # AI校正器（如果可用）
        self.ai_corrector = None
        if AI_CORRECTOR_AVAILABLE:
            try:
                self.ai_corrector = AICorrector()
            except Exception as e:
                print(f"AI校正器初始化失败: {e}")
                self.ai_corrector = None

        super().__init__()

    def _get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        if self.ai_corrector is None:
            return ModelInfo(
                name="Hybrid-Enhanced-SnowNLP",
                version="1.0.0",
                provider="SnowNLP + Weibo Rules",
                supports_gpu=False,
                batch_size=8,
                description="本地混合模型：SnowNLP 结合微博热词、表情、反问及讽刺规则",
            )
        return ModelInfo(
            name="Hybrid-Sentiment-Analyzer",
            version="1.0.0",
            provider="SnowNLP + AI校正",
            supports_gpu=self.ai_corrector is not None,
            batch_size=8,  # 混合模型处理较慢
            description="结合SnowNLP快速分析和AI语义校正的混合模型，显著提升微博语言识别准确率"
        )

    def analyze(self, text: str, **kwargs) -> SentimentResult:
        """
        混合模型情感分析流程：
        1. SnowNLP初步分析
        2. 低置信度或特定场景文本启用AI校正
        3. 综合两个模型结果
        4. 微博场景增强

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
            # 第一步：SnowNLP快速分析
            base_result = self.base_analyzer.analyze(text)
            base_time = time.time() - start_time

            # 第二步：判断是否需要AI校正
            needs_correction = self._need_ai_correction(text, base_result)

            if needs_correction and self.ai_corrector:
                # AI语义校正
                ai_result = self._apply_ai_correction(text, base_result)
                final_result = self._combine_results(base_result, ai_result)
                correction_applied = True
            else:
                final_result = base_result
                correction_applied = False

            # 第三步：微博场景增强
            if self.use_enhancement and self.enhancer:
                final_result = self.enhancer.enhance_sentiment(text, final_result)

            # 更新处理时间和分析说明
            final_result.model_time = time.time() - start_time
            final_result.analysis = self._generate_analysis(
                base_result, correction_applied, final_result.analysis
            )

            return final_result

        except Exception as e:
            log.error("混合模型分析失败: %s", e)
            raise RuntimeError(f"混合模型分析失败: {e}") from e

    def analyze_batch(self, texts: List[str], **kwargs) -> List[SentimentResult]:
        """
        批量混合模型情感分析
        对每个样本单独判断是否需要AI校正

        Args:
            texts: 待分析文本列表
            **kwargs: 额外参数

        Returns:
            List[SentimentResult]: 情感分析结果列表
        """
        if not texts:
            return []

        batch_start_time = time.time()
        results = []

        try:
            # 第一步：批量SnowNLP分析
            base_results = self.base_analyzer.analyze_batch(texts)

            # 第二步：逐个判断是否需要AI校正
            for i, (text, base_result) in enumerate(zip(texts, base_results)):
                if not text or not text.strip():
                    results.append(base_result)
                    continue

                # 判断是否需要AI校正
                needs_correction = self._need_ai_correction(text, base_result)

                if needs_correction and self.ai_corrector:
                    # AI语义校正
                    ai_result = self._apply_ai_correction(text, base_result)
                    final_result = self._combine_results(base_result, ai_result)
                    correction_applied = True
                else:
                    final_result = base_result
                    correction_applied = False

                # 微博场景增强
                if self.use_enhancement and self.enhancer:
                    final_result = self.enhancer.enhance_sentiment(text, final_result)

                results.append(final_result)

            # 统一设置批量处理时间
            batch_time = time.time() - batch_start_time
            if results:
                avg_time = batch_time / len(results)
                for result in results:
                    result.model_time = avg_time

            return results

        except Exception as e:
            log.error("混合模型批量分析失败: %s", e)
            raise RuntimeError(f"混合模型批量分析失败: {e}") from e

    def _need_ai_correction(self, text: str, base_result: SentimentResult) -> bool:
        """
        判断文本是否需要AI校正，基于多种条件：
        1. SnowNLP置信度低
        2. 包含微博网络语言特征
        3. 情感模糊的文本

        Args:
            text: 待分析文本
            base_result: SnowNLP分析结果

        Returns:
            bool: 是否需要AI校正
        """
        if not self.ai_corrector:
            return False

        # 条件1：SnowNLP置信度低（0.4-0.6区间）
        if 0.4 <= base_result.score <= 0.6:
            return True

        # 条件2：包含微博网络语言特征
        if self.enhancer and self.enhancer.constants.contains_trending_words(text):
            return True

        # 条件3：粉圈或体育文本
        if (self.enhancer and
            (self.enhancer.constants.is_fan_circle_text(text) or
             self.enhancer.constants.is_sports_text(text))):
            return True

        # 条件4：包含特定模式（讽刺、反问等）
        if self._contains_ambiguous_patterns(text):
            return True

        return False

    def _contains_ambiguous_patterns(self, text: str) -> bool:
        """检测情感模糊的模式"""
        ambiguous_patterns = [
            # 讽刺模式
            r'真.*不错', r'太.*好了', r'棒棒哒',
            # 反问模式
            r'难道.*不', r'怎么.*会', r'谁.*会',
            # 不确定模式
            r'啊这', r'就这', r'还好', r'还行', r'一般',
            # 极端程度词
            r'无敌', r'神仙', r'封神', r'绝了',
            r'垃圾', r'废物', r'离谱', r'逆天'
        ]

        import re
        for pattern in ambiguous_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _apply_ai_correction(self, text: str, base_result: SentimentResult) -> SentimentResult:
        """应用AI语义校正"""
        try:
            # 使用AI校正器进行语义分析
            ai_analysis = self.ai_corrector.correct_sentiment(text, base_result)

            # 解析AI校正结果
            corrected_label = ai_analysis.get('corrected_label', base_result.label)
            confidence_boost = ai_analysis.get('confidence_boost', 0.0)
            correction_reason = ai_analysis.get('reason', '')

            # 创建校正后结果
            corrected_result = SentimentResult(
                label=corrected_label,
                score=base_result.score,  # 保持SnowNLP分数，但更新标签
                confidence=min(1.0, base_result.confidence + confidence_boost),
                model_time=base_result.model_time,
                analysis=f"AI校正: {correction_reason}"
            )

            return corrected_result

        except Exception as e:
            print(f"AI校正失败: {e}")
            # 校正失败时返回原始结果
            return base_result

    def _combine_results(self, base_result: SentimentResult, ai_result: SentimentResult) -> SentimentResult:
        """综合SnowNLP和AI校正结果"""
        # 优先使用AI校正的标签
        final_label = ai_result.label

        # 综合置信度（取较高者）
        combined_confidence = max(base_result.confidence, ai_result.confidence)

        # 综合时间（取平均值）
        combined_time = (base_result.model_time + ai_result.model_time) / 2

        return SentimentResult(
            label=final_label,
            score=base_result.score,  # 保持SnowNLP分数
            confidence=combined_confidence,
            model_time=combined_time,
            analysis=f"SnowNLP基础分析 + {ai_result.analysis}"
        )

    def _generate_analysis(self, base_result: SentimentResult,
                          correction_applied: bool,
                          enhanced_analysis: str) -> str:
        """生成分析说明"""
        analysis_parts = ["混合模型分析"]

        if correction_applied:
            analysis_parts.append("AI语义校正已应用")
        else:
            analysis_parts.append("SnowNLP基础分析")

        if enhanced_analysis:
            analysis_parts.append(enhanced_analysis)

        return "; ".join(analysis_parts)

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
            'ai_correction_threshold': self.ai_correction_threshold,
            'use_enhancement': self.use_enhancement,
            'ai_corrector_available': self.ai_corrector is not None,
            'model_info': self.model_info.to_dict()
        }


# 如果没有AI校正器，降级使用增强的SnowNLP
class FallbackHybridAnalyzer(SentimentAnalyzer):
    """降级混合模型分析器（使用增强的SnowNLP）"""

    def __init__(self):
        self.snow_analyzer = AnalyzerFactory.create_analyzer("snownlp", use_enhancement=True)
        super().__init__()

    def _get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="Fallback-Hybrid-Analyzer",
            version="1.0.0",
            provider="Enhanced-SnowNLP",
            supports_gpu=False,
            batch_size=1,
            description="AI校正不可用时的降级方案，使用增强版SnowNLP"
        )

    def analyze(self, text: str, **kwargs) -> SentimentResult:
        return self.snow_analyzer.analyze(text, **kwargs)

    def analyze_batch(self, texts: List[str], **kwargs) -> List[SentimentResult]:
        return self.snow_analyzer.analyze_batch(texts, **kwargs)
