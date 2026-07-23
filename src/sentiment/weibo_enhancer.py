"""
微博场景情感分析增强器
对微博特有的网络语言、粉圈语言、体育表达进行语义分析和情感增强
"""

import re
from typing import Dict, Any, Tuple, List
from .constants import WeiboConstants, WeiboLanguageType
from .base import SentimentResult


class WeiboEnhancer:
    """微博场景情感分析增强器"""

    def __init__(self):
        self.constants = WeiboConstants()

    def enhance_sentiment(self, text: str, base_result: SentimentResult) -> SentimentResult:
        """
        增强情感分析结果，针对微博场景优化

        Args:
            text: 原始文本
            base_result: 基础分析结果

        Returns:
            SentimentResult: 增强后的结果
        """
        if not text or not base_result:
            return base_result

        enhanced_result = SentimentResult(
            label=base_result.label,
            score=base_result.score,
            confidence=base_result.confidence,
            model_time=base_result.model_time,
            analysis=base_result.analysis,
            enhanced=False
        )

        try:
            # 表情符号分析
            emoji_score = self._analyze_emojis(text)

            # 网络热词分析
            trending_score = self._analyze_trending_words(text)

            # 粉圈语言增强
            fan_circle_enhancement = self._enhance_fan_circle_text(text, base_result)

            # 体育语言增强
            sports_enhancement = self._enhance_sports_text(text, base_result)

            # 讽刺表达检测
            sarcasm_adjustment = self._detect_sarcasm(text)

            # 综合调整情感得分
            final_score = self._combine_adjustments(
                base_result.score,
                emoji_score,
                trending_score,
                fan_circle_enhancement,
                sports_enhancement,
                sarcasm_adjustment
            )

            # 更新结果
            enhanced_result.score = max(0.0, min(1.0, final_score))
            enhanced_result.label = self._classify_enhanced_sentiment(enhanced_result.score)
            enhanced_result.confidence = min(1.0, base_result.confidence + 0.1)
            enhanced_result.enhanced = True
            enhanced_result.analysis = self._generate_enhanced_analysis(
                text, base_result, emoji_score, trending_score
            )

        except Exception as e:
            # 增强失败时返回原始结果
            print(f"微博场景增强失败: {e}")
            return base_result

        return enhanced_result

    def _analyze_emojis(self, text: str) -> float:
        """分析文本中的表情符号情感倾向"""
        total_score = 0.0
        emoji_count = 0

        for emoji, score in self.constants.EMOJI_SENTIMENT.items():
            if emoji in text:
                count = text.count(emoji)
                total_score += score * count
                emoji_count += count

        if emoji_count > 0:
            return total_score / emoji_count
        return 0.0

    def _analyze_trending_words(self, text: str) -> float:
        """分析网络热词的情感倾向"""
        total_score = 0.0
        word_count = 0

        for word, semantic in self.constants.TRENDING_SEMANTICS.items():
            if word in text:
                total_score += semantic['sentiment']
                word_count += 1

        if word_count > 0:
            # 网络热词对情感影响较大，加权处理
            return (total_score / word_count) * 0.3  # 调整权重
        return 0.0

    def _enhance_fan_circle_text(self, text: str, base_result: SentimentResult) -> float:
        """粉圈文本情感增强"""
        if not self.constants.is_fan_circle_text(text):
            return 0.0

        adjustment = 0.0

        # 正向表达增强
        positive_count = sum(1 for word in self.constants.POSITIVE_WORDS
                           if word in text and self.constants.is_fan_circle_text(word))

        # 负向表达减弱
        negative_count = sum(1 for word in self.constants.NEGATIVE_WORDS
                           if word in text and self.constants.is_fan_circle_text(word))

        # 粉圈语言通常表达强烈情感
        if positive_count > negative_count:
            adjustment += 0.15  # 粉圈正向表达更强烈
        elif negative_count > positive_count:
            adjustment -= 0.15  # 粉圈负向表达更强烈

        return adjustment

    def _enhance_sports_text(self, text: str, base_result: SentimentResult) -> float:
        """体育文本情感增强"""
        if not self.constants.is_sports_text(text):
            return 0.0

        adjustment = 0.0

        # 检测胜利相关的词汇
        victory_words = {'赢', '胜利', '冠军', '夺冠', '王者', '牛逼', '给力'}
        defeat_words = {'输', '失败', '失利', '崩了', '垃圾', '菜'}

        victory_count = sum(1 for word in victory_words if word in text)
        defeat_count = sum(1 for word in defeat_words if word in text)

        if victory_count > defeat_count:
            adjustment += 0.2  # 体育胜利表达很强烈
        elif defeat_count > victory_count:
            adjustment -= 0.2  # 体育失败表达很强烈

        return adjustment

    def _detect_sarcasm(self, text: str) -> float:
        """检测讽刺表达"""
        # 一些讽刺表达的模式
        sarcasm_patterns = [
            r'真.*不错',    # 真的不错（实际表示很差）
            r'太.*好了',    # 太好了（反话）
            r'棒棒哒',      # 棒棒哒（常用于讽刺）
            r'就这',        # 就这（轻蔑）
            r'啊这',        # 啊这（表示无语）
        ]

        sarcasm_count = sum(1 for pattern in sarcasm_patterns
                          if re.search(pattern, text))

        # 检测反问句
        rhetorical_count = sum(1 for pattern in self.constants.RHETORICAL_PATTERNS
                              if re.search(pattern, text))

        # 讽刺表达通常需要反转情感
        if sarcasm_count > 0 or rhetorical_count > 0:
            return -0.3  # 中等程度的负向调整

        return 0.0

    def _combine_adjustments(self, base_score: float, *adjustments: float) -> float:
        """综合各个调整项"""
        total_adjustment = sum(adjustments)

        # 限制调整幅度，避免过度偏离
        max_adjustment = 0.4
        total_adjustment = max(-max_adjustment, min(max_adjustment, total_adjustment))

        return base_score + total_adjustment

    def _classify_enhanced_sentiment(self, score: float) -> str:
        """根据增强后的得分分类情感"""
        if score >= 0.6:
            return "积极"
        elif score <= 0.4:
            return "消极"
        else:
            return "中性"

    def _generate_enhanced_analysis(self, text: str, base_result: SentimentResult,
                                   emoji_score: float, trending_score: float) -> str:
        """生成增强分析说明"""
        analysis_parts = []
        language_types = self.constants.detect_language_type(text)

        if language_types:
            type_names = [t.value for t in language_types]
            analysis_parts.append(f"检测到{', '.join(type_names)}特征")

        if abs(emoji_score) > 0.1:
            sentiment = "正向" if emoji_score > 0 else "负向"
            analysis_parts.append(f"表情符号表达{abs(emoji_score):.1f}程度的{sentiment}情绪")

        if abs(trending_score) > 0.1:
            sentiment = "正向" if trending_score > 0 else "负向"
            analysis_parts.append(f"网络热词包含{abs(trending_score):.1f}程度的{sentiment}倾向")

        if self.constants.is_fan_circle_text(text):
            analysis_parts.append("包含粉圈语言表达")

        if self.constants.is_sports_text(text):
            analysis_parts.append("包含体育赛事相关表达")

        if analysis_parts:
            return "; ".join(analysis_parts) + "; " + base_result.analysis
        else:
            return base_result.analysis

    def batch_enhance(self, texts: List[str], base_results: List[SentimentResult]) -> List[SentimentResult]:
        """批量增强情感分析结果"""
        enhanced_results = []

        for text, base_result in zip(texts, base_results):
            enhanced_result = self.enhance_sentiment(text, base_result)
            enhanced_results.append(enhanced_result)

        return enhanced_results
