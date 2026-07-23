"""
模型评估和性能监控模块
支持多模型对比、性能监控、报告生成等功能
"""

import time
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from .base import SentimentAnalyzer, AnalyzerFactory, SentimentResult
from config import EVALUATION_SAMPLE_SIZE


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_time: float = 0.0
    avg_time_per_sample: float = 0.0
    samples_processed: int = 0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class AccuracyMetrics:
    """准确率指标（需要标注数据）"""
    accuracy: float = 0.0
    precision_positive: float = 0.0
    precision_negative: float = 0.0
    recall_positive: float = 0.0
    recall_negative: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Dict[str, int] = None


@dataclass
class ModelComparison:
    """模型对比结果"""
    model_name: str
    performance: PerformanceMetrics
    sample_results: List[Dict[str, Any]]
    agreement_stats: Dict[str, float] = None


class ModelEvaluator:
    """模型评估器"""

    def __init__(self):
        self.sample_size = EVALUATION_SAMPLE_SIZE
        self.comparison_results = {}

    def compare_models(self, texts: List[str], analyzer_types: List[str] = None,
                      **kwargs) -> Dict[str, ModelComparison]:
        """
        对比多个模型在相同数据集上的表现

        Args:
            texts: 测试文本列表
            analyzer_types: 分析器类型列表，默认比较所有可用模型
            **kwargs: 分析器参数

        Returns:
            Dict[str, ModelComparison]: 各模型的对比结果
        """
        if analyzer_types is None:
            analyzer_types = AnalyzerFactory.get_supported_analyzers()

        # 抽样测试数据
        if len(texts) > self.sample_size:
            import random
            texts = random.sample(texts, self.sample_size)

        comparison_results = {}
        analyzers = kwargs.pop("analyzers", {})

        for analyzer_type in analyzer_types:
            try:
                print(f"评估模型: {analyzer_type}")
                comparison = self._evaluate_analyzer(
                    analyzer_type, texts,
                    analyzer=analyzers.get(analyzer_type), **kwargs
                )
                comparison_results[analyzer_type] = comparison
            except Exception as e:
                print(f"模型 {analyzer_type} 评估失败: {e}")
                comparison_results[analyzer_type] = None

        self.comparison_results = comparison_results
        return comparison_results

    def _evaluate_analyzer(self, analyzer_type: str, texts: List[str],
                          analyzer: SentimentAnalyzer = None, **kwargs) -> ModelComparison:
        """评估单个分析器"""
        start_time = time.time()

        # 创建分析器
        analyzer = analyzer or AnalyzerFactory.create_analyzer(analyzer_type, **kwargs)
        model_info = analyzer.get_model_info()

        # 执行批量分析
        results = analyzer.analyze_batch(texts)

        # 计算性能指标
        total_time = time.time() - start_time
        avg_time = total_time / len(texts) if texts else 0

        performance = PerformanceMetrics(
            total_time=total_time,
            avg_time_per_sample=avg_time,
            samples_processed=len(texts)
        )

        # 收集样本结果
        sample_results = []
        for i, (text, result) in enumerate(zip(texts, results)):
            sample_results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'label': result.label,
                'score': result.score,
                'confidence': result.confidence,
                'model_time': result.model_time,
                'analysis': result.analysis
            })

        return ModelComparison(
            model_name=f"{model_info.name} ({analyzer_type})",
            performance=performance,
            sample_results=sample_results
        )

    def calculate_agreement(self, comparison_results: Dict[str, ModelComparison]) -> Dict[str, float]:
        """
        计算模型间的一致性

        Args:
            comparison_results: 模型对比结果

        Returns:
            Dict[str, float]: 一致性统计
        """
        valid_models = {k: v for k, v in comparison_results.items() if v is not None}
        if len(valid_models) < 2:
            return {}

        # 收集所有模型的预测结果
        model_predictions = {}
        sample_count = len(list(valid_models.values())[0].sample_results)

        for model_name, comparison in valid_models.items():
            predictions = [result['label'] for result in comparison.sample_results]
            model_predictions[model_name] = predictions

        # 计算模型间一致性
        agreement_stats = {}
        model_names = list(model_predictions.keys())

        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if i < j:
                    # 计算两个模型的一致性
                    agreement = self._calculate_pairwise_agreement(
                        model_predictions[model_a],
                        model_predictions[model_b]
                    )
                    pair_key = f"{model_a}_vs_{model_b}"
                    agreement_stats[pair_key] = agreement

        # 计算平均一致性
        if agreement_stats:
            avg_agreement = sum(agreement_stats.values()) / len(agreement_stats)
            agreement_stats['average_agreement'] = avg_agreement

        return agreement_stats

    def _calculate_pairwise_agreement(self, predictions_a: List[str],
                                     predictions_b: List[str]) -> float:
        """计算两个模型预测结果的一致性"""
        if len(predictions_a) != len(predictions_b):
            return 0.0

        agreements = 0
        for pred_a, pred_b in zip(predictions_a, predictions_b):
            if pred_a == pred_b:
                agreements += 1

        return agreements / len(predictions_a)

    def generate_comparison_report(self, comparison_results: Dict[str, ModelComparison],
                                  output_path: str = None) -> str:
        """
        生成模型对比报告

        Args:
            comparison_results: 模型对比结果
            output_path: 报告保存路径（可选）

        Returns:
            str: Markdown格式的报告内容
        """
        agreement_stats = self.calculate_agreement(comparison_results)

        # 构建Markdown报告
        report = ["# 情感分析模型对比报告\n"]
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**样本数量**: {self.sample_size}\n")
        report.append("\n## 性能对比\n")

        # 性能表格
        report.append("| 模型 | 总耗时(秒) | 平均耗时(秒/条) | 处理样本数 |")
        report.append("|------|------------|----------------|------------|")

        for model_type, comparison in comparison_results.items():
            if comparison is not None:
                perf = comparison.performance
                report.append(f"| {model_type} | {perf.total_time:.2f} | {perf.avg_time_per_sample:.4f} | {perf.samples_processed} |")

        # 一致性分析
        if agreement_stats:
            report.append("\n## 模型一致性分析\n")
            report.append("| 模型对 | 一致性 |")
            report.append("|-------|---------|")

            for pair, agreement in agreement_stats.items():
                if pair != 'average_agreement':
                    report.append(f"| {pair} | {agreement:.3f} |")

            if 'average_agreement' in agreement_stats:
                report.append(f"\n**平均一致性**: {agreement_stats['average_agreement']:.3f}\n")

        # 样本结果展示
        report.append("\n## 样本分析结果\n")
        report.append("选择前3个样本展示各模型分析结果:\n")

        sample_count = min(3, self.sample_size)
        valid_models = {k: v for k, v in comparison_results.items() if v is not None}

        if valid_models:
            first_model = list(valid_models.values())[0]
            for i in range(sample_count):
                if i < len(first_model.sample_results):
                    sample = first_model.sample_results[i]
                    report.append(f"\n### 样本 {i+1}: `{sample['text']}`\n")
                    report.append("| 模型 | 情感标签 | 得分 | 置信度 | 分析说明 |")
                    report.append("|------|----------|------|--------|----------|")

                    for model_type, comparison in valid_models.items():
                        if i < len(comparison.sample_results):
                            result = comparison.sample_results[i]
                            report.append(f"| {model_type} | {result['label']} | {result['score']:.3f} | {result['confidence']:.3f} | {result['analysis']} |")

        # 结论和建议
        report.append("\n## 结论和建议\n")
        report.append(self._generate_recommendations(comparison_results))

        report_content = "\n".join(report)

        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"报告已保存至: {output_path}")

        return report_content

    def _generate_recommendations(self, comparison_results: Dict[str, ModelComparison]) -> str:
        """生成模型选择建议"""
        valid_results = {k: v for k, v in comparison_results.items() if v is not None}
        if not valid_results:
            return "所有模型评估均失败，请检查依赖和环境配置。"

        recommendations = []

        # 性能最佳模型
        fastest_model = min(valid_results.items(),
                           key=lambda x: x[1].performance.avg_time_per_sample)
        recommendations.append(f"- **性能最佳**: {fastest_model[0]} (平均 {fastest_model[1].performance.avg_time_per_sample:.4f} 秒/条)")

        # 检查模型一致性
        agreement_stats = self.calculate_agreement(comparison_results)
        if agreement_stats.get('average_agreement', 0) < 0.7:
            recommendations.append("- **注意**: 模型间一致性较低，建议人工检查分歧样本")

        # 可用性提示；没有人工标签时不宣称任何模型更准确。
        if 'paddle' in valid_results:
            recommendations.append("- **PaddleNLP 可用**：最终模型选择须以人工标注集的 Macro-F1 为依据")
        elif 'snownlp' in valid_results:
            recommendations.append("- **SnowNLP 可用**：兼容性较好，但仍需微博领域标注集验证")

        return "\n".join(recommendations)

    def real_time_monitor(self, analyzer: SentimentAnalyzer, texts: List[str],
                         interval: int = 5) -> Dict[str, Any]:
        """
        实时性能监控

        Args:
            analyzer: 分析器实例
            texts: 测试文本
            interval: 监控间隔（秒）

        Returns:
            Dict[str, Any]: 监控数据
        """
        import psutil
        import threading
        from collections import deque

        monitor_data = {
            'timestamps': deque(maxlen=100),
            'processing_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'sample_counts': deque(maxlen=100)
        }

        def monitor_loop():
            while True:
                try:
                    # 记录时间戳
                    monitor_data['timestamps'].append(datetime.now())

                    # 记录内存使用
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    monitor_data['memory_usage'].append(memory_mb)

                    time.sleep(interval)
                except:
                    break

        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

        # 执行分析并记录性能
        start_time = time.time()
        results = analyzer.analyze_batch(texts)
        total_time = time.time() - start_time

        # 记录性能数据
        monitor_data['processing_times'].append(total_time)
        monitor_data['sample_counts'].append(len(texts))

        return {
            'total_samples': len(texts),
            'total_time': total_time,
            'avg_time_per_sample': total_time / len(texts) if texts else 0,
            'peak_memory_mb': max(monitor_data['memory_usage']) if monitor_data['memory_usage'] else 0,
            'monitor_data': dict(monitor_data)
        }
