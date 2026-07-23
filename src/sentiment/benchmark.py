"""Ground-truth benchmarking for Weibo sentiment models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .base import AnalyzerFactory


LABELS = ["积极", "中性", "消极"]
LABEL_ALIASES = {
    "positive": "积极", "pos": "积极", "正面": "积极", "积极": "积极",
    "neutral": "中性", "neu": "中性", "中立": "中性", "中性": "中性",
    "negative": "消极", "neg": "消极", "负面": "消极", "消极": "消极",
}


@dataclass
class BenchmarkMetrics:
    model: str
    sample_size: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_label(value: Any) -> str:
    label = LABEL_ALIASES.get(str(value).strip().lower())
    if label is None:
        raise ValueError(f"未知人工标签: {value!r}；允许值为 {LABELS}")
    return label


def load_labeled_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV containing 评论内容 and 人工标签."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"评论内容", "人工标签"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"标注集缺少列: {sorted(missing)}")
    frame = frame.dropna(subset=["评论内容", "人工标签"]).copy()
    frame["评论内容"] = frame["评论内容"].astype(str).str.strip()
    frame = frame[frame["评论内容"].str.len() > 0]
    frame["人工标签"] = frame["人工标签"].map(normalize_label)
    if frame.empty:
        raise ValueError("标注集中没有可评估样本")
    return frame


def build_labeling_sample(
    frame: pd.DataFrame, sample_size: int = 500, random_state: int = 42
) -> pd.DataFrame:
    """Create a reproducible annotation worksheet from comment data."""
    if "评论内容" not in frame.columns:
        raise ValueError("数据缺少 '评论内容' 列")
    source = frame.copy()
    source["评论内容"] = source["评论内容"].astype(str).str.strip()
    source = source[source["评论内容"].str.len() > 0]
    if "duplicate_count" not in source.columns:
        source["duplicate_count"] = source.groupby("评论内容")["评论内容"].transform("size")
    source = source.drop_duplicates("评论内容")
    if len(source) > sample_size:
        source = source.sample(n=sample_size, random_state=random_state)
    source = source.reset_index(drop=True)
    return pd.DataFrame({
        "sample_id": [f"WB-{index + 1:04d}" for index in range(len(source))],
        "评论内容": source["评论内容"],
        "原始重复次数": source["duplicate_count"].fillna(1).astype(int),
        "人工标签": "",
        "标注人": "",
        "备注": "",
    })


def calculate_metrics(y_true: list[str], y_pred: list[str], model: str) -> BenchmarkMetrics:
    """Calculate three-class metrics without conflating agreement with accuracy."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("真实标签与预测标签数量必须一致且非空")
    y_true = [normalize_label(label) for label in y_true]
    y_pred = [normalize_label(label) for label in y_pred]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    per_class = {
        label: {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    matrix_dict = {
        true_label: {pred_label: int(matrix[i, j]) for j, pred_label in enumerate(LABELS)}
        for i, true_label in enumerate(LABELS)
    }
    return BenchmarkMetrics(
        model=model,
        sample_size=len(y_true),
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        macro_precision=round(float(precision.mean()), 4),
        macro_recall=round(float(recall.mean()), 4),
        macro_f1=round(float(f1.mean()), 4),
        per_class=per_class,
        confusion_matrix=matrix_dict,
    )


def evaluate_model(frame: pd.DataFrame, model_type: str) -> BenchmarkMetrics:
    """Run one configured model against a validated ground-truth DataFrame."""
    required = {"评论内容", "人工标签"}
    if not required.issubset(frame.columns):
        raise ValueError("评估数据必须包含 '评论内容' 和 '人工标签' 列")
    analyzer = AnalyzerFactory.create_analyzer(model_type)
    predictions = analyzer.analyze_batch(frame["评论内容"].astype(str).tolist())
    if len(predictions) != len(frame):
        raise RuntimeError(f"模型返回 {len(predictions)} 条结果，预期 {len(frame)} 条")
    predicted_labels = [result.label for result in predictions]
    return calculate_metrics(frame["人工标签"].tolist(), predicted_labels, model_type)
