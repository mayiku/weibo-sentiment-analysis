import unittest

import pandas as pd

from src.sentiment.benchmark import build_labeling_sample, calculate_metrics


class BenchmarkTests(unittest.TestCase):
    def test_macro_metrics_and_confusion_matrix(self):
        metrics = calculate_metrics(
            ["积极", "积极", "中性", "消极"],
            ["积极", "中性", "中性", "消极"],
            "test-model",
        )
        self.assertEqual(metrics.sample_size, 4)
        self.assertEqual(metrics.accuracy, 0.75)
        self.assertEqual(metrics.confusion_matrix["积极"]["中性"], 1)
        self.assertIn("中性", metrics.per_class)

    def test_labeling_sample_is_reproducible_and_deduplicated(self):
        frame = pd.DataFrame({"评论内容": ["好", "好", "一般", "差"]})
        sample = build_labeling_sample(frame, sample_size=10)
        self.assertEqual(len(sample), 3)
        self.assertEqual(sample.loc[sample["评论内容"] == "好", "原始重复次数"].iloc[0], 2)
        self.assertEqual(sample.columns.tolist(), [
            "sample_id", "评论内容", "原始重复次数", "人工标签", "标注人", "备注"
        ])


if __name__ == "__main__":
    unittest.main()
