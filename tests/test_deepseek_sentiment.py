import json
import unittest

import pandas as pd

from src.sentiment.deepseek_analyzer import DeepSeekSentimentAnalyzer
from src.sentiment.compatibility import analyze_sentiment


class FakeDeepSeekClient:
    last_error = None

    def __init__(self):
        self.calls = 0

    def chat(self, prompt, **kwargs):
        self.calls += 1
        marker = "输入："
        items = json.loads(prompt.split(marker, 1)[1])
        labels = ["积极", "消极", "中性"]
        return json.dumps([
            {
                "id": item["id"],
                "label": labels[item["id"] % len(labels)],
                "confidence": 0.9,
                "reason": "测试",
            }
            for item in items
        ], ensure_ascii=False)


class DeepSeekSentimentTests(unittest.TestCase):
    def test_batch_results_keep_input_order(self):
        client = FakeDeepSeekClient()
        analyzer = DeepSeekSentimentAnalyzer(client=client, batch_size=3)
        results = analyzer.analyze_batch(["很好", "太差", "发布公告"])
        self.assertEqual([item.label for item in results], ["积极", "消极", "中性"])
        self.assertEqual(client.calls, 1)
        self.assertGreater(results[0].score, 0.6)
        self.assertLess(results[1].score, 0.4)
        self.assertEqual(results[2].score, 0.5)

    def test_invalid_response_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "预期 2 条"):
            DeepSeekSentimentAnalyzer._parse_response(
                '[{"id": 0, "label": "积极", "confidence": 0.9}]', 2
            )

    def test_malformed_batch_is_retried_without_global_fallback(self):
        class MalformedOnceClient(FakeDeepSeekClient):
            def chat(self, prompt, **kwargs):
                if self.calls == 0:
                    self.calls += 1
                    return '[{"id": 0, "label": "积极", "confidence": 0.9, "text": "未闭合}]'
                return super().chat(prompt, **kwargs)

        client = MalformedOnceClient()
        analyzer = DeepSeekSentimentAnalyzer(
            client=client, batch_size=3, batch_retries=2
        )
        results = analyzer.analyze_batch(["很好", "太差", "发布公告"])

        self.assertEqual(len(results), 3)
        self.assertEqual(client.calls, 2)
        self.assertEqual(analyzer.partial_fallback_count, 0)

    def test_persistently_bad_batch_is_split_before_fallback(self):
        class SplitClient(FakeDeepSeekClient):
            def chat(self, prompt, **kwargs):
                items = json.loads(prompt.split("输入：", 1)[1])
                if len(items) > 1:
                    self.calls += 1
                    return '[{"id": 0, "label": "积极"}]'
                return super().chat(prompt, **kwargs)

        client = SplitClient()
        analyzer = DeepSeekSentimentAnalyzer(
            client=client, batch_size=2, batch_retries=1
        )
        results = analyzer.analyze_batch(["很好", "太差"])

        self.assertEqual(len(results), 2)
        self.assertEqual(client.calls, 3)
        self.assertEqual(analyzer.partial_fallback_count, 0)

    def test_only_irrecoverable_single_item_uses_snownlp(self):
        class AlwaysMalformedClient(FakeDeepSeekClient):
            def chat(self, prompt, **kwargs):
                self.calls += 1
                return '[{"id": 0, "label": "积极", "confidence": "未闭合}]'

        client = AlwaysMalformedClient()
        frame = pd.DataFrame({'评论内容': ['这个结果实在太差了']})
        result = analyze_sentiment(
            frame,
            model_type='deepseek',
            client=client,
            batch_size=1,
            batch_retries=1,
        )

        metadata = result.attrs['analysis_metadata']
        self.assertEqual(metadata['effective_model'], 'deepseek+snownlp')
        self.assertEqual(metadata['partial_fallback_count'], 1)
        self.assertTrue(metadata['fallback_used'])
        self.assertIn('局部降级', metadata['fallback_reason'])


if __name__ == "__main__":
    unittest.main()
