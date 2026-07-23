import json
import unittest

from src.sentiment.deepseek_analyzer import DeepSeekSentimentAnalyzer


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


if __name__ == "__main__":
    unittest.main()
