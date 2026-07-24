import unittest
from unittest.mock import patch

from src.ai_agent.deepseek_client import DeepSeekClient


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def post(self, *_args, **_kwargs):
        self.calls += 1
        return next(self.responses)


class DeepSeekClientTests(unittest.TestCase):
    def test_success_records_usage_and_latency(self):
        client = DeepSeekClient(api_key="test")
        client.session = _Session([_Response(200, {
            "choices": [{"message": {"content": "报告"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        })])
        with patch(
            "src.ai_agent.deepseek_client.time.perf_counter",
            side_effect=[10.0, 11.25],
        ):
            result = client.chat("prompt")
        self.assertEqual(result, "报告")
        self.assertEqual(client.last_usage["total_tokens"], 30)
        self.assertEqual(client.last_latency_seconds, 1.25)

    def test_last_failed_retry_does_not_sleep(self):
        client = DeepSeekClient(api_key="test")
        client.session = _Session([_Response(500), _Response(500)])
        with patch("src.ai_agent.deepseek_client.DEEPSEEK_MAX_RETRIES", 2), patch(
            "src.ai_agent.deepseek_client.time.sleep"
        ) as sleep_mock:
            self.assertIsNone(client.chat("prompt"))
        self.assertEqual(client.session.calls, 2)
        self.assertIn("HTTP 500", client.last_error)
        sleep_mock.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
