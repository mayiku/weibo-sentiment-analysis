import unittest

from src.weibo_api import WeiboAPIClient


class WeiboApiReportingTests(unittest.TestCase):
    def test_count_probe_is_reused_for_the_session(self):
        client = WeiboAPIClient()
        calls = []
        client._adaptive_wait = lambda **_kwargs: None
        client.get_comments_page = lambda *_args, **kwargs: (
            calls.append(kwargs["count"]) or
            {"ok": 1, "data": [{"id": str(i)} for i in range(20)], "total_number": 100}
        )
        self.assertEqual(client._detect_max_count("first"), 20)
        self.assertEqual(client._detect_max_count("second"), 20)
        self.assertEqual(calls, [60, 50, 40, 30, 20])

    def test_empty_page_is_reported_as_visible_window_limit(self):
        client = WeiboAPIClient()
        client._detect_max_count = lambda _mid: 20
        client._adaptive_wait = lambda **_kwargs: None
        client._extract_all_records = lambda _comment: [
            {"comment_id": "1", "text": "评论", "parent_id": "", "depth": 0}
        ]
        responses = iter([
            {"ok": 1, "data": [{"id": "1"}], "max_id": 123, "total_number": 100},
            {"ok": 1, "data": [], "max_id": 0, "total_number": 100},
        ])
        client.get_comments_page = lambda *_args, **_kwargs: next(responses)

        comments, report = client.get_all_comments("mid", max_pages=10)

        self.assertEqual(comments, ["评论"])
        self.assertEqual(report["stop_reason"], "empty_page")
        self.assertTrue(report["incomplete"])
        self.assertTrue(report["visible_window_limited"])
        self.assertFalse(report["truncated_by_pages"])

    def test_incremental_scan_stops_after_checkpoint_pages(self):
        client = WeiboAPIClient()
        client._detect_max_count = lambda _mid: 20
        client._adaptive_wait = lambda **_kwargs: None
        client._extract_all_records = lambda comment: [{
            "comment_id": str(comment["id"]), "text": "旧评论",
            "parent_id": "", "depth": 0,
        }]
        responses = iter([
            {"ok": 1, "data": [{"id": "1"}], "max_id": 10, "total_number": 100},
            {"ok": 1, "data": [{"id": "2"}], "max_id": 20, "total_number": 100},
        ])
        client.get_comments_page = lambda *_args, **_kwargs: next(responses)

        comments, report = client.get_all_comments(
            "mid", max_pages=10, known_comment_ids={"1", "2"},
            stop_after_known_pages=2,
        )

        self.assertEqual(comments, [])
        self.assertEqual(report["pages"], 2)
        self.assertEqual(report["stop_reason"], "checkpoint_reached")
        self.assertEqual(report["known_records_seen"], 2)

    def test_single_known_terminal_page_is_an_incremental_checkpoint(self):
        client = WeiboAPIClient()
        client._detect_max_count = lambda _mid: 20
        client._adaptive_wait = lambda **_kwargs: None
        client._extract_all_records = lambda comment: [{
            "comment_id": str(comment["id"]), "text": "旧评论",
            "parent_id": "", "depth": 0,
        }]
        client.get_comments_page = lambda *_args, **_kwargs: {
            "ok": 1, "data": [{"id": "1"}],
            "max_id": 0, "total_number": 1,
        }

        comments, report = client.get_all_comments(
            "mid", known_comment_ids={"1"}, stop_after_known_pages=2,
        )

        self.assertEqual(comments, [])
        self.assertTrue(report["checkpoint_reached"])
        self.assertTrue(report["request_succeeded"])
        self.assertFalse(report["incomplete"])


if __name__ == "__main__":
    unittest.main()
