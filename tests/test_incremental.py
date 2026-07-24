import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import database
from src.incremental import begin_run, get_series, merge_snapshot


class IncrementalCollectionTests(unittest.TestCase):
    def test_comment_ids_are_merged_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "incremental.db"
            with patch.object(database, "DATABASE_PATH", db_path):
                database.init_db()
                series_id, run1 = begin_run("测试话题", enabled=True, interval_hours=6)
                first_posts = [{
                    "weibo_id": "post-1", "username": "用户", "post_content": "帖子",
                    "comment_count": 10, "comments": ["相同文本", "相同文本"],
                    "comment_records": [
                        {"comment_id": "c1", "text": "相同文本"},
                        {"comment_id": "c2", "text": "相同文本"},
                    ],
                    "fetch_report": {"stop_reason": "empty_page"},
                    "fetch_method": "api_pc",
                }]
                cumulative1, meta1 = merge_snapshot(series_id, run1, first_posts)
                self.assertEqual(meta1["new_comments"], 2)
                self.assertEqual(len(cumulative1[0]["comments"]), 2)
                self.assertIsNotNone(meta1["next_run_at"])

                _, run2 = begin_run("测试话题", enabled=True, interval_hours=6)
                second_posts = [{
                    **first_posts[0],
                    "comments": ["相同文本", "新增评论"],
                    "comment_records": [
                        {"comment_id": "c2", "text": "相同文本"},
                        {"comment_id": "c3", "text": "新增评论"},
                    ],
                }]
                cumulative2, meta2 = merge_snapshot(series_id, run2, second_posts)
                self.assertEqual(meta2["new_comments"], 1)
                self.assertEqual(meta2["total_unique_comments"], 3)
                self.assertEqual(len(cumulative2[0]["comments"]), 3)
                self.assertEqual(get_series("测试话题")["last_run"]["status"], "completed")

    def test_snapshot_uses_constant_number_of_remote_round_trips(self):
        class CountingConnection:
            def __init__(self, connection):
                self.connection = connection
                self.execute_calls = 0
                self.executemany_calls = 0

            def execute(self, *args, **kwargs):
                self.execute_calls += 1
                return self.connection.execute(*args, **kwargs)

            def executemany(self, *args, **kwargs):
                self.executemany_calls += 1
                return self.connection.executemany(*args, **kwargs)

            def commit(self):
                return self.connection.commit()

            def rollback(self):
                return self.connection.rollback()

            def close(self):
                return self.connection.close()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "batched-incremental.db"
            with patch.object(database, "DATABASE_PATH", db_path):
                database.init_db()
                series_id, run_id = begin_run("批量话题")
                counted = CountingConnection(database.get_connection())
                records = [
                    {"comment_id": f"c{i}", "text": f"评论 {i}"}
                    for i in range(120)
                ]
                posts = [{
                    "weibo_id": "post-batch",
                    "comment_count": len(records),
                    "comments": [record["text"] for record in records],
                    "comment_records": records,
                    "fetch_report": {"stop_reason": "max_id_zero"},
                    "fetch_method": "api_pc",
                }]

                with patch("src.incremental.get_connection", return_value=counted):
                    cumulative, metadata = merge_snapshot(series_id, run_id, posts)

                self.assertEqual(metadata["new_comments"], 120)
                self.assertEqual(len(cumulative[0]["comments"]), 120)
                self.assertEqual(counted.executemany_calls, 2)
                self.assertLessEqual(counted.execute_calls, 8)


if __name__ == "__main__":
    unittest.main()
