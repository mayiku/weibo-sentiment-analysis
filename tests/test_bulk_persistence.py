import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import database


class CountingConnection:
    def __init__(self, connection):
        self.connection = connection
        self.comment_insert_calls = 0

    def execute(self, sql, parameters=()):
        if sql.startswith("INSERT INTO comments"):
            self.comment_insert_calls += 1
        return self.connection.execute(sql, parameters)

    def commit(self):
        return self.connection.commit()

    def close(self):
        return self.connection.close()


class BulkPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = database.DATABASE_PATH
        database.DATABASE_PATH = Path(self.temp_dir.name) / "bulk.db"
        database.init_db()
        self.task_id = database.create_task("批量写入测试", source="upload")

    def tearDown(self):
        database.DATABASE_PATH = self.original_path
        self.temp_dir.cleanup()

    def test_large_comment_set_uses_fixed_size_multi_value_chunks(self):
        row_count = 1_203
        frame = pd.DataFrame({
            "评论内容": [f"评论 {index}" for index in range(row_count)],
            "clean_text": [f"评论 {index}" for index in range(row_count)],
            "nlp_result": ["中性"] * row_count,
            "nlp_score": [0.5] * row_count,
            "nlp_confidence": [0.75] * row_count,
            "duplicate_count": [1] * row_count,
        })
        counted = CountingConnection(database.get_connection())

        with patch.object(database, "get_connection", return_value=counted):
            database.insert_comments(self.task_id, frame)

        self.assertEqual(counted.comment_insert_calls, math.ceil(row_count / 500))
        conn = database.get_connection()
        try:
            saved = conn.execute(
                "SELECT COUNT(*) FROM comments WHERE task_id=?", (self.task_id,)
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(saved, row_count)

    def test_recovery_clear_removes_only_derived_task_rows(self):
        other_task_id = database.create_task("另一个任务", source="upload")
        database.insert_posts(self.task_id, [{"weibo_id": "post-1"}])
        database.insert_keywords(self.task_id, [("世界杯", 5)])
        database.insert_comments(
            self.task_id,
            pd.DataFrame({"评论内容": ["测试"], "clean_text": ["测试"]}),
        )
        database.insert_comments(
            other_task_id,
            pd.DataFrame({"评论内容": ["保留"], "clean_text": ["保留"]}),
        )

        database.clear_task_analysis_data(self.task_id)

        conn = database.get_connection()
        try:
            for table in ("posts", "comments", "keywords"):
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE task_id=?", (self.task_id,)
                ).fetchone()[0]
                self.assertEqual(count, 0)
            other_count = conn.execute(
                "SELECT COUNT(*) FROM comments WHERE task_id=?", (other_task_id,)
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(other_count, 1)


if __name__ == "__main__":
    unittest.main()
