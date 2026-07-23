import tempfile
import unittest
import sqlite3
from pathlib import Path

import pandas as pd

from src import database
from src.cleaner import clean_dataframe
from src.sentiment.compatibility import analyze_sentiment
from src.sentiment.compatibility import get_sentiment_stats


class ProvenanceTests(unittest.TestCase):
    def test_cleaner_records_row_lineage(self):
        frame = pd.DataFrame({"评论内容": ["很好", "很好", "", "一般"]})
        cleaned = clean_dataframe(frame)
        self.assertEqual(cleaned.attrs["cleaning_metadata"]["raw_comments"], 4)
        self.assertEqual(cleaned.attrs["cleaning_metadata"]["cleaned_comments"], 2)

    def test_cleaner_preserves_emoji_and_records_duplicate_weight(self):
        frame = pd.DataFrame({"评论内容": ["😍", "😍", "！！！", "一般"]})
        cleaned = clean_dataframe(frame)
        self.assertEqual(cleaned["评论内容"].tolist(), ["😍", "一般"])
        weights = dict(zip(cleaned["评论内容"], cleaned["duplicate_count"]))
        self.assertEqual(weights["😍"], 2)

    def test_sentiment_stats_report_volume_and_unique_views(self):
        frame = pd.DataFrame({
            "nlp_result": ["积极", "消极", "中性"],
            "duplicate_count": [8, 1, 1],
        })
        stats = get_sentiment_stats(frame)
        self.assertEqual(stats["total"], 10)
        self.assertEqual(stats["unique_total"], 3)
        self.assertEqual(stats["pos_pct"], 80.0)
        self.assertEqual(stats["unique_pos_pct"], 33.3)

    def test_sentiment_records_effective_model_and_confidence(self):
        frame = pd.DataFrame({"评论内容": ["非常喜欢", "非常失望"]})
        result = analyze_sentiment(frame, model_type="snownlp")
        self.assertIn("nlp_confidence", result.columns)
        self.assertEqual(result.attrs["analysis_metadata"]["requested_model"], "snownlp")
        self.assertEqual(result.attrs["analysis_metadata"]["effective_model"], "snownlp")
        self.assertFalse(result.attrs["analysis_metadata"]["fallback_used"])

    def test_database_persists_quality_and_report_metadata(self):
        original_path = database.DATABASE_PATH
        with tempfile.TemporaryDirectory() as tmp:
            database.DATABASE_PATH = Path(tmp) / "test.db"
            try:
                database.init_db()
                task_id = database.create_task("测试", source="upload")
                database.update_task_results(
                    task_id, total=2, pos=1, neg=1, neu=0,
                    raw_comments=3, requested_model="paddle",
                    effective_model="snownlp", fallback_used=True,
                    fallback_reason="test", quality_status="warning",
                    quality_issues_json='[{"code":"model_fallback"}]',
                    processing_time=1.25, model_memory=8.5,
                )
                database.update_task_report(task_id, "/tmp/report.md", "deepseek")
                task = database.get_task(task_id)
                self.assertEqual(task["effective_model"], "snownlp")
                self.assertEqual(task["quality_status"], "warning")
                self.assertEqual(task["report_provider"], "deepseek")
                self.assertEqual(task["processing_time"], 1.25)
                self.assertEqual(task["model_memory"], 8.5)
            finally:
                database.DATABASE_PATH = original_path

    def test_task_result_write_repairs_stale_schema(self):
        original_path = database.DATABASE_PATH
        with tempfile.TemporaryDirectory() as tmp:
            database.DATABASE_PATH = Path(tmp) / "stale.db"
            try:
                conn = sqlite3.connect(database.DATABASE_PATH)
                conn.execute("""
                    CREATE TABLE tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'crawler',
                        status TEXT NOT NULL DEFAULT 'pending',
                        total_comments INTEGER DEFAULT 0,
                        pos_count INTEGER DEFAULT 0,
                        neg_count INTEGER DEFAULT 0,
                        neu_count INTEGER DEFAULT 0,
                        wordcloud_path TEXT,
                        keywords_json TEXT
                    )
                """)
                task_id = conn.execute(
                    "INSERT INTO tasks(topic) VALUES ('旧结构')"
                ).lastrowid
                conn.commit()
                conn.close()

                database.update_task_results(
                    task_id, total=2, pos=1, neg=0, neu=1,
                    processing_time=2.5, model_memory=4.0,
                )
                task = database.get_task(task_id)
                self.assertEqual(task["processing_time"], 2.5)
                self.assertEqual(task["model_memory"], 4.0)
            finally:
                database.DATABASE_PATH = original_path


if __name__ == "__main__":
    unittest.main()
