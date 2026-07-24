import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import task_lifecycle


class TaskLifecycleCompatibilityTests(unittest.TestCase):
    def test_fallback_repairs_old_schema_and_reconciles_stale_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "old.db"

            def connection():
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                return conn

            conn = connection()
            conn.execute("""
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            conn.execute(
                "INSERT INTO tasks VALUES (1, 'crawling', NULL, ?, NULL)",
                ("2026-07-24 00:00:00",),
            )
            conn.commit()
            conn.close()

            stale_database = SimpleNamespace(get_connection=connection)
            with patch.object(task_lifecycle, "_database", stale_database):
                reconciled = task_lifecycle.reconcile_stale_tasks(
                    45, now=datetime(2026, 7, 24, 2, 0, tzinfo=timezone.utc)
                )

            self.assertEqual(reconciled, 1)
            conn = connection()
            row = conn.execute("SELECT * FROM tasks WHERE id=1").fetchone()
            columns = {
                item[1] for item in conn.execute("PRAGMA table_info(tasks)").fetchall()
            }
            conn.close()
            self.assertIn("updated_at", columns)
            self.assertEqual(row["status"], "failed")
            self.assertIsNotNone(row["completed_at"])


if __name__ == "__main__":
    unittest.main()
