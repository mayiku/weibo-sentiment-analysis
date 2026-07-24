import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import libsql
from src import database
from src.db_connection import LibSQLConnection, MappingRow


class LibSQLConnectionTests(unittest.TestCase):
    def setUp(self):
        database.clear_turso_connection_cache()

    def tearDown(self):
        database.clear_turso_connection_cache()

    def test_mapping_row_supports_tuple_and_named_access(self):
        row = MappingRow(["id", "name"], [1, "Alice"])
        self.assertEqual(row[0], 1)
        self.assertEqual(row["name"], "Alice")
        self.assertEqual(dict(row), {"id": 1, "name": "Alice"})

    def test_libsql_adapter_supports_existing_database_contract(self):
        conn = LibSQLConnection(libsql.connect(":memory:"))
        conn.executescript("""
            CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
            INSERT INTO users(name) VALUES ('Alice');
        """)
        conn.executemany(
            "INSERT INTO users(name) VALUES (?)", [("Bob",), ("Carol",)]
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users ORDER BY id").fetchone()
        self.assertEqual(row["name"], "Alice")
        cursor = conn.execute("SELECT * FROM users ORDER BY id")
        self.assertEqual(
            [row["name"] for row in cursor.fetchall()],
            ["Alice", "Bob", "Carol"],
        )
        conn.close()

    def test_database_uses_turso_when_both_credentials_are_configured(self):
        raw = libsql.connect(":memory:")
        with patch.object(database, "TURSO_DATABASE_URL", "libsql://example.turso.io"), patch.object(
            database, "TURSO_AUTH_TOKEN", "secret"
        ), patch("libsql.connect", return_value=raw) as connect_mock:
            conn = database.get_connection()
        self.assertTrue(conn.is_turso)
        connect_mock.assert_called_once_with(
            database="libsql://example.turso.io", auth_token="secret"
        )
        conn.close()

    def test_turso_connection_is_reused_across_database_operations(self):
        raw = libsql.connect(":memory:")
        with patch.object(database, "TURSO_DATABASE_URL", "libsql://reuse.turso.io"), patch.object(
            database, "TURSO_AUTH_TOKEN", "secret"
        ), patch("libsql.connect", return_value=raw) as connect_mock:
            first = database.get_connection()
            first.close()
            second = database.get_connection()
            second.execute("SELECT 1").fetchone()

        self.assertIs(first, second)
        connect_mock.assert_called_once()

    def test_partial_turso_configuration_fails_explicitly(self):
        with patch.object(database, "TURSO_DATABASE_URL", "libsql://example.turso.io"), patch.object(
            database, "TURSO_AUTH_TOKEN", ""
        ):
            with self.assertRaisesRegex(RuntimeError, "配置不完整"):
                database.get_connection()

    def test_database_schema_and_crud_run_through_turso_adapter(self):
        real_connect = libsql.connect
        with tempfile.TemporaryDirectory() as tmp:
            db_file = Path(tmp) / "turso-compatible.db"

            def local_libsql(**_kwargs):
                return real_connect(str(db_file))

            with patch.object(
                database, "TURSO_DATABASE_URL", "libsql://example.turso.io"
            ), patch.object(database, "TURSO_AUTH_TOKEN", "secret"), patch(
                "libsql.connect", side_effect=local_libsql
            ):
                database.init_db()
                task_id = database.create_task("Turso测试")
                database.update_task_status(task_id, "crawling")
                database.update_task_results(
                    task_id, total=2, pos=1, neg=0, neu=1
                )
                database.update_task_status(task_id, "completed")
                task = database.get_task(task_id)

            self.assertEqual(task["topic"], "Turso测试")
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["total_comments"], 2)


if __name__ == "__main__":
    unittest.main()
