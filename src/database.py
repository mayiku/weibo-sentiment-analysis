"""
SQLite 数据库模块 — 存储任务记录、帖子、评论数据和分析结果

v2.0: 新增 posts 表，支持帖子↔评论关联，为 Agent API 分析做准备
"""
import sqlite3
import json
import pandas as pd
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from config import (
    DATABASE_PATH,
    REPORT_DIR,
    TURSO_AUTH_TOKEN,
    TURSO_DATABASE_URL,
)

from src.db_connection import LibSQLConnection
from src.logger import get_logger
from src.quality import assess_result_quality, load_crawl_metadata

log = get_logger(__name__)

DATABASE_SCHEMA_VERSION = 5
ACTIVE_TASK_STATUSES = (
    'pending', 'crawling', 'cleaning', 'analyzing', 'generating_wordcloud'
)
TASK_COLUMN_MIGRATIONS = [
    ('total_posts', 'INTEGER DEFAULT 0'),
    ('structured_json', 'TEXT'),
    ('raw_comments', 'INTEGER DEFAULT 0'),
    ('unique_comments', 'INTEGER DEFAULT 0'),
    ('expected_comments', 'INTEGER'),
    ('fetched_comments', 'INTEGER'),
    ('coverage_pct', 'REAL'),
    ('requested_model', 'TEXT'),
    ('effective_model', 'TEXT'),
    ('model_version', 'TEXT'),
    ('fallback_used', 'INTEGER DEFAULT 0'),
    ('fallback_reason', 'TEXT'),
    ('processing_time', 'REAL'),
    ('model_memory', 'REAL'),
    ('quality_status', "TEXT DEFAULT 'unknown'"),
    ('quality_issues_json', 'TEXT'),
    ('sampling_json', 'TEXT'),
    ('representation_status', "TEXT DEFAULT 'unknown'"),
    ('report_path', 'TEXT'),
    ('report_provider', 'TEXT'),
    ('updated_at', 'TIMESTAMP'),
]


def _ensure_columns(conn: sqlite3.Connection, table: str,
                    definitions: list[tuple[str, str]]) -> None:
    """Idempotently add missing columns, including after Streamlit hot reloads."""
    existing = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for column, column_type in definitions:
        if column not in existing:
            log.info("Schema migration: adding %s to %s...", column, table)
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
            existing.add(column)


def _create_turso_connection(database_url: str, auth_token: str):
    """Create a short-lived Turso connection for one database operation."""
    try:
        import libsql
    except ImportError as exc:
        raise RuntimeError("已配置 Turso，但缺少 libsql 依赖。") from exc
    raw = libsql.connect(database=database_url, auth_token=auth_token)
    raw.execute("PRAGMA foreign_keys=ON")
    # Hrana streams are server-scoped and can become invalid after Streamlit
    # hot reloads. Short-lived handles avoid stale snapshots and stream-not-
    # found failures; higher-level UI reads and bulk writes are cached/batched.
    return LibSQLConnection(raw)


def get_connection():
    """获取数据库连接"""
    if bool(TURSO_DATABASE_URL) != bool(TURSO_AUTH_TOKEN):
        raise RuntimeError(
            "Turso 配置不完整：TURSO_DATABASE_URL 和 TURSO_AUTH_TOKEN 必须同时设置。"
        )
    if TURSO_DATABASE_URL:
        return _create_turso_connection(TURSO_DATABASE_URL, TURSO_AUTH_TOKEN)

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                topic           TEXT    NOT NULL,
                source          TEXT    NOT NULL DEFAULT 'crawler',
                status          TEXT    NOT NULL DEFAULT 'pending',
                total_comments  INTEGER DEFAULT 0,
                unique_comments INTEGER DEFAULT 0,
                total_posts     INTEGER DEFAULT 0,
                pos_count       INTEGER DEFAULT 0,
                neg_count       INTEGER DEFAULT 0,
                neu_count       INTEGER DEFAULT 0,
                wordcloud_path  TEXT,
                keywords_json   TEXT,
                structured_json TEXT,
                raw_comments    INTEGER DEFAULT 0,
                expected_comments INTEGER,
                fetched_comments INTEGER,
                coverage_pct    REAL,
                requested_model TEXT,
                effective_model TEXT,
                model_version   TEXT,
                fallback_used   INTEGER DEFAULT 0,
                fallback_reason TEXT,
                processing_time REAL,
                model_memory    REAL,
                quality_status  TEXT DEFAULT 'unknown',
                quality_issues_json TEXT,
                sampling_json   TEXT,
                representation_status TEXT DEFAULT 'unknown',
                report_path     TEXT,
                report_provider TEXT,
                error_message   TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at    TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS posts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER NOT NULL,
                weibo_id        TEXT    NOT NULL,
                username        TEXT    DEFAULT '',
                content         TEXT    DEFAULT '',
                comment_count   INTEGER DEFAULT 0,
                post_time       TEXT    DEFAULT '',
                url             TEXT    DEFAULT '',
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS comments (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER NOT NULL,
                post_id         INTEGER,
                content         TEXT    NOT NULL,
                cleaned_content TEXT,
                nlp_result      TEXT,
                nlp_score       REAL,
                nlp_confidence  REAL,
                duplicate_count INTEGER DEFAULT 1,
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
                FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS keywords (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER NOT NULL,
                word            TEXT    NOT NULL,
                frequency       INTEGER NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS crawl_series (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                topic           TEXT NOT NULL UNIQUE,
                enabled         INTEGER NOT NULL DEFAULT 0,
                interval_hours  INTEGER NOT NULL DEFAULT 6,
                next_run_at     TIMESTAMP,
                last_run_at     TIMESTAMP,
                total_unique_comments INTEGER NOT NULL DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS crawl_checkpoints (
                series_id       INTEGER NOT NULL,
                weibo_id        TEXT NOT NULL,
                expected_total  INTEGER,
                observed_total  INTEGER NOT NULL DEFAULT 0,
                last_cursor     TEXT,
                stop_reason     TEXT,
                metadata_json   TEXT,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (series_id, weibo_id),
                FOREIGN KEY (series_id) REFERENCES crawl_series(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS crawl_observations (
                series_id       INTEGER NOT NULL,
                weibo_id        TEXT NOT NULL,
                comment_key     TEXT NOT NULL,
                comment_id      TEXT,
                content         TEXT NOT NULL,
                first_seen_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (series_id, weibo_id, comment_key),
                FOREIGN KEY (series_id) REFERENCES crawl_series(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS crawl_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id       INTEGER NOT NULL,
                task_id         INTEGER,
                status          TEXT NOT NULL DEFAULT 'running',
                new_comments    INTEGER NOT NULL DEFAULT 0,
                total_unique_comments INTEGER NOT NULL DEFAULT 0,
                started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at    TIMESTAMP,
                error_message   TEXT,
                FOREIGN KEY (series_id) REFERENCES crawl_series(id) ON DELETE CASCADE,
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_comments_task ON comments(task_id);
            CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
            CREATE INDEX IF NOT EXISTS idx_posts_task    ON posts(task_id);
            CREATE INDEX IF NOT EXISTS idx_keywords_task ON keywords(task_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_topic   ON tasks(topic);
            CREATE INDEX IF NOT EXISTS idx_crawl_runs_series ON crawl_runs(series_id);
            CREATE INDEX IF NOT EXISTS idx_crawl_observations_series ON crawl_observations(series_id);
        """)
        conn.commit()
        log.info(
            "数据库初始化完成: %s",
            "Turso Cloud" if TURSO_DATABASE_URL else DATABASE_PATH,
        )

        # Schema migration for v2.0 — handle existing databases
        _migrate_schema(conn)
        _backfill_unique_comment_counts(conn)
        _backfill_quality_metadata(conn)
        _backfill_sampling_metadata(conn)
        _backfill_missing_posts(conn)
        _backfill_report_links(conn)
        conn.commit()
    except Exception as e:
        log.error("数据库初始化失败: %s", e)
        raise
    finally:
        conn.close()


def _migrate_schema(conn: sqlite3.Connection):
    """增量迁移旧数据库到最新 schema"""
    # Check if posts table exists
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='posts'"
    )
    if not cur.fetchone():
        log.info("Schema migration: creating posts table...")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS posts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER NOT NULL,
                weibo_id        TEXT    NOT NULL,
                username        TEXT    DEFAULT '',
                content         TEXT    DEFAULT '',
                comment_count   INTEGER DEFAULT 0,
                post_time       TEXT    DEFAULT '',
                url             TEXT    DEFAULT '',
                FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_posts_task ON posts(task_id);
        """)

    # Check if comments has post_id column
    cur = conn.execute("PRAGMA table_info(comments)")
    columns = [row[1] for row in cur.fetchall()]
    if 'post_id' not in columns:
        log.info("Schema migration: adding post_id to comments...")
        conn.execute("ALTER TABLE comments ADD COLUMN post_id INTEGER REFERENCES posts(id) ON DELETE SET NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id)")

    # Ensure tasks has all versioned columns.
    _ensure_columns(conn, 'tasks', TASK_COLUMN_MIGRATIONS)

    cur = conn.execute("PRAGMA table_info(comments)")
    comment_cols = [row[1] for row in cur.fetchall()]
    if 'nlp_confidence' not in comment_cols:
        log.info("Schema migration: adding nlp_confidence to comments...")
        conn.execute("ALTER TABLE comments ADD COLUMN nlp_confidence REAL")
    if 'duplicate_count' not in comment_cols:
        log.info("Schema migration: adding duplicate_count to comments...")
        conn.execute("ALTER TABLE comments ADD COLUMN duplicate_count INTEGER DEFAULT 1")

    conn.commit()


def _backfill_quality_metadata(conn: sqlite3.Connection):
    """Populate evidence fields for tasks created before provenance was added."""
    rows = conn.execute(
        "SELECT * FROM tasks WHERE quality_status IS NULL OR quality_status='unknown'"
    ).fetchall()
    for row in rows:
        task = dict(row)
        crawl = load_crawl_metadata(task.get('structured_json'))
        quality = assess_result_quality(
            total=task.get('total_comments') or 0,
            positive=task.get('pos_count') or 0,
            negative=task.get('neg_count') or 0,
            neutral=task.get('neu_count') or 0,
            coverage_pct=crawl.get('coverage_pct'),
            raw_comments=task.get('raw_comments') or task.get('total_comments') or 0,
        )
        conn.execute(
            """UPDATE tasks SET raw_comments=?, expected_comments=?, fetched_comments=?,
               coverage_pct=?, quality_status=?, quality_issues_json=? WHERE id=?""",
            (
                task.get('raw_comments') or task.get('total_comments') or 0,
                crawl.get('expected_comments'), crawl.get('fetched_comments'),
                crawl.get('coverage_pct'), quality['status'],
                json.dumps(quality['issues'], ensure_ascii=False), task['id'],
            ),
        )
    if rows:
        log.info("Schema migration: backfilled quality metadata for %d tasks", len(rows))


def _backfill_sampling_metadata(conn: sqlite3.Connection):
    """Persist per-post sampling diagnostics for historical crawler tasks."""
    rows = conn.execute(
        "SELECT id, structured_json, sampling_json FROM tasks WHERE structured_json IS NOT NULL"
    ).fetchall()
    updated = 0
    for row in rows:
        existing = {}
        try:
            existing = json.loads(row['sampling_json'] or '{}')
        except json.JSONDecodeError:
            pass
        required_sampling_fields = {
            'median_post_coverage_pct', 'active_post_count', 'zero_comment_post_count'
        }
        if required_sampling_fields.issubset(existing):
            continue
        crawl = load_crawl_metadata(row['structured_json'])
        conn.execute(
            "UPDATE tasks SET sampling_json=?, representation_status=? WHERE id=?",
            (json.dumps(crawl, ensure_ascii=False), crawl.get('representation_status', 'unknown'), row['id']),
        )
        updated += 1
    if updated:
        log.info("Schema migration: backfilled sampling metadata for %d tasks", updated)


def _backfill_missing_posts(conn: sqlite3.Connection):
    """Restore zero-comment posts omitted by the legacy CSV-based insert path."""
    tasks = conn.execute(
        "SELECT id, structured_json FROM tasks WHERE structured_json IS NOT NULL"
    ).fetchall()
    inserted = 0
    for task in tasks:
        path = Path(task['structured_json'])
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        existing = {
            str(row['weibo_id']) for row in conn.execute(
                "SELECT weibo_id FROM posts WHERE task_id=?", (task['id'],)
            ).fetchall()
        }
        for post in payload.get('posts') or []:
            weibo_id = str(post.get('weibo_id', ''))
            if not weibo_id or weibo_id in existing:
                continue
            conn.execute(
                """INSERT INTO posts
                   (task_id, weibo_id, username, content, comment_count, post_time, url)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    task['id'], weibo_id, post.get('username', ''), post.get('content', ''),
                    int(post.get('comment_count_on_card', 0) or 0),
                    post.get('post_time', ''), post.get('url', ''),
                ),
            )
            existing.add(weibo_id)
            inserted += 1
    if inserted:
        log.info("Schema migration: restored %d zero-comment or missing posts", inserted)


def _backfill_unique_comment_counts(conn: sqlite3.Connection):
    """Record stored unique texts for tasks created before duplicate weights."""
    rows = conn.execute(
        "SELECT id, total_comments FROM tasks WHERE unique_comments IS NULL OR unique_comments=0"
    ).fetchall()
    updated = 0
    for row in rows:
        stored = conn.execute(
            "SELECT COUNT(*) FROM comments WHERE task_id=?", (row['id'],)
        ).fetchone()[0]
        unique_count = stored or row['total_comments'] or 0
        conn.execute("UPDATE tasks SET unique_comments=? WHERE id=?", (unique_count, row['id']))
        updated += 1
    if updated:
        log.info("Schema migration: backfilled unique counts for %d tasks", updated)


def _backfill_report_links(conn: sqlite3.Connection):
    """Associate legacy cached reports with the task completed just before them."""
    reports = []
    for path in REPORT_DIR.glob("report_*.md"):
        try:
            header = path.read_text(encoding="utf-8")[:500]
        except OSError:
            continue
        topic_match = re.search(r"<!--\s*话题:\s*(.*?)\s*-->", header)
        if topic_match:
            reports.append({"path": path, "topic": topic_match.group(1), "mtime": path.stat().st_mtime})

    # Repair duplicate links created by older migration logic.  Keep the task
    # whose completion time is closest to the report file timestamp.
    linked_rows = conn.execute(
        "SELECT id, completed_at, report_path FROM tasks WHERE report_path IS NOT NULL"
    ).fetchall()
    linked_by_path = {}
    for row in linked_rows:
        linked_by_path.setdefault(row['report_path'], []).append(dict(row))
    report_mtimes = {str(report['path']): report['mtime'] for report in reports}
    for report_path, linked_tasks in linked_by_path.items():
        if len(linked_tasks) < 2 or report_path not in report_mtimes:
            continue

        def link_distance(task):
            try:
                completed_ts = datetime.fromisoformat(task['completed_at']).replace(tzinfo=timezone.utc).timestamp()
                delta = report_mtimes[report_path] - completed_ts
                return delta if delta >= 0 else float('inf')
            except (TypeError, ValueError):
                return float('inf')

        keeper = min(linked_tasks, key=link_distance)
        duplicate_ids = [task['id'] for task in linked_tasks if task['id'] != keeper['id']]
        conn.executemany("UPDATE tasks SET report_path=NULL WHERE id=?", [(task_id,) for task_id in duplicate_ids])
        log.info("Schema migration: removed %d duplicate report links", len(duplicate_ids))

    used_paths = {
        Path(row[0]) for row in conn.execute(
            "SELECT DISTINCT report_path FROM tasks WHERE report_path IS NOT NULL"
        ).fetchall()
    }
    tasks = conn.execute(
        """SELECT id, topic, completed_at FROM tasks
           WHERE status='completed' AND report_path IS NULL AND completed_at IS NOT NULL
           ORDER BY completed_at DESC"""
    ).fetchall()
    linked = 0
    for row in tasks:
        task = dict(row)
        try:
            completed_ts = datetime.fromisoformat(task['completed_at']).replace(tzinfo=timezone.utc).timestamp()
        except (TypeError, ValueError):
            continue
        candidates = [
            report for report in reports
            if report['path'] not in used_paths
            and report['topic'] == task['topic']
            and 0 <= report['mtime'] - completed_ts <= 20 * 60
        ]
        if not candidates:
            continue
        report = min(candidates, key=lambda item: item['mtime'] - completed_ts)
        conn.execute("UPDATE tasks SET report_path=? WHERE id=?", (str(report['path']), task['id']))
        used_paths.add(report['path'])
        linked += 1
    if linked:
        log.info("Schema migration: linked %d legacy AI reports", linked)


def create_task(topic: str, source: str = "crawler") -> int:
    """创建新任务，返回 task_id"""
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO tasks (topic, source, status, updated_at)
               VALUES (?, ?, 'pending', CURRENT_TIMESTAMP)""",
            (topic, source)
        )
        conn.commit()
        task_id = cur.lastrowid
        log.info("创建任务 #%d: 话题='%s', 来源=%s", task_id, topic, source)
        return task_id
    finally:
        conn.close()


def update_task_status(task_id: int, status: str, error_message: str = None):
    """更新任务状态"""
    conn = get_connection()
    try:
        _ensure_columns(conn, 'tasks', TASK_COLUMN_MIGRATIONS)
        if status in {"completed", "failed"}:
            conn.execute(
                """UPDATE tasks SET status=?, error_message=?,
                   updated_at=CURRENT_TIMESTAMP, completed_at=CURRENT_TIMESTAMP
                   WHERE id=?""",
                (status, error_message, task_id)
            )
        else:
            conn.execute(
                """UPDATE tasks SET status=?, error_message=?,
                   updated_at=CURRENT_TIMESTAMP, completed_at=NULL WHERE id=?""",
                (status, error_message, task_id)
            )
        conn.commit()
        log.info("任务 #%d 状态更新: %s", task_id, status)
    finally:
        conn.close()


def touch_task(task_id: int) -> bool:
    """刷新运行中任务的心跳；任务已结束时不再改写。"""
    conn = get_connection()
    try:
        _ensure_columns(conn, 'tasks', TASK_COLUMN_MIGRATIONS)
        placeholders = ','.join('?' for _ in ACTIVE_TASK_STATUSES)
        cur = conn.execute(
            f"""UPDATE tasks SET updated_at=CURRENT_TIMESTAMP
                WHERE id=? AND status IN ({placeholders})""",
            (task_id, *ACTIVE_TASK_STATUSES),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def fail_task_if_active(task_id: int, error_message: str) -> bool:
    """仅在任务仍运行时将其结案，避免覆盖已经完成的结果。"""
    conn = get_connection()
    try:
        _ensure_columns(conn, 'tasks', TASK_COLUMN_MIGRATIONS)
        placeholders = ','.join('?' for _ in ACTIVE_TASK_STATUSES)
        cur = conn.execute(
            f"""UPDATE tasks SET status='failed', error_message=?,
                updated_at=CURRENT_TIMESTAMP, completed_at=CURRENT_TIMESTAMP
                WHERE id=? AND status IN ({placeholders})""",
            (error_message, task_id, *ACTIVE_TASK_STATUSES),
        )
        conn.commit()
        if cur.rowcount:
            log.warning("任务 #%d 已从运行状态安全结案: %s", task_id, error_message)
        return cur.rowcount > 0
    finally:
        conn.close()


def reconcile_stale_tasks(stale_after_minutes: int = 45,
                          now: datetime = None) -> int:
    """将长时间没有心跳的运行中任务标记为失败。"""
    if stale_after_minutes <= 0:
        raise ValueError("stale_after_minutes must be positive")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is not None:
        current = current.astimezone(timezone.utc).replace(tzinfo=None)
    cutoff = current - timedelta(minutes=stale_after_minutes)
    cutoff_text = cutoff.strftime('%Y-%m-%d %H:%M:%S')
    completed_text = current.strftime('%Y-%m-%d %H:%M:%S')

    conn = get_connection()
    try:
        _ensure_columns(conn, 'tasks', TASK_COLUMN_MIGRATIONS)
        placeholders = ','.join('?' for _ in ACTIVE_TASK_STATUSES)
        message = (
            f"任务超过 {stale_after_minutes} 分钟没有进度，可能因页面刷新或会话中断而停止。"
        )
        cur = conn.execute(
            f"""UPDATE tasks SET status='failed', error_message=?,
                updated_at=?, completed_at=?
                WHERE status IN ({placeholders})
                  AND COALESCE(updated_at, created_at) < ?""",
            (message, completed_text, completed_text, *ACTIVE_TASK_STATUSES, cutoff_text),
        )
        conn.commit()
        if cur.rowcount:
            log.warning("自动结束 %d 个陈旧任务", cur.rowcount)
        return cur.rowcount
    finally:
        conn.close()


def update_task_results(task_id: int, total: int, pos: int, neg: int, neu: int,
                        wordcloud_path: str = None, keywords_json: str = None,
                        structured_json: str = None, total_posts: int = 0,
                        raw_comments: int = 0, expected_comments: int = None,
                        fetched_comments: int = None, coverage_pct: float = None,
                        requested_model: str = None, effective_model: str = None,
                        model_version: str = None, fallback_used: bool = False,
                        fallback_reason: str = None, quality_status: str = "unknown",
                        quality_issues_json: str = None, unique_comments: int = None,
                        sampling_json: str = None, representation_status: str = "unknown",
                        processing_time: float = None, model_memory: float = None):
    """更新任务的统计结果"""
    conn = get_connection()
    try:
        # cache_resource may survive a source hot reload. Ensure the write schema
        # at the point of use so a stale initialization cache cannot break tasks.
        _ensure_columns(conn, 'tasks', TASK_COLUMN_MIGRATIONS)
        conn.execute(
            """UPDATE tasks SET total_comments=?, unique_comments=?, total_posts=?, pos_count=?, neg_count=?, neu_count=?,
               wordcloud_path=?, keywords_json=?, structured_json=?, raw_comments=?,
               expected_comments=?, fetched_comments=?, coverage_pct=?, requested_model=?,
               effective_model=?, model_version=?, fallback_used=?, fallback_reason=?,
               processing_time=?, model_memory=?, quality_status=?, quality_issues_json=?,
               sampling_json=?, representation_status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?""",
            (total, unique_comments if unique_comments is not None else total, total_posts, pos, neg, neu, wordcloud_path, keywords_json, structured_json,
             raw_comments, expected_comments, fetched_comments, coverage_pct, requested_model,
             effective_model, model_version, int(bool(fallback_used)), fallback_reason,
             processing_time, model_memory, quality_status, quality_issues_json,
             sampling_json, representation_status, task_id)
        )
        conn.commit()
        log.info("任务 #%d 统计已更新: 帖子%d, 评论%d, 积极%d, 消极%d, 中性%d",
                 task_id, total_posts, total, pos, neg, neu)
    finally:
        conn.close()


def insert_posts(task_id: int, posts: list[dict]) -> list[int]:
    """
    批量插入帖子数据，返回 post_id 列表。

    Args:
        task_id: 任务ID
        posts: [{'weibo_id': str, 'username': str, 'content': str,
                 'comment_count': int, 'post_time': str, 'url': str}, ...]

    Returns:
        [post_id, ...] 按插入顺序
    """
    conn = get_connection()
    post_ids = []
    try:
        for p in posts:
            cur = conn.execute(
                """INSERT INTO posts (task_id, weibo_id, username, content, comment_count, post_time, url)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (task_id, p['weibo_id'], p.get('username', ''),
                 p.get('content', ''), p.get('comment_count', 0),
                 p.get('post_time', ''), p.get('url', ''))
            )
            post_ids.append(cur.lastrowid)
        conn.commit()
        log.info("任务 #%d: 插入 %d 条帖子", task_id, len(post_ids))
        return post_ids
    finally:
        conn.close()


def insert_comments(task_id: int, df: pd.DataFrame):
    """批量插入评论数据（支持可选的 post_id 列）"""
    conn = get_connection()
    try:
        has_post_id = 'post_id' in df.columns
        records = []
        for _, row in df.iterrows():
            record = [
                task_id,
                str(row.get("评论内容", "")),
                str(row.get("clean_text", "")),
                str(row.get("nlp_result", "")),
                float(row.get("nlp_score", 0)),
                float(row.get("nlp_confidence", 0)),
                max(int(row.get("duplicate_count", 1)), 1),
            ]
            if has_post_id:
                record.append(int(row.get("post_id", 0)) if pd.notna(row.get("post_id")) else None)
            records.append(record)

        if has_post_id:
            conn.executemany(
                "INSERT INTO comments (task_id, content, cleaned_content, nlp_result, nlp_score, nlp_confidence, duplicate_count, post_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", records)
        else:
            conn.executemany(
                "INSERT INTO comments (task_id, content, cleaned_content, nlp_result, nlp_score, nlp_confidence, duplicate_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", records)
        conn.commit()
        log.info("任务 #%d: 插入 %d 条评论", task_id, len(records))
    finally:
        conn.close()


def insert_keywords(task_id: int, keywords: list):
    """批量插入关键词"""
    conn = get_connection()
    try:
        conn.executemany(
            "INSERT INTO keywords (task_id, word, frequency) VALUES (?, ?, ?)",
            [(task_id, w, f) for w, f in keywords]
        )
        conn.commit()
        log.info("任务 #%d: 插入 %d 个关键词", task_id, len(keywords))
    finally:
        conn.close()


def update_task_report(task_id: int, report_path: str, provider: str = None):
    """Persist an AI report so historical tasks can restore it."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE tasks SET report_path=?, report_provider=? WHERE id=?",
            (report_path, provider, task_id),
        )
        conn.commit()
        log.info("任务 #%d AI 报告已关联: %s", task_id, report_path)
    finally:
        conn.close()


def get_task(task_id: int) -> dict:
    """获取单个任务详情"""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_tasks(limit: int = 50) -> list:
    """获取历史任务列表"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_task_comments(task_id: int) -> pd.DataFrame:
    """获取某个任务的所有评论"""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM comments WHERE task_id=? ORDER BY id", (task_id,)
        )
        columns = [column[0] for column in (cursor.description or [])]
        return pd.DataFrame(
            [tuple(row) for row in cursor.fetchall()], columns=columns
        )
    finally:
        conn.close()


def get_task_keywords(task_id: int) -> list:
    """获取某个任务的关键词"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT word, frequency FROM keywords WHERE task_id=? ORDER BY frequency DESC",
            (task_id,)
        ).fetchall()
        return [(r["word"], r["frequency"]) for r in rows]
    finally:
        conn.close()


def delete_task(task_id: int):
    """删除任务及其关联数据"""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        conn.commit()
        log.info("任务 #%d 已删除", task_id)
    finally:
        conn.close()


def get_structured_data(task_id: int) -> dict:
    """
    获取帖子↔评论结构化数据 — 供 Agent API 消费。

    Returns:
        {
            'topic': str,
            'total_posts': int,
            'total_comments': int,
            'pos/neg/neu_count': int,
            'posts': [
                {
                    'weibo_id': str,
                    'username': str,
                    'content': str,
                    'post_time': str,
                    'comment_count': int,
                    'comments': [
                        {'content': str, 'nlp_result': str, 'nlp_score': float},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    conn = get_connection()
    try:
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not task:
            return None
        task = dict(task)

        posts = conn.execute(
            "SELECT * FROM posts WHERE task_id=? ORDER BY id", (task_id,)
        ).fetchall()

        result = {
            'topic': task['topic'],
            'source': task['source'],
            'total_posts': task['total_posts'],
            'total_comments': task['total_comments'],
            'pos_count': task['pos_count'],
            'neg_count': task['neg_count'],
            'neu_count': task['neu_count'],
            'created_at': task['created_at'],
            'requested_model': task.get('requested_model'),
            'effective_model': task.get('effective_model'),
            'coverage_pct': task.get('coverage_pct'),
            'quality_status': task.get('quality_status'),
            'posts': [],
        }

        for p in posts:
            p = dict(p)
            post_comments = conn.execute(
                "SELECT content, cleaned_content, nlp_result, nlp_score, nlp_confidence, duplicate_count "
                "FROM comments WHERE post_id=? ORDER BY id",
                (p['id'],)
            ).fetchall()

            result['posts'].append({
                'weibo_id': p['weibo_id'],
                'username': p['username'],
                'content': p['content'],
                'post_time': p['post_time'],
                'comment_count': p['comment_count'],
                'url': p['url'],
                'comments': [dict(c) for c in post_comments],
            })

        return result
    finally:
        conn.close()


def get_task_posts(task_id: int) -> list:
    """获取某个任务的所有帖子"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM posts WHERE task_id=? ORDER BY id", (task_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
