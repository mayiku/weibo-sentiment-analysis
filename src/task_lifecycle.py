"""Hot-reload-safe task lifecycle operations.

Streamlit Cloud can retain an older ``src.database`` module while reloading
``app.py``.  These wrappers use the native implementation when available and
fall back to compatible SQL when the cached module predates schema v5.
"""
from datetime import datetime, timedelta, timezone

from src import database as _database
from src.logger import get_logger

log = get_logger(__name__)

ACTIVE_TASK_STATUSES = (
    "pending", "crawling", "cleaning", "analyzing", "generating_wordcloud"
)


def _ensure_updated_at(conn) -> None:
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
    }
    if "updated_at" not in columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN updated_at TIMESTAMP")


def touch_task(task_id: int) -> bool:
    native = getattr(_database, "touch_task", None)
    if native:
        return native(task_id)
    conn = _database.get_connection()
    try:
        _ensure_updated_at(conn)
        placeholders = ",".join("?" for _ in ACTIVE_TASK_STATUSES)
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
    native = getattr(_database, "fail_task_if_active", None)
    if native:
        return native(task_id, error_message)
    conn = _database.get_connection()
    try:
        _ensure_updated_at(conn)
        placeholders = ",".join("?" for _ in ACTIVE_TASK_STATUSES)
        cur = conn.execute(
            f"""UPDATE tasks SET status='failed', error_message=?,
                updated_at=CURRENT_TIMESTAMP, completed_at=CURRENT_TIMESTAMP
                WHERE id=? AND status IN ({placeholders})""",
            (error_message, task_id, *ACTIVE_TASK_STATUSES),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def reconcile_stale_tasks(stale_after_minutes: int = 45,
                          now: datetime = None) -> int:
    native = getattr(_database, "reconcile_stale_tasks", None)
    if native:
        return native(stale_after_minutes=stale_after_minutes, now=now)
    if stale_after_minutes <= 0:
        raise ValueError("stale_after_minutes must be positive")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is not None:
        current = current.astimezone(timezone.utc).replace(tzinfo=None)
    cutoff = current - timedelta(minutes=stale_after_minutes)
    current_text = current.strftime("%Y-%m-%d %H:%M:%S")
    cutoff_text = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    conn = _database.get_connection()
    try:
        _ensure_updated_at(conn)
        placeholders = ",".join("?" for _ in ACTIVE_TASK_STATUSES)
        message = (
            f"任务超过 {stale_after_minutes} 分钟没有进度，"
            "可能因页面刷新或会话中断而停止。"
        )
        cur = conn.execute(
            f"""UPDATE tasks SET status='failed', error_message=?,
                updated_at=?, completed_at=?
                WHERE status IN ({placeholders})
                  AND COALESCE(updated_at, created_at) < ?""",
            (
                message, current_text, current_text,
                *ACTIVE_TASK_STATUSES, cutoff_text,
            ),
        )
        conn.commit()
        if cur.rowcount:
            log.warning("兼容模式自动结束 %d 个陈旧任务", cur.rowcount)
        return cur.rowcount
    finally:
        conn.close()
