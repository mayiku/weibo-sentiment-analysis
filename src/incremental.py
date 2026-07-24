"""Persistent incremental crawl checkpoints and cumulative snapshots."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from src.database import get_connection
from src.logger import get_logger

log = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stable_key(weibo_id: str, text: str, comment_id: str = "") -> str:
    if comment_id:
        return f"id:{comment_id}"
    digest = hashlib.sha256(f"{weibo_id}|{text.strip()}".encode("utf-8")).hexdigest()
    return f"hash:{digest[:32]}"


def configure_series(topic: str, *, enabled: bool = False,
                     interval_hours: int = 6) -> dict[str, Any]:
    """Create or update an incremental collection series."""
    interval_hours = max(int(interval_hours), 1)
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO crawl_series (topic, enabled, interval_hours)
               VALUES (?, ?, ?)
               ON CONFLICT(topic) DO UPDATE SET
                   enabled=excluded.enabled,
                   interval_hours=excluded.interval_hours""",
            (topic, int(enabled), interval_hours),
        )
        conn.commit()
        return dict(conn.execute(
            "SELECT * FROM crawl_series WHERE topic=?", (topic,)
        ).fetchone())
    finally:
        conn.close()


def begin_run(topic: str, *, task_id: int | None = None,
              enabled: bool = False, interval_hours: int = 6) -> tuple[int, int]:
    series = configure_series(topic, enabled=enabled, interval_hours=interval_hours)
    conn = get_connection()
    try:
        claimed_next_run = None
        if enabled:
            claimed_next_run = (
                _utc_now() + timedelta(hours=max(int(interval_hours), 1))
            ).isoformat(timespec="seconds")
            conn.execute(
                "UPDATE crawl_series SET next_run_at=? WHERE id=?",
                (claimed_next_run, series["id"]),
            )
        cur = conn.execute(
            "INSERT INTO crawl_runs (series_id, task_id) VALUES (?, ?)",
            (series["id"], task_id),
        )
        conn.commit()
        return int(series["id"]), int(cur.lastrowid)
    finally:
        conn.close()


def fail_run(run_id: int, error: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE crawl_runs SET status='failed', error_message=?,
               completed_at=CURRENT_TIMESTAMP WHERE id=?""",
            (str(error)[:1000], run_id),
        )
        conn.commit()
    finally:
        conn.close()


def merge_snapshot(series_id: int, run_id: int,
                   posts: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Merge one crawl pass into persistent observations and return cumulative posts."""
    conn = get_connection()
    now = _utc_now().isoformat(timespec="seconds")
    try:
        # Fetch existing keys once. The previous implementation performed an
        # INSERT, optional UPDATE and COUNT query for every comment/post. That
        # was acceptable for local SQLite but caused hundreds of sequential
        # network round trips with Turso.
        existing_keys = {
            (str(row["weibo_id"]), str(row["comment_key"]))
            for row in conn.execute(
                """SELECT weibo_id, comment_key FROM crawl_observations
                   WHERE series_id=?""",
                (series_id,),
            ).fetchall()
        }

        observation_rows = []
        current_keys = set()
        checkpoint_payloads = []
        for post in posts:
            weibo_id = str(post.get("weibo_id", ""))
            if not weibo_id:
                continue
            records = post.get("comment_records") or []
            if not records:
                records = [{"text": text, "comment_id": ""}
                           for text in post.get("comments", []) if text]

            for record in records:
                text = str(record.get("text", "")).strip()
                if not text:
                    continue
                comment_id = str(record.get("comment_id", ""))
                key = _stable_key(weibo_id, text, comment_id)
                compound_key = (weibo_id, key)
                if compound_key in current_keys:
                    continue
                current_keys.add(compound_key)
                observation_rows.append(
                    (series_id, weibo_id, key, comment_id, text, now, now)
                )

            report = dict(post.get("fetch_report") or {})
            report.pop("comment_records", None)
            metadata = {
                "weibo_id": weibo_id,
                "username": post.get("username", ""),
                "post_content": post.get("post_content", post.get("content", "")),
                "post_time": post.get("post_time", ""),
                "comment_count": post.get("comment_count", -1),
                "url": post.get("url", ""),
                "fetch_method": post.get("fetch_method", "unknown"),
                "fetch_report": report,
            }
            checkpoint_payloads.append((post, report, metadata))

        new_comments = len(current_keys - existing_keys)
        if observation_rows:
            log.info("【增量】批量写入 %d 条观测记录...", len(observation_rows))
            conn.executemany(
                """INSERT INTO crawl_observations
                   (series_id, weibo_id, comment_key, comment_id, content,
                    first_seen_at, last_seen_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(series_id, weibo_id, comment_key) DO UPDATE SET
                       last_seen_at=excluded.last_seen_at""",
                observation_rows,
            )

        observed_counts = {
            str(row["weibo_id"]): int(row["observed_total"])
            for row in conn.execute(
                """SELECT weibo_id, COUNT(*) AS observed_total
                   FROM crawl_observations WHERE series_id=?
                   GROUP BY weibo_id""",
                (series_id,),
            ).fetchall()
        }
        checkpoint_rows = []
        for post, report, metadata in checkpoint_payloads:
            weibo_id = str(post.get("weibo_id", ""))
            checkpoint_rows.append((
                series_id,
                weibo_id,
                post.get("comment_count", -1),
                observed_counts.get(weibo_id, 0),
                str(report.get("last_cursor", "")),
                report.get("stop_reason", "unknown"),
                json.dumps(metadata, ensure_ascii=False),
                now,
            ))
        if checkpoint_rows:
            conn.executemany(
                """INSERT INTO crawl_checkpoints
                   (series_id, weibo_id, expected_total, observed_total,
                    last_cursor, stop_reason, metadata_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(series_id, weibo_id) DO UPDATE SET
                       expected_total=excluded.expected_total,
                       observed_total=excluded.observed_total,
                       last_cursor=excluded.last_cursor,
                       stop_reason=excluded.stop_reason,
                       metadata_json=excluded.metadata_json,
                       updated_at=excluded.updated_at""",
                checkpoint_rows,
            )

        total_unique = conn.execute(
            "SELECT COUNT(*) FROM crawl_observations WHERE series_id=?", (series_id,)
        ).fetchone()[0]
        series = dict(conn.execute(
            "SELECT * FROM crawl_series WHERE id=?", (series_id,)
        ).fetchone())
        next_run = None
        if series["enabled"]:
            next_run = (_utc_now() + timedelta(hours=series["interval_hours"])).isoformat(
                timespec="seconds"
            )
        conn.execute(
            """UPDATE crawl_series SET last_run_at=?, next_run_at=?,
               total_unique_comments=? WHERE id=?""",
            (now, next_run, total_unique, series_id),
        )
        conn.execute(
            """UPDATE crawl_runs SET status='completed', new_comments=?,
               total_unique_comments=?, completed_at=? WHERE id=?""",
            (new_comments, total_unique, now, run_id),
        )
        conn.commit()

        cumulative_posts = []
        checkpoints = conn.execute(
            "SELECT * FROM crawl_checkpoints WHERE series_id=? ORDER BY expected_total DESC",
            (series_id,),
        ).fetchall()
        observations_by_post = {}
        observation_results = conn.execute(
            """SELECT weibo_id, comment_id, content
               FROM crawl_observations WHERE series_id=?
               ORDER BY weibo_id, first_seen_at, comment_key""",
            (series_id,),
        ).fetchall()
        for row in observation_results:
            observations_by_post.setdefault(str(row["weibo_id"]), []).append(row)

        for checkpoint in checkpoints:
            meta = json.loads(checkpoint["metadata_json"] or "{}")
            rows = observations_by_post.get(str(checkpoint["weibo_id"]), [])
            comments = [row["content"] for row in rows]
            records = [
                {"comment_id": row["comment_id"], "text": row["content"]}
                for row in rows
            ]
            report = dict(meta.get("fetch_report") or {})
            expected = checkpoint["expected_total"]
            report.update({
                "actual_fetched": len(comments),
                "coverage_pct": round(min(len(comments) / expected * 100, 100.0), 1)
                if expected and expected > 0 else None,
                "stop_reason": checkpoint["stop_reason"],
                "incomplete": bool(expected and len(comments) < expected),
                "truncated": bool(expected and len(comments) < expected),
            })
            cumulative_posts.append({
                "weibo_id": checkpoint["weibo_id"],
                "username": meta.get("username", ""),
                "post_content": meta.get("post_content", ""),
                "post_time": meta.get("post_time", ""),
                "comment_count": expected,
                "url": meta.get("url", ""),
                "fetch_method": meta.get("fetch_method", "unknown"),
                "fetch_report": report,
                "comments": comments,
                "comment_records": records,
            })

        return cumulative_posts, {
            "series_id": series_id,
            "run_id": run_id,
            "new_comments": new_comments,
            "total_unique_comments": total_unique,
            "interval_hours": series["interval_hours"],
            "schedule_enabled": bool(series["enabled"]),
            "last_run_at": now,
            "next_run_at": next_run,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_series(topic: str) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM crawl_series WHERE topic=?", (topic,)).fetchone()
        if not row:
            return None
        result = dict(row)
        last_run = conn.execute(
            "SELECT * FROM crawl_runs WHERE series_id=? ORDER BY id DESC LIMIT 1",
            (result["id"],),
        ).fetchone()
        result["last_run"] = dict(last_run) if last_run else None
        return result
    finally:
        conn.close()


def get_series_snapshot(topic: str) -> list[dict[str, Any]]:
    """Rebuild the durable post/comment snapshot for disaster recovery."""
    conn = get_connection()
    try:
        series = conn.execute(
            "SELECT id FROM crawl_series WHERE topic=?", (topic,)
        ).fetchone()
        if not series:
            return []
        series_id = int(series["id"])
        checkpoints = conn.execute(
            "SELECT * FROM crawl_checkpoints WHERE series_id=? ORDER BY expected_total DESC",
            (series_id,),
        ).fetchall()
        observations = conn.execute(
            """SELECT weibo_id, comment_id, content
               FROM crawl_observations WHERE series_id=?
               ORDER BY weibo_id, first_seen_at, comment_key""",
            (series_id,),
        ).fetchall()
        observations_by_post: dict[str, list] = {}
        for row in observations:
            observations_by_post.setdefault(str(row["weibo_id"]), []).append(row)

        posts = []
        for checkpoint in checkpoints:
            meta = json.loads(checkpoint["metadata_json"] or "{}")
            rows = observations_by_post.get(str(checkpoint["weibo_id"]), [])
            if not rows:
                continue
            posts.append({
                "weibo_id": str(checkpoint["weibo_id"]),
                "username": meta.get("username", ""),
                "post_content": meta.get("post_content", ""),
                "post_time": meta.get("post_time", ""),
                "comment_count": checkpoint["expected_total"],
                "url": meta.get("url", ""),
                "comment_records": [
                    {"comment_id": row["comment_id"], "text": row["content"]}
                    for row in rows
                ],
            })
        return posts
    finally:
        conn.close()


def get_known_comment_ids(series_id: int, weibo_id: str) -> set[str]:
    """Return stable IDs already observed for one post."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT comment_id FROM crawl_observations
               WHERE series_id=? AND weibo_id=? AND comment_id IS NOT NULL
               AND comment_id!=''""",
            (series_id, str(weibo_id)),
        ).fetchall()
        return {str(row["comment_id"]) for row in rows}
    finally:
        conn.close()


def get_due_series() -> list[dict[str, Any]]:
    now = _utc_now().isoformat(timespec="seconds")
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM crawl_series WHERE enabled=1
               AND next_run_at IS NOT NULL AND next_run_at<=? ORDER BY next_run_at""",
            (now,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
