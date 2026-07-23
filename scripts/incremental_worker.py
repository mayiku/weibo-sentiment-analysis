#!/usr/bin/env python3
"""Run due incremental Weibo collection series.

Examples:
    python3 scripts/incremental_worker.py --once
    python3 scripts/incremental_worker.py --poll-seconds 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crawler import crawl_topic_v2
from src.database import init_db
from src.incremental import get_due_series
from src.logger import get_logger

log = get_logger("incremental_worker")


def run_due() -> int:
    due = get_due_series()
    for series in due:
        topic = series["topic"]
        try:
            log.info("【定时增量】开始: %s", topic)
            path, comments = crawl_topic_v2(
                topic,
                incremental=True,
                schedule_enabled=True,
                interval_hours=series["interval_hours"],
            )
            log.info("【定时增量】完成: %s | 累计 %d 条 | %s", topic, len(comments), path)
        except Exception as exc:
            log.exception("【定时增量】失败: %s | %s", topic, exc)
    return len(due)


def main() -> int:
    parser = argparse.ArgumentParser(description="执行到期的微博增量采集任务")
    parser.add_argument("--once", action="store_true", help="检查一次后退出")
    parser.add_argument(
        "--poll-seconds", type=int, default=60,
        help="常驻模式下的队列检查间隔，默认 60 秒",
    )
    args = parser.parse_args()
    init_db()
    while True:
        count = run_due()
        if args.once:
            log.info("队列检查完成，到期任务 %d 个", count)
            return 0
        time.sleep(max(args.poll_seconds, 10))


if __name__ == "__main__":
    raise SystemExit(main())
