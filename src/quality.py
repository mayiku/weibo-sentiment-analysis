"""Analysis quality and provenance helpers.

The UI must not present a precise sentiment percentage without also carrying
the evidence needed to interpret it.  This module keeps those checks outside
Streamlit so they can be regression-tested and reused by other entry points.
"""
from __future__ import annotations

import json
from statistics import median
from pathlib import Path
from typing import Any


def load_crawl_metadata(structured_path: str | None) -> dict[str, Any]:
    """Return sampling metadata from a crawler structured sidecar."""
    empty = {
        "expected_comments": None,
        "fetched_comments": None,
        "coverage_pct": None,
        "total_posts": 0,
        "active_post_count": 0,
        "zero_comment_post_count": 0,
        "representation_status": "unknown",
        "dominant_post_share_pct": None,
        "dominant_post_coverage_pct": None,
        "coverage_excluding_dominant_pct": None,
        "median_post_coverage_pct": None,
        "per_post": [],
        "incremental": None,
    }
    if not structured_path:
        return empty

    path = Path(structured_path)
    if not path.exists():
        return empty

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return empty

    posts = payload.get("posts") or []
    expected_values = []
    per_post = []
    for post in posts:
        value = post.get("comment_count_on_card", post.get("comment_count", -1))
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = -1
        if value >= 0:
            expected_values.append(value)
        fetched_post = post.get("fetched_comment_count")
        if fetched_post is None:
            fetched_post = len(post.get("comments") or [])
        try:
            fetched_post = int(fetched_post or 0)
        except (TypeError, ValueError):
            fetched_post = 0
        post_coverage = post.get("coverage_pct")
        if post_coverage is None and value > 0:
            post_coverage = round(min(fetched_post / value * 100, 100.0), 1)
        per_post.append({
            "weibo_id": str(post.get("weibo_id", "")),
            "username": str(post.get("username", "")),
            "expected_comments": value if value >= 0 else None,
            "fetched_comments": fetched_post,
            "coverage_pct": post_coverage,
            "fetch_method": post.get("fetch_method", "unknown"),
            "stop_reason": post.get("stop_reason", "unknown"),
            "incomplete": bool(post.get("incomplete", value > fetched_post >= 0)),
            "pages": post.get("pages"),
        })

    fetched = payload.get("total_comments")
    if fetched is None:
        fetched = sum(len(post.get("comments") or []) for post in posts)
    fetched = int(fetched or 0)

    expected = sum(expected_values) if expected_values else None
    coverage = None
    if expected and expected > 0:
        coverage = round(min(fetched / expected * 100, 100.0), 1)

    dominant_share = None
    dominant_coverage = None
    coverage_excluding_dominant = None
    if expected and expected_values:
        dominant_expected = max(expected_values)
        dominant_share = round(dominant_expected / expected * 100, 1)
        dominant = next(
            (p for p in per_post if p["expected_comments"] == dominant_expected), None
        )
        dominant_coverage = dominant.get("coverage_pct") if dominant else None
        remaining_expected = expected - dominant_expected
        remaining_fetched = fetched - (dominant.get("fetched_comments", 0) if dominant else 0)
        if remaining_expected > 0:
            coverage_excluding_dominant = round(
                min(max(remaining_fetched, 0) / remaining_expected * 100, 100.0), 1
            )

    coverage_values = [
        float(item["coverage_pct"]) for item in per_post
        if item.get("coverage_pct") is not None and (item.get("expected_comments") or 0) > 0
    ]
    median_coverage = round(median(coverage_values), 1) if coverage_values else None
    active_post_count = sum(
        1 for item in per_post if (item.get("expected_comments") or 0) > 0
    )
    zero_comment_post_count = sum(
        1 for item in per_post if item.get("expected_comments") == 0
    )

    if coverage is None:
        representation = "unknown"
    elif coverage < 20 or (
        dominant_share is not None and dominant_share >= 50
        and dominant_coverage is not None and dominant_coverage < 20
    ):
        representation = "limited"
    elif coverage < 60:
        representation = "partial"
    else:
        representation = "good"

    for item in per_post:
        item_expected = item.get("expected_comments")
        item["expected_share_pct"] = (
            round(item_expected / expected * 100, 1)
            if expected and item_expected is not None else None
        )
    per_post.sort(key=lambda item: item.get("expected_comments") or -1, reverse=True)

    return {
        "expected_comments": expected,
        "fetched_comments": fetched,
        "coverage_pct": coverage,
        "total_posts": int(payload.get("total_posts") or len(posts)),
        "active_post_count": active_post_count,
        "zero_comment_post_count": zero_comment_post_count,
        "representation_status": representation,
        "dominant_post_share_pct": dominant_share,
        "dominant_post_coverage_pct": dominant_coverage,
        "coverage_excluding_dominant_pct": coverage_excluding_dominant,
        "median_post_coverage_pct": median_coverage,
        "per_post": per_post,
        "incremental": payload.get("incremental"),
    }


def assess_result_quality(
    *,
    total: int,
    positive: int,
    negative: int,
    neutral: int,
    coverage_pct: float | None = None,
    fallback_used: bool = False,
    raw_comments: int | None = None,
) -> dict[str, Any]:
    """Classify whether a result is safe to present as a completed analysis."""
    issues: list[dict[str, str]] = []
    counts_total = int(positive) + int(negative) + int(neutral)

    if total <= 0:
        issues.append({"code": "empty_result", "severity": "invalid", "message": "没有可分析的有效评论。"})
    if counts_total != total:
        issues.append({
            "code": "count_mismatch",
            "severity": "invalid",
            "message": f"情绪分类合计 {counts_total} 与样本数 {total} 不一致。",
        })
    if total > 0 and max(positive, negative, neutral) == total:
        issues.append({
            "code": "single_class_distribution",
            "severity": "warning",
            "message": "全部评论被判定为同一种情绪，建议复核模型和输入数据。",
        })
    if 0 < total < 30:
        issues.append({
            "code": "small_sample",
            "severity": "warning",
            "message": f"有效样本仅 {total} 条，结论稳定性有限。",
        })
    if coverage_pct is not None and coverage_pct < 20:
        issues.append({
            "code": "low_coverage",
            "severity": "warning",
            "message": f"评论采集覆盖率仅 {coverage_pct:.1f}%，结果可能存在抽样偏差。",
        })
    if fallback_used:
        issues.append({
            "code": "model_fallback",
            "severity": "warning",
            "message": "所选模型运行失败，本次结果由备用模型生成。",
        })
    if raw_comments is not None and raw_comments > 0 and total / raw_comments < 0.5:
        issues.append({
            "code": "high_cleaning_loss",
            "severity": "warning",
            "message": f"清洗后仅保留 {total}/{raw_comments} 条评论，请检查去重和过滤规则。",
        })

    if any(issue["severity"] == "invalid" for issue in issues):
        status = "invalid"
    elif issues:
        status = "warning"
    else:
        status = "good"

    return {"status": status, "issues": issues}
