import json
import tempfile
import unittest
from pathlib import Path

from src.quality import (
    assess_result_quality,
    enrich_quality_with_sampling,
    load_crawl_metadata,
)


class QualityAssessmentTests(unittest.TestCase):
    def test_low_coverage_and_fallback_are_warnings(self):
        result = assess_result_quality(
            total=100,
            positive=60,
            negative=20,
            neutral=20,
            coverage_pct=4.3,
            fallback_used=True,
            raw_comments=110,
        )
        self.assertEqual(result["status"], "warning")
        self.assertEqual(
            {issue["code"] for issue in result["issues"]},
            {"low_coverage", "model_fallback"},
        )

    def test_single_class_distribution_is_flagged(self):
        result = assess_result_quality(
            total=100, positive=0, negative=0, neutral=100
        )
        self.assertEqual(result["status"], "warning")
        self.assertIn("single_class_distribution", [i["code"] for i in result["issues"]])

    def test_partial_model_fallback_reports_affected_scope(self):
        result = assess_result_quality(
            total=100,
            positive=50,
            negative=30,
            neutral=20,
            fallback_used=True,
            fallback_count=2,
        )
        fallback_issue = next(
            issue for issue in result['issues'] if issue['code'] == 'model_fallback'
        )
        self.assertIn('2 条评论局部使用备用模型', fallback_issue['message'])

    def test_count_mismatch_is_invalid(self):
        result = assess_result_quality(
            total=10, positive=5, negative=2, neutral=2
        )
        self.assertEqual(result["status"], "invalid")

    def test_structured_sidecar_calculates_weighted_coverage(self):
        payload = {
            "total_posts": 2,
            "total_comments": 30,
            "posts": [
                {"comment_count_on_card": 100, "comments": ["a"] * 20},
                {"comment_count_on_card": 50, "comments": ["b"] * 10},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = load_crawl_metadata(str(path))
        self.assertEqual(result["expected_comments"], 150)
        self.assertEqual(result["fetched_comments"], 30)
        self.assertEqual(result["coverage_pct"], 20.0)
        self.assertEqual(result["representation_status"], "partial")
        self.assertEqual(result["dominant_post_share_pct"], 66.7)
        self.assertEqual(result["dominant_post_coverage_pct"], 20.0)
        self.assertEqual(result["coverage_excluding_dominant_pct"], 20.0)
        self.assertEqual(result["median_post_coverage_pct"], 20.0)
        self.assertEqual(result["total_posts"], 2)
        self.assertEqual(result["active_post_count"], 2)
        self.assertEqual(result["zero_comment_post_count"], 0)
        self.assertEqual(result["per_post"][0]["fetched_comments"], 20)

    def test_dominant_undercovered_post_limits_representativeness(self):
        payload = {
            "posts": [
                {"weibo_id": "hot", "comment_count_on_card": 900, "comments": ["a"] * 90},
                {"weibo_id": "small", "comment_count_on_card": 100, "comments": ["b"] * 100},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = load_crawl_metadata(str(path))
        self.assertEqual(result["coverage_pct"], 19.0)
        self.assertEqual(result["representation_status"], "limited")
        self.assertEqual(result["per_post"][0]["expected_share_pct"], 90.0)
        self.assertEqual(result["coverage_excluding_dominant_pct"], 100.0)

    def test_partial_sampling_changes_an_otherwise_good_quality_verdict(self):
        base = assess_result_quality(
            total=231, positive=124, negative=64, neutral=43,
            coverage_pct=59.8,
        )
        enriched = enrich_quality_with_sampling(base, {
            "coverage_pct": 59.8,
            "representation_status": "partial",
            "dominant_post_share_pct": 30.2,
            "dominant_post_coverage_pct": 12.8,
        })
        self.assertEqual(base["status"], "good")
        self.assertEqual(enriched["status"], "warning")
        issue = next(i for i in enriched["issues"] if i["code"] == "sampling_bias")
        self.assertIn("12.8%", issue["message"])
        self.assertIn("仅代表本次可见评论", issue["message"])


if __name__ == "__main__":
    unittest.main()
