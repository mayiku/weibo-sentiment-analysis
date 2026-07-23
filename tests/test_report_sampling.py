import unittest

from src.ai_agent.prompts import build_analysis_prompt
from src.ai_agent.report_generator import ReportGenerator


class ReportSamplingTests(unittest.TestCase):
    def test_low_coverage_prompt_sets_evidence_boundary(self):
        prompt = build_analysis_prompt(
            "世界杯",
            {"total": 1606, "positive": 500, "neutral": 800, "negative": 306},
            posts=[], keywords=[], samples={},
            sampling={
                "expected_comments": 37982,
                "fetched_comments": 1633,
                "coverage_pct": 4.3,
                "representation_status": "limited",
            },
        )
        self.assertIn("采集覆盖率 4.3%", prompt)
        self.assertIn("不能代表整体舆情", prompt)
        self.assertIn("不得把横截面数据写成时间趋势", prompt)

    def test_prompt_distinguishes_nominal_and_analyzed_post_counts(self):
        prompt = build_analysis_prompt(
            "世界杯", {"total": 379, "unique_total": 370, "positive": 200, "neutral": 100, "negative": 79},
            posts=[{"weibo_id": "hot", "username": "品牌", "content": "帖子", "comment_count": 32565}],
            keywords=[], samples={},
            sampling={"coverage_pct": 1.2, "expected_comments": 32565, "fetched_comments": 379,
                      "representation_status": "limited", "per_post": [{
                          "weibo_id": "hot", "expected_comments": 32565,
                          "fetched_comments": 379, "coverage_pct": 1.2,
                      }]},
        )
        self.assertIn("微博标称 32565 条 | 实际分析 379 条 | 覆盖率 1.2%", prompt)
        self.assertIn("唯一评论文本**: 370", prompt)

    def test_report_validator_rejects_trend_and_nominal_confusion(self):
        generator = ReportGenerator.__new__(ReportGenerator)
        sampling = {"per_post": [{
            "expected_comments": 32565, "fetched_comments": 379,
        }]}
        issues = generator._validate_report(
            "用户关注点正在从赛事转向明星，该帖获得32565条评论。", sampling
        )
        self.assertIn("unsupported_temporal_claim", issues)
        self.assertIn("nominal_count_presented_as_sample", issues)
        safe = generator._validate_report(
            "本次样本同时涉及赛事和明星。微博标称32565条评论，实际分析379条。", sampling
        )
        self.assertEqual(safe, [])

    def test_report_validator_rejects_incorrect_report_date(self):
        generator = ReportGenerator.__new__(ReportGenerator)
        issues = generator._validate_report(
            "**报告日期**: 2026-07-23", {}, "2026-07-22"
        )
        self.assertIn("incorrect_report_date", issues)


if __name__ == "__main__":
    unittest.main()
