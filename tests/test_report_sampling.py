import unittest
from pathlib import Path

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
        conditional = generator._validate_report(
            "如果负面反馈持续上升，则应启动专项回应。", sampling
        )
        self.assertEqual(conditional, [])
        postfix_conditional = generator._validate_report(
            "负面反馈持续上升的可能性取决于后续回应。", sampling
        )
        self.assertEqual(postfix_conditional, [])
        operational = generator._validate_report(
            "建议团队将持续监测关键词变化。", sampling
        )
        self.assertEqual(operational, [])

    def test_report_validator_rejects_incorrect_report_date(self):
        generator = ReportGenerator.__new__(ReportGenerator)
        issues = generator._validate_report(
            "**报告日期**: 2026-07-23", {}, "2026-07-22"
        )
        self.assertIn("incorrect_report_date", issues)

    def test_single_snapshot_report_gets_visible_non_blocking_notice(self):
        generator = ReportGenerator.__new__(ReportGenerator)
        report = generator._add_sampling_notice(
            "## 趋势展望\n讨论热度正在上升。", {}
        )
        self.assertIn("趋势判断基于单次采样，仅供参考", report)
        self.assertIn("讨论热度正在上升", report)

    def test_temporal_evidence_does_not_add_single_snapshot_notice(self):
        generator = ReportGenerator.__new__(ReportGenerator)
        report = generator._add_sampling_notice(
            "## 趋势展望\n讨论热度正在上升。",
            {"temporal_evidence": True},
        )
        self.assertNotIn("趋势判断基于单次采样", report)

    def test_temporal_claim_is_displayed_without_repair_call(self):
        class FakeClient:
            api_key = 'test-key'

            def __init__(self):
                self.calls = 0

            def chat(self, **kwargs):
                self.calls += 1
                return "## 趋势展望\n讨论热度持续上升。"

        generator = ReportGenerator.__new__(ReportGenerator)
        generator.provider = 'deepseek'
        generator.client = FakeClient()
        generator._save_cache = lambda *args: Path('/tmp/report.md')

        result = generator.generate(
            topic='测试',
            stats={'total': 10, 'positive': 5, 'neutral': 3, 'negative': 2},
            df=None,
            posts=[],
            keywords=[],
            use_cache=False,
            sampling={},
        )

        self.assertTrue(result['success'])
        self.assertEqual(generator.client.calls, 1)
        self.assertIn('趋势判断基于单次采样，仅供参考', result['report'])
        self.assertIn('unsupported_temporal_claim', result['usage_info']['guardrail_warnings'])

    def test_nominal_count_confusion_is_displayed_with_scope_notice(self):
        class FakeClient:
            api_key = 'test-key'

            def __init__(self):
                self.calls = 0

            def chat(self, **kwargs):
                self.calls += 1
                return "## 样本\n该帖子获得32565条评论。"

        generator = ReportGenerator.__new__(ReportGenerator)
        generator.provider = 'deepseek'
        generator.client = FakeClient()
        generator._save_cache = lambda *args: Path('/tmp/report.md')

        result = generator.generate(
            topic='测试',
            stats={'total': 379, 'positive': 200, 'neutral': 100, 'negative': 79},
            df=None,
            posts=[],
            keywords=[],
            use_cache=False,
            sampling={'per_post': [{
                'expected_comments': 32565,
                'fetched_comments': 379,
            }]},
        )

        self.assertTrue(result['success'])
        self.assertEqual(generator.client.calls, 1)
        self.assertIn('微博卡片标称值', result['report'])
        self.assertIn('32565条评论', result['report'])
        self.assertIn(
            'nominal_count_presented_as_sample',
            result['usage_info']['guardrail_warnings'],
        )


if __name__ == "__main__":
    unittest.main()
