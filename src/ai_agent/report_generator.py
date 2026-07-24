"""
报告生成器 — 整合情绪分析结果，调用 AI Provider 生成舆情报告。

核心流程:
  1. 从情绪分析 DataFrame 提取统计 + 样本
  2. 从数据库/JSON 获取帖子内容
  3. 构建 Prompt → AI Provider → Markdown 报告
  4. 缓存到 data/reports/ → 返回报告路径

支持 Provider:
  - DeepSeek (默认)
  - SiliconFlow (可选)

用法:
    from src.ai_agent.report_generator import ReportGenerator, generate_report

    gen = ReportGenerator()
    report = gen.generate(topic, stats, df, posts, keywords)
    print(report)  # Markdown 文本
"""

import json
import time
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import REPORT_DIR, AI_PROVIDER, CURRENT_API_KEY
from src.ai_agent.ai_provider import create_ai_client
from src.ai_agent.prompts import build_analysis_prompt, build_quick_prompt, SYSTEM_PROMPT
from src.logger import get_logger

log = get_logger(__name__)


class ReportGenerator:
    """
    舆情报告生成器。

    特性:
      - 自动提取评论样本（正/中/负各 8 条）
      - 缓存机制 (同一话题+数据哈希 → 复用报告)
      - 失败不影响原有流程（返回错误信息而非抛异常）
      - 支持多 AI Provider (SiliconFlow/DeepSeek)
    """

    _SOFT_GUARDRAIL_ISSUES = {
        'unsupported_temporal_claim',
        'nominal_count_presented_as_sample',
    }
    _SAMPLING_NOTICE = '> **证据边界提示：** 趋势判断基于单次采样，仅供参考。'
    _NOMINAL_COUNT_NOTICE = (
        '> **数据口径提示：** 帖子评论数为微博卡片标称值；'
        '实际分析样本数以本报告的采样说明和评论总数为准。'
    )

    def __init__(self, provider: str = None):
        """
        Args:
            provider: "siliconflow" 或 "deepseek"，默认使用环境变量 AI_PROVIDER
        """
        self.provider = provider or AI_PROVIDER
        self.client = create_ai_client(self.provider)

    # ── 主入口 ──────────────────────────────────────────

    def generate(self, topic: str, stats: dict, df,
                 posts: list = None, keywords: list = None,
                 use_cache: bool = True, quick: bool = False,
                 sampling: dict = None) -> dict:
        """
        生成舆情分析报告。

        Args:
            topic: 话题名称
            stats: get_sentiment_stats() 的返回
            df: 情感分析后的 DataFrame (含 nlp_result, 评论内容)
            posts: 帖子列表（来自 structured JSON 或数据库）
            keywords: [(word, freq), ...] TOP 关键词
            use_cache: 是否使用缓存
            quick: 是否快速模式（短报告）

        Returns:
            {
                'success': bool,
                'report': str,          # Markdown 报告文本
                'report_path': str,     # 缓存文件路径
                'from_cache': bool,
                'usage_info': dict,     # token 用量等
                'error': str or None,
            }
        """
        # ── 0. 检查 API Key ──
        if not getattr(self.client, 'api_key', CURRENT_API_KEY):
            return {
                'success': False,
                'report': '',
                'report_path': '',
                'from_cache': False,
                'usage_info': {},
                'error': f'AI API Key 未配置。请在 .env 文件中设置 {self.provider.upper()}_API_KEY',
            }

        # ── 1. 准备输入数据 ──
        if posts is None:
            posts = []
        if keywords is None:
            keywords = []
        stats = dict(stats or {})
        if df is not None:
            stats.setdefault('unique_total', len(df))

        samples = self._extract_samples(df) if df is not None else {
            'positive': [], 'neutral': [], 'negative': []
        }
        crawl_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # ── 2. 缓存检查 ──
        sampling = sampling or {}
        cache_key = self._cache_key(topic, stats, posts, keywords, sampling)
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached:
                log.info("【报告】命中缓存: %s", cache_key[:16])
                cached_issues = self._validate_report(cached, sampling)
                cached = self._add_guardrail_notices(cached, sampling, cached_issues)
                return {
                    'success': True,
                    'report': cached,
                    'report_path': str(self._cache_path(cache_key)),
                    'from_cache': True,
                    'usage_info': {},
                    'error': None,
                }

        # ── 3. 构建 Prompt ──
        if quick:
            prompt = build_quick_prompt(topic, stats, posts, keywords, sampling=sampling)
            temperature = 0.5
        else:
            prompt = build_analysis_prompt(
                topic, stats, posts, keywords, samples, crawl_time, sampling=sampling
            )
            temperature = 0.3

        log.info("【报告】Prompt 长度: %d 字符", len(prompt))

        # ── 4. 调用 LLM ──
        try:
            report = self.client.chat(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=temperature,
            )
        except Exception as e:
            log.error("【报告】LLM 调用异常: %s", e)
            report = None

        if not report:
            provider_error = getattr(self.client, 'last_error', None)
            return {
                'success': False,
                'report': '',
                'report_path': '',
                'from_cache': False,
                'usage_info': {},
                'error': provider_error or f'{self.provider.upper()} API 调用失败。请检查 API Key 和网络连接。',
            }

        expected_report_date = crawl_time[:10]
        guardrail_issues = self._validate_report(report, sampling, expected_report_date)
        guardrail_warnings = [
            issue for issue in guardrail_issues
            if issue in self._SOFT_GUARDRAIL_ISSUES
        ]
        blocking_issues = [
            issue for issue in guardrail_issues
            if issue not in self._SOFT_GUARDRAIL_ISSUES
        ]
        guardrail_repaired = False
        if guardrail_warnings:
            log.warning(
                "【报告校验】非阻断证据提示: %s",
                ", ".join(guardrail_warnings),
            )

        if blocking_issues and not quick:
            log.warning("【报告校验】发现数据一致性问题: %s", ", ".join(blocking_issues))
            repair_prompt = self._build_repair_prompt(
                report, blocking_issues, sampling, expected_report_date
            )
            try:
                repaired = self.client.chat(
                    prompt=repair_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.1,
                )
            except Exception as exc:
                log.error("【报告校验】修订调用失败: %s", exc)
                repaired = None
            repaired_issues = (
                self._validate_report(repaired, sampling, expected_report_date)
                if repaired else ['empty_repair']
            )
            repaired_blocking_issues = [
                issue for issue in repaired_issues
                if issue not in self._SOFT_GUARDRAIL_ISSUES
            ]
            if repaired and not repaired_blocking_issues:
                report = repaired
                guardrail_repaired = True
                guardrail_warnings = [
                    issue for issue in repaired_issues
                    if issue in self._SOFT_GUARDRAIL_ISSUES
                ]
            else:
                log.warning(
                    "【报告校验】修订后仍未通过: %s",
                    ", ".join(repaired_blocking_issues),
                )
                return {
                    'success': False,
                    'report': '', 'report_path': '', 'from_cache': False,
                    'usage_info': {
                        'guardrail_issues': blocking_issues,
                        'repaired_issues': repaired_issues,
                    },
                    'error': 'AI 报告存在数据一致性问题，自动修订失败。请重试。',
                }

        report = self._add_guardrail_notices(report, sampling, guardrail_warnings)

        # ── 5. 保存缓存 ──
        cache_path = self._save_cache(cache_key, report, topic)

        log.info("【报告】生成成功: %d 字符 → %s", len(report), cache_path)

        return {
            'success': True,
            'report': report,
            'report_path': str(cache_path),
            'from_cache': False,
            'usage_info': {
                'guardrail_repaired': guardrail_repaired,
                'guardrail_warnings': guardrail_warnings,
            },
            'error': None,
        }

    # ── 样本提取 ────────────────────────────────────────

    def _extract_samples(self, df, n: int = 8) -> dict:
        """从情感分析 DataFrame 提取各情绪样本评论"""
        samples = {'positive': [], 'neutral': [], 'negative': []}

        def comment_text(row) -> str:
            # Pipeline frames use 评论内容; persisted history frames use content.
            return str(row.get('评论内容') or row.get('content') or '').strip()

        sentiment_map = {
            '积极': 'positive', 'positive': 'positive',
            '中性': 'neutral', 'neutral': 'neutral',
            '消极': 'negative', 'negative': 'negative',
        }

        for _, row in df.iterrows():
            result = str(row.get('nlp_result', '')).strip()
            key = sentiment_map.get(result)
            if key and len(samples[key]) < n:
                comment = comment_text(row)
                if comment and comment not in samples[key]:
                    samples[key].append(comment)

        # 补齐不足的样本
        for key in samples:
            if len(samples[key]) < n:
                # 从其他行随机补充
                for _, row in df.iterrows():
                    if len(samples[key]) >= n:
                        break
                    comment = comment_text(row)
                    if comment and comment not in samples[key]:
                        samples[key].append(comment)

        return samples

    # ── 缓存机制 ────────────────────────────────────────

    _TEMPORAL_PATTERNS = (
        r"正在从.{0,30}(?:转向|转移)",
        r"(?:仍将|将逐渐|将持续|将进一步)[^。！？\n]{0,16}"
        r"(?:上升|下降|升温|降温|扩散|转向|转移|扩大|收窄|加剧|缓解)",
        r"(?:正在|持续)(?:上升|下降|升温|降温|扩散)",
    )
    _CONDITIONAL_MARKERS = (
        r"(?:若|如果|如若|一旦|假设|可能|或将)",
        r"在[^。！？\n]{0,30}情况下",
        r"(?:取决于|视)[^。！？\n]{0,20}",
    )

    def _has_unconditional_temporal_claim(self, report: str) -> bool:
        """Return True only for trend language stated as fact, not scenarios."""
        for pattern in self._TEMPORAL_PATTERNS:
            for match in re.finditer(pattern, report):
                sentence_start = max(
                    report.rfind(mark, 0, match.start()) for mark in "。！？\n"
                ) + 1
                sentence_ends = [
                    end for mark in "。！？\n"
                    if (end := report.find(mark, match.end())) >= 0
                ]
                sentence_end = min(sentence_ends) if sentence_ends else len(report)
                context = report[sentence_start:sentence_end]
                if not any(
                    re.search(marker, context) for marker in self._CONDITIONAL_MARKERS
                ):
                    return True
        return False

    def _validate_report(
        self, report: str, sampling: dict, expected_report_date: str | None = None
    ) -> list[str]:
        """Reject unsupported trend claims and nominal/sample count confusion."""
        issues = []
        if not sampling.get('temporal_evidence'):
            if self._has_unconditional_temporal_claim(report):
                issues.append('unsupported_temporal_claim')

        if expected_report_date:
            date_match = re.search(r"报告日期\*{0,2}\s*[:：]\s*(\d{4}-\d{2}-\d{2})", report)
            if date_match and date_match.group(1) != expected_report_date:
                issues.append('incorrect_report_date')

        for post in sampling.get('per_post') or []:
            expected = post.get('expected_comments')
            fetched = post.get('fetched_comments')
            if not expected or fetched is None or expected <= fetched:
                continue
            number = f"{int(expected)}"
            for match in re.finditer(rf"{re.escape(number)}\s*条评论", report):
                context = report[max(0, match.start() - 24):match.end() + 12]
                if not any(label in context for label in ('标称', '显示', '卡片', '约')):
                    issues.append('nominal_count_presented_as_sample')
                    break
            if 'nominal_count_presented_as_sample' in issues:
                break
        return issues

    def _add_sampling_notice(self, report: str, sampling: dict) -> str:
        """Attach a visible caveat without hiding an otherwise useful report."""
        if sampling.get('temporal_evidence') or self._SAMPLING_NOTICE in report:
            return report
        return f"{self._SAMPLING_NOTICE}\n\n{report}"

    def _add_guardrail_notices(
        self, report: str, sampling: dict, issues: list[str]
    ) -> str:
        """Turn recoverable evidence issues into visible report caveats."""
        report = self._add_sampling_notice(report, sampling)
        if (
            'nominal_count_presented_as_sample' in issues
            and self._NOMINAL_COUNT_NOTICE not in report
        ):
            report = f"{self._NOMINAL_COUNT_NOTICE}\n\n{report}"
        return report

    def _build_repair_prompt(
        self, report: str, issues: list[str], sampling: dict,
        expected_report_date: str | None = None,
    ) -> str:
        post_evidence = [
            {
                'weibo_id': item.get('weibo_id'),
                'expected': item.get('expected_comments'),
                'analyzed': item.get('fetched_comments'),
                'coverage_pct': item.get('coverage_pct'),
            }
            for item in (sampling.get('per_post') or [])[:10]
        ]
        return f"""请修订以下舆情报告并返回完整 Markdown 报告。

必须修复的问题：{', '.join(issues)}
逐帖证据口径：{json.dumps(post_evidence, ensure_ascii=False)}

强制规则：
1. 微博标称评论数不能写成已分析评论数；每次引用必须同时标明实际分析数。
2. 本数据没有时间序列，不得使用“正在转向、仍将、将逐渐、持续上升/下降/扩散”等趋势断言。
3. 趋势章节只能写带明确条件的情景假设。
4. 保留原报告的数据、结构和有依据的洞察，不新增事实。
5. 报告日期必须为 {expected_report_date or '输入数据所示日期'}。

待修订报告：
---
{report}
---"""

    def _cache_key(self, topic: str, stats: dict, posts: list,
                   keywords: list, sampling: dict = None) -> str:
        """生成缓存键（基于输入数据的哈希）"""
        raw = json.dumps({
            'report_format': 7,
            'topic': topic,
            'stats': {k: v for k, v in stats.items() if k in ['positive', 'negative', 'neutral', 'total', 'unique_total']},
            'posts_count': len(posts),
            'keywords_top10': [w for w, _ in keywords[:10]],
            'sampling': {
                key: (sampling or {}).get(key) for key in (
                    'expected_comments', 'fetched_comments', 'coverage_pct',
                    'representation_status', 'dominant_post_share_pct'
                )
            },
            'per_post': [
                [item.get('weibo_id'), item.get('expected_comments'), item.get('fetched_comments')]
                for item in (sampling or {}).get('per_post', [])[:10]
            ],
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _cache_path(self, cache_key: str) -> Path:
        return REPORT_DIR / f"report_{cache_key}.md"

    def _save_cache(self, cache_key: str, report: str, topic: str) -> Path:
        """保存报告到缓存"""
        path = self._cache_path(cache_key)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = (
            f"<!-- 话题: {topic} -->\n"
            f"<!-- 生成时间: {timestamp} -->\n"
            f"<!-- cache_key: {cache_key} -->\n\n"
            f"{report}"
        )
        path.write_text(content, encoding='utf-8')
        log.info("【缓存】报告已保存: %s", path)
        return path

    def _load_cache(self, cache_key: str) -> Optional[str]:
        """加载缓存报告"""
        path = self._cache_path(cache_key)
        if path.exists():
            content = path.read_text(encoding='utf-8')
            # 去掉 HTML 注释头部
            lines = content.split('\n')
            report_lines = []
            in_header = True
            for line in lines:
                if in_header and line.startswith('<!--'):
                    continue
                in_header = False
                report_lines.append(line)
            return '\n'.join(report_lines).strip()
        return None

    def clear_cache(self):
        """清空所有缓存报告"""
        count = 0
        for f in REPORT_DIR.glob("report_*.md"):
            f.unlink()
            count += 1
        log.info("【缓存】清空 %d 个报告", count)
        return count


# ============================================================================
# 便捷函数
# ============================================================================

def generate_report(topic: str, stats: dict, df,
                    posts: list = None, keywords: list = None,
                    quick: bool = False, sampling: dict = None) -> dict:
    """
    一键生成舆情报告。

    Args:
        topic: 话题名称
        stats: 情绪统计
        df: 情感分析 DataFrame
        posts: 帖子列表
        keywords: 关键词
        quick: 快速模式

    Returns:
        {'success': bool, 'report': str, 'error': str or None}
    """
    gen = ReportGenerator()
    return gen.generate(
        topic, stats, df, posts=posts, keywords=keywords, quick=quick,
        sampling=sampling,
    )
