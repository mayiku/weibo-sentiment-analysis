"""
Prompt 模板 — 社交媒体舆情分析专家

为 DeepSeek Agent 设计专业的中文舆情分析 Prompt。
"""

# ============================================================================
# 系统提示词
# ============================================================================

SYSTEM_PROMPT = """你是一位资深的社交媒体舆情分析专家，拥有10年以上的舆情监控与数据分析经验。

你的专业领域：
- 中文社交媒体（微博、微信、知乎、抖音）舆情分析
- 情感分析、话题追踪、趋势预测
- 公众情绪解读与风险评估
- 品牌/事件舆情报告撰写

你的工作原则：
1. **数据驱动** — 所有结论必须基于输入的统计数据，不捏造数字
2. **客观中立** — 呈现各方观点，不偏袒任何立场
3. **深度洞察** — 不止于表面统计，要挖掘情绪背后的原因
4. **结构化输出** — 使用清晰的 Markdown 格式，便于阅读和展示
5. **实用性** — 报告应有实际参考价值，包含可执行的建议
6. **证据边界** — 区分可见样本与总体；单次采集不能证明时间趋势或因果关系

输出要求：
- 使用中文
- Markdown 格式（标题、列表、表格、粗体等）
- 引用具体数据（百分比、排名等）
- 每条结论对应数据支撑"""


# ============================================================================
# 分析报告 Prompt 模板
# ============================================================================

ANALYSIS_REPORT_PROMPT = """## 任务

请根据以下微博话题的评论数据和情绪分析结果，生成一份**专业的舆情分析报告**。

---

## 输入数据

### 1. 基本信息

- **话题名称**: {topic}
- **采集时间**: {crawl_time}
- **分析帖子数**: {total_posts} 条
- **评论总数**: {total_comments} 条
- **唯一评论文本**: {unique_comments} 条（情绪比例按原始声量加权）

### 采样与代表性声明

{sampling_context}

### 2. 原始帖子内容（按热度排序）

以下是该话题下讨论度最高的帖子原文，请结合这些帖子内容分析用户评论：

{posts_content}

### 3. 情绪分布

| 情绪类别 | 数量 | 占比 |
|---------|------|------|
| 积极 | {pos_count} | {pos_pct}% |
| 中性 | {neu_count} | {neu_pct}% |
| 消极 | {neg_count} | {neg_pct}% |

### 4. TOP 20 高频关键词

{keywords_table}

### 5. 正面评论样本（{pos_sample_count} 条）

{pos_samples}

### 6. 中性评论样本（{neu_sample_count} 条）

{neu_samples}

### 7. 负面评论样本（{neg_sample_count} 条）

{neg_samples}

---

## 分析要求

请**特别结合"原始帖子内容"和"评论内容"共同分析**。不要只分析评论数据。

请按以下专业咨询报告结构输出：

### Executive Summary｜执行摘要

- 用 2-3 句话总结整体舆情态势
- 指出情绪分布的特点（如：以正面为主/负面情绪集中/两极分化等）
- 引用具体比例数据

### Key Findings｜关键发现

- 列出 3-5 个用户最关注的核心议题
- 每个议题说明热度（引用高频词数据）
- 分析这些议题与原始帖子内容的关联
- 说明为什么这些议题引发关注
- 分析积极情绪的主要来源（用户为什么开心/支持/期待？）
- 分析消极情绪的主要来源（用户为什么愤怒/失望/担忧？）
- 将情绪原因与帖子内容关联分析
- 引用代表性评论佐证
- 正面观点 TOP 3（列出具体的正面评价维度）
- 负面观点 TOP 3（列出具体的负面评价维度）
- 说明正负面观点的博弈与平衡
- 仅描述本次可见样本中的观点差异；没有多时点数据时，不得声称观点发生转变

### Risk Analysis｜风险分析

- 识别潜在的舆情风险点
- 识别负面情绪风险信号；没有多时点数据时，不得声称情绪正在扩散
- 是否有引发争议的话题点
- 是否需要关注敏感议题

### Trend Outlook｜趋势展望

- 改为情景展望：明确假设和触发条件，不作确定性预测
- 用户关注点是否会转移
- 情绪是否会升温或降温
- 建议关注的时间窗口
- 3-5 条核心结论
- 每条结论简洁有力，可直接用于汇报
- 整体舆情评级（正面/中性/负面/预警）

---

## 格式要求

- 使用 Markdown 格式
- 数据引用格式: `积极占比 65.3%（425/651）`
- 关键词用 **粗体** 标注
- 风险等级使用“低 / 中 / 高”文字标注，不使用 emoji
- 评论引用用 > 引用块格式
- 每个章节之间用 --- 分隔
- 若输出报告日期，必须使用采集日期 {crawl_date}，不得自行推算或改写日期
- 所有总体判断必须写作“本次可见样本中”；覆盖率低于 20% 时明确标注“探索性结论，不能代表整体舆情”
- 不得把微博标称评论数当作已分析样本数，不得把横截面数据写成时间趋势
- 描述单帖热度时必须同时写“微博标称 X 条、实际分析 Y 条”；不得使用“该帖获得 X 条评论”代替这两个口径

---

请开始生成报告。"""


# ============================================================================
# 简版分析 Prompt (用于快速预览)
# ============================================================================

QUICK_ANALYSIS_PROMPT = """## 任务

对以下微博话题进行快速舆情分析。

**话题**: {topic}
**评论总数**: {total_comments}
**采样声明**: {sampling_context}
**情绪分布**: 积极 {pos_pct}% | 中性 {neu_pct}% | 消极 {neg_pct}%
**TOP 关键词**: {top_keywords}

**核心帖子内容**:
{posts_summary}

请用 300 字以内给出：
1. 整体舆情判断（一句话）
2. 用户关注焦点（2-3 点）
3. 主要风险提示（1-2 点）
4. 舆情评级（正面/中性/负面/预警）

所有判断仅限本次可见样本；没有多时点数据时不得声称趋势。"""


# ============================================================================
# Prompt 构建函数
# ============================================================================

def build_analysis_prompt(topic: str, stats: dict, posts: list,
                          keywords: list, samples: dict,
                          crawl_time: str = "", sampling: dict | None = None) -> str:
    """
    根据实际数据构建舆情分析 Prompt。

    Args:
        topic: 话题名称
        stats: 情绪统计 {'positive': int, 'negative': int, 'neutral': int, 'total': int}
        posts: 帖子列表 [{'username': str, 'content': str, 'comment_count': int}, ...]
        keywords: [(word, frequency), ...]  TOP 20 关键词
        samples: {'positive': [str, ...], 'neutral': [str, ...], 'negative': [str, ...]}
        crawl_time: 采集时间字符串

    Returns:
        格式化的完整 prompt 字符串
    """
    total = stats.get('total', 1) or 1
    pos = stats.get('positive', 0)
    neu = stats.get('neutral', 0)
    neg = stats.get('negative', 0)
    sampling = sampling or {}
    coverage = sampling.get('coverage_pct')
    expected = sampling.get('expected_comments')
    fetched = sampling.get('fetched_comments', total)
    representation = sampling.get('representation_status', 'unknown')
    representation_label = {
        'limited': '代表性有限', 'partial': '部分代表',
        'good': '覆盖较好', 'unknown': '无法判断',
    }.get(representation, str(representation))
    if coverage is None:
        sampling_context = "上传数据，无法计算对平台总体评论的采集覆盖率；结论仅适用于所上传样本。"
    else:
        sampling_context = (
            f"微博标称评论约 {expected or 0} 条，实际抓取 {fetched or 0} 条，"
            f"采集覆盖率 {coverage:.1f}%，代表性状态为“{representation_label}”。"
        )
        if coverage < 20 or representation == 'limited':
            sampling_context += " 本报告属于探索性分析，不能代表整体舆情。"

    # ── 帖子内容 ──
    per_post = {
        str(item.get('weibo_id', '')): item
        for item in (sampling.get('per_post') or [])
    }
    posts_lines = []
    ranked_posts = sorted(
        posts,
        key=lambda post: int(post.get('comment_count', post.get('comment_count_on_card', 0)) or 0),
        reverse=True,
    )
    for i, p in enumerate(ranked_posts[:10], 1):  # TOP 10 帖子
        username = p.get('username', '未知用户')
        content = p.get('content', p.get('post_content', ''))
        cc = p.get('comment_count', p.get('comment_count_on_card', 0))
        post_sampling = per_post.get(str(p.get('weibo_id', '')), {})
        fetched_post = post_sampling.get('fetched_comments')
        coverage_post = post_sampling.get('coverage_pct')
        if content:
            evidence = f"微博标称 {cc} 条"
            if fetched_post is not None:
                evidence += f" | 实际分析 {fetched_post} 条"
            if coverage_post is not None:
                evidence += f" | 覆盖率 {coverage_post:.1f}%"
            posts_lines.append(f"**帖子 {i}** (@{username} | {evidence})\n\n{content}\n")
    posts_content = "\n".join(posts_lines) if posts_lines else "（无帖子数据）"

    # ── 关键词表格 ──
    kw_lines = ["| 排名 | 关键词 | 频次 |", "|------|--------|------|"]
    for rank, (word, freq) in enumerate(keywords[:20], 1):
        kw_lines.append(f"| {rank} | {word} | {freq} |")
    keywords_table = "\n".join(kw_lines)

    # ── 评论样本 ──
    def format_samples(comments, max_n=8):
        if not comments:
            return "（无样本）"
        lines = []
        for i, c in enumerate(comments[:max_n], 1):
            # Truncate long comments
            text = c if len(c) <= 200 else c[:200] + "..."
            lines.append(f"{i}. {text}")
        return "\n".join(lines)

    pos_samples = format_samples(samples.get('positive', []))
    neu_samples = format_samples(samples.get('neutral', []))
    neg_samples = format_samples(samples.get('negative', []))

    # ── 构建 Prompt ──
    prompt = ANALYSIS_REPORT_PROMPT.format(
        topic=topic,
        crawl_time=crawl_time or "最近采集",
        crawl_date=(crawl_time or "最近采集")[:10],
        total_posts=len(posts),
        total_comments=total,
        unique_comments=stats.get('unique_total', total),
        sampling_context=sampling_context,
        pos_count=pos,
        neu_count=neu,
        neg_count=neg,
        pos_pct=round(pos / total * 100, 1),
        neu_pct=round(neu / total * 100, 1),
        neg_pct=round(neg / total * 100, 1),
        posts_content=posts_content,
        keywords_table=keywords_table,
        pos_sample_count=len(samples.get('positive', [])),
        neu_sample_count=len(samples.get('neutral', [])),
        neg_sample_count=len(samples.get('negative', [])),
        pos_samples=pos_samples,
        neu_samples=neu_samples,
        neg_samples=neg_samples,
    )

    return prompt


def build_quick_prompt(topic: str, stats: dict, posts: list,
                       keywords: list, sampling: dict | None = None) -> str:
    """构建快速分析 Prompt（300字以内输出）"""
    total = stats.get('total', 1) or 1
    pos_pct = round(stats.get('positive', 0) / total * 100, 1)
    neu_pct = round(stats.get('neutral', 0) / total * 100, 1)
    neg_pct = round(stats.get('negative', 0) / total * 100, 1)
    sampling = sampling or {}
    coverage = sampling.get('coverage_pct')
    sampling_context = (
        f"实际抓取 {sampling.get('fetched_comments', total)} / 标称 {sampling.get('expected_comments')} 条，"
        f"覆盖率 {coverage:.1f}%，仅代表可见样本"
        if coverage is not None else "覆盖率未知，仅代表输入样本"
    )

    top_kw = ", ".join([w for w, _ in keywords[:10]])

    posts_summary = ""
    for p in posts[:3]:
        content = p.get('content', p.get('post_content', ''))
        if content:
            posts_summary += f"- @{p.get('username', '?')}: {content[:100]}...\n"

    return QUICK_ANALYSIS_PROMPT.format(
        topic=topic,
        total_comments=total,
        sampling_context=sampling_context,
        pos_pct=pos_pct,
        neu_pct=neu_pct,
        neg_pct=neg_pct,
        top_keywords=top_kw,
        posts_summary=posts_summary or "（无）",
    )
