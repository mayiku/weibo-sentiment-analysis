"""
微博情绪分析系统 — Streamlit Web 应用
"""
import sys
import json
import re
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

import streamlit as st

# 确保项目根目录在 import 路径中
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    TOP_KEYWORDS_N,
    TURSO_DATABASE_URL,
    WEIBO_COOKIE,
)
from src.logger import get_logger, get_log_file
from src.database import (
    DATABASE_SCHEMA_VERSION,
    init_db, create_task, update_task_status, update_task_results,
    insert_comments, insert_keywords, insert_posts,
    get_task, get_all_tasks, get_task_comments, get_task_posts,
    get_structured_data, delete_task, update_task_report,
    clear_task_analysis_data,
)
from src.task_lifecycle import (
    fail_task_if_active,
    reconcile_stale_tasks,
    touch_task,
)
from src.cleaner import clean_dataframe, clean_csv
from src.quality import assess_result_quality, load_crawl_metadata
from src import quality as quality_module
# 延迟导入sentiment模块，避免启动时PyTorch加载错误
try:
    from src.sentiment import (
        analyze_sentiment,
        check_model_health,
        extract_top_keywords,
        get_available_models,
        get_configured_models,
        get_model_health_report,
        get_model_info,
        get_sentiment_stats,
    )
except (ImportError, OSError, RuntimeError) as e:
    print(f"[WARNING] 情感分析模块加载失败: {e}")
    print("[INFO] 系统将使用基础功能，BERT模型可能不可用")

    # 创建备用的分析函数
    def analyze_sentiment(df, model_type="snownlp", **kwargs):
        print(f"警告: 使用备用的SnowNLP分析，请求的模型 {model_type} 不可用")
        # 这里可以实现基础的SnowNLP分析
        return df

    def get_sentiment_stats(df):
        return {'total': len(df), 'positive': 0, 'negative': 0, 'neutral': len(df)}

    def check_model_health(model_type):
        return {"model": model_type, "available": model_type == "snownlp", "error": None}

    check_model_health.cache_clear = lambda: None

    def extract_top_keywords(df, top_n=20):
        return []

    def get_available_models():
        return ["snownlp"]

    def get_configured_models():
        return ["snownlp"]

    def get_model_info(model_type):
        return {
            "name": "SnowNLP",
            "description": "基础 SnowNLP 模型",
            "provider": "SnowNLP Library",
            "supports_gpu": False,
        }

    def get_model_health_report():
        return {
            "snownlp": {"available": True, "error": None},
            "paddle": {"available": False, "error": "模型模块未加载"},
            "bert": {"available": False, "error": "模型模块未加载"},
            "hybrid": {"available": False, "error": "模型模块未加载"},
        }
from src.wordcloud_gen import (
    generate_wordcloud, generate_sentiment_distribution,
    generate_sentiment_wordclouds, find_chinese_font, has_chinese_font,
)

log = get_logger(__name__)

# ── 页面配置 ───────────────────────────────────────────
st.set_page_config(
    page_title="微博舆情分析平台",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def _log_runtime_configuration_once(cookie_configured: bool, python_version: str):
    """Emit deployment configuration once per process, not once per session."""
    log.info("微博 Cookie 配置状态: %s", "已配置" if cookie_configured else "未配置")
    if python_version != "3.12":
        log.warning(
            "运行时 Python %s 与项目声明 3.12 不一致，依赖兼容性可能受影响。",
            python_version,
        )
    else:
        log.info("运行时 Python %s 与项目声明一致", python_version)
    return True


_log_runtime_configuration_once(
    bool(WEIBO_COOKIE), f"{sys.version_info.major}.{sys.version_info.minor}"
)

# ── 自定义 CSS ─────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --canvas: #f7f8fa;
        --surface: #ffffff;
        --ink: #111827;
        --muted: #667085;
        --line: #e5e7eb;
        --accent: #2563eb;
        --accent-soft: #eff6ff;
        --radius: 10px;
    }

    html, body, [class*="css"] {
        font-family: Inter, "SF Pro Text", "Segoe UI", "PingFang SC", sans-serif;
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] { background: var(--canvas); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] .block-container {
        max-width: 1240px;
        padding: 3.25rem 3rem 6rem;
    }
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.25rem; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #475467;
    }
    h1, h2, h3 {
        font-family: "SF Pro Display", Inter, "PingFang SC", sans-serif;
        color: var(--ink);
        letter-spacing: -.025em;
    }
    h2 { font-size: 1.28rem !important; margin-top: 0 !important; }
    h3 { font-size: 1rem !important; }
    p { color: #344054; line-height: 1.65; }

    .product-header {
        display: grid;
        grid-template-columns: 3px 1fr;
        gap: 1.25rem;
        align-items: stretch;
        margin-bottom: 3rem;
    }
    .signal-rail { background: var(--accent); border-radius: 999px; }
    .product-eyebrow {
        color: var(--muted);
        font: 600 .7rem/1.2 ui-monospace, "SFMono-Regular", monospace;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .65rem;
    }
    h1.product-title {
        color: var(--ink);
        font: 650 2.25rem/1.12 "SF Pro Display", Inter, "PingFang SC", sans-serif;
        font-size: 2.25rem !important;
        line-height: 1.12 !important;
        letter-spacing: -.04em;
        margin: 0;
    }
    .product-subtitle {
        color: var(--muted);
        font-size: .98rem;
        margin: .7rem 0 0;
        max-width: 680px;
    }
    .section-heading { margin: 3.5rem 0 1.25rem; }
    .section-heading h2 { margin: 0 0 .35rem !important; }
    .section-heading p { color: var(--muted); font-size: .88rem; margin: 0; }

    .stat-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 1.35rem 1.4rem;
        min-height: 112px;
    }
    .stat-number {
        color: var(--ink);
        font-size: 1.85rem;
        font-weight: 650;
        letter-spacing: -.04em;
    }
    .stat-label {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .02em;
        margin-bottom: .7rem;
    }
    .stat-meta { color: #98a2b3; font-size: .74rem; margin-top: .45rem; }
    .model-summary {
        background: #fafafa;
        border: 1px solid var(--line);
        border-radius: 8px;
        color: #475467;
        font-size: .78rem;
        line-height: 1.65;
        margin-top: .8rem;
        padding: .8rem .9rem;
    }
    .health-row {
        align-items: center;
        border-bottom: 1px solid #f0f1f3;
        color: #475467;
        display: flex;
        font-size: .78rem;
        justify-content: space-between;
        padding: .55rem 0;
    }
    .health-row:last-child { border-bottom: 0; }
    .health-state { color: var(--muted); font-family: ui-monospace, monospace; }
    .history-context {
        background: var(--accent-soft);
        border-left: 3px solid var(--accent);
        color: #344054;
        font-size: .86rem;
        margin-bottom: 1.5rem;
        padding: .8rem 1rem;
    }
    .empty-state {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        margin-top: 1rem;
        padding: 4.5rem 4rem;
    }
    .empty-kicker {
        color: var(--accent);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .1em;
        margin-bottom: .9rem;
        text-transform: uppercase;
    }
    .empty-state h2 { font-size: 1.55rem !important; max-width: 520px; }
    .empty-state p { color: var(--muted); max-width: 620px; }

    hr { border-color: var(--line) !important; margin: 2rem 0 !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--line) !important;
        border-radius: var(--radius);
    }
    [data-testid="stImage"] img { border-radius: 8px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.45rem; }
    [data-testid="stAlert"] { border-radius: 8px; }
    [data-testid="stExpander"] { background: var(--surface); border-color: var(--line); }
    [data-testid="stFileUploaderDropzone"] { background: #fafafa; border-color: #d0d5dd; }

    .stButton > button, .stDownloadButton > button {
        border-color: #d0d5dd;
        border-radius: 7px;
        box-shadow: none;
        font-weight: 600;
        min-height: 2.45rem;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"] p { color: #ffffff !important; }
    .stButton > button:disabled {
        background: #f2f4f7 !important;
        border-color: #e4e7ec !important;
        color: #98a2b3 !important;
    }
    .stButton > button:disabled p { color: #98a2b3 !important; }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #98a2b3;
        color: var(--ink);
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
    input:focus-visible, textarea:focus-visible {
        outline: 2px solid #93c5fd !important;
        outline-offset: 2px;
    }
    [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--line); }
    [data-baseweb="tab"] { padding-left: 0; padding-right: 0; }

    @media (max-width: 1100px) {
        [data-testid="stMain"] .block-container { padding: 2.25rem 1.5rem 5rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 calc(50% - 1rem) !important;
            min-width: 260px !important;
        }
    }
    @media (max-width: 800px) {
        [data-testid="stMain"] .block-container { padding: 2rem 1rem 4rem; }
        .product-header { gap: .9rem; margin-bottom: 2rem; }
        h1.product-title { font-size: 1.65rem !important; line-height: 1.15 !important; letter-spacing: -.055em; }
        .product-subtitle { font-size: .9rem; }
        .empty-state { padding: 2.5rem 1.5rem; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex-basis: 100% !important;
            min-width: 0 !important;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
    }
</style>
""", unsafe_allow_html=True)


# ── 初始化数据库 ───────────────────────────────────────
@st.cache_resource
def _initialize_database_once(schema_version: int):
    # schema_version deliberately participates in the cache key.
    init_db()
    return True


_initialize_database_once(DATABASE_SCHEMA_VERSION)


@st.cache_data(ttl=300, show_spinner=False)
def _reconcile_stale_tasks_periodically(schema_version: int):
    # Avoid a remote UPDATE on every Streamlit rerun when Turso is enabled.
    return reconcile_stale_tasks(stale_after_minutes=45)


@st.cache_data(ttl=5, show_spinner=False)
def _get_recent_tasks(limit: int = 20):
    # Sidebar interactions rerun the whole script; a short cache removes
    # redundant remote reads while keeping task history effectively current.
    return get_all_tasks(limit=limit)


def _find_recovery_candidate(tasks: list[dict]):
    """Find the newest interrupted crawler task with a preserved raw CSV."""
    recoverable_statuses = {
        'pending', 'crawling', 'cleaning', 'analyzing',
        'generating_wordcloud', 'failed',
    }
    csv_files = list(DATA_DIR.glob("weibo_topic_*.csv"))
    newest_task_by_topic = {}
    for task in tasks:
        newest_task_by_topic.setdefault(str(task.get('topic', '')), task.get('id'))
    for task in tasks:
        topic = str(task.get('topic', ''))
        if (
            task.get('source') != 'crawler'
            or task.get('status') not in recoverable_statuses
            or int(task.get('total_comments') or 0) > 0
            or newest_task_by_topic.get(topic) != task.get('id')
        ):
            continue
        prefix = f"weibo_topic_{topic}_"
        matches = [path for path in csv_files if path.stem.startswith(prefix)]
        if not matches:
            try:
                from src.incremental import get_series_snapshot
                posts = get_series_snapshot(str(task.get('topic', '')))
                rows = []
                for post in posts:
                    for record in post.get('comment_records', []):
                        rows.append({
                            '评论内容': str(record.get('text', '')),
                            '评论ID': str(record.get('comment_id', '')),
                            '帖子ID': str(post.get('weibo_id', '')),
                            '用户名': str(post.get('username', '')),
                            '帖子内容': str(post.get('post_content', '')),
                            '帖子评论数': int(post.get('comment_count', 0) or 0),
                            '发布时间': str(post.get('post_time', '')),
                        })
                if rows:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    recovered_csv = DATA_DIR / (
                        f"weibo_topic_{task.get('topic', '')}_{timestamp}_turso_recovery.csv"
                    )
                    pd.DataFrame(rows).to_csv(
                        recovered_csv, index=False, encoding='utf-8-sig'
                    )
                    structured_posts = [{
                        'weibo_id': post.get('weibo_id', ''),
                        'username': post.get('username', ''),
                        'content': post.get('post_content', ''),
                        'comment_count_on_card': post.get('comment_count', 0),
                        'post_time': post.get('post_time', ''),
                        'url': post.get('url', ''),
                        'comments': [
                            record.get('text', '')
                            for record in post.get('comment_records', [])
                        ],
                        'fetched_comment_count': len(post.get('comment_records', [])),
                    } for post in posts]
                    recovered_csv.with_name(
                        f"{recovered_csv.stem}_structured.json"
                    ).write_text(
                        json.dumps({
                            'posts': structured_posts,
                            'total_posts': len(structured_posts),
                            'total_comments': len(rows),
                        }, ensure_ascii=False),
                        encoding='utf-8',
                    )
                    log.warning(
                        "任务 #%s 从 Turso 增量观测恢复 %d 条评论",
                        task.get('id'), len(rows),
                    )
                    matches = [recovered_csv]
            except Exception as exc:
                log.warning("Turso 中断恢复快照生成失败: %s", exc)
        if matches:
            return {
                'task': task,
                'csv_path': max(matches, key=lambda path: path.stat().st_mtime),
            }
    return None


_reconcile_stale_tasks_periodically(DATABASE_SCHEMA_VERSION)


# ── 辅助函数 ───────────────────────────────────────────
def run_pipeline(topic: str, csv_path: str, task_id: int, model_type: str = None) -> dict:
    """
    执行完整的分析流水线: 清洗 → 情感分析 → 词云

    新增: 若 CSV 含 帖子ID 列，则插入帖子↔评论关联数据供 Agent API 消费。

    Args:
        topic: 话题名称
        csv_path: CSV文件路径
        task_id: 任务ID
        model_type: 情感分析模型类型

    Returns:
        dict: 分析结果统计
    """
    # Step 1: 清洗
    update_task_status(task_id, 'cleaning')
    log.info("--- 步骤 1/3: 数据清洗 ---")
    df = clean_csv(csv_path)
    cleaning_metadata = dict(df.attrs.get('cleaning_metadata', {}))

    if len(df) == 0:
        update_task_status(task_id, 'failed', 'CSV 文件中没有评论数据（0 行）')
        raise ValueError("CSV 文件中没有评论数据。请确认话题有搜索结果，或尝试其他关键词。")

    # Step 2: 情感分析
    update_task_status(task_id, 'analyzing')
    log.info("--- 步骤 2/3: 情感分析 ---")

    # 确定模型类型
    if model_type is None:
        model_type = st.session_state.get('selected_model', 'hybrid')

    log.info(f"使用情感分析模型: {model_type}")
    analysis_started = time.perf_counter()
    memory_before_mb = None
    try:
        import psutil
        memory_before_mb = psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        pass

    df = analyze_sentiment(df, model_type=model_type)
    processing_time = time.perf_counter() - analysis_started
    model_memory = None
    if memory_before_mb is not None:
        try:
            memory_after_mb = psutil.Process().memory_info().rss / 1024 / 1024
            model_memory = max(0.0, memory_after_mb - memory_before_mb)
        except Exception:
            pass
    analysis_metadata = dict(df.attrs.get('analysis_metadata', {}))
    analysis_metadata['processing_time'] = processing_time
    analysis_metadata['model_memory'] = model_memory

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_csv = OUTPUT_DIR / f"result_{topic}_{timestamp}.csv"
    df.to_csv(result_csv, index=False, encoding='utf-8-sig')

    # Step 3: 词云 + 统计
    update_task_status(task_id, 'generating_wordcloud')
    log.info("--- 步骤 3/3: 词云与可视化 ---")
    stats = get_sentiment_stats(df)

    font_path = find_chinese_font()

    wc_path = str(OUTPUT_DIR / f"wordcloud_{topic}_{timestamp}.png")
    generate_wordcloud(df['clean_text'], wc_path, font_path=font_path)

    dist_path = str(OUTPUT_DIR / f"distribution_{topic}_{timestamp}.png")
    generate_sentiment_distribution(stats, dist_path)

    sentiment_wc = generate_sentiment_wordclouds(
        df,
        str(OUTPUT_DIR),
        font_path=font_path,
        filename_suffix=f"task_{task_id}",
    )

    keywords = extract_top_keywords(df, top_n=TOP_KEYWORDS_N)

    # 优先读取爬虫结构化侧车；它包含零评论帖子和逐帖采样信息。
    structured_json = None
    structured_payload = {}
    json_sidecar = DATA_DIR / f"{Path(csv_path).stem}_structured.json"
    if json_sidecar.exists():
        structured_json = str(json_sidecar)
        try:
            structured_payload = json.loads(json_sidecar.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            structured_payload = {}

    # 插入完整帖子集合；CSV 仅包含有评论帖子，因此作为上传数据的回退来源。
    # A stopped recovery may leave committed posts but no comments. Reset only
    # this task's derived rows before rebuilding them from the preserved CSV.
    clear_task_analysis_data(task_id)
    weibo_id_to_db_post_id = {}
    total_posts = 0
    posts_to_insert = []
    if structured_payload.get('posts'):
        for post in structured_payload['posts']:
            posts_to_insert.append({
                'weibo_id': str(post.get('weibo_id', '')),
                'username': str(post.get('username', '')),
                'content': str(post.get('content', '')),
                'comment_count': int(post.get('comment_count_on_card', 0) or 0),
                'post_time': str(post.get('post_time', '')),
                'url': str(post.get('url', '')),
            })
    elif '帖子ID' in df.columns:
        seen_pids = set()
        for _, row in df.iterrows():
            wid = str(row.get('帖子ID', ''))
            if wid and wid not in seen_pids:
                seen_pids.add(wid)
                posts_to_insert.append({
                    'weibo_id': wid,
                    'username': str(row.get('用户名', '')),
                    'content': str(row.get('帖子内容', '')),
                    'comment_count': int(row.get('帖子评论数', 0)) if pd.notna(row.get('帖子评论数')) else 0,
                    'post_time': str(row.get('发布时间', '')),
                    'url': '',
                })
    posts_to_insert = [post for post in posts_to_insert if post['weibo_id']]
    if posts_to_insert:
        post_ids = insert_posts(task_id, posts_to_insert)
        for i, post in enumerate(posts_to_insert):
            weibo_id_to_db_post_id[post['weibo_id']] = post_ids[i]
        total_posts = len(post_ids)
        log.info("插入 %d 条帖子到数据库", total_posts)

    # ★ 给评论 DataFrame 加上 post_id 列
    if weibo_id_to_db_post_id and '帖子ID' in df.columns:
        df['post_id'] = df['帖子ID'].apply(
            lambda wid: weibo_id_to_db_post_id.get(str(wid)) if pd.notna(wid) else None
        )

    # 存入数据库
    insert_comments(task_id, df)
    insert_keywords(task_id, keywords)

    keywords_json = json.dumps(
        [{"word": w, "freq": f} for w, f in keywords], ensure_ascii=False
    )

    crawl_metadata = load_crawl_metadata(structured_json)
    if crawl_metadata['total_posts']:
        total_posts = crawl_metadata['total_posts']
    quality = assess_result_quality(
        total=stats['total'],
        positive=stats['positive'],
        negative=stats['negative'],
        neutral=stats['neutral'],
        coverage_pct=crawl_metadata['coverage_pct'],
        fallback_used=analysis_metadata.get('fallback_used', False),
        raw_comments=cleaning_metadata.get('raw_comments'),
    )
    # Newer quality modules include per-post representativeness checks. Keep
    # hot reload compatible if Streamlit still holds an older module object.
    enrich_sampling = getattr(quality_module, 'enrich_quality_with_sampling', None)
    if enrich_sampling:
        quality = enrich_sampling(quality, crawl_metadata)
    # Keep this call compatible with a stale quality module during Streamlit
    # hot reloads. Enrich the warning here instead of requiring a new keyword
    # argument to have loaded atomically across modules.
    partial_fallback_count = int(
        analysis_metadata.get('partial_fallback_count', 0) or 0
    )
    if partial_fallback_count:
        for issue in quality.get('issues', []):
            if issue.get('code') == 'model_fallback':
                issue['message'] = (
                    f"DeepSeek 输出异常，{partial_fallback_count} 条评论"
                    "局部使用备用模型。"
                )

    update_task_results(
        task_id,
        total=stats['total'],
        pos=stats['positive'],
        neg=stats['negative'],
        neu=stats['neutral'],
        wordcloud_path=wc_path,
        keywords_json=keywords_json,
        structured_json=structured_json,
        total_posts=total_posts,
        raw_comments=cleaning_metadata.get('raw_comments', stats['total']),
        expected_comments=crawl_metadata['expected_comments'],
        fetched_comments=crawl_metadata['fetched_comments'],
        coverage_pct=crawl_metadata['coverage_pct'],
        requested_model=analysis_metadata.get('requested_model', model_type),
        effective_model=analysis_metadata.get('effective_model', model_type),
        model_version=analysis_metadata.get('model_version'),
        fallback_used=analysis_metadata.get('fallback_used', False),
        fallback_reason=analysis_metadata.get('fallback_reason'),
        quality_status=quality['status'],
        quality_issues_json=json.dumps(quality['issues'], ensure_ascii=False),
        unique_comments=stats.get('unique_total', stats['total']),
        sampling_json=json.dumps(crawl_metadata, ensure_ascii=False),
        representation_status=crawl_metadata.get('representation_status', 'unknown'),
        processing_time=processing_time,
        model_memory=model_memory,
    )
    if quality['status'] == 'invalid':
        message = '；'.join(issue['message'] for issue in quality['issues'])
        update_task_status(task_id, 'failed', message)
        raise RuntimeError(f"分析结果未通过质量校验：{message}")
    update_task_status(task_id, 'completed')

    log.info("流水线完成! 任务 #%d (帖子 %d)", task_id, total_posts)

    # 获取帖子数据（供 AI 报告使用）
    posts_data = []
    try:
        posts_data = get_task_posts(task_id)
    except Exception:
        pass

    return {
        **stats,
        'df': df,
        'posts': posts_data,
        'source': 'crawler' if structured_json else 'upload',
        'cleaning_metadata': cleaning_metadata,
        'analysis_metadata': analysis_metadata,
        'processing_time': processing_time,
        'model_memory': model_memory,
        'crawl_metadata': crawl_metadata,
        'quality': quality,
        'wc_path': wc_path,
        'dist_path': dist_path,
        'sentiment_wc': sentiment_wc,
        'keywords': keywords,
        'result_csv': str(result_csv),
    }


# ── 页面标题 ───────────────────────────────────────────
st.markdown("""
<div class="product-header">
    <div class="signal-rail"></div>
    <div>
        <div class="product-eyebrow">Social Intelligence Workspace</div>
        <h1 class="product-title">微博舆情分析平台</h1>
        <p class="product-subtitle">输入话题或上传评论数据，完成采集、情绪识别与舆情研判。</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 侧边栏 ─────────────────────────────────────────────
with st.sidebar:
    st.header("分析工作台")

    # 日志文件信息
    st.caption(f"当前实例日志 · {get_log_file().name}")
    st.caption(
        f"数据存储 · {'Turso Cloud' if TURSO_DATABASE_URL else '本地 SQLite'}"
    )

    st.divider()

    # 输入模式
    input_mode = st.radio(
        "选择数据来源",
        ["上传 CSV", "爬取微博话题"],
        help="上传已有CSV或输入话题关键词进行爬取"
    )

    topic_keyword = ""
    uploaded_file = None
    page_num = 3
    scroll_times = 10
    use_mock = False
    incremental_enabled = True
    schedule_enabled = False
    interval_hours = 6

    if input_mode == "爬取微博话题":
        topic_keyword = st.text_input(
            "话题关键词",
            placeholder="例如: 陈奕迅, 福州车祸...",
            help="输入要分析的微博话题（不需要加 # 号）",
        )
        if WEIBO_COOKIE:
            st.caption("微博 Cookie 已配置 · 云端将自动恢复登录状态。")
        else:
            st.caption("云端采集需要在 Streamlit Secrets 中配置微博 Cookie。")
        if topic_keyword.strip():
            try:
                from src.incremental import get_series
                series_status = get_series(topic_keyword.strip())
                if series_status:
                    schedule_text = (
                        f"下一轮 {series_status.get('next_run_at')}"
                        if series_status.get('enabled') and series_status.get('next_run_at')
                        else "未启用定时队列"
                    )
                    st.caption(
                        f"历史累计 {series_status.get('total_unique_comments', 0):,} 条 · {schedule_text}"
                    )
            except Exception:
                pass

    else:
        uploaded_file = st.file_uploader(
            "上传评论 CSV",
            type=['csv'],
            help="CSV 文件必须包含 '评论内容' 列",
        )

    model_descriptions = {
        "snownlp": "传统统计方法，兼容性好",
        "paddle": "PaddleNLP 二分类模型",
        "bert": "预训练二分类模型",
        "hybrid": "SnowNLP 与微博语义规则增强",
        "deepseek": "DeepSeek 语义三分类，高准确度模式",
    }
    with st.expander("高级设置", expanded=False):
        try:
            configured_models = get_configured_models() or ["snownlp"]
        except Exception:
            configured_models = ["snownlp"]

        selected_model = st.session_state.get('selected_model', 'hybrid')
        if selected_model not in configured_models:
            selected_model = "hybrid" if "hybrid" in configured_models else configured_models[0]
        selected_model = st.radio(
            "情绪分析模型",
            configured_models,
            index=configured_models.index(selected_model),
            format_func=lambda model: f"{model.upper()} — {model_descriptions.get(model, '')}",
            key="sentiment_model_selector",
        )
        st.session_state['selected_model'] = selected_model
        st.caption("模型不可用时会自动降级，并在结果的数据质量区域明确记录。")

        if input_mode == "爬取微博话题":
            col1, col2 = st.columns(2)
            with col1:
                page_num = st.number_input("搜索页数", 1, 10, 3)
            with col2:
                scroll_times = st.number_input("滚动次数", 5, 100, 10)
            use_mock = st.checkbox(
                "模拟数据模式",
                value=False,
                help="跳过真实爬取，仅用于演示与流程测试",
            )
            st.markdown("##### 增量采集")
            incremental_enabled = st.checkbox(
                "合并历史采集断点",
                value=True,
                help="按评论 ID 合并同话题历史数据，本次分析使用累计唯一评论。",
            )
            schedule_enabled = st.checkbox(
                "加入定时采集队列",
                value=False,
                disabled=not incremental_enabled,
                help="保存下一轮执行时间；由项目的增量 worker 执行。",
            )
            interval_hours = st.selectbox(
                "采集间隔",
                [1, 6, 12, 24],
                index=1,
                format_func=lambda value: f"每 {value} 小时",
                disabled=not schedule_enabled,
            )

        st.markdown("##### 模型运行自检")
        if st.button("运行模型自检", key="run_model_health_check", width='stretch'):
            with st.spinner("正在执行真实单条与批量推理..."):
                check_model_health.cache_clear()
                st.session_state['model_health_report'] = get_model_health_report()

        model_health = st.session_state.get('model_health_report')
        if model_health:
            for model_name, health in model_health.items():
                state = (
                    "DEGRADED" if health.get("degraded") else
                    "READY" if health.get("available") else "UNAVAILABLE"
                )
                detail = health.get("detail") or (
                    "推理校验通过" if health.get("available") else health.get("error", "不可用")
                )
                st.markdown(
                    f'<div class="health-row"><span>{model_name.upper()} · {detail}</span>'
                    f'<span class="health-state">{state}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("按需运行，不影响首屏加载速度。")

    st.divider()

    # 启动按钮
    can_start = (
        (input_mode == "上传 CSV" and uploaded_file is not None) or
        (input_mode == "爬取微博话题" and topic_keyword.strip())
    )
    start_btn = st.button(
        "开始分析", type="primary", width='stretch',
        disabled=not can_start,
    )

    st.divider()

    # 历史记录
    st.subheader("历史任务")
    tasks = _get_recent_tasks(limit=20)
    recovery_candidate = _find_recovery_candidate(tasks)
    recovery_btn = False
    auto_recovery = False
    if recovery_candidate:
        recovery_task = recovery_candidate['task']
        recovery_csv = recovery_candidate['csv_path']
        auto_recovery = (
            recovery_task.get('status') == 'generating_wordcloud'
            and recovery_csv.stem.endswith('_turso_recovery')
        )
        st.warning(
            f"发现可恢复的中断任务 #{recovery_task['id']} · "
            f"{recovery_task['topic']} · {recovery_csv.name}"
        )
        recovery_btn = st.button(
            "恢复中断分析", type="primary", width='stretch'
        )
        st.divider()
    if tasks:
        for t in tasks:
            status_label = {
                'pending': '等待中', 'crawling': '采集中', 'cleaning': '清洗中',
                'analyzing': '分析中', 'generating_wordcloud': '生成中',
                'completed': '已完成', 'failed': '失败',
            }.get(t['status'], t['status'])
            quality_label = {
                'good': '质量正常', 'warning': '需复核', 'invalid': '不可用',
            }.get(t.get('quality_status'), '')
            coverage_label = (
                f"覆盖 {t['coverage_pct']:.1f}%" if t.get('coverage_pct') is not None else ""
            )
            detail_parts = [f"{t.get('total_comments', 0):,} 条", coverage_label, quality_label]
            detail = ' · '.join(part for part in detail_parts if part)
            if st.button(
                f"{t['topic'][:18]} · {status_label}" + (f" · {detail}" if detail else ""),
                key=f"task_{t['id']}",
                width='stretch',
            ):
                st.session_state.selected_task = t['id']
                st.session_state.pop('result', None)
                st.session_state['current_topic'] = t['topic']
                st.session_state.pop('ai_report', None)
                report_path = t.get('report_path')
                if report_path and Path(report_path).exists():
                    st.session_state['ai_report'] = Path(report_path).read_text(encoding='utf-8')
                    st.session_state['ai_report_path'] = report_path
                st.rerun()

        if not st.session_state.get('confirm_clear_history'):
            if st.button("清空历史", width='stretch'):
                st.session_state['confirm_clear_history'] = True
                st.rerun()
        else:
            st.warning("将删除当前显示的全部历史任务，此操作不可撤销。")
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("确认清空", type="primary", width='stretch'):
                for t in tasks:
                    delete_task(t['id'])
                st.session_state.pop('confirm_clear_history', None)
                st.session_state.pop('selected_task', None)
                st.rerun()
            if cancel_col.button("取消", width='stretch'):
                st.session_state.pop('confirm_clear_history', None)
                st.rerun()
    else:
        st.caption("暂无历史记录")


# ── 主区域 ─────────────────────────────────────────────
# 处理分析请求
if recovery_btn or auto_recovery:
    recovery_task = recovery_candidate['task']
    recovery_csv = recovery_candidate['csv_path']
    recovery_task_id = int(recovery_task['id'])
    recovery_topic = str(recovery_task['topic'])
    recovery_model = recovery_task.get('requested_model') or st.session_state.get(
        'selected_model', 'hybrid'
    )
    with st.status("正在恢复中断分析", expanded=True) as recovery_status:
        st.write(f"从保留的 CSV 恢复 · {recovery_csv.name}")
        try:
            recovered_result = run_pipeline(
                recovery_topic,
                str(recovery_csv),
                recovery_task_id,
                model_type=recovery_model,
            )
        except BaseException as exc:
            fail_task_if_active(
                recovery_task_id,
                f"中断恢复失败：{str(exc) or type(exc).__name__}",
            )
            raise
        recovery_status.update(label="中断分析已恢复", state="complete")
        st.session_state.result = recovered_result
        st.session_state.task_id = recovery_task_id
        st.session_state['current_topic'] = recovery_topic
        _get_recent_tasks.clear()
        st.rerun()

if start_btn:
    if input_mode == "上传 CSV":
        # 上传 CSV 模式
        topic = Path(uploaded_file.name).stem
        st.session_state['current_topic'] = topic
        csv_path = DATA_DIR / f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(csv_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        task_id = create_task(topic, source="upload")

        with st.status("处理上传数据", expanded=True) as pipeline_status:
            pipeline_status.update(label="步骤 1/2 · 数据清洗与情绪分析", state="running")
            model_type = st.session_state.get('selected_model', 'hybrid')
            try:
                result = run_pipeline(topic, csv_path, task_id, model_type=model_type)
            except BaseException as exc:
                fail_task_if_active(
                    task_id,
                    f"上传分析被中断：{str(exc) or type(exc).__name__}",
                )
                raise
            st.write(f"分析完成 · {result['total']} 条评论 · {model_type.upper()}")

            pipeline_status.update(label="步骤 2/2 · 完成", state="complete")
            st.session_state.result = result
            st.session_state.task_id = task_id
            st.rerun()

    else:
        # 爬虫模式 — V2 (API) 优先, V1 (Selenium) 回退
        from src.crawler import crawl_topic, crawl_topic_v2

        topic = topic_keyword.strip()
        st.session_state['current_topic'] = topic
        task_id = create_task(topic, source="crawler")
        update_task_status(task_id, 'crawling')

        # 增强的进度管理系统
        pipeline_progress = st.progress(0)
        progress_steps = [
            "微博登录",
            "搜索帖子",
            "抓取评论",
            "数据清洗",
            "情绪分析",
            "生成词云",
            "AI 舆情分析"
        ]

        with st.status("正在执行分析流程", expanded=True) as pipeline_status:
            login_msgs = st.empty()

            def login_cb(msg):
                login_msgs.info(msg)

            def update_progress(step_index, label_suffix=""):
                touch_task(task_id)
                progress = int((step_index / len(progress_steps)) * 100)
                pipeline_progress.progress(progress)
                step_label = f"{step_index + 1}/7 · {progress_steps[step_index]}"
                if label_suffix:
                    step_label += f" {label_suffix}"
                pipeline_status.update(label=step_label, state="running")

            # 步骤1: 微博登录
            update_progress(0)

            # 步骤2-3: 搜索和抓取 (通过回调函数更新进度)
            crawl_status_msg = st.empty()

            def enhanced_login_cb(msg):
                login_msgs.info(msg)
                # 动态解析状态消息来更新进度
                if "使用缓存登录状态" in msg:
                    update_progress(1, "· 已使用缓存登录")  # 步骤2开始
                elif "登录成功" in msg:
                    update_progress(1, "· 搜索帖子")  # 步骤2开始
                elif "搜索" in msg or "帖" in msg:
                    update_progress(1, "· 进行中")  # 步骤2进行中
                elif "抓取" in msg or "评论" in msg:
                    update_progress(2, "· 进行中")  # 步骤3进行中

            try:
                csv_path, all_comments = crawl_topic_v2(
                    topic, page_num=page_num, scroll_times=scroll_times,
                    use_mock=use_mock, status_callback=enhanced_login_cb,
                    incremental=incremental_enabled,
                    schedule_enabled=schedule_enabled,
                    interval_hours=interval_hours,
                    task_id=task_id,
                )
                # 爬取完成后更新进度
                update_progress(3)  # 步骤4开始: 数据清洗
                crawl_status_msg.success(f"采集完成 · {len(all_comments)} 条评论")
            except Exception as e:
                log.warning("V2 失败: %s, 回退 V1", e)
                update_progress(3, "· V2 失败，回退 V1")
                try:
                    csv_path, all_comments = crawl_topic(
                        topic, page_num=page_num, scroll_times=scroll_times,
                        use_mock=use_mock, status_callback=enhanced_login_cb,
                    )
                    crawl_status_msg.success(f"采集完成 · V1 回退 · {len(all_comments)} 条评论")
                except Exception as e2:
                    pipeline_status.update(label="采集失败", state="error")
                    update_task_status(task_id, 'failed', str(e2))
                    st.error(f"采集失败：{e2}")
                    st.stop()
                except BaseException as interrupted:
                    fail_task_if_active(
                        task_id,
                        f"采集会话被中断：{str(interrupted) or type(interrupted).__name__}",
                    )
                    raise
            except BaseException as interrupted:
                fail_task_if_active(
                    task_id,
                    f"采集会话被中断：{str(interrupted) or type(interrupted).__name__}",
                )
                raise

            # 步骤4-7: 后续处理
            update_progress(3, "· 数据清洗")
            model_type = st.session_state.get('selected_model', 'hybrid')
            try:
                result = run_pipeline(topic, csv_path, task_id, model_type=model_type)
            except BaseException as exc:
                fail_task_if_active(
                    task_id,
                    f"分析流程被中断：{str(exc) or type(exc).__name__}",
                )
                raise
            st.write(f"{result['total']} 条评论 · 积极 {result['pos_pct']}% · 中性 {result['neu_pct']}% · 消极 {result['neg_pct']}%")

            # 步骤5-6: 情感分析和生成词云
            update_progress(4, "· 情绪分析完成")
            update_progress(5, "· 词云生成完成")

            # AI 报告
            from config import CURRENT_API_AVAILABLE, AI_PROVIDER
            if CURRENT_API_AVAILABLE:
                update_progress(6, f"· AI 舆情分析 ({AI_PROVIDER})")
                try:
                    from src.ai_agent import ReportGenerator
                    gen = ReportGenerator()
                    ai_result = gen.generate(
                        topic=topic,
                        stats={k: result[k] for k in ['total', 'positive', 'negative', 'neutral']},
                        df=result.get('df'), posts=result.get('posts', []),
                        keywords=result.get('keywords', []), use_cache=True,
                        sampling=result.get('crawl_metadata', {}),
                    )
                    if ai_result['success']:
                        st.session_state['ai_report'] = ai_result['report']
                        st.session_state['ai_report_path'] = ai_result['report_path']
                        update_task_report(task_id, ai_result['report_path'], AI_PROVIDER)
                        st.write(f"AI 报告完成 · {len(ai_result['report'])} 字 · {AI_PROVIDER}")
                    else:
                        st.warning(f"AI 报告失败：{ai_result['error']}")
                except Exception as e:
                    st.write(f"AI 报告已跳过：{e}")

            # 步骤7: 完成
            update_progress(6, "· 分析完成")
            pipeline_progress.progress(100)
            pipeline_status.update(label="7/7 · 分析完成", state="complete")
            st.session_state.result = result
            st.session_state.task_id = task_id
            st.rerun()



# ── 结果展示 ───────────────────────────────────────────
# 优先展示当前结果，否则展示选中的历史任务
result = st.session_state.get('result')

if result is None and 'selected_task' in st.session_state:
    task_id = st.session_state.selected_task
    task = get_task(task_id)
    if task and task['status'] == 'completed':
        df = get_task_comments(task_id)
        keywords_data = json.loads(task['keywords_json']) if task['keywords_json'] else []

        # Classified word clouds used to exist only in session memory. Give
        # every task deterministic paths and rebuild older/missing artifacts
        # from the comments already stored in SQLite.
        sentiment_wc = {
            sentiment: str(OUTPUT_DIR / f"wordcloud_{sentiment}_task_{task_id}.png")
            for sentiment in ['积极', '消极', '中性']
        }
        existing_sentiment_wc = {
            sentiment: path if Path(path).exists() else None
            for sentiment, path in sentiment_wc.items()
        }
        available_sentiments = set(df['nlp_result'].dropna()) if 'nlp_result' in df.columns else set()
        missing_required_cloud = any(
            sentiment in available_sentiments and not existing_sentiment_wc[sentiment]
            for sentiment in sentiment_wc
        )
        if missing_required_cloud and not df.empty:
            history_wc_df = df.copy()
            if 'clean_text' not in history_wc_df.columns:
                history_wc_df['clean_text'] = history_wc_df.get('cleaned_content', '')
            try:
                existing_sentiment_wc = generate_sentiment_wordclouds(
                    history_wc_df,
                    str(OUTPUT_DIR),
                    font_path=find_chinese_font(),
                    filename_suffix=f"task_{task_id}",
                )
            except Exception as exc:
                log.warning("重建任务 #%s 分类词云失败: %s", task_id, exc)

        result = {
            'total': task['total_comments'],
            'unique_total': task.get('unique_comments') or task['total_comments'],
            'positive': task['pos_count'],
            'negative': task['neg_count'],
            'neutral': task['neu_count'],
            'pos_pct': round(task['pos_count'] / task['total_comments'] * 100, 1) if task['total_comments'] else 0,
            'neg_pct': round(task['neg_count'] / task['total_comments'] * 100, 1) if task['total_comments'] else 0,
            'neu_pct': round(task['neu_count'] / task['total_comments'] * 100, 1) if task['total_comments'] else 0,
            'df': df,
            'wc_path': task['wordcloud_path'],
            'sentiment_wc': existing_sentiment_wc,
            'keywords': [(k['word'], k['freq']) for k in keywords_data],
            'posts': get_task_posts(task_id),
            'source': task.get('source'),
            'task_id': task_id,
            'cleaning_metadata': {
                'raw_comments': task.get('raw_comments') or task['total_comments'],
                'cleaned_comments': task.get('unique_comments') or task['total_comments'],
                'valid_comments': task['total_comments'],
                'unique_comments': task.get('unique_comments') or task['total_comments'],
                'removed_comments': max((task.get('raw_comments') or task['total_comments']) - task['total_comments'], 0),
            },
            'analysis_metadata': {
                'requested_model': task.get('requested_model'),
                'effective_model': task.get('effective_model'),
                'model_version': task.get('model_version'),
                'fallback_used': bool(task.get('fallback_used')),
                'fallback_reason': task.get('fallback_reason'),
            },
            'processing_time': task.get('processing_time'),
            'model_memory': task.get('model_memory'),
            'crawl_metadata': {
                'expected_comments': task.get('expected_comments'),
                'fetched_comments': task.get('fetched_comments'),
                'coverage_pct': task.get('coverage_pct'),
                'total_posts': task.get('total_posts', 0),
                **(json.loads(task['sampling_json']) if task.get('sampling_json') else {}),
            },
            'quality': {
                'status': task.get('quality_status') or 'unknown',
                'issues': json.loads(task['quality_issues_json']) if task.get('quality_issues_json') else [],
            },
        }
        st.markdown(
            f'<div class="history-context">历史分析 · {task["topic"]} '
            f'· {task["created_at"]}</div>',
            unsafe_allow_html=True,
        )

if result:
    st.markdown("""
    <div class="section-heading">
        <h2>分析结果</h2>
        <p>核心情绪结构与样本规模</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">总评论数</div>
            <div class="stat-number">{result['total']:,}</div>
            <div class="stat-meta">{result.get('unique_total', result['total']):,} 条去重样本</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">积极率</div>
            <div class="stat-number">{result['pos_pct']}%</div>
            <div class="stat-meta">{result['positive']:,} 条评论</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">中性率</div>
            <div class="stat-number">{result['neu_pct']}%</div>
            <div class="stat-meta">{result['neutral']:,} 条评论</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">消极率</div>
            <div class="stat-number">{result['neg_pct']}%</div>
            <div class="stat-meta">{result['negative']:,} 条评论</div>
        </div>
        """, unsafe_allow_html=True)

    quality = result.get('quality', {})
    analysis_meta = result.get('analysis_metadata', {})
    crawl_meta = result.get('crawl_metadata', {})
    cleaning_meta = result.get('cleaning_metadata', {})
    effective_model = analysis_meta.get('effective_model') or analysis_meta.get('requested_model') or '历史未记录'
    requested_model = analysis_meta.get('requested_model') or effective_model
    coverage_pct = crawl_meta.get('coverage_pct')

    with st.container(border=True):
        st.markdown("#### 数据质量与分析方法")
        method_col, sample_col, coverage_col = st.columns(3)
        method_col.metric(
            "实际运行模型",
            str(effective_model).upper(),
            f"请求 {str(requested_model).upper()}" if requested_model != effective_model else None,
        )
        sample_col.metric(
            "有效评论声量",
            f"{result.get('total', 0):,}",
            f"去重样本 {result.get('unique_total', result.get('total', 0)):,}",
        )
        coverage_col.metric(
            "采集覆盖率",
            f"{coverage_pct:.1f}%" if coverage_pct is not None else "不适用",
            f"预计 {crawl_meta.get('expected_comments', 0):,} 条" if crawl_meta.get('expected_comments') else None,
        )

        issues = quality.get('issues') or []
        if quality.get('status') == 'invalid':
            st.error("结果未通过质量校验：" + "；".join(issue['message'] for issue in issues))
        elif issues:
            st.warning("需要谨慎解读：" + "；".join(issue['message'] for issue in issues))
        else:
            st.caption("本次结果未触发自动质量警告。情绪占比仅代表本次有效样本的模型分类结果。")

        per_post = crawl_meta.get('per_post') or []
        incremental_meta = crawl_meta.get('incremental') or {}
        if incremental_meta:
            next_run = incremental_meta.get('next_run_at')
            st.caption(
                f"增量轮次 #{incremental_meta.get('run_id')} · "
                f"本轮新增 {incremental_meta.get('new_comments', 0):,} 条 · "
                f"累计 {incremental_meta.get('total_unique_comments', 0):,} 条"
                + (f" · 下一轮 {next_run}" if next_run else "")
            )
        if crawl_meta.get('dominant_post_share_pct') is not None:
            dominant_coverage = crawl_meta.get('dominant_post_coverage_pct')
            st.caption(
                f"覆盖结构 · 最大帖占标称评论 {crawl_meta['dominant_post_share_pct']:.1f}%"
                + (
                    f"，该帖覆盖 {dominant_coverage:.1f}%"
                    if dominant_coverage is not None else ""
                )
                + (
                    f" · 排除最大帖后覆盖 {crawl_meta['coverage_excluding_dominant_pct']:.1f}%"
                    if crawl_meta.get('coverage_excluding_dominant_pct') is not None else ""
                )
                + (
                    f" · 单帖覆盖中位数 {crawl_meta['median_post_coverage_pct']:.1f}%"
                    if crawl_meta.get('median_post_coverage_pct') is not None else ""
                )
            )
        if per_post:
            dominant_share = crawl_meta.get('dominant_post_share_pct')
            status_text = {
                'limited': '代表性有限', 'partial': '部分代表',
                'good': '覆盖较好', 'unknown': '无法判断',
            }.get(crawl_meta.get('representation_status'), '无法判断')
            st.caption(
                f"采样代表性：{status_text} · 最大帖子占标称评论 "
                f"{dominant_share:.1f}%" if dominant_share is not None
                else f"采样代表性：{status_text}"
            )
            with st.expander("逐帖采样诊断", expanded=False):
                rows = []
                method_labels = {
                    'api_pc': '微博 PC API', 'api_mobile': '微博移动 API',
                    'selenium': '页面滚动', 'skipped_zero': '零评论跳过', 'unknown': '历史未记录',
                }
                stop_labels = {
                    'empty_page': 'API 返回空页', 'max_id_zero': '游标结束',
                    'cursor_stalled': '游标停滞', 'no_new_comments': '无新增评论',
                    'max_pages': '达到页数上限', 'selenium_scroll_limit': '页面滚动结束',
                    'card_zero': '卡片标称零评论', 'unknown': '历史未记录',
                }
                for item in per_post:
                    rows.append({
                        '账号': item.get('username') or '未知',
                        '标称评论': item.get('expected_comments'),
                        '实际采集': item.get('fetched_comments'),
                        '覆盖率': f"{item['coverage_pct']:.1f}%" if item.get('coverage_pct') is not None else '—',
                        '采集方式': method_labels.get(item.get('fetch_method'), item.get('fetch_method')),
                        '终止原因': stop_labels.get(item.get('stop_reason'), item.get('stop_reason')),
                    })
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    with st.expander("分析方法与模型对比", expanded=False):
        try:
            from src.sentiment import ModelEvaluator, compare_models_on_dataframe
            current_model = effective_model
            st.caption(f"实际模型 · {str(current_model).upper()}")
            col1, col2, col3 = st.columns(3)
            processing_time = result.get('processing_time')
            model_memory = result.get('model_memory')
            col1.metric("分析耗时", f"{processing_time:.2f}s" if processing_time is not None else "未记录")
            col2.metric("内存增量", f"{model_memory:.1f}MB" if model_memory is not None else "未记录")
            col3.metric("处理样本", f"{result.get('total', 0)} 条")

            if st.button("多模型对比", type="secondary"):
                with st.spinner("正在对比模型性能..."):
                    comparison_result = compare_models_on_dataframe(result['df'], sample_size=50)
                    if 'error' not in comparison_result:
                        st.session_state['model_comparison'] = comparison_result
                        st.success("模型对比完成。")
                    else:
                        st.error(f"模型对比失败：{comparison_result['error']}")

            st.caption("模型一致性仅表示预测相似程度，不代表准确率；准确率需使用人工标注集评估。")

            if 'model_comparison' in st.session_state:
                comparison = st.session_state['model_comparison']
                comparison_data = []
                for model, data in comparison.get('comparison', {}).items():
                    if data and data.performance:
                        perf = data.performance
                        comparison_data.append({
                            '模型': model,
                            '总耗时(s)': f"{perf.total_time:.2f}",
                            '平均耗时(s/条)': f"{perf.avg_time_per_sample:.4f}",
                            '处理样本数': perf.samples_processed,
                        })
                if comparison_data:
                    st.table(comparison_data)
                agreement = comparison.get('agreement', {})
                if agreement and 'average_agreement' in agreement:
                    st.metric("平均一致性", f"{agreement['average_agreement']:.3f}")
        except Exception as e:
            st.warning(f"模型性能信息不可用：{e}")

    # ── 第二行: 情绪分布图 + 词云 ──
    st.markdown("""
    <div class="section-heading">
        <h2>可视化分析</h2>
        <p>情绪分布与关键语义概览</p>
    </div>
    """, unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 情绪分布")
        if 'dist_path' in result and Path(result['dist_path']).exists():
            st.image(result['dist_path'], width='stretch')
        else:
            # 动态生成
            stats = {k: result[k] for k in ['total', 'positive', 'negative', 'neutral']}
            dist_path = str(OUTPUT_DIR / f"dist_fallback_{datetime.now().timestamp()}.png")
            generate_sentiment_distribution(stats, dist_path)
            st.image(dist_path, width='stretch')

    with col_right:
        st.markdown("#### 词云图")
        wc_path = result.get('wc_path')
        if wc_path and Path(wc_path).exists():
            st.image(wc_path, width='stretch')
        else:
            st.info("词云图不可用")

    # ── 第三行: TOP 20 高频词 + 数据表格 ──
    st.markdown("""
    <div class="section-heading">
        <h2>高频关键词</h2>
        <p>样本中出现频率最高的 20 个词项</p>
    </div>
    """, unsafe_allow_html=True)
    keywords = result.get('keywords', [])
    if keywords:
        words = [w for w, _ in keywords[:20]]
        freqs = [f for _, f in keywords[:20]]

        # 使用 matplotlib 绘制水平条形图
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        font_path = find_chinese_font()
        if font_path:
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams["font.family"] = prop.get_name()

        fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
        colors_grad = ['#2563eb' if i < 3 else '#94a3b8' for i in range(len(words))]
        bars = ax.barh(range(len(words)), freqs, color=colors_grad, edgecolor='none')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('频次', fontsize=11, color='#667085')
        ax.set_title('TOP 20 高频关键词', fontsize=14, fontweight='semibold', pad=16, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        ax.grid(axis='x', color='#eef0f3', linewidth=.8)
        ax.set_axisbelow(True)

        for bar, freq in zip(bars, freqs):
            ax.text(bar.get_width() + max(freqs) * 0.02, bar.get_y() + bar.get_height() / 2,
                    str(freq), ha='left', va='center', fontsize=10)

        st.pyplot(fig)
        plt.close(fig)

    # ── 第四行: 评论详情 ──
    st.markdown("""
    <div class="section-heading">
        <h2>评论详情</h2>
        <p>按情绪类别查看与导出原始样本</p>
    </div>
    """, unsafe_allow_html=True)
    df = result.get('df')
    if df is not None and not df.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["全部", "积极", "中性", "消极"])

        tabs_map = {'全部': df, '积极': df[df['nlp_result'] == '积极'],
                    '中性': df[df['nlp_result'] == '中性'],
                    '消极': df[df['nlp_result'] == '消极']}

        for tab_name, tab_obj in [('全部', tab1), ('积极', tab2),
                                   ('中性', tab3), ('消极', tab4)]:
            with tab_obj:
                subset = tabs_map[tab_name]
                subset_volume = int(subset.get('duplicate_count', pd.Series(1, index=subset.index)).sum())
                st.caption(f"{len(subset)} 条去重样本 · 原始声量 {subset_volume} 条")
                display_cols = ['评论内容', 'nlp_result', 'nlp_score', 'nlp_confidence', 'duplicate_count']
                available = [c for c in display_cols if c in subset.columns]
                st.dataframe(
                    subset[available].head(100),
                    width='stretch',
                    height=400,
                )

                if len(subset) > 0:
                    csv_data = subset[available].to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        f"下载{tab_name}评论",
                        csv_data,
                        f"comments_{tab_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                    )

    # ── 分类词云 ──
    if 'sentiment_wc' in result:
        with st.expander("分类情绪词云", expanded=False):
            cols = st.columns(3)
            for i, sentiment in enumerate(['积极', '消极', '中性']):
                with cols[i]:
                    st.markdown(f"#### {sentiment}")
                    wc_path = result['sentiment_wc'].get(sentiment)
                    if wc_path and Path(wc_path).exists():
                        st.image(wc_path, width='stretch')
                    else:
                        st.info("无数据")

    # ── AI 舆情报告 ──
    current_topic = st.session_state.get('current_topic', '')
    if current_topic:
        st.markdown("""
        <div class="section-heading">
            <h2>AI 分析报告</h2>
            <p>以专业咨询报告结构呈现核心发现、风险与趋势判断</p>
        </div>
        """, unsafe_allow_html=True)

        from config import CURRENT_API_AVAILABLE, AI_PROVIDER, REPORT_DIR
        ai_available = CURRENT_API_AVAILABLE

        if not ai_available:
            st.info(
                f"**AI 报告尚未启用** — "
                "请在项目根目录的 `.env` 文件中设置 AI_PROVIDER 和相应的 API Key 以启用 AI 分析。\n\n"
                f"当前 Provider: {AI_PROVIDER}\n"
                "获取 API Key:\n"
                "- SiliconFlow: https://cloud.siliconflow.cn/console\n"
                "- DeepSeek: https://platform.deepseek.com/api_keys"
            )
        else:
            import hashlib
            cache_raw = json.dumps({
                'report_format': 7,
                'topic': current_topic, 'keywords_top10': [w for w, _ in (result.get('keywords', []) or [])[:10]],
                'stats': {k: result.get(k, 0) for k in ['positive', 'negative', 'neutral', 'total', 'unique_total']},
                'posts_count': len(result.get('posts', [])),
                'sampling': {
                    key: (result.get('crawl_metadata') or {}).get(key) for key in (
                        'expected_comments', 'fetched_comments', 'coverage_pct',
                        'representation_status', 'dominant_post_share_pct'
                    )
                },
                'per_post': [
                    [item.get('weibo_id'), item.get('expected_comments'), item.get('fetched_comments')]
                    for item in (result.get('crawl_metadata') or {}).get('per_post', [])[:10]
                ],
        }, sort_keys=True, ensure_ascii=False)
        cache_key = hashlib.md5(cache_raw.encode()).hexdigest()[:12]
        cache_path = REPORT_DIR / f"report_{cache_key}.md"
        from_cache = cache_path.exists()

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            generate_btn = st.button(
                "生成 AI 报告" if not from_cache else "查看 AI 报告",
                type="primary", width='stretch',
            )
        with col_status:
            if from_cache:
                st.caption(f"已缓存 · {datetime.fromtimestamp(cache_path.stat().st_mtime).strftime('%m-%d %H:%M')}")
            else:
                st.caption("首次生成约需 30-60 秒，后续从缓存加载")

        if generate_btn:
            with st.spinner(f"{AI_PROVIDER} 正在生成分析报告..."):
                try:
                    from src.ai_agent import ReportGenerator

                    gen = ReportGenerator()
                    ai_result = gen.generate(
                        topic=current_topic,
                        stats={k: result[k] for k in ['total', 'positive', 'negative', 'neutral']},
                        df=result.get('df'),
                        posts=result.get('posts', []),
                        keywords=result.get('keywords', []),
                        use_cache=True, quick=False,
                        sampling=result.get('crawl_metadata', {}),
                    )

                    if ai_result['success']:
                        st.session_state['ai_report'] = ai_result['report']
                        st.session_state['ai_report_path'] = ai_result['report_path']
                        st.session_state['ai_from_cache'] = ai_result['from_cache']
                        active_task_id = result.get('task_id') or st.session_state.get('task_id')
                        if active_task_id:
                            update_task_report(active_task_id, ai_result['report_path'], AI_PROVIDER)
                        if ai_result['from_cache']:
                            st.success("已从缓存加载报告。")
                        else:
                            st.success("AI 报告生成完成。")
                    else:
                        st.error(ai_result['error'])
                except Exception as e:
                    st.error(f"AI 报告生成失败：{e}")
                    log.error("AI 报告异常: %s", e)

        ai_report = st.session_state.get('ai_report')
        if ai_report:
            report_for_display = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", ai_report)
            with st.container(border=True):
                st.markdown(report_for_display)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "下载报告 (Markdown)",
                    ai_report.encode('utf-8'),
                    f"AI_Report_{current_topic}_{datetime.now().strftime('%Y%m%d')}.md",
                    "text/markdown", width='stretch',
                )
            with c2:
                if st.button("清除报告", width='stretch'):
                    st.session_state.pop('ai_report', None)
                    st.rerun()

else:
    # 欢迎页面
    st.markdown("""
    <div class="empty-state">
        <div class="empty-kicker">Start an analysis</div>
        <h2>从话题或评论数据开始</h2>
        <p>在左侧选择数据来源，输入微博话题或上传包含“评论内容”列的 CSV 文件，然后开始分析。</p>
    </div>
    """, unsafe_allow_html=True)

    # 系统状态
    with st.expander("运行环境", expanded=False):
        status_cols = st.columns(4)
        with status_cols[0]:
            font_ok = has_chinese_font()
            st.metric("中文字体", "可用" if font_ok else "未找到")
        with status_cols[1]:
            from src.webdriver_manager import find_chrome_binary, find_chromedriver
            driver_ok = bool(find_chrome_binary() and find_chromedriver())
            st.metric("ChromeDriver", "就绪" if driver_ok else "缺失")
        with status_cols[2]:
            if TURSO_DATABASE_URL:
                st.metric("数据库", "Turso Cloud")
            else:
                from config import DATABASE_PATH
                db_ok = Path(DATABASE_PATH).exists()
                st.metric("SQLite 数据库", "已初始化" if db_ok else "待初始化")
        with status_cols[3]:
            st.metric("最近任务", f"{len(tasks)} 条")
