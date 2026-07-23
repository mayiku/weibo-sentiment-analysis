"""
词云生成模块 — 生成词云图并提取高频词
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import matplotlib.font_manager as fm
from pathlib import Path

from config import WORDCLOUD_MAX_WORDS, WORDCLOUD_WIDTH, WORDCLOUD_HEIGHT
from src.logger import get_logger

log = get_logger(__name__)

# ── 中文字体管理 ────────────────────────────────────────

# 常见中文字体路径（Windows / macOS / Linux）
_CHINESE_FONT_CANDIDATES = [
    # Windows
    'C:/Windows/Fonts/simhei.ttf',        # 黑体（首选，字形完整）
    'C:/Windows/Fonts/msyh.ttc',          # 微软雅黑
    'C:/Windows/Fonts/msyhbd.ttc',        # 微软雅黑 Bold
    'C:/Windows/Fonts/simsun.ttc',        # 宋体
    'C:/Windows/Fonts/kaiu.ttf',          # 楷体
    'C:/Windows/Fonts/Deng.ttf',          # 等线
    'C:/Windows/Fonts/Dengb.ttf',         # 等线 Bold
    # macOS
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
    # Linux
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
]

# 缓存已找到的字体
_CACHED_FONT_PATH = None
_CACHED_FONT_NAME = None


def _register_chinese_font() -> str | None:
    """
    查找并注册中文字体到 matplotlib 全局配置。

    只在首次调用时查找，后续使用缓存。

    Returns:
        字体文件路径，未找到则 None
    """
    global _CACHED_FONT_PATH, _CACHED_FONT_NAME

    if _CACHED_FONT_PATH is not None:
        return _CACHED_FONT_PATH

    font_path = None
    font_name = None

    # 1. 直接检查常见路径（最快）
    for path in _CHINESE_FONT_CANDIDATES:
        if os.path.exists(path):
            font_path = path
            log.info("找到中文字体: %s", path)
            break

    # 2. 回退：扫描系统所有字体
    if not font_path:
        log.info("常见路径未找到中文字体，扫描系统中...")
        chinese_keywords = ['SimHei', 'WenQuanYi', 'Heiti', 'YaHei', 'SimSun',
                            'KaiTi', 'PingFang', 'CJK', 'DroidSans', 'uming']
        for fp in fm.findSystemFonts():
            try:
                name = fm.FontProperties(fname=fp).get_name()
                if any(kw.lower() in name.lower() for kw in chinese_keywords):
                    font_path = fp
                    log.info("扫描找到中文字体: %s (%s)", name, fp)
                    break
            except Exception:
                continue

    if not font_path:
        log.warning("未找到中文字体！图表和词云将无法正确显示中文")
        _CACHED_FONT_PATH = None
        _CACHED_FONT_NAME = None
        return None

    # ★ 关键：注册字体到 matplotlib 并设置全局 rcParams
    try:
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        log.info("字体名称: %s", font_name)

        # Register only the selected font. Rebuilding the full system font
        # cache here added 10-20 seconds to every fresh application process.
        fm.fontManager.addfont(font_path)

        # ★ 设置全局默认字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        log.info("中文字体已注册到 matplotlib: %s", font_name)
    except Exception as e:
        log.warning("注册字体到 matplotlib 失败: %s，回退使用路径", e)
        font_name = None

    _CACHED_FONT_PATH = font_path
    _CACHED_FONT_NAME = font_name
    return font_path


def find_chinese_font() -> str | None:
    """查找系统中可用的中文字体（供 wordcloud 使用）"""
    return _register_chinese_font()


def has_chinese_font() -> bool:
    """Check font availability without importing properties or rebuilding caches."""
    return any(os.path.exists(path) for path in _CHINESE_FONT_CANDIDATES)


def get_font_properties():
    """
    获取 FontProperties 对象，供 matplotlib 图表使用。

    应在每次图表生成前调用，确保字体已注册。
    """
    _register_chinese_font()
    if _CACHED_FONT_PATH:
        return fm.FontProperties(fname=_CACHED_FONT_PATH)
    return None


def generate_wordcloud(text_series, output_path: str = 'output/wordcloud.png',
                       mask_image: str = None, font_path: str = None) -> WordCloud:
    """
    生成词云图

    参数:
        text_series: 分词后的文本 Series 或空格分隔的字符串
        output_path: 输出图片路径
        mask_image: 遮罩图片路径（可选）
        font_path: 字体路径（可选，自动检测）

    返回:
        WordCloud 对象
    """
    log.info("生成词云...")

    # 合并文本
    if hasattr(text_series, 'str'):
        all_text = " ".join(text_series.dropna().astype(str))
    else:
        all_text = str(text_series)

    if not all_text.strip():
        log.warning("词云输入文本为空 — 生成空白词云")
        all_text = "无数据"

    # 字体
    if font_path is None:
        font_path = find_chinese_font()

    # 遮罩
    mask = None
    if mask_image and os.path.exists(mask_image):
        mask = np.array(Image.open(mask_image))
        log.info("使用遮罩图片: %s", mask_image)

    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        colormap='Blues',
        max_words=WORDCLOUD_MAX_WORDS,
        max_font_size=100,
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        mask=mask,
        collocations=False,
    )

    wc.generate(all_text)

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(output_path)
    log.info("词云已保存: %s", output_path)

    return wc


def generate_sentiment_distribution(stats: dict, output_path: str) -> str:
    """
    生成情感分布柱状图
    """
    log.info("生成情感分布图...")

    # ★ 确保中文字体已注册
    font_prop = get_font_properties()

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    labels = ['积极', '中性', '消极']
    values = [stats['positive'], stats['neutral'], stats['negative']]
    colors = ['#2563EB', '#94A3B8', '#475569']

    bars = ax.bar(labels, values, color=colors, edgecolor='none', width=.58)

    # ★ 每个文字元素显式指定 FontProperties
    ax.set_title('情绪分布', fontsize=15, fontweight='semibold', pad=16, loc='left',
                 fontproperties=font_prop)
    ax.set_ylabel('评论数量', fontsize=11, color='#667085', fontproperties=font_prop)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=12)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 在柱子上标注数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                str(val), ha='center', va='bottom', fontsize=13, fontweight='bold',
                fontproperties=font_prop)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.grid(axis='y', color='#eef0f3', linewidth=.8)
    ax.set_axisbelow(True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("分布图已保存: %s", output_path)

    return output_path


def generate_sentiment_wordclouds(df, output_dir: str = 'output',
                                  font_path: str = None) -> dict[str, str]:
    """
    按情绪分类生成词云

    返回: {'积极': path, '消极': path, '中性': path}
    """
    log.info("按情绪分类生成词云...")
    paths = {}
    for sentiment in ['积极', '消极', '中性']:
        subset = df[df['nlp_result'] == sentiment]
        if not subset.empty and subset['clean_text'].str.strip().sum():
            out_path = f"{output_dir}/wordcloud_{sentiment}.png"
            generate_wordcloud(subset['clean_text'], out_path, font_path=font_path)
            paths[sentiment] = out_path
        else:
            log.info("无 '%s' 评论，跳过", sentiment)
            paths[sentiment] = None
    return paths
