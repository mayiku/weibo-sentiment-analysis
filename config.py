"""
全局配置文件
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass


def _env_bool(name: str, default: bool) -> bool:
    """Read a deployment-friendly boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = ROOT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "model"
DEBUG_DIR = ROOT_DIR / "debug"          # 失败页面 HTML + 截图
COOKIE_DIR = ROOT_DIR / "cookies"       # Cookie 缓存

# ChromeDriver 路径 (webdriver-manager 自动管理时的后备)
CHROMEDRIVER_PATH = ROOT_DIR / "chromedriver-win64" / "chromedriver.exe"

# SQLite 数据库路径
DATABASE_PATH = DATA_DIR / "sentiment_system.db"

# Cookie 缓存文件
COOKIE_FILE = COOKIE_DIR / "weibo_cookies.pkl"

# 确保目录存在
for d in [DATA_DIR, OUTPUT_DIR, LOG_DIR, MODEL_DIR, DEBUG_DIR, COOKIE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 情感分析阈值
SENTIMENT_POS_THRESHOLD = 0.6
SENTIMENT_NEG_THRESHOLD = 0.4

# 爬虫配置 - 用户体验优化
CRAWLER_PAGE_NUM = 3
CRAWLER_SCROLL_TIMES = 10
CRAWLER_HEADLESS = _env_bool("CRAWLER_HEADLESS", not bool(os.getenv("DISPLAY")))
CRAWLER_AUTO_LOGIN_TIMEOUT = 180   # 登录超时延长至180秒，方便用户扫码
CRAWLER_MOCK_FALLBACK = False      # 真实爬取失败时明确报错，避免把模拟数据当成真实结果
CRAWLER_REQUEST_DELAY_MIN = 3.0    # 请求间隔最小值（防反爬）
CRAWLER_REQUEST_DELAY_MAX = 7.0    # 请求间隔最大值
CRAWLER_PAGE_LOAD_TIMEOUT = 60     # 页面加载超时
CRAWLER_ELEMENT_WAIT_TIMEOUT = 15  # 元素等待超时
CRAWLER_MAX_RETRIES = 3            # 重试次数，增加用户体验稳定性
CRAWLER_SELENIUM_SCROLL_DELAY_MIN = 2.0
CRAWLER_SELENIUM_SCROLL_DELAY_MAX = 4.0

# 云端登录凭据。建议仅在 Streamlit Secrets 中配置，严禁提交到 GitHub。
# 支持浏览器 Cookie Header 格式："SUB=...; SUBP=...; XSRF-TOKEN=..."
WEIBO_COOKIE = os.getenv("WEIBO_COOKIE", "").strip()

# API 爬虫配置（requests 方式）
CRAWLER_API_ENABLED = True          # 是否启用 API 方式（优先）
CRAWLER_API_COMMENTS_PER_PAGE = 20  # 每页评论数（最大值）
CRAWLER_API_MAX_PAGES = None        # 最大翻页数（None=直到末尾）
CRAWLER_API_DELAY_MIN = 0.1          # API 请求间隔最小值（秒）- 自适应退避
CRAWLER_API_DELAY_MAX = 0.5          # API 请求间隔最大值（秒）
CRAWLER_API_RETRIES = 3             # API 请求重试次数
CRAWLER_API_TIMEOUT = 15            # API 请求超时（秒）
CRAWLER_API_ALTERNATE_FLOW_ENABLED = True  # 低覆盖高评论帖尝试另一排序窗口
CRAWLER_API_ALTERNATE_FLOW_MIN_COMMENTS = 50
CRAWLER_API_ALTERNATE_FLOW_COVERAGE_THRESHOLD = 25.0
CRAWLER_API_ALTERNATE_FLOW_MAX_POSTS = 3

# 词云配置
WORDCLOUD_MAX_WORDS = 200
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 600

# 高频词 TOP N
TOP_KEYWORDS_N = 20

# 停用词表
STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '那', '这个', '那个', '啊', '吧',
    '呢', '呀', '吗', '哦', '嗯', '哈', '嘛', '啦', '哟', '哎', '唉',
    '还是', '但是', '因为', '所以', '如果', '虽然', '可以', '知道',
    '觉得', '真的', '就是', '不是', '什么', '怎么', '怎么样', '为什么',
    '然后', '已经', '比较', '开始', '一直', '可能', '应该', '不过',
    '全部', '一点', '一些', '有点', '有点', '的话', '而已', '而且',
    '还有', '还是', '或者', '只是', '但是', '嗯嗯', '哈哈哈', '哈哈',
])

# ============================================================================
# AI Agent 配置 - 多 Provider 支持
# ============================================================================

# 报告存储目录
REPORT_DIR = ROOT_DIR / "data" / "reports"

# 确保目录存在
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 新情感分析模块配置
# ============================================================================

# 默认情感分析模型
DEFAULT_SENTIMENT_MODEL = "hybrid"  # 云端默认使用轻量微博规则增强模型

# 情感分析阈值（兼容原有配置）
SENTIMENT_POS_THRESHOLD = 0.6
SENTIMENT_NEG_THRESHOLD = 0.4

# 批量处理配置
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_GPU_ENABLED = _env_bool("SENTIMENT_GPU_ENABLED", True)
SENTIMENT_MAX_TEXT_LENGTH = 512

# 微博场景优化配置
WEIBO_ENHANCEMENT_ENABLED = True
WEIBO_TRENDING_WORDS_ENABLED = True
WEIBO_EMOJI_ANALYSIS_ENABLED = True
WEIBO_FAN_CIRCLE_ENHANCEMENT = True
WEIBO_SPORTS_ENHANCEMENT = True

# 模型缓存配置
MODEL_CACHE_SIZE = 3  # 最大缓存模型数量
MODEL_LOAD_TIMEOUT = 30  # 模型加载超时（秒）

# 模型文件路径
PADDLE_MODEL_PATH = MODEL_DIR / "paddle_sentiment"
BERT_MODEL_PATH = MODEL_DIR / "bert_sentiment"

# 确保模型目录存在
PADDLE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
BERT_MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 评估配置
EVALUATION_SAMPLE_SIZE = 100  # 评估抽样数量
EVALUATION_METRICS_ENABLED = True
EVALUATION_REPORT_ENABLED = True

# 环境变量配置（.env 已在模块顶部加载）
import os as _os

# AI Provider 选择
AI_PROVIDER = _os.getenv("AI_PROVIDER", "deepseek").lower()  # deepseek 或 siliconflow

# DeepSeek 配置 (兼容模式)
DEEPSEEK_API_KEY = _os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = _os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_MAX_TOKENS = int(_os.getenv("DEEPSEEK_MAX_TOKENS", "4096"))
DEEPSEEK_TEMPERATURE = float(_os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))
DEEPSEEK_TIMEOUT = int(_os.getenv("DEEPSEEK_TIMEOUT", "60"))
DEEPSEEK_MAX_RETRIES = int(_os.getenv("DEEPSEEK_MAX_RETRIES", "2"))
DEEPSEEK_SENTIMENT_BATCH_SIZE = int(_os.getenv("DEEPSEEK_SENTIMENT_BATCH_SIZE", "30"))
DEEPSEEK_SENTIMENT_BATCH_RETRIES = int(_os.getenv("DEEPSEEK_SENTIMENT_BATCH_RETRIES", "2"))
DEEPSEEK_SENTIMENT_MAX_TEXT_LENGTH = int(_os.getenv("DEEPSEEK_SENTIMENT_MAX_TEXT_LENGTH", "280"))
AI_REPORT_MAX_TOKENS = int(_os.getenv("AI_REPORT_MAX_TOKENS", "2600"))
AI_QUICK_REPORT_MAX_TOKENS = int(_os.getenv("AI_QUICK_REPORT_MAX_TOKENS", "700"))
AI_REPAIR_MAX_TOKENS = int(_os.getenv("AI_REPAIR_MAX_TOKENS", "1800"))

# SiliconFlow 配置 (可选)
SILICONFLOW_API_KEY = _os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = _os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = _os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3.1-Terminus")
SILICONFLOW_MAX_TOKENS = int(_os.getenv("SILICONFLOW_MAX_TOKENS", "4096"))
SILICONFLOW_TEMPERATURE = float(_os.getenv("SILICONFLOW_TEMPERATURE", "0.3"))
SILICONFLOW_TIMEOUT = int(_os.getenv("SILICONFLOW_TIMEOUT", "120"))
SILICONFLOW_MAX_RETRIES = int(_os.getenv("SILICONFLOW_MAX_RETRIES", "3"))

# 当前有效的 API 配置 (根据 AI_PROVIDER 选择)
CURRENT_API_KEY = SILICONFLOW_API_KEY if AI_PROVIDER == "siliconflow" else DEEPSEEK_API_KEY
CURRENT_API_AVAILABLE = bool(CURRENT_API_KEY)
