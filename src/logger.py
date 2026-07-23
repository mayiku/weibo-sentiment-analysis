"""
统一日志模块 — 同时输出到文件和控制台
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from config import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

# 每次启动创建新的日志文件
_LOG_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FILE = LOG_DIR / f"run_{_LOG_TIMESTAMP}.log"

# 根 logger
_logger = logging.getLogger("weibo_sentiment")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False

# 防止重复添加 handler
if not _logger.handlers:
    # 文件 handler — 详细日志
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _logger.addHandler(fh)

    # 控制台 handler — INFO 以上
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    _logger.addHandler(ch)


def get_logger(name: str = "weibo_sentiment") -> logging.Logger:
    """获取子 logger"""
    if name == "weibo_sentiment":
        return _logger
    # All module loggers must be children of the configured application logger;
    # otherwise console output appears, but the configured file stays empty.
    child = _logger.getChild(name)
    child.setLevel(logging.DEBUG)
    return child


def get_log_file() -> Path:
    """返回当前日志文件路径"""
    return _LOG_FILE
