"""
微博话题爬虫模块 — 使用 Selenium 自动化爬取微博评论

v3.0 自动诊断模式:
  - webdriver-manager 自动匹配 Chrome 版本
  - 基于真实微博 HTML 结构的多重 CSS + XPath 选择器
  - 每一步记录详细日志（登录态/卡片数/评论入口数/评论数）
  - 失败时自动保存 debug.html + debug.png
  - 评论数为0时自动分析HTML并输出根因诊断
  - 登录状态检测 + Cookie 持久化缓存
  - Mock 数据回退模式（系统可演示）
"""
import csv
import json
import time
import random
import re
import pickle
import os
from datetime import datetime
from urllib.parse import quote
from pathlib import Path
from typing import Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options

from config import (
    CHROMEDRIVER_PATH,
    CRAWLER_PAGE_NUM,
    CRAWLER_SCROLL_TIMES,
    CRAWLER_HEADLESS,
    CRAWLER_MOCK_FALLBACK,
    CRAWLER_AUTO_LOGIN_TIMEOUT,
    CRAWLER_REQUEST_DELAY_MIN,
    CRAWLER_REQUEST_DELAY_MAX,
    CRAWLER_PAGE_LOAD_TIMEOUT,
    CRAWLER_ELEMENT_WAIT_TIMEOUT,
    CRAWLER_API_ENABLED,
    CRAWLER_API_MAX_PAGES,
    CRAWLER_API_DELAY_MIN,
    CRAWLER_API_DELAY_MAX,
    DATA_DIR,
    DEBUG_DIR,
    COOKIE_DIR,
    COOKIE_FILE,
)
from src.logger import get_logger
from src.webdriver_manager import find_chrome_binary, find_chromedriver

log = get_logger(__name__)

# ============================================================================
# 选择器注册表 — 基于真实微博 HTML 结构 (2026-07)
# ============================================================================

# ── 搜索页：微博卡片容器 ──
SEARCH_CARD_SELECTORS = [
    (By.CSS_SELECTOR, "div[action-type='feed_list_item']", "feed_list_item 属性"),
    (By.CSS_SELECTOR, "div.card-wrap",                    "card-wrap 类名"),
    (By.CSS_SELECTOR, "div.card",                          "card 类名"),
    (By.XPATH, "//div[contains(@class, 'card-wrap')]",    "XPATH: card-wrap"),
    (By.XPATH, "//div[@action-type='feed_list_item']",    "XPATH: feed_list_item"),
]

# ── 搜索页：提取微博帖子 URL（多种方式） ──
# 方式A: 从 .from 区域的发布时间链接提取
POST_URL_SELECTORS_FROM_LINK = [
    (By.CSS_SELECTOR, "div.from a[href*='weibo.com']",        "from区微博链接"),
    (By.XPATH, ".//div[contains(@class,'from')]//a[contains(@href,'weibo.com')]", "XPATH: from区链接"),
]

# 方式B: 从卡片中任意指向 weibo.com/{uid}/{code} 的链接提取
POST_URL_SELECTORS_ANY = [
    (By.CSS_SELECTOR, "a[href*='//weibo.com/'][href*='refer_flag']", "微博正文链接"),
    (By.XPATH, ".//a[contains(@href,'//weibo.com/') and contains(@href,'refer_flag')]", "XPATH: 微博正文链接"),
]

# ── 搜索页：评论区入口（备用：构造URL时不需要） ──
COMMENT_ENTRY_SELECTORS = [
    (By.CSS_SELECTOR, "a[action-type='feed_list_comment']", "评论按钮 (feed_list_comment)"),
    (By.XPATH, ".//a[@action-type='feed_list_comment']",    "XPATH: 评论按钮"),
]

# ── 搜索页：下一页 ──
NEXT_PAGE_SELECTORS = [
    (By.CSS_SELECTOR, "a.next",                              "下一页 (a.next)"),
    (By.CSS_SELECTOR, "a[href*='page='].next",               "下一页 (href含page)"),
    (By.XPATH, "//a[contains(@class, 'next')]",              "XPATH: class含next"),
    (By.XPATH, "//a[text()='下一页']",                       "XPATH: text=下一页"),
    (By.CSS_SELECTOR, ".m-page a.next",                      "下一页 (m-page)"),
]

# ── 评论页：评论列表容器 ──
COMMENT_LIST_SELECTORS = [
    (By.CSS_SELECTOR, "div[class*='comment_li']",             "comment_li"),
    (By.CSS_SELECTOR, "div.list_li",                          "list_li"),
    (By.CSS_SELECTOR, ".wbpro-scroller-item",                 "wbpro-scroller"),
    (By.CSS_SELECTOR, "div.WB_feed_repeat",                   "WB_feed_repeat"),
    (By.CSS_SELECTOR, "div[class*='Comment_item']",           "Comment_item"),
    (By.XPATH, "//div[contains(@class, 'comment_li')]",       "XPATH: comment_li"),
    (By.XPATH, "//div[contains(@class, 'list_li')]",          "XPATH: list_li"),
    (By.XPATH, "//div[contains(@class, 'Comment')]",          "XPATH: Comment"),
]

# ── 评论页：评论文本 ──
COMMENT_TEXT_SELECTORS = [
    (By.CSS_SELECTOR, "div.text",                             "div.text"),
    (By.CSS_SELECTOR, "div.txt",                              "div.txt"),
    (By.CSS_SELECTOR, "span.text",                            "span.text"),
    (By.CSS_SELECTOR, ".WB_text",                             "WB_text"),
    (By.CSS_SELECTOR, "div[class*='text']",                   "class含text的div"),
    (By.XPATH, ".//div[contains(@class, 'text')]",            "XPATH: class含text"),
    (By.XPATH, ".//*[contains(@class, 'txt')]",               "XPATH: class含txt"),
    (By.XPATH, ".//p[contains(@class, 'txt')]",               "XPATH: p含txt"),
]

# ── 卡片内：帖子正文 ──
POST_CONTENT_SELECTORS = [
    (By.CSS_SELECTOR, "p[node-type='feed_list_content'].txt",   "p.txt (feed_list_content)"),
    (By.CSS_SELECTOR, "p.txt",                                   "p.txt"),
    (By.CSS_SELECTOR, "div.content p.txt",                       "div.content p.txt"),
    (By.XPATH, ".//p[@node-type='feed_list_content']",          "XPATH: feed_list_content"),
    (By.XPATH, ".//p[contains(@class, 'txt')]",                 "XPATH: p含txt"),
    (By.XPATH, ".//div[@node-type='like']//p",                  "XPATH: content区首段p"),
]

# ── 卡片内：用户名 ──
USERNAME_SELECTORS = [
    (By.CSS_SELECTOR, "a.name",                                  "a.name"),
    (By.CSS_SELECTOR, "div.info a[nick-name]",                   "a[nick-name]"),
    (By.XPATH, ".//a[@nick-name]",                              "XPATH: nick-name"),
    (By.XPATH, ".//div[contains(@class,'info')]//a[contains(@class,'name')]", "XPATH: a.name"),
]

# ── 卡片内：评论数 ──
COMMENT_COUNT_SELECTORS = [
    (By.CSS_SELECTOR, "a[action-type='feed_list_comment']",     "评论按钮"),
    (By.XPATH, ".//a[@action-type='feed_list_comment']",        "XPATH: 评论按钮"),
]

# ── 卡片内：发布时间 ──
POST_TIME_SELECTORS = [
    (By.CSS_SELECTOR, "div.from a[href*='refer_flag']",         "from区时间链接"),
    (By.XPATH, ".//div[contains(@class,'from')]//a[contains(@href,'weibo.com')]", "XPATH: from区链接"),
]

# ============================================================================
# 诊断工具 — 页面结构分析
# ============================================================================

def _log_all_selectors():
    """将所有选择器输出到日志"""
    groups = [
        ("SEARCH_CARD", SEARCH_CARD_SELECTORS),
        ("POST_URL (from)", POST_URL_SELECTORS_FROM_LINK),
        ("POST_URL (any)", POST_URL_SELECTORS_ANY),
        ("NEXT_PAGE", NEXT_PAGE_SELECTORS),
        ("COMMENT_LIST", COMMENT_LIST_SELECTORS),
        ("COMMENT_TEXT", COMMENT_TEXT_SELECTORS),
        ("POST_CONTENT", POST_CONTENT_SELECTORS),
        ("USERNAME", USERNAME_SELECTORS),
        ("COMMENT_COUNT", COMMENT_COUNT_SELECTORS),
    ]
    log.debug("【选择器配置清单】")
    for name, selectors in groups:
        log.debug("  %s: %d 条", name, len(selectors))
        for _, selector, description in selectors:
            log.debug("    - %s: %s", description, selector)


def analyze_page_structure(driver: webdriver.Chrome) -> str:
    """
    自动分析当前页面 HTML 结构，输出诊断报告。

    检测内容:
      - 页面标题/URL
      - 是否存在关键 CSS 类名
      - 当前页面属于哪个阶段（搜索/登录/验证码/错误）
      - 所有可用选择器的匹配状况
    """
    try:
        page_source = driver.page_source or ""
        current_url = driver.current_url or "unknown"
        title = driver.title or "unknown"
    except Exception:
        page_source = ""
        current_url = "unknown"
        title = "unknown"

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("【页面结构自动诊断报告】")
    report_lines.append(f"  URL: {current_url[:200]}")
    report_lines.append(f"  Title: {title}")
    report_lines.append(f"  页面大小: {len(page_source)} 字符")
    report_lines.append("-" * 60)

    # 1. 判断页面类型
    if 'login' in current_url.lower() or 'passport' in current_url.lower():
        report_lines.append("  [*] 页面类型: 登录页（需要登录）")
    elif 's.weibo.com/weibo' in current_url or 's.weibo.com' in current_url:
        report_lines.append("  [*] 页面类型: 微博搜索页")
    elif 's.weibo.com/realtime' in current_url:
        report_lines.append("  [*] 页面类型: 微博实时搜索页")
    elif 'weibo.com' in current_url and ('/comment/' in current_url or 'refer_flag' in current_url):
        report_lines.append("  [*] 页面类型: 微博帖子详情/评论页")
    elif 'verify' in current_url.lower() or 'captcha' in current_url.lower():
        report_lines.append("  [*] 页面类型: [WARN] 验证码/人机验证页")
    else:
        report_lines.append("  [*] 页面类型: 未知")

    # 2. 关键 CSS 类名检测
    key_classes = [
        'card-wrap', 'card-feed', 'feed_list_item', 'comment_li',
        'list_li', 'WB_feed_repeat', 'wbpro-scroller-item',
        'm-page', 'next', 'card-no-result', 'no-result',
        'gn_nav_login', 'gn_name', 'WB_miniblog-fb',
    ]
    report_lines.append("-" * 60)
    report_lines.append("  【关键CSS类名检测】")
    for cls in key_classes:
        found = cls in page_source
        marker = "[OK]" if found else "[FAIL]"
        report_lines.append(f"    {marker} .{cls}")

    # 3. 选择器逐一检测结果
    report_lines.append("-" * 60)
    report_lines.append("  【选择器匹配检测】")
    all_selector_groups = [
        ("SEARCH_CARD", SEARCH_CARD_SELECTORS),
        ("POST_URL(from)", POST_URL_SELECTORS_FROM_LINK),
        ("NEXT_PAGE", NEXT_PAGE_SELECTORS),
        ("COMMENT_LIST", COMMENT_LIST_SELECTORS),
        ("COMMENT_ENTRY", COMMENT_ENTRY_SELECTORS),
    ]
    for group_name, selectors in all_selector_groups:
        for by, selector, desc in selectors:
            try:
                elements = driver.find_elements(by, selector)
                count = len(elements)
                marker = f"[OK] {count}个" if count > 0 else "[FAIL]"
                report_lines.append(f"    {marker} [{group_name}] {desc}: {selector}")
            except Exception as e:
                report_lines.append(f"    [FAIL] [{group_name}] {desc}: 异常({e})")

    # 4. 根因推断
    report_lines.append("-" * 60)
    report_lines.append("  【根因推断】")

    has_cards = 'card-wrap' in page_source or 'feed_list_item' in page_source
    has_comments = any(c in page_source for c in ['comment_li', 'list_li', 'WB_feed_repeat'])
    has_login_btn = 'gn_nav_login' in page_source
    has_verify = any(w in page_source for w in ['验证码', '请输入验证码', '滑块验证', '访问过于频繁'])
    has_no_result = any(w in page_source for w in ['未找到', '没有找到', '暂无相关'])

    reasons = []
    if has_verify:
        reasons.append("[WARN] 检测到验证码/反爬拦截 → 需要手动在浏览器中完成验证后重试")
    if has_login_btn and 's.weibo.com' in current_url:
        reasons.append("[WARN] 未登录 → 微博搜索需要登录态，请先登录")
    if has_no_result:
        reasons.append("[WARN] 搜索无结果 → 话题不存在或拼写有误")
    if not has_cards and 's.weibo.com' in current_url:
        reasons.append("[WARN] 未检测到微博卡片 → 页面结构可能已变更，需要更新选择器")
    if has_cards and not has_comments and 'comment' in current_url:
        reasons.append("[WARN] 检测到帖子但无评论 → 1)该帖无评论 2)评论区需滚动加载 3)评论区结构变更")
    if not reasons:
        reasons.append("未检测到明显异常，可能是选择器需要更新")

    for r in reasons:
        report_lines.append(f"  {r}")

    report_lines.append("=" * 60)
    return "\n".join(report_lines)


def _save_debug_info(driver: webdriver.Chrome, tag: str) -> Path:
    """保存 debug.html + debug.png + 诊断报告到 debug 目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = re.sub(r'[\\/*?:"<>|]', '_', tag)
    debug_dir = DEBUG_DIR / f"{timestamp}_{safe_tag}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存诊断报告
    try:
        report = analyze_page_structure(driver)
        report_path = debug_dir / "diagnostic_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        log.info("诊断报告已保存: %s", report_path)
        # Also print the report to log
        for line in report.split('\n'):
            log.info("  [DIAG] %s", line)
    except Exception as e:
        log.warning("生成诊断报告失败: %s", e)

    # 2. 保存 HTML
    try:
        html_path = debug_dir / "debug.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        log.info("debug.html 已保存: %s (%d 字符)", html_path, len(driver.page_source))
    except Exception as e:
        log.warning("保存 HTML 失败: %s", e)

    # 3. 保存截图
    try:
        png_path = debug_dir / "debug.png"
        driver.save_screenshot(str(png_path))
        log.info("debug.png 已保存: %s", png_path)
    except Exception as e:
        log.warning("保存截图失败: %s", e)

    # 4. 保存 URL
    try:
        url_path = debug_dir / "current_url.txt"
        with open(url_path, 'w', encoding='utf-8') as f:
            f.write(f"{driver.current_url}\n{driver.title}")
    except Exception:
        pass

    return debug_dir


# ============================================================================
# Cookie 管理
# ============================================================================

class CookieManager:
    """微博 Cookie 持久化缓存"""

    def __init__(self, cookie_file: Path = COOKIE_FILE):
        self.cookie_file = cookie_file

    def save(self, driver: webdriver.Chrome) -> bool:
        try:
            cookies = driver.get_cookies()
            if not cookies:
                log.warning("没有可保存的 Cookie（空列表）")
                return False
            self.cookie_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cookie_file, 'wb') as f:
                pickle.dump(cookies, f)
            log.info("Cookie 已保存: %s (%d 条)", self.cookie_file, len(cookies))
            return True
        except Exception as e:
            log.error("保存 Cookie 失败: %s", e)
            return False

    def load(self, driver: webdriver.Chrome) -> bool:
        if not self.cookie_file.exists():
            log.info("Cookie 文件不存在: %s", self.cookie_file)
            return False
        try:
            with open(self.cookie_file, 'rb') as f:
                cookies = pickle.load(f)
            if not cookies:
                log.warning("Cookie 文件为空")
                return False
            driver.get('https://weibo.com')
            time.sleep(2)
            loaded = 0
            for cookie in cookies:
                try:
                    cookie.pop('sameSite', None)
                    cookie.pop('httpOnly', None)
                    driver.add_cookie(cookie)
                    loaded += 1
                except Exception:
                    pass
            driver.refresh()
            log.info("Cookie 已加载: %d/%d 条", loaded, len(cookies))
            return loaded > 0
        except Exception as e:
            log.error("加载 Cookie 失败: %s", e)
            return False

    def delete(self):
        try:
            if self.cookie_file.exists():
                self.cookie_file.unlink()
                log.info("Cookie 缓存已删除: %s", self.cookie_file)
        except Exception as e:
            log.warning("删除 Cookie 文件失败: %s", e)


# ============================================================================
# WebDriver 管理 — webdriver-manager 自动匹配
# ============================================================================

def _create_driver(headless: bool = None) -> webdriver.Chrome:
    """
    创建 ChromeDriver — webdriver-manager 自动匹配 Chrome 版本。
    失败时回退到手动路径。
    """
    if headless is None:
        headless = CRAWLER_HEADLESS

    options = Options()
    chrome_binary = find_chrome_binary()
    if chrome_binary:
        options.binary_location = chrome_binary
        log.info("检测到 Chrome/Chromium: %s", chrome_binary)
    options.page_load_strategy = 'eager'  # ★ DOM 就绪即返回，不等图片/视频加载
    if headless:
        options.add_argument('--headless=new')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    errors = []

    # 策略1: 云端/Linux 优先使用系统配套的 Chromium + chromedriver
    system_driver = find_chromedriver()
    if system_driver:
        try:
            log.info(">>> 使用系统 ChromeDriver: %s", system_driver)
            driver = webdriver.Chrome(
                service=Service(system_driver), options=options
            )
            log.info("[OK] ChromeDriver 启动成功 (系统路径)")
            _apply_stealth(driver)
            return driver
        except Exception as exc:
            errors.append(f"系统驱动: {exc}")
            log.warning("系统 ChromeDriver 失败: %s", exc)

    # 策略2: webdriver-manager（本地环境自动下载）
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service as ChromeService

        log.info(">>> 使用 webdriver-manager 自动匹配 ChromeDriver...")
        driver_path = ChromeDriverManager().install()
        log.info("    ChromeDriver 路径: %s", driver_path)
        service = ChromeService(driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        log.info("[OK] ChromeDriver 启动成功 (webdriver-manager)")
        _apply_stealth(driver)
        return driver
    except Exception as e:
        errors.append(f"webdriver-manager: {e}")
        log.warning("webdriver-manager 失败: %s", e)

    # 策略3: 兼容旧版 Windows 项目内驱动
    if CHROMEDRIVER_PATH.exists() and str(CHROMEDRIVER_PATH) != system_driver:
        try:
            log.info(">>> 回退到项目内 ChromeDriver: %s", CHROMEDRIVER_PATH)
            service = Service(str(CHROMEDRIVER_PATH))
            driver = webdriver.Chrome(service=service, options=options)
            log.info("[OK] ChromeDriver 启动成功 (项目内路径)")
            _apply_stealth(driver)
            return driver
        except Exception as exc:
            errors.append(f"项目内驱动: {exc}")
            log.error("项目内 ChromeDriver 失败: %s", exc)

    # Both failed
    raise RuntimeError(
        "无法启动 Chromium/ChromeDriver。"
        f" 浏览器={chrome_binary or '未检测到'}；"
        f"驱动={system_driver or '未检测到'}。"
        " 云端请在 packages.txt 安装 chromium 与 chromium-driver；"
        "本地可安装 Chrome，或运行 pip install webdriver-manager。"
        f" 诊断: {' | '.join(errors)[:600]}"
    )


def _apply_stealth(driver: webdriver.Chrome):
    """隐藏自动化特征"""
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    driver.set_page_load_timeout(60)  # 增加到 60s，避免微博页面加载超时
    driver.set_script_timeout(30)


# ============================================================================
# 状态检测
# ============================================================================

def check_login_state(driver: webdriver.Chrome) -> dict:
    """检测微博登录状态 - 增强版自动检测"""
    result = {
        'is_logged_in': False,
        'indicators': [],
        'details': '',
        'confidence': 0,  # 新增：登录信心度 (0-100)
    }

    # URL分析 - 最高优先级
    current_url = driver.current_url or ''
    if 'weibo.com' in current_url and 'login' not in current_url and 'passport' not in current_url:
        result['confidence'] += 40
        result['indicators'].append("已登录:URL非登录页")

    # 已登录标志（高权重）
    logged_in_checks = [
        (By.CSS_SELECTOR, "span.gn_name", "用户名", 30),
        (By.CSS_SELECTOR, "div.WB_miniblog-fb", "发布框", 25),
        (By.XPATH, "//span[contains(@class, 'gn_name')]", "XPATH:用户名", 25),
        (By.CSS_SELECTOR, "i.WB_icon_member", "VIP图标", 20),
        (By.XPATH, "//em[contains(@class, 'W_icon')]", "XPATH:VIP徽章", 15),
    ]

    # 未登录标志（高权重）
    not_logged_checks = [
        (By.CSS_SELECTOR, "a.gn_nav_login", "登录按钮", 30),
        (By.XPATH, "//a[contains(text(),'登录')]", "XPATH:登录链接", 25),
        (By.XPATH, "//a[contains(text(),'注册')]", "XPATH:注册链接", 20),
    ]

    # 检查未登录标志
    for by, sel, desc, weight in not_logged_checks:
        try:
            elements = driver.find_elements(by, sel)
            for el in elements:
                if el and el.is_displayed():
                    result['confidence'] -= weight
                    result['indicators'].append(f"未登录:{desc}")
                    break
        except Exception:
            pass

    # 检查已登录标志
    for by, sel, desc, weight in logged_in_checks:
        try:
            elements = driver.find_elements(by, sel)
            for el in elements:
                if el and el.is_displayed():
                    result['confidence'] += weight
                    result['indicators'].append(f"已登录:{desc}")
                    break
        except Exception:
            pass

    # 动态行为检测 - 检查是否有发布、点赞、关注等互动功能
    has_userinfo = any(
        'username' in (el.get_attribute('outerHTML') or '')
        for el in driver.find_elements(By.CSS_SELECTOR, "[class*='user']")
    )
    has_interactive = any(
        'action' in (el.get_attribute('outerHTML') or '')
        for el in driver.find_elements(By.CSS_SELECTOR, "[action-type]")
    )

    if has_interactive and has_userinfo:
        result['confidence'] += 10
        result['indicators'].append("已登录:互动功能")

    # 决定登录状态
    if result['confidence'] >= 50:
        result['is_logged_in'] = True
        result['details'] = f'[OK] 已登录 (信心度:{result["confidence"]}%)'
    elif result['confidence'] <= -30:
        result['is_logged_in'] = False
        result['details'] = f'[FAIL] 未登录 (信心度:{abs(result["confidence"])}%)'
    else:
        result['is_logged_in'] = result['confidence'] > 0
        confidence_text = f'(信心度:{abs(result["confidence"])}%)'
        result['details'] = f'~ 疑似已登录 {confidence_text}' if result['is_logged_in'] else f'? 状态不明 {confidence_text}'

    log.info("【登录状态】 %s | 指标数: %d", result['details'], len(result['indicators']))
    return result


def check_search_results(driver: webdriver.Chrome) -> dict:
    """检测搜索结果页"""
    result = {'has_results': False, 'is_empty': False, 'is_blocked': False,
              'card_count': 0, 'details': ''}

    page_text = driver.page_source or ''

    # 拦截检测
    block_patterns = [
        ('请输入验证码', '验证码'), ('访问过于频繁', '频率限制'),
        ('滑块验证', '滑块验证'), ('安全验证', '安全验证'),
        ('帐号存在异常', '账号异常'), ('请输入手机号', '手机验证'),
    ]
    for pattern, desc in block_patterns:
        if pattern in page_text:
            result['is_blocked'] = True
            result['details'] = f"[WARN] 反爬拦截: {desc}"
            log.warning("【搜索结果】%s", result['details'])
            return result

    # 空结果检测
    for pattern in ['未找到', '没有找到', '暂无相关', '抱歉，未找到']:
        if pattern in page_text:
            result['is_empty'] = True
            result['details'] = f"搜索结果为空: 含'{pattern}'"
            log.warning("【搜索结果】%s", result['details'])
            return result

    # 计数卡片
    cards = _safe_find_elements(driver, SEARCH_CARD_SELECTORS, timeout=5)
    result['card_count'] = len(cards)
    if cards:
        result['has_results'] = True
        result['details'] = f"[OK] 找到 {len(cards)} 条微博卡片"
    else:
        if any(t in page_text for t in ['card-wrap', 'feed_list', 'comment']):
            result['has_results'] = True
            result['details'] = f"? 页面含预期关键词但选择器未匹配（可能结构变更）"
        else:
            result['is_empty'] = True
            result['details'] = "[FAIL] 未检测到搜索结果"

    log.info("【搜索结果】 %s", result['details'])
    return result


def check_comment_area(driver: webdriver.Chrome) -> dict:
    """检测评论区是否加载"""
    result = {'is_loaded': False, 'comment_count': 0, 'details': ''}

    comments = _safe_find_elements(driver, COMMENT_LIST_SELECTORS, timeout=5)
    result['comment_count'] = len(comments)
    if comments:
        result['is_loaded'] = True
        result['details'] = f"[OK] 评论区已加载: {len(comments)} 个元素"
    else:
        page_text = driver.page_source or ''
        if any(kw in page_text for kw in ['comment', 'WB_text', 'list_li', '评论']):
            result['details'] = "? 含评论关键词但选择器未匹配（结构已变更）"
        else:
            result['details'] = "[FAIL] 未检测到评论区"

    log.info("【评论区】 %s", result['details'])
    return result


# ============================================================================
# 通用选择器工具
# ============================================================================

def _try_selectors(driver, selectors: list, multiple: bool = False,
                   timeout: int = None, context=None):
    """依次尝试多个选择器，返回第一个匹配结果"""
    if timeout is None:
        timeout = CRAWLER_ELEMENT_WAIT_TIMEOUT
    search_root = context if context else driver

    for by, selector, desc in selectors:
        try:
            if timeout > 0:
                if multiple:
                    WebDriverWait(search_root, timeout).until(
                        EC.presence_of_all_elements_located((by, selector)))
                else:
                    WebDriverWait(search_root, timeout).until(
                        EC.presence_of_element_located((by, selector)))
            if multiple:
                elements = search_root.find_elements(by, selector)
            else:
                elements = search_root.find_element(by, selector)
            if elements:
                count = len(elements) if isinstance(elements, list) else 1
                log.debug("  [OK] 选择器 [%s] → %d 个", desc, count)
                return elements
        except (TimeoutException, NoSuchElementException):
            log.debug("  [FAIL] 选择器 [%s] 无匹配", desc)
            continue
        except Exception as e:
            log.debug("  ! 选择器 [%s] 异常: %s", desc, e)
            continue

    return [] if multiple else None


def _safe_find_element(driver, selectors: list, context=None, timeout: int = None):
    return _try_selectors(driver, selectors, multiple=False,
                          timeout=timeout, context=context)


def _safe_find_elements(driver, selectors: list, context=None, timeout: int = None):
    result = _try_selectors(driver, selectors, multiple=True,
                            timeout=timeout, context=context)
    return result if result else []


# ============================================================================
# 登录流程
# ============================================================================

def login_weibo(driver: webdriver.Chrome, status_callback=None) -> bool:
    """
    登录微博 — 智能检测流程：Cookie优先 → 自动登录检测 → 实时进度反馈。

    返回登录状态，但无论成功与否都会继续执行爬取流程。
    """
    cookie_mgr = CookieManager()

    def _report(msg, level='info'):
        """统一状态报告函数"""
        if level == 'success':
            prefix = "[OK]"
            log_func = log.info
        elif level == 'warning':
            prefix = "[WARNING]"
            log_func = log.warning
        elif level == 'error':
            prefix = "[FAILED]"
            log_func = log.error
        else:
            prefix = "[SEARCH]"
            log_func = log.info

        full_msg = f"{prefix} {msg}"
        log_func(f"【登录】{msg}")
        if status_callback:
            status_callback(full_msg)

    # ── 步骤1: Cookie缓存检查 ──
    _report("检查缓存的登录状态...", 'info')
    driver.get('https://weibo.com')
    time.sleep(2)

    if cookie_mgr.load(driver):
        time.sleep(2)
        state = check_login_state(driver)
        if state['is_logged_in']:
            _report(f"使用缓存登录状态 (信心度:{state['confidence']}%)", 'success')
            return True
        else:
            _report(f"Cookie已过期 (信心度:{state['confidence']}%)", 'warning')
            cookie_mgr.delete()
    else:
        _report("未找到缓存Cookie", 'info')

    # ── 步骤2: 自动扫码登录流程 ──
    _report("正在打开微博登录页...", 'info')
    driver.get('https://weibo.com/login.php')
    _report("请在浏览器中扫码登录，系统会自动检测登录状态 🔐", 'info')

    deadline = time.time() + CRAWLER_AUTO_LOGIN_TIMEOUT
    last_state_update = time.time()
    state_counts = {'login_detected': 0, 'not_login': 0, 'unknown': 0}

    while time.time() < deadline:
        elapsed = int(time.time() - (deadline - CRAWLER_AUTO_LOGIN_TIMEOUT))
        remaining = CRAWLER_AUTO_LOGIN_TIMEOUT - elapsed
        state = check_login_state(driver)

        # 状态计数器，用于判断稳定状态
        if state['confidence'] >= 50:
            state_counts['login_detected'] += 1
        elif state['confidence'] <= -30:
            state_counts['not_login'] += 1
        else:
            state_counts['unknown'] += 1

        # 连续5次检测到登录状态，视为稳定登录
        if state_counts['login_detected'] >= 5:
            cookie_mgr.save(driver)
            _report(f"登录成功！(用时{elapsed}s，信心度:{state['confidence']}%)", 'success')
            return True

        # 动态反馈 - 根据状态变化频率调整
        current_time = time.time()
        if current_time - last_state_update >= 2:
            if elapsed == 0:
                # 首次报告
                status_text = f"等待扫码登录... (0/{CRAWLER_AUTO_LOGIN_TIMEOUT}s)"
            else:
                # 根据状态智能报告
                if state_counts['login_detected'] > 0:
                    status_text = f"检测到登录状态... ({state_counts['login_detected']}次)"
                elif state_counts['not_login'] > 0:
                    status_text = f"仍需要登录... ({remaining}s 剩余)"
                else:
                    status_text = f"检测中... ({elapsed}s / {CRAWLER_AUTO_LOGIN_TIMEOUT}s)"

            _report(status_text, 'info')
            last_state_update = current_time

        time.sleep(1.5)  # 较短的检测间隔，提高响应速度

    # ── 步骤3: 超时处理 ──
    final_state = check_login_state(driver)
    if final_state['confidence'] >= 30:
        # 即使超时但信心度足够高，也保存并继续
        cookie_mgr.save(driver)
        _report(f"登录成功 (超时但检测有效，信心度:{final_state['confidence']}%)", 'success')
        return True
    else:
        _report(f"登录等待超时 ({CRAWLER_AUTO_LOGIN_TIMEOUT}s) - 继续尝试爬取", 'warning')
        try:
            cookie_mgr.save(driver)  # 保存当前状态以备后续使用
        except Exception:
            pass
        return False  # 返回实际登录状态


# ============================================================================
# Mock 数据
# ============================================================================

MOCK_COMMENTS = [
    "这个真的不错，支持一下！", "终于有人说了，说出了我的心声",
    "太棒了吧，期待很久了", "感觉还行，但还有进步空间",
    "不太赞同，保持观望态度", "什么时候能实现啊，等的花都谢了",
    "点赞！👍", "这个确实需要改进，希望能看到变化",
    "作为一个老用户，感觉这次变化很大", "说得好，支持你",
    "有道理，但是实际情况可能更复杂", "加油，相信会越来越好的",
    "第一次看到这个话题，学习了", "已经转发，让更多人看到",
    "评论区里都是明白人", "不如预期，有点失望",
    "希望官方能重视起来", "太真实了哈哈",
    "这是今天看到最有价值的讨论", "坐等后续进展",
    "说实话有点意外", "观点独特，值得深思",
    "真的假的？不太敢相信", "每天都在关注这个话题",
    "确实是这样，感同身受", "要是能早点解决就好了",
    "支持正能量！", "理性讨论，不要带节奏",
    "客观来说，各有利弊", "这个角度我没想过，受教了",
]


def get_mock_comments(topic_keyword: str, count: int = 40) -> list[str]:
    """生成模拟评论数据"""
    import random as _random
    _random.seed(hash(topic_keyword) % (2 ** 31))

    topic_specific = [
        f"关于{topic_keyword}，我有不同的看法",
        f"{topic_keyword}这个话题最近很火啊",
        f"我觉得{topic_keyword}的发展前景很好",
        f"一直在关注{topic_keyword}，终于有人讨论了",
        f"{topic_keyword}的问题确实需要重视",
        f"对{topic_keyword}很感兴趣，蹲一个后续",
        f"{topic_keyword}这个方向是对的",
        f"说实话{topic_keyword}比我想象的好",
        f"关于{topic_keyword}，我补充一点",
        f"{topic_keyword}值得更多人了解",
    ]
    all_templates = MOCK_COMMENTS + topic_specific
    comments = [_random.choice(all_templates) for _ in range(count)]
    log.info("【Mock】生成 %d 条模拟评论 (话题: %s)", count, topic_keyword)
    return comments


def _generate_mock_posts(topic_keyword: str, comments: list[str]) -> list[dict]:
    """为 Mock 模式生成模拟的帖子↔评论结构化数据"""
    import random as _random
    _random.seed(hash(topic_keyword + "posts") % (2 ** 31))

    mock_usernames = ["体育观察员", "吃瓜群众", "娱乐记者小李",
                      "数码控小王", "财经老张", "文化评论员",
                      "美食探索者", "旅行达人", "影视发烧友", "历史爱好者"]

    mock_post_contents = [
        f"关于{topic_keyword}的最新动态，大家怎么看？这个话题最近讨论度很高。",
        f"分享一个关于{topic_keyword}的观点，个人觉得非常有道理。#{topic_keyword}#",
        f"说实话，{topic_keyword}这方面还需要更多的关注和讨论。",
        f"{topic_keyword}这个事件让我想起了之前的一些类似经历，感同身受。",
        f"整理了{topic_keyword}相关的最新资讯，希望对大家有帮助。",
    ]

    # Split comments across 3-5 mock posts
    n_posts = min(_random.randint(3, 5), max(1, len(comments) // 5))
    comments_per_post, remainder = divmod(len(comments), n_posts)

    posts = []
    idx = 0
    for p in range(n_posts):
        post_size = comments_per_post + (1 if p < remainder else 0)
        post_comments = comments[idx:idx + post_size]
        idx += post_size
        if not post_comments:
            break
        posts.append({
            'weibo_id': f"mock_{hash(topic_keyword) % 1000000}_{p}",
            'url': f'https://weibo.com/comment/hot/mock_{p}',
            'post_content': _random.choice(mock_post_contents),
            'username': _random.choice(mock_usernames),
            'post_time': datetime.now().strftime('%m月%d日 %H:%M'),
            'comment_count': len(post_comments),
            'comments': post_comments,
        })

    return posts


# ============================================================================
# 核心爬取逻辑
# ============================================================================

def _random_delay(min_s: float = None, max_s: float = None):
    if min_s is None: min_s = CRAWLER_REQUEST_DELAY_MIN
    if max_s is None: max_s = CRAWLER_REQUEST_DELAY_MAX
    delay = random.uniform(min_s, max_s)
    log.debug("  [T] 等待 %.1fs", delay)
    time.sleep(delay)


def _extract_mid(card) -> Optional[str]:
    """从微博卡片提取 mid"""
    try:
        mid = card.get_attribute('mid')
        if mid:
            log.debug("  [OK] 提取 mid: %s (from mid属性)", mid)
            return mid
    except Exception:
        pass

    # 回退: action-data 中的 id=
    try:
        ad = card.get_attribute('action-data') or ''
        m = re.search(r'mid=(\d+)', ad)
        if m:
            log.debug("  [OK] 提取 mid: %s (from action-data)", m.group(1))
            return m.group(1)
    except Exception:
        pass

    # 回退: /detail/ 链接
    try:
        links = card.find_elements(By.CSS_SELECTOR, 'a[href*="/detail/"]')
        for link in links:
            href = link.get_attribute('href') or ''
            m = re.search(r'/detail/(\d+)', href)
            if m:
                log.debug("  [OK] 提取 mid: %s (from /detail/ link)", m.group(1))
                return m.group(1)
    except Exception:
        pass

    return None


def _extract_post_url(card) -> Optional[str]:
    """
    从微博卡片提取帖子详情页 URL。

    策略:
      1. .from 区域的发布时间链接 (最可靠)
      2. 卡片中任意 weibo.com/{uid}/{code} 链接
    """
    # 策略1: .from 区域
    for by, selector, desc in POST_URL_SELECTORS_FROM_LINK:
        try:
            links = card.find_elements(by, selector)
            for link in links:
                href = link.get_attribute('href') or ''
                # 匹配 //weibo.com/{uid}/{code}?refer_flag=...
                if 'weibo.com' in href and 'refer_flag' in href:
                    if href.startswith('//'):
                        href = 'https:' + href
                    log.debug("  [OK] 提取帖子URL (from): %s", href[:80])
                    return href
        except Exception:
            continue

    # 策略2: 任意链接
    for by, selector, desc in POST_URL_SELECTORS_ANY:
        try:
            links = card.find_elements(by, selector)
            for link in links:
                href = link.get_attribute('href') or ''
                if 'weibo.com' in href and 'refer_flag' in href:
                    if href.startswith('//'):
                        href = 'https:' + href
                    log.debug("  [OK] 提取帖子URL (any): %s", href[:80])
                    return href
        except Exception:
            continue

    return None


def _construct_comment_url(mid: str, uid: str = None) -> str:
    """用 mid 构造微博评论页 URL"""
    # 方案1: 热门评论页 (通常不需要登录即可看)
    return f"https://weibo.com/comment/hot/{mid}"


def _extract_post_content(card) -> str:
    """从卡片提取帖子正文"""
    for by, selector, desc in POST_CONTENT_SELECTORS:
        try:
            el = card.find_element(by, selector)
            text = el.text.strip()
            if text and len(text) > 5:
                log.debug("  [OK] 帖子正文 (via %s): %s", desc, text[:60])
                return text
        except Exception:
            continue
    return ""


def _extract_username(card) -> str:
    """从卡片提取用户名"""
    for by, selector, desc in USERNAME_SELECTORS:
        try:
            el = card.find_element(by, selector)
            name = el.text.strip() or el.get_attribute('nick-name') or ''
            if name:
                return name
        except Exception:
            continue
    return ""


def _extract_comment_count(card) -> int:
    """从卡片提取评论数。返回 -1 表示无法解析"""
    for by, selector, desc in COMMENT_COUNT_SELECTORS:
        try:
            el = card.find_element(by, selector)
            text = el.text.strip()
            # 可能是 "评论" (=0条), "1", "6", "123" 等
            if text == '评论':
                return 0
            try:
                return int(text)
            except ValueError:
                # 可能含 "评论 1" 之类的，尝试提取数字
                nums = re.findall(r'\d+', text)
                if nums:
                    return int(nums[0])
                return 0
        except Exception:
            continue
    return -1  # 无法检测


def _extract_post_time(card) -> str:
    """从卡片提取发布时间"""
    for by, selector, desc in POST_TIME_SELECTORS:
        try:
            el = card.find_element(by, selector)
            text = el.text.strip()
            if text:
                return text
        except Exception:
            continue
    return ""


def search_topic(driver: webdriver.Chrome, topic_keyword: str,
                 page_num: int = None) -> list[dict]:
    """
    搜索话题并获取相关微博的完整卡片信息。

    Returns:
        [{
            'weibo_id': str,
            'url': str,
            'post_content': str,
            'username': str,
            'post_time': str,
            'comment_count': int,   # 卡片上显示的评论数, -1=无法检测
        }, ...]
    """
    if page_num is None:
        page_num = CRAWLER_PAGE_NUM

    encoded_topic = quote(topic_keyword)
    search_url = f"https://s.weibo.com/weibo?q=%23{encoded_topic}%23"

    log.info("=" * 50)
    log.info("【搜索阶段】关键词: #%s#", topic_keyword)
    log.info("  URL: %s", search_url)
    log.info("  搜索页数: %d", page_num)
    log.info("=" * 50)

    driver.get(search_url)
    _random_delay(3, 5)

    search_check = check_search_results(driver)
    log.info("【搜索阶段】卡片数: %d, 有结果: %s, 被拦截: %s, 空: %s",
             search_check['card_count'], search_check['has_results'],
             search_check['is_blocked'], search_check['is_empty'])

    if search_check['is_blocked']:
        log.error("【搜索阶段】[FAIL] 被反爬拦截")
        _save_debug_info(driver, f"blocked_{topic_keyword}")
        return []
    if search_check['is_empty']:
        log.warning("【搜索阶段】[FAIL] 搜索结果为空")
        _save_debug_info(driver, f"empty_{topic_keyword}")
        return []

    posts = []
    seen_ids = set()
    total_cards = 0
    skipped_no_content = 0
    skipped_no_comment_count = 0

    for page in range(page_num):
        log.info("--- 搜索页 %d/%d ---", page + 1, page_num)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        _random_delay(3, 5)

        weibo_cards = _safe_find_elements(driver, SEARCH_CARD_SELECTORS, timeout=8)
        card_count = len(weibo_cards)
        total_cards += card_count
        log.info("  【卡片数】第 %d 页检测到 %d 个微博卡片 (累计 %d)",
                 page + 1, card_count, total_cards)

        if card_count == 0:
            log.warning("  【卡片数】第 %d 页为0，停止搜索", page + 1)
            _save_debug_info(driver, f"no_cards_p{page + 1}")
            break

        for i in range(card_count):
            try:
                weibo_cards = _safe_find_elements(driver, SEARCH_CARD_SELECTORS, timeout=5)
                if i >= len(weibo_cards):
                    continue

                card = weibo_cards[i]
                mid = _extract_mid(card)
                if not mid or mid in seen_ids:
                    continue
                seen_ids.add(mid)

                # 提取帖子正文
                post_content = _extract_post_content(card)

                # 提取用户名
                username = _extract_username(card)

                # 提取评论数
                comment_count = _extract_comment_count(card)

                # 提取发布时间
                post_time = _extract_post_time(card)

                # 提取帖子URL
                post_url = _extract_post_url(card)
                if not post_url:
                    post_url = _construct_comment_url(mid)
                    log.debug("  [TOOL] 使用构造URL: %s", post_url[:80])

                log.info("  微博 %d: mid=%s | 用户=%s | 评论数=%d | %s",
                         i + 1, mid, username, comment_count, post_content[:40])

                posts.append({
                    'weibo_id': mid,
                    'url': post_url,
                    'post_content': post_content,
                    'username': username,
                    'post_time': post_time,
                    'comment_count': comment_count,
                })

            except Exception as e:
                log.warning("  处理第 %d 条微博异常: %s", i + 1, e)
                continue

        # 翻页
        if page < page_num - 1:
            next_btn = _safe_find_element(driver, NEXT_PAGE_SELECTORS, timeout=5)
            if next_btn:
                try:
                    driver.execute_script("arguments[0].click();", next_btn)
                    log.info("  >> 翻到第 %d 页", page + 2)
                    _random_delay(3, 5)
                    _safe_find_elements(driver, SEARCH_CARD_SELECTORS, timeout=8)
                except Exception as e:
                    log.warning("  翻页失败: %s", e)
                    break
            else:
                log.info("  无下一页按钮，搜索结束")
                break

    # 统计摘要
    with_comment = sum(1 for p in posts if p['comment_count'] > 0)
    without_comment = sum(1 for p in posts if p['comment_count'] == 0)
    unknown_comment = sum(1 for p in posts if p['comment_count'] < 0)

    log.info("【搜索完成】帖子: %d | 有评论: %d | 无评论: %d | 未知: %d",
             len(posts), with_comment, without_comment, unknown_comment)
    return posts


def get_weibo_comments(driver: webdriver.Chrome, post_info: dict,
                       scroll_times: int = None) -> list[str]:
    """
    加载单条微博的评论。若卡片标注评论数为0则直接跳过。

    Args:
        post_info: 来自 search_topic() 的帖子信息，含 url, weibo_id, comment_count
    """
    if scroll_times is None:
        scroll_times = CRAWLER_SCROLL_TIMES

    url = post_info['url']
    mid = post_info.get('weibo_id', '?')
    card_cc = post_info.get('comment_count', -1)

    log.info("--- 评论: mid=%s card评论数=%d", mid, card_cc)

    # ★ 卡片标注评论数为0 → 跳过，不浪费页面加载时间
    if card_cc == 0:
        log.info("  [SKIP] 卡片标注评论数=0，跳过此帖")
        return []

    try:
        driver.get(url)
    except TimeoutException:
        log.warning("  [FAIL] 页面加载超时: %s", url[:80])
        _save_debug_info(driver, f"timeout_{mid}")
        return []
    except WebDriverException as e:
        log.error("  [FAIL] 页面加载异常: %s", str(e)[:150])
        return []

    _random_delay(3, 5)

    area_check = check_comment_area(driver)
    if not area_check['is_loaded']:
        log.warning("  评论区首次未检测到，等待更久...")
        _random_delay(5, 8)
        area_check = check_comment_area(driver)
        if not area_check['is_loaded']:
            log.error("  [FAIL] 评论区未加载 (mid=%s)", mid)
            _save_debug_info(driver, f"no_comment_area_{mid}")
            report = analyze_page_structure(driver)
            for line in report.split('\n'):
                log.info("  [DIAG] %s", line)
            return []

    all_comments_raw = []
    last_height = driver.execute_script("return document.body.scrollHeight")
    no_change_count = 0

    for scroll_idx in range(scroll_times):
        log.debug("  滚动 %d/%d (当前 %d 条)", scroll_idx + 1, scroll_times, len(all_comments_raw))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.randint(8, 15))
        try:
            for el in _safe_find_elements(driver, COMMENT_LIST_SELECTORS, timeout=3):
                try:
                    text = _extract_comment_text(el)
                    if text and text not in all_comments_raw:
                        all_comments_raw.append(text)
                except Exception:
                    pass
        except Exception as e:
            log.warning("  提取评论异常 (scroll %d): %s", scroll_idx + 1, e)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            no_change_count += 1
            if no_change_count >= 2:
                break
        else:
            no_change_count = 0
        last_height = new_height

    all_comments = []
    for text in all_comments_raw:
        if ':' in text and len(text.split(':', 1)[0]) < 30:
            text = text.split(':', 1)[-1].strip()
        if text:
            all_comments.append(text)

    log.info("  【评论数】mid=%s → %d 条 (卡片标注=%d)", mid, len(all_comments), card_cc)

    if len(all_comments) == 0 and card_cc != 0:
        log.warning("  [WARN] 卡片标注有评论(%d)但提取为0！启动诊断...", card_cc)
        report = analyze_page_structure(driver)
        for line in report.split('\n'):
            log.info("  [DIAG] %s", line)
        _save_debug_info(driver, f"zero_extracted_{mid}")

    return all_comments


def _extract_comment_text(element) -> Optional[str]:
    """从评论元素提取文本"""
    for by, selector, desc in COMMENT_TEXT_SELECTORS:
        try:
            text_el = element.find_element(by, selector)
            text = text_el.text.strip()
            if text:
                return text
        except Exception:
            continue
    # 回退: 整个元素文本
    try:
        text = element.text.strip()
        if text:
            return text
    except Exception:
        pass
    return None


# ============================================================================
# 主入口
# ============================================================================

def crawl_topic(topic_keyword: str, page_num: int = None,
                scroll_times: int = None,
                use_mock: bool = False,
                status_callback=None) -> tuple[str, list[str]]:
    """
    爬取指定话题的微博评论 + 帖子内容。

    CSV 新增列: 帖子ID, 用户名, 帖子内容, 帖子评论数, 发布时间
    同时保存 structured.json 供 Agent API 消费。

    Returns:
        (csv_file_path, flat_comments_list) — 保持向后兼容
    """
    if page_num is None: page_num = CRAWLER_PAGE_NUM
    if scroll_times is None: scroll_times = CRAWLER_SCROLL_TIMES

    log.info("=" * 60)
    log.info("【开始爬取】话题: #%s# | 搜索页: %d | 滚动: %d | Mock: %s",
             topic_keyword, page_num, scroll_times, use_mock)
    log.info("=" * 60)

    _log_all_selectors()

    # Mock 模式
    if use_mock:
        log.info("【模式】Mock 演示模式")
        comments = get_mock_comments(topic_keyword, count=35)
        # 生成模拟的帖子↔评论结构化数据
        mock_posts = _generate_mock_posts(topic_keyword, comments)
        csv_path = _save_comments_to_csv(topic_keyword, comments, mock_posts)
        _save_structured_json(topic_keyword, mock_posts, csv_path)
        return str(csv_path), comments

    driver = None
    all_comments = []          # 扁平化评论列表（向后兼容）
    posts_with_comments = []   # [{post_info, comments: [...]}, ...]

    try:
        driver = _create_driver()

        log.info("【步骤1/3】登录微博")
        login_weibo(driver, status_callback=status_callback)

        log.info("【步骤2/3】搜索话题")
        posts = search_topic(driver, topic_keyword, page_num=page_num)
        log.info("【帖子数量】%d (含帖子内容)", len(posts))

        if not posts:
            log.warning("【结果】搜索到的帖子数为 0")
            if CRAWLER_MOCK_FALLBACK:
                log.info("→ 回退 Mock 数据")
                comments = get_mock_comments(topic_keyword, count=35)
                csv_path = _save_comments_to_csv(topic_keyword, comments, None)
                return str(csv_path), comments
            raise RuntimeError(f"未搜索到 '#{topic_keyword}#' 的相关微博")

        # 统计跳过情况
        skipped_zero = 0
        scraped = 0

        log.info("【步骤3/3】爬取评论 (%d 条微博)", len(posts))
        for i, post_info in enumerate(posts):
            cc = post_info.get('comment_count', -1)
            log.info(">> 微博 %d/%d (mid=%s, card评论数=%d, 用户=%s)",
                     i + 1, len(posts), post_info['weibo_id'], cc,
                     post_info.get('username', '?'))

            _random_delay(CRAWLER_REQUEST_DELAY_MIN, CRAWLER_REQUEST_DELAY_MAX)

            # get_weibo_comments 内部处理 card_cc==0 的快速跳过
            comments = get_weibo_comments(driver, post_info, scroll_times=scroll_times)

            if cc == 0:
                skipped_zero += 1

            posts_with_comments.append({
                **post_info,
                'comments': comments,
            })

            if comments:
                all_comments.extend(comments)
                scraped += 1
                log.info("  [OK] 本贴 %d 条 | 累计 %d 条", len(comments), len(all_comments))
            else:
                reason = "卡片标注0评论" if cc == 0 else "提取失败或确实无评论"
                log.info("  [SKIP] 本贴 0 条 (%s) | 累计 %d 条", reason, len(all_comments))

            _random_delay(5, 10)

        # 最终统计
        log.info("=" * 50)
        log.info("【爬取完成】帖子: %d | 成功爬取: %d | 跳过(0评论): %d | 总评论: %d",
                 len(posts), scraped, skipped_zero, len(all_comments))
        log.info("=" * 50)

        if len(all_comments) == 0:
            log.warning("【结果】总评论数为 0")
            _save_debug_info(driver, f"zero_total_{topic_keyword}")
            if CRAWLER_MOCK_FALLBACK:
                log.info("→ 回退 Mock 数据")
                comments = get_mock_comments(topic_keyword, count=35)
                csv_path = _save_comments_to_csv(topic_keyword, comments, None)
                return str(csv_path), comments
            raise RuntimeError(f"未能爬取到 '#{topic_keyword}#' 的任何评论")

        # 保存 CSV (含帖子内容) + JSON (结构化)
        csv_path = _save_comments_to_csv(topic_keyword, all_comments, posts_with_comments)
        _save_structured_json(topic_keyword, posts_with_comments, csv_path)

        return str(csv_path), all_comments

    except Exception as e:
        log.exception("【异常】%s: %s", type(e).__name__, str(e)[:200])
        if driver:
            try:
                _save_debug_info(driver, f"exception_{type(e).__name__}")
            except Exception:
                pass
        if CRAWLER_MOCK_FALLBACK and not use_mock:
            log.info("→ 异常后回退 Mock 数据")
            comments = get_mock_comments(topic_keyword, count=35)
            csv_path = _save_comments_to_csv(topic_keyword, comments, None)
            return str(csv_path), comments
        raise

    finally:
        if driver:
            try:
                driver.quit()
                log.info("ChromeDriver 已关闭")
            except Exception:
                pass


def _save_comments_to_csv(topic_keyword: str, comments: list[str],
                          posts_with_comments: list[dict] = None) -> Path:
    """
    保存 CSV。如果有帖子关联数据则写入完整列：
    评论内容, 帖子ID, 用户名, 帖子内容, 帖子评论数, 发布时间
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[\\/*?:"<>|]', '_', topic_keyword)
    filename = f"weibo_topic_{safe_topic}_{timestamp}.csv"
    csv_path = DATA_DIR / filename

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if posts_with_comments:
            # 新格式: 每行一条评论 + 帖子关联信息
            writer.writerow(['评论内容', '评论ID', '帖子ID', '用户名', '帖子内容',
                             '帖子评论数', '发布时间'])
            for post in posts_with_comments:
                post_id = post.get('weibo_id', '')
                username = post.get('username', '')
                content = post.get('post_content', '')
                cc = post.get('comment_count', '')
                pt = post.get('post_time', '')
                records = post.get('comment_records') or []
                if records and len(records) == len(post.get('comments', [])):
                    pairs = [(r.get('text', ''), r.get('comment_id', '')) for r in records]
                else:
                    pairs = [(c, '') for c in post.get('comments', [])]
                for c, comment_id in pairs:
                    writer.writerow([c, comment_id, post_id, username, content, cc, pt])
        else:
            writer.writerow(['评论内容'])
            for comment in comments:
                writer.writerow([comment])

    log.info("【CSV】已保存: %s (%d 条)", csv_path, len(comments))
    return csv_path


def _save_structured_json(topic_keyword: str, posts_with_comments: list[dict],
                          csv_path: Path, incremental_metadata: dict = None) -> Path:
    """保存结构化 JSON 供 Agent API 消费"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[\\/*?:"<>|]', '_', topic_keyword)
    json_path = DATA_DIR / f"weibo_topic_{safe_topic}_{timestamp}_structured.json"

    structured = {
        'topic': topic_keyword,
        'crawled_at': datetime.now().isoformat(),
        'csv_file': str(csv_path),
        'total_posts': len(posts_with_comments),
        'total_comments': sum(len(p['comments']) for p in posts_with_comments),
        'posts': []
    }
    if incremental_metadata:
        structured['incremental'] = incremental_metadata

    for post in posts_with_comments:
        report = post.get('fetch_report') or {}
        expected = report.get('expected_total', post.get('comment_count', -1))
        fetched = report.get('actual_fetched', len(post.get('comments', [])))
        coverage = report.get('coverage_pct')
        if coverage is None and isinstance(expected, int) and expected > 0:
            coverage = round(fetched / expected * 100, 1)
        structured['posts'].append({
            'weibo_id': post.get('weibo_id', ''),
            'username': post.get('username', ''),
            'content': post.get('post_content', ''),
            'post_time': post.get('post_time', ''),
            'comment_count_on_card': post.get('comment_count', -1),
            'fetched_comment_count': fetched,
            'coverage_pct': coverage,
            'fetch_method': post.get('fetch_method', 'unknown'),
            'stop_reason': report.get('stop_reason', 'unknown'),
            'incomplete': report.get('incomplete', bool(expected and fetched < expected)),
            'pages': report.get('pages'),
            'fetch_report': report,
            'url': post.get('url', ''),
            'comments': post.get('comments', []),
            'comment_records': post.get('comment_records', []),
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    log.info("【JSON】结构化数据已保存: %s (%d 帖, %d 评论)",
             json_path, structured['total_posts'], structured['total_comments'])
    return json_path


# ============================================================================
# 环境验证
# ============================================================================

def verify_crawler_env() -> dict:
    """验证爬虫环境"""
    result = {
        'webdriver_ok': False,
        'webdriver_manager_ok': False,
        'manual_chromedriver_ok': False,
        'network_ok': False,
        'details': '',
    }

    try:
        import webdriver_manager
        result['webdriver_manager_ok'] = True
    except ImportError:
        pass

    if find_chromedriver():
        result['manual_chromedriver_ok'] = True

    if find_chrome_binary() and (result['webdriver_manager_ok'] or result['manual_chromedriver_ok']):
        try:
            driver = _create_driver(headless=True)
            driver.get('https://www.baidu.com')
            result['network_ok'] = True
            driver.quit()
            result['webdriver_ok'] = True
            result['details'] = '[OK] 爬虫环境正常'
        except Exception as e:
            result['details'] = f'[FAIL] 浏览器启动失败: {e}'
    else:
        result['details'] = '[FAIL] 无可用 ChromeDriver。pip install webdriver-manager'

    return result


# ============================================================================
# V2: API-based 评论抓取 (requests 替代 Selenium scroll)
# ============================================================================

def crawl_topic_v2(topic_keyword: str, page_num: int = None,
                   scroll_times: int = None,
                   use_mock: bool = False,
                   status_callback=None, *, incremental: bool = False,
                   schedule_enabled: bool = False, interval_hours: int = 6,
                   task_id: int = None) -> tuple[str, list[str]]:
    """
    V2 爬虫 — Selenium 登录+搜索 + requests API 评论抓取。

    核心改进:
      - Selenium 仅用于登录和搜索（保留 UI 交互能力）
      - 评论抓取使用微博 PC/mobile API，~15x 提速
      - 自动分页遍历全部评论（1000+ 条覆盖 90%+）
      - API 失败时自动逐帖回退到 Selenium

    Returns:
        (csv_file_path, flat_comments_list) — 与 v1 相同接口
    """
    from src.weibo_api import WeiboAPIClient

    if page_num is None: page_num = CRAWLER_PAGE_NUM
    if scroll_times is None: scroll_times = CRAWLER_SCROLL_TIMES

    log.info("=" * 60)
    log.info("【V2 API模式】话题: #%s# | 搜索页: %d | Mock: %s",
             topic_keyword, page_num, use_mock)
    log.info("=" * 60)
    _log_all_selectors()

    if use_mock:
        log.info("【模式】Mock 演示模式")
        comments = get_mock_comments(topic_keyword, count=35)
        mock_posts = _generate_mock_posts(topic_keyword, comments)
        csv_path = _save_comments_to_csv(topic_keyword, comments, mock_posts)
        _save_structured_json(topic_keyword, mock_posts, csv_path)
        return str(csv_path), comments

    driver = None
    all_comments = []
    posts_with_comments = []
    series_id = None
    run_id = None
    incremental_metadata = None

    if incremental:
        from src.incremental import begin_run
        series_id, run_id = begin_run(
            topic_keyword, task_id=task_id, enabled=schedule_enabled,
            interval_hours=interval_hours,
        )

    try:
        driver = _create_driver()

        log.info("【步骤1/4】Selenium 登录")
        login_weibo(driver, status_callback=status_callback)

        if status_callback:
            status_callback("正在搜索微博帖子...")
        log.info("【步骤2/4】Selenium 搜索")
        posts = search_topic(driver, topic_keyword, page_num=page_num)
        log.info("【帖子数量】%d", len(posts))
        if status_callback:
            status_callback(f"找到 {len(posts)} 条相关微博")

        if not posts:
            if CRAWLER_MOCK_FALLBACK:
                comments = get_mock_comments(topic_keyword, count=35)
                mock_posts = _generate_mock_posts(topic_keyword, comments)
                csv_path = _save_comments_to_csv(topic_keyword, comments, mock_posts)
                _save_structured_json(topic_keyword, mock_posts, csv_path)
                return str(csv_path), comments
            raise RuntimeError(f"未搜索到 '#{topic_keyword}#' 的相关微博")

        log.info("【步骤3/4】初始化 API 客户端")
        api = WeiboAPIClient()
        cookie_count = api.load_cookies_from_selenium(driver)
        conn = api.test_connection()
        log.info("  Cookies: %d | PC API: %s | Mobile API: %s",
                 cookie_count, conn['pc_api'], conn['mobile_api'])

        # ★ API 端点自动发现 (用第一个有评论的帖子做探测)
        discovery = None
        if posts:
            for p in posts:
                if p.get('comment_count', -1) != 0:
                    try:
                        discovery = api.discover_endpoint(p['weibo_id'])
                        log.info("  API 发现: %s (total=%d)",
                                 discovery.get('endpoint_name', '?'),
                                 discovery.get('total_test', 0))
                    except Exception as e:
                        log.warning("  API 发现失败: %s", e)
                    break  # 只探测第一个有评论的帖子

        log.info("【步骤4/4】抓取评论 (%d 帖, API优先)", len(posts))
        if status_callback:
            status_callback(f"开始抓取 {len(posts)} 条微博的评论...")
        api_ok = 0
        se_ok = 0
        skipped_zero = 0
        api_consecutive_fails = 0  # ★ 连续 API 失败计数
        api_disabled = False       # ★ 3 次连续失败后禁用 API

        for i, post_info in enumerate(posts):
            mid = post_info['weibo_id']
            cc = post_info.get('comment_count', -1)

            log.info(">> 帖 %d/%d mid=%s card_cc=%d", i + 1, len(posts), mid, cc)
            _random_delay(CRAWLER_API_DELAY_MIN, CRAWLER_API_DELAY_MAX)

            if cc == 0:
                skipped_zero += 1
                log.info("  [SKIP] 卡片标注0评论")
                posts_with_comments.append({
                    **post_info, 'comments': [], 'fetch_method': 'skipped_zero',
                    'fetch_report': {
                        'expected_total': 0, 'actual_fetched': 0, 'coverage_pct': 100.0,
                        'pages': 0, 'stop_reason': 'card_zero', 'incomplete': False,
                    },
                })
                continue

            comments = []
            comment_records = []
            api_done = False
            fetch_method = 'selenium'
            fetch_report = None

            # ★ 尝试 API (连续失败 3 次则禁用)
            if CRAWLER_API_ENABLED and not api_disabled:
                try:
                    known_ids = set()
                    if incremental and series_id is not None:
                        from src.incremental import get_known_comment_ids
                        known_ids = get_known_comment_ids(series_id, mid)
                    comments, report = api.get_all_comments(
                        mid, known_comment_ids=known_ids,
                        stop_after_known_pages=2,
                    )
                    comment_records = report.pop('comment_records', [])
                    if comments or report.get('checkpoint_reached'):
                        fetch_method = 'api_mobile' if report.get('use_mobile') else 'api_pc'
                        fetch_report = report
                        api_ok += 1
                        api_done = True
                        api_consecutive_fails = 0
                        if report.get('checkpoint_reached'):
                            log.info("  [API] 命中增量断点 | 本轮新增 %d 条 | 扫描 %d 页",
                                     len(comments), report.get('pages', 0))
                        else:
                            log.info("  [API] %d 条 | 覆盖率 %.1f%% | %d 页 x %d 条",
                                     len(comments), report.get('coverage_pct', 0),
                                     report.get('pages', 0), report.get('count_per_page', 0))
                        if report.get('nested_replies', 0) > 0:
                            log.info("    含 %d 条二级回复", report.get('nested_replies', 0))
                        if report.get('truncated_by_pages'):
                            log.warning("    ⚠ 被 MAX_PAGES 截断")
                        elif report.get('incomplete'):
                            log.info("    平台可见窗口结束，未达到标称评论数")
                    else:
                        api_consecutive_fails += 1
                        log.warning("  [API] 返回空 (连续失败 %d/3)", api_consecutive_fails)
                except Exception as e:
                    api_consecutive_fails += 1
                    log.warning("  [API FAIL] %s (连续失败 %d/3)", str(e)[:80], api_consecutive_fails)

                if api_consecutive_fails >= 3:
                    api_disabled = True
                    log.warning("  ⚠ API 连续失败 3 次，后续帖子全部使用 Selenium")

            # ★ API 失败 → Selenium
            if not api_done:
                log.info("  -> Selenium fallback...")
                try:
                    comments = get_weibo_comments(driver, post_info, scroll_times=scroll_times)
                    se_ok += 1
                    log.info("  [SEL] %d 条", len(comments))
                except Exception as e:
                    log.error("  [SEL FAIL] %s", str(e)[:80])
                    comments = []

                expected = cc if isinstance(cc, int) and cc >= 0 else None
                coverage = round(len(comments) / expected * 100, 1) if expected else None
                fetch_report = {
                    'expected_total': expected, 'actual_fetched': len(comments),
                    'coverage_pct': coverage, 'pages': scroll_times,
                    'stop_reason': 'selenium_scroll_limit',
                    'incomplete': bool(expected and len(comments) < expected),
                }
                comment_records = []

            if comments:
                all_comments.extend(comments)
            posts_with_comments.append({
                **post_info, 'comments': comments, 'fetch_method': fetch_method,
                'fetch_report': fetch_report or {},
                'comment_records': comment_records if api_done else [],
            })

            if status_callback and (i + 1) % 3 == 0:
                status_callback(f"已处理 {i + 1}/{len(posts)} 帖, 累计 {len(all_comments)} 条评论")

        if incremental:
            from src.incremental import merge_snapshot
            posts_with_comments, incremental_metadata = merge_snapshot(
                series_id, run_id, posts_with_comments
            )
            all_comments = [
                comment for post in posts_with_comments for comment in post.get('comments', [])
            ]
            log.info("【增量】本轮新增 %d 条，累计唯一评论 %d 条",
                     incremental_metadata['new_comments'],
                     incremental_metadata['total_unique_comments'])

        csv_path = _save_comments_to_csv(topic_keyword, all_comments, posts_with_comments)
        _save_structured_json(
            topic_keyword, posts_with_comments, csv_path,
            incremental_metadata=incremental_metadata,
        )

        log.info("=" * 50)
        log.info("【V2完成】帖:%d 跳过0评论:%d 总评论:%d | API:%d帖 SEL:%d帖",
                 len(posts), skipped_zero, len(all_comments), api_ok, se_ok)
        log.info("=" * 50)

        if len(all_comments) == 0:
            if CRAWLER_MOCK_FALLBACK:
                comments = get_mock_comments(topic_keyword, count=35)
                mock_posts = _generate_mock_posts(topic_keyword, comments)
                csv_path = _save_comments_to_csv(topic_keyword, comments, mock_posts)
                _save_structured_json(topic_keyword, mock_posts, csv_path)
                return str(csv_path), comments
            raise RuntimeError(f"未能爬取到 '#{topic_keyword}#' 的任何评论")

        return str(csv_path), all_comments

    except Exception as e:
        if incremental and run_id is not None:
            try:
                from src.incremental import fail_run
                fail_run(run_id, str(e))
            except Exception:
                pass
        log.error("【异常】%s: %s", type(e).__name__, str(e)[:200])
        if driver:
            try:
                _save_debug_info(driver, f"exception_{type(e).__name__}")
            except Exception:
                pass
        if CRAWLER_MOCK_FALLBACK and not use_mock:
            comments = get_mock_comments(topic_keyword, count=35)
            mock_posts = _generate_mock_posts(topic_keyword, comments)
            csv_path = _save_comments_to_csv(topic_keyword, comments, mock_posts)
            return str(csv_path), comments
        raise

    finally:
        if driver:
            try:
                driver.quit()
                log.info("ChromeDriver 已关闭")
            except Exception:
                pass
