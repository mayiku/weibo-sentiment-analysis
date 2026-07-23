"""
微博评论 API 网络分析器 — 从浏览器 Network 请求中识别评论 API。

用法:
    python -m src.weibo_analyzer <post_url_or_mid>

或用作库:
    from src.weibo_analyzer import capture_comment_api, analyze_api
    report = capture_comment_api(driver, post_url)
"""
import json
import time
import pprint
from urllib.parse import urlparse, parse_qs, urlencode
from typing import Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import requests

from config import (
    CHROMEDRIVER_PATH, CRAWLER_HEADLESS,
    CRAWLER_PAGE_LOAD_TIMEOUT,
)
from src.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# 评论相关的 API 模式匹配
# ============================================================================

# 已知的评论 API URL 关键词（用于过滤 Network 请求）
COMMENT_API_PATTERNS = [
    'buildComments',      # PC 微博
    'comments/hotflow',    # 移动微博
    'statuses/show',      # 帖子详情（含评论数）
    'statuses/comments',  # 可能的变体
    'commentsByHot',      # 热门评论
    'commentsByTime',     # 时间序评论
    'getIndex',           # 移动端容器
    'statuses/extend',    # 扩展信息
    'detail/',            # 移动端详情
    'comments',           # 通用评论关键词
]

# 需要排除的非评论 URL 模式
EXCLUDE_PATTERNS = [
    'beacon', 'suda', 'analytics', 'log', 'track',
    'advertisement', 'unread', 'notification',
    '.css', '.js', '.png', '.jpg', '.gif', '.woff',
    'getCommentList',  # 有时候是其他模块
]


class WeiboNetworkAnalyzer:
    """
    微博网络分析器 — 启动 Chrome + Performance Log，捕获评论 API 请求。

    工作流程:
      1. 启动带 Performance Logging 的 Chrome
      2. 导航到帖子页
      3. 等待评论加载
      4. 收集所有 XHR/Fetch 请求
      5. 过滤出评论相关的 API 调用
      6. 逐一测试每个 API
    """

    def __init__(self, driver: webdriver.Chrome = None):
        self.driver = driver
        self._own_driver = False
        self.captured_requests = []  # 原始日志条目
        self.comment_apis = []       # 过滤后的评论 API

    def create_driver(self, headless: bool = None) -> webdriver.Chrome:
        """创建带 Performance Logging 的 Chrome"""
        if headless is None:
            headless = CRAWLER_HEADLESS

        options = Options()
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

        # ★ 关键: 启用 Performance Logging
        options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

        # 使用 webdriver-manager
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service as ChromeService
            service = ChromeService(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        except Exception:
            service = Service(str(CHROMEDRIVER_PATH))
            self.driver = webdriver.Chrome(service=service, options=options)

        self.driver.set_page_load_timeout(CRAWLER_PAGE_LOAD_TIMEOUT)
        self.driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        self._own_driver = True
        log.info("Chrome + Performance Log 已启动")
        return self.driver

    def load_cookies_from_file(self, cookie_file):
        """加载持久化 cookie (避免重新登录)"""
        import pickle
        if not cookie_file.exists():
            log.warning("Cookie 文件不存在: %s", cookie_file)
            return False
        try:
            with open(cookie_file, 'rb') as f:
                cookies = pickle.load(f)
            self.driver.get('https://weibo.com')
            time.sleep(2)
            for c in cookies:
                try:
                    c.pop('sameSite', None)
                    c.pop('httpOnly', None)
                    self.driver.add_cookie(c)
                except Exception:
                    pass
            self.driver.refresh()
            log.info("Cookie 已加载")
            return True
        except Exception as e:
            log.error("加载 Cookie 失败: %s", e)
            return False

    def capture(self, post_url: str, wait_seconds: float = 8.0) -> list[dict]:
        """
        导航到帖子页并捕获所有 Network 请求。

        Args:
            post_url: 微博帖子 URL (如 https://weibo.com/xxx/xxx 或 mid)
            wait_seconds: 等待评论加载的秒数

        Returns:
            comment API 请求列表
        """
        # 清除旧日志
        try:
            self.driver.get_log('performance')
        except Exception:
            pass
        self.captured_requests = []
        self.comment_apis = []

        # 如果传的是 mid，构造 URL
        if post_url.isdigit():
            mid = post_url
            # 先尝试 PC
            post_url = f'https://weibo.com/comment/hot/{mid}'
        else:
            # 提取 mid
            import re
            m = re.search(r'/(\d{10,})', post_url)
            mid = m.group(1) if m else None

        log.info("=" * 60)
        log.info("【网络分析】打开帖子: %s", post_url)
        log.info("=" * 60)

        # 导航
        self.driver.get(post_url)
        time.sleep(3)

        # 尝试滚动加载评论
        for _ in range(3):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(wait_seconds / 3)

        # 收集 Performance Log
        try:
            raw_logs = self.driver.get_log('performance')
            self.captured_requests = raw_logs
            log.info("收集到 %d 条 Performance Log", len(raw_logs))
        except Exception as e:
            log.error("无法获取 Performance Log: %s", e)
            return []

        # 解析并过滤
        self._parse_and_filter()

        # 打印报告
        self.print_report()

        return self.comment_apis

    def _parse_and_filter(self):
        """解析 Performance Log，过滤出评论 API 请求"""
        seen_urls = set()

        for entry in self.captured_requests:
            try:
                msg = json.loads(entry['message'])
                method = msg.get('message', {}).get('method', '')
                params = msg.get('message', {}).get('params', {})

                request_info = None

                if method == 'Network.requestWillBeSent':
                    req = params.get('request', {})
                    url = req.get('url', '')
                    req_method = req.get('method', 'GET')
                    headers = req.get('headers', {})
                    post_data = req.get('postData', '')
                    request_id = params.get('requestId', '')

                    request_info = {
                        'url': url,
                        'method': req_method,
                        'request_headers': {k: v for k, v in headers.items()
                                           if k.lower() in ['content-type', 'x-requested-with',
                                                            'x-xsrf-token', 'referer']},
                        'post_data': post_data,
                        'request_id': request_id,
                        'parsed_url': urlparse(url),
                        'query_params': parse_qs(urlparse(url).query),
                        'response': None,
                    }

                elif method == 'Network.responseReceived':
                    resp = params.get('response', {})
                    url = resp.get('url', '')
                    status = resp.get('status', 0)
                    resp_headers = resp.get('headers', {})

                    request_info = {
                        'url': url,
                        'response_status': status,
                        'response_headers': {k: v for k, v in resp_headers.items()
                                            if k.lower() in ['content-type', 'content-length']},
                        'parsed_url': urlparse(url),
                    }

                if not request_info:
                    continue

                url = request_info['url']
                parsed = request_info['parsed_url']
                path = parsed.path

                # 检查是否是评论相关
                is_comment_api = any(p in path for p in COMMENT_API_PATTERNS)
                is_excluded = any(p in url for p in EXCLUDE_PATTERNS)

                if is_comment_api and not is_excluded:
                    url_key = f"{request_info.get('method', '')}:{url.split('?')[0]}"
                    if url_key not in seen_urls:
                        seen_urls.add(url_key)
                        self.comment_apis.append(request_info)

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        log.info("过滤得到 %d 个评论 API 请求", len(self.comment_apis))

    def print_report(self):
        """打印分析报告"""
        print("\n" + "=" * 70)
        print("  微博评论 API 网络分析报告")
        print("=" * 70)

        if not self.comment_apis:
            print("\n  ⚠ 未捕获到评论 API 请求！")
            print("  可能原因: 1) 未登录 2) 页面未加载 3) API 域名不在匹配列表")
            print("\n  尝试手动分析...")
            self._print_all_filtered_requests()
            return

        for i, api in enumerate(self.comment_apis):
            print(f"\n  ── API #{i+1} ──")
            parsed = api.get('parsed_url', '')
            print(f"  请求URL: {api.get('url', '?')[:150]}")
            if parsed:
                print(f"  端点:     {parsed.path}")
            print(f"  方法:     {api.get('method', 'GET')}")

            qp = api.get('query_params', {})
            if qp:
                print(f"  请求参数:")
                for k, v in qp.items():
                    print(f"    {k}: {v[0] if len(v) == 1 else v}")

            if api.get('response_status'):
                print(f"  响应状态: {api['response_status']}")

            if api.get('response_headers'):
                ct = api['response_headers'].get('content-type', '?')
                print(f"  Content-Type: {ct}")

        print("\n" + "=" * 70)

    def _print_all_filtered_requests(self):
        """打印所有捕获到的 XHR 请求（调试用）"""
        xhr_urls = set()
        for entry in self.captured_requests:
            try:
                msg = json.loads(entry['message'])
                method = msg.get('message', {}).get('method', '')
                if method == 'Network.requestWillBeSent':
                    url = msg['message']['params']['request']['url']
                    if url not in xhr_urls and any(
                        t in url.lower() for t in ['ajax', 'api', 'comment', 'status', 'flow']
                    ):
                        xhr_urls.add(url)
                        print(f"  {url[:200]}")
            except Exception:
                pass

    def test_api_directly(self, cookies: dict = None) -> list[dict]:
        """
        用 requests 直接测试每个捕获到的 API 端点。

        Returns:
            [{url, params, response_sample, pagination_info, status}, ...]
        """
        results = []
        session = requests.Session()

        if cookies:
            for name, value in cookies.items():
                session.cookies.set(name, value,
                                    domain='.weibo.com', path='/')

        for api in self.comment_apis:
            url = api['url']
            params = {k: v[0] for k, v in api.get('query_params', {}).items()}

            log.info("直接测试 API: %s", api.get('parsed_url', {}).get('path', url[:80]))

            result = {
                'url': url.split('?')[0],
                'path': api.get('parsed_url', {}).get('path', ''),
                'params_used': params,
                'working': False,
                'response_sample': None,
                'pagination_info': None,
                'total_number': None,
                'error': None,
            }

            try:
                resp = session.get(url, params=params, timeout=15,
                                   headers={
                                       'Referer': 'https://weibo.com/',
                                       'X-Requested-With': 'XMLHttpRequest',
                                   })
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except Exception:
                        result['error'] = f"非JSON响应: {resp.text[:100]}"
                        results.append(result)
                        continue

                    result['response_sample'] = data

                    if data.get('ok') == 1:
                        result['working'] = True

                        # 分析分页机制
                        pagination = {}
                        if 'max_id' in data:
                            pagination['cursor_field'] = 'max_id'
                            pagination['current_max_id'] = data['max_id']
                        if 'since_id' in data:
                            pagination['since_id'] = data.get('since_id', 0)
                        if 'total_number' in data:
                            pagination['total'] = data['total_number']
                            result['total_number'] = data['total_number']

                        # Check for nested data
                        inner = data.get('data', {})
                        if isinstance(inner, dict):
                            if 'max_id' in inner:
                                pagination['cursor_field'] = 'data.max_id'
                            if 'total_number' in inner:
                                pagination['total'] = inner['total_number']
                                result['total_number'] = inner['total_number']
                            inner_array = inner.get('data', [])
                            if isinstance(inner_array, list):
                                pagination['comments_per_page'] = len(inner_array)
                        elif isinstance(inner, list):
                            pagination['comments_per_page'] = len(inner)

                        result['pagination_info'] = pagination
                    else:
                        result['error'] = f"ok={data.get('ok')}, errno={data.get('errno')}, msg={data.get('msg')}"
                else:
                    result['error'] = f"HTTP {resp.status_code}: {resp.text[:100]}"

            except Exception as e:
                result['error'] = str(e)[:200]

            results.append(result)

        return results

    def close(self):
        """关闭浏览器"""
        if self._own_driver and self.driver:
            self.driver.quit()
            self.driver = None


# ============================================================================
# 独立分析脚本
# ============================================================================

def analyze_weibo_api(post_url_or_mid: str, cookie_file=None) -> dict:
    """
    一键分析微博评论 API。

    Args:
        post_url_or_mid: 帖子 URL 或 mid
        cookie_file: Cookie 缓存文件（避免手动登录）

    Returns:
        {
            'captured_apis': [...],      # 从 Network 捕获的 API 列表
            'direct_test_results': [...], # 直接测试结果
            'working_endpoints': [...],   # 正常工作的端点
            'recommendation': str,        # 推荐使用哪个端点
        }
    """
    from config import COOKIE_FILE

    analyzer = WeiboNetworkAnalyzer()
    analyzer.create_driver(headless=False)

    if cookie_file is None:
        cookie_file = COOKIE_FILE
    analyzer.load_cookies_from_file(cookie_file)

    captured = analyzer.capture(post_url_or_mid)

    # 直接测试
    test_results = analyzer.test_api_directly()

    # 汇总
    working = [r for r in test_results if r['working']]
    not_working = [r for r in test_results if not r['working']]

    report = {
        'captured_apis': captured,
        'direct_test_results': test_results,
        'working_endpoints': [r['path'] for r in working],
        'failed_endpoints': [{'path': r['path'], 'error': r['error']} for r in not_working],
        'recommendation': '',
    }

    if working:
        report['recommendation'] = f"推荐使用: {working[0]['path']}"
        pag = working[0].get('pagination_info', {})
        if pag:
            report['recommendation'] += f" (分页: {pag.get('cursor_field', '?')}, 每页 {pag.get('comments_per_page', '?')} 条)"
    else:
        report['recommendation'] = "所有端点均失败，需要登录或更新 API"

    analyzer.close()

    # 打印最终报告
    print("\n" + "=" * 70)
    print("  最终分析结论")
    print("=" * 70)
    print(f"  捕获 API: {len(captured)} 个")
    print(f"  可用端点: {len(working)} 个")
    for w in working:
        print(f"    ✓ {w['path']}")
        print(f"      分页: {w.get('pagination_info', {})}")
        print(f"      总数: {w.get('total_number', '?')}")
    print(f"  不可用: {len(not_working)} 个")
    for f in not_working:
        print(f"    ✗ {f['path']}: {f['error']}")
    print(f"  推荐: {report['recommendation']}")
    print("=" * 70)

    return report


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='微博评论 API 分析器')
    parser.add_argument('post', nargs='?',
                        help='帖子 mid 或完整 URL')
    parser.add_argument('--cookie', '-c', default=None,
                        help='Cookie 文件路径')
    parser.add_argument('--mid', '-m', default=None,
                        help='直接指定 mid')
    args = parser.parse_args()

    if args.mid:
        target = args.mid
    elif args.post:
        target = args.post
    else:
        # 默认: 用 buildComments 已知端点直接测试
        print("未指定帖子URL/mid，使用已知 API 端点进行连通性测试\n")
        target = None

    if target:
        cookie_file = None
        if args.cookie:
            from pathlib import Path
            cookie_file = Path(args.cookie)

        report = analyze_weibo_api(target, cookie_file)
        print("\n完整报告:", json.dumps(report, ensure_ascii=False, indent=2,
                           default=str)[:3000])
    else:
        # 直接测试已知端点
        print("连通性测试（需要先登录获取 cookie）...")
        print("请先启动 crawler login，然后提供有效的 cookie 文件")
