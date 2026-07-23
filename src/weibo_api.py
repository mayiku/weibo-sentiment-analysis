"""
微博 API 客户端 v2.0 — 高性能评论抓取

优化:
  1. 自动探测 buildComments 最大 count 参数
  2. 每页评论数提升至允许上限 (20→60)
  3. API 请求间隔降至 0.1~0.5s (自适应退避)
  4. 覆盖率追踪 (预期/实际/百分比)
  5. 递归提取二级回复 (nested replies)
  6. MAX_PAGES 截断检测与警告
  7. 不改变架构前提下的最大性能

架构:
  Selenium (登录 + 搜索) → 提取 cookies
                          ↓
                   WeiboAPIClient (requests.Session)
                          ↓
                   PC API 端点 → 自动分页获取全部评论
                          ↓ (失败时)
                   Mobile API 回退

关键端点:
  - PC Comments:  /ajax/statuses/buildComments
  - PC Post:      /ajax/statuses/show
  - PC LongText:  /ajax/statuses/longtext
  - Mobile Comments: /comments/hotflow
  - Mobile Post:     /statuses/show
"""
import json
import hashlib
import time
import random
import re
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    CRAWLER_API_COMMENTS_PER_PAGE,
    CRAWLER_API_MAX_PAGES,
    CRAWLER_API_DELAY_MIN,
    CRAWLER_API_DELAY_MAX,
    CRAWLER_API_RETRIES,
    CRAWLER_API_TIMEOUT,
)
from src.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# WeiboAPIClient v2.0
# ============================================================================

class WeiboAPIClient:
    """
    微博 API 客户端 v2.0 — 优化版。

    用法:
        client = WeiboAPIClient()
        client.load_cookies_from_selenium(driver)

        # 自动探测最优 count + 分页抓取全部评论 + 递归提取二级回复
        comments, report = client.get_all_comments('5322786954281260')
        # report = {
        #   'expected_total': 456, 'actual_fetched': 432,
        #   'coverage_pct': 94.7, 'pages': 8, 'count_per_page': 60,
        #   'truncated': False, 'nested_replies': 12, ...
        # }
    """

    # ── API 端点 ──
    PC_BASE = 'https://weibo.com'
    PC_COMMENTS_URL = f'{PC_BASE}/ajax/statuses/buildComments'
    PC_POST_URL = f'{PC_BASE}/ajax/statuses/show'
    PC_LONGTEXT_URL = f'{PC_BASE}/ajax/statuses/longtext'

    MOBILE_BASE = 'https://m.weibo.cn'
    MOBILE_COMMENTS_URL = f'{MOBILE_BASE}/comments/hotflow'
    MOBILE_POST_URL = f'{MOBILE_BASE}/statuses/show'

    # ★ 所有已知的评论 API 端点 (按优先级排列) — 用于自动发现
    KNOWN_COMMENT_ENDPOINTS = [
        # (标签, URL模板, 参数构建函数, 响应解析函数)
        ('PC buildComments', PC_COMMENTS_URL,
         lambda mid, max_id, count, flow: {
             'flow': flow, 'is_reload': 1, 'id': str(mid),
             'is_show_bulletin': 2, 'is_mix': 0,
             'count': count, 'max_id': max_id, 'fetch_level': 0,
         },
         lambda data: {
             'data': data.get('data', []), 'max_id': data.get('max_id', 0),
             'total_number': data.get('total_number', 0),
         }),

        ('PC commentsByHot', f'{PC_BASE}/ajax/statuses/commentsByHot',
         lambda mid, max_id, count, flow: {
             'id': str(mid), 'max_id': max_id, 'count': count,
         },
         lambda data: {
             'data': data.get('data', []), 'max_id': data.get('max_id', 0),
             'total_number': data.get('total_number', 0),
         }),

        ('PC commentsByTime', f'{PC_BASE}/ajax/statuses/commentsByTime',
         lambda mid, max_id, count, flow: {
             'id': str(mid), 'max_id': max_id, 'count': count,
         },
         lambda data: {
             'data': data.get('data', []), 'max_id': data.get('max_id', 0),
             'total_number': data.get('total_number', 0),
         }),

        ('Mobile hotflow', MOBILE_COMMENTS_URL,
         lambda mid, max_id, count, flow: {
             'id': str(mid), 'mid': str(mid), 'max_id': max_id,
             'max_id_type': 0, 'count': count,
         },
         lambda data: {
             'data': data.get('data', {}).get('data', []),
             'max_id': data.get('data', {}).get('max_id', 0),
             'total_number': data.get('data', {}).get('total_number', 0) or data.get('data', {}).get('max', 0),
         }),

    ]

    # ★ count 探测序列（从高到低试探）
    _COUNT_PROBE_SEQUENCE = [60, 50, 40, 30, 20]

    # ★ 自适应延迟范围
    _DELAY_MIN = 0.08
    _DELAY_MAX = 5.0
    _DELAY_INITIAL = 0.15

    def __init__(self):
        self.session = requests.Session()
        self._stats = {
            'api_calls': 0, 'comments_fetched': 0, 'errors': 0,
            'nested_replies': 0, 'rate_limits': 0,
        }
        self._adaptive_delay = self._DELAY_INITIAL
        self._detected_max_count = None   # 缓存探测结果
        self._setup_session()

    def _setup_session(self):
        """配置 requests session"""
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
        })

        retry_strategy = Retry(
            total=CRAWLER_API_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['GET'],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

    # ======================================================================
    # Cookie 管理 (不变)
    # ======================================================================

    def load_cookies_from_selenium(self, driver) -> int:
        """从 Selenium WebDriver 提取 cookies 到 requests session"""
        try:
            selenium_cookies = driver.get_cookies()
        except Exception as e:
            log.error("无法从 Selenium 获取 cookies: %s", e)
            return 0

        loaded = 0
        for cookie in selenium_cookies:
            name = cookie.get('name', '')
            value = cookie.get('value', '')
            if not name or not value:
                continue
            self.session.cookies.set(name, value,
                                     domain=cookie.get('domain', '.weibo.com'),
                                     path=cookie.get('path', '/'))
            loaded += 1

        xsrf = self.session.cookies.get('XSRF-TOKEN', '')
        if xsrf:
            self.session.headers['X-XSRF-TOKEN'] = xsrf
            log.info("  XSRF-TOKEN: %s...", xsrf[:20])
        log.info("Cookies: %d/%d 条加载完成", loaded, len(selenium_cookies))
        return loaded

    def load_cookies_from_file(self, cookie_data: list[dict]) -> int:
        """从持久化 cookie 数据加载"""
        loaded = 0
        for cookie in cookie_data:
            name = cookie.get('name', '')
            value = cookie.get('value', '')
            if not name or not value:
                continue
            self.session.cookies.set(name, value,
                                     domain=cookie.get('domain', '.weibo.com'),
                                     path=cookie.get('path', '/'))
            loaded += 1
        xsrf = self.session.cookies.get('XSRF-TOKEN', '')
        if xsrf:
            self.session.headers['X-XSRF-TOKEN'] = xsrf
        log.info("Cookies 从文件加载: %d 条", loaded)
        return loaded

    def export_cookies(self) -> list[dict]:
        """导出 cookies 用于持久化存储"""
        cookies = []
        for cookie in self.session.cookies:
            cookies.append({
                'name': cookie.name, 'value': cookie.value,
                'domain': cookie.domain, 'path': cookie.path,
            })
        return cookies

    # ======================================================================
    # 请求核心
    # ======================================================================

    def _api_request(self, url: str, params: dict = None,
                     headers: dict = None, label: str = 'API') -> Optional[requests.Response]:
        """带统计的 API 请求"""
        self._stats['api_calls'] += 1
        try:
            resp = self.session.get(url, params=params, headers=headers,
                                    timeout=CRAWLER_API_TIMEOUT)
            return resp
        except requests.RequestException as e:
            self._stats['errors'] += 1
            log.warning("  [%s] 请求异常: %s", label, str(e)[:100])
            return None

    def _parse_json(self, resp: requests.Response, label: str = 'API') -> Optional[dict]:
        """安全解析 JSON"""
        if not resp:
            return None
        try:
            data = resp.json()
            if data.get('ok') == 1:
                return data
            errno = data.get('errno', '?')
            errmsg = data.get('msg', data.get('errmsg', '?'))
            log.warning("  [%s] errno=%s msg=%s", label, errno, errmsg)
            return data
        except json.JSONDecodeError:
            log.warning("  [%s] JSON 解析失败: %s", label, resp.text[:150])
            return None
        except Exception as e:
            log.warning("  [%s] 解析异常: %s", label, e)
            return None

    def _adaptive_wait(self, rate_limited: bool = False):
        """自适应等待：成功时缩短，429时加长"""
        if rate_limited:
            self._adaptive_delay = min(self._DELAY_MAX, self._adaptive_delay * 2.0)
            self._stats['rate_limits'] += 1
            log.warning("  [RATE] 限流! 延迟增至 %.2fs", self._adaptive_delay)
        else:
            # 缓慢向初始延迟收敛
            self._adaptive_delay = max(
                self._DELAY_MIN,
                self._adaptive_delay * 0.95 + self._DELAY_INITIAL * 0.05
            )
        jitter = random.uniform(-0.05, 0.05)
        delay = max(0.05, self._adaptive_delay + jitter)
        time.sleep(delay)

    # ======================================================================
    # ★ 新增: count 上限自动探测
    # ======================================================================

    def _detect_max_count(self, mid: str) -> int:
        """
        探测 buildComments API 允许的最大 count 参数。

        策略: 从 60 → 50 → 40 → 30 → 20 依次请求 count 页，
              取实际返回条数 ≥ 请求数的最大值。
              缓存结果避免重复探测。

        Returns: 最大可用 count (最小 20)
        """
        if self._detected_max_count is not None:
            return self._detected_max_count

        log.info("  [PROBE] 探测 max_count (mid=%s)...", mid)

        best_count = 20  # fallback minimum
        for test_count in self._COUNT_PROBE_SEQUENCE:
            data = self.get_comments_page(mid, max_id=0, count=test_count)
            if not data:
                log.info("    count=%d → API 无响应", test_count)
                self._adaptive_wait(rate_limited=False)
                continue

            returned = len(data.get('data', []))
            total = data.get('total_number', 0)

            log.info("    count=%d → 返回 %d 条 (总 %d)", test_count, returned, total)

            if returned >= test_count:
                # API 返回了我们请求的数量 → 这个 count 可行
                best_count = test_count
                log.info("    ✓ count=%d 可用 (返回 %d 条)", test_count, returned)
                break  # 从高到低，第一个可行的就是最大值
            elif returned >= min(test_count, total):
                # 帖子评论总数不够，无法判断
                # 取 min(test_count, total) 是因为total可能<test_count
                pass

            self._adaptive_wait(rate_limited=False)

        self._detected_max_count = best_count
        log.info("  [PROBE] 最优 count = %d (每页)", best_count)
        return best_count

    # ======================================================================
    # ★ 新增: 递归提取评论 + 二级回复
    # ======================================================================

    def _extract_comment_text(self, comment: dict) -> str:
        """从单条评论提取纯文本"""
        text = comment.get('text_raw', '') or comment.get('text', '')
        # 1. 去除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 2. 去除 "回复 @xxx:" 或 "@xxx:" 回复前缀
        text = re.sub(r'(?:回复\s*)?@\S+\s*:', '', text).strip()
        # 3. 清理多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_all_texts(self, comment: dict, depth: int = 0) -> list[str]:
        """
        递归提取评论 + 所有嵌套二级回复 (replies)。

        Weibo buildComments API 中每条评论的 `comments` 字段
        包含该评论下的回复列表，可能有多层嵌套。
        """
        texts = []

        # 主评论文本
        main_text = self._extract_comment_text(comment)
        if main_text:
            if depth > 0:
                self._stats['nested_replies'] += 1
            texts.append(main_text)

        # 递归提取嵌套回复
        replies = comment.get('comments', [])
        if isinstance(replies, list):
            for reply in replies:
                texts.extend(self._extract_all_texts(reply, depth + 1))

        return texts

    def _extract_all_records(self, comment: dict, depth: int = 0,
                             parent_id: str = '') -> list[dict]:
        """提取带稳定标识的评论记录，供增量采集按 ID 去重。"""
        records = []
        text = self._extract_comment_text(comment)
        raw_id = str(comment.get('id') or comment.get('idstr') or '')
        if text:
            if depth > 0:
                self._stats['nested_replies'] += 1
            fallback = hashlib.sha256(
                f"{parent_id}|{depth}|{text}".encode('utf-8')
            ).hexdigest()[:24]
            records.append({
                'comment_id': raw_id or f'hash:{fallback}',
                'text': text,
                'parent_id': parent_id,
                'depth': depth,
            })
        current_parent = raw_id or parent_id
        replies = comment.get('comments', [])
        if isinstance(replies, list):
            for reply in replies:
                records.extend(self._extract_all_records(reply, depth + 1, current_parent))
        return records

    # ======================================================================
    # 帖子信息 API (不变)
    # ======================================================================

    def get_post_detail(self, mid: str) -> Optional[dict]:
        """获取帖子详情 (PC → Mobile fallback)"""
        params = {'id': str(mid)}
        headers = {'Referer': 'https://weibo.com/', 'X-Requested-With': 'XMLHttpRequest'}
        resp = self._api_request(self.PC_POST_URL, params=params,
                                 headers=headers, label='PostDetail')
        data = self._parse_json(resp, 'PostDetail')
        if data and data.get('ok') == 1:
            return data

        log.debug("  PC Post API 失败，尝试 Mobile...")
        params = {'id': str(mid)}
        headers = {'Referer': f'https://m.weibo.cn/detail/{mid}',
                   'X-Requested-With': 'XMLHttpRequest'}
        resp = self._api_request(self.MOBILE_POST_URL, params=params,
                                 headers=headers, label='PostDetail(M)')
        data = self._parse_json(resp, 'PostDetail(M)')
        if data and data.get('ok') == 1:
            return data
        return None

    def get_comment_count(self, mid: str) -> int:
        """获取帖子评论总数（不抓取内容）"""
        post = self.get_post_detail(mid)
        if post:
            count = post.get('comments_count', 0)
            if count:
                return int(count)
            inner = post.get('data', {}) or post.get('status', {})
            count = inner.get('comments_count', 0)
            if count:
                return int(count)

        data = self.get_comments_page(mid, max_id=0, count=1)
        if data and data.get('total_number'):
            return int(data['total_number'])
        return -1

    # ======================================================================
    # 评论页 API
    # ======================================================================

    def get_comments_page(self, mid: str, max_id=0, count: int = None,
                          flow: int = 0) -> Optional[dict]:
        """获取一页评论 (PC API)"""
        if count is None:
            count = CRAWLER_API_COMMENTS_PER_PAGE

        params = {
            'flow': flow, 'is_reload': 1, 'id': str(mid),
            'is_show_bulletin': 2, 'is_mix': 0,
            'count': count, 'max_id': max_id, 'fetch_level': 0,
        }
        headers = {
            'Referer': 'https://weibo.com/',
            'X-Requested-With': 'XMLHttpRequest',
        }
        resp = self._api_request(self.PC_COMMENTS_URL, params=params,
                                 headers=headers, label=f'C(max_id={max_id})')
        data = self._parse_json(resp, 'C')
        if data and data.get('ok') == 1 and data.get('data') is not None:
            return data
        return None

    def get_comments_page_mobile(self, mid: str, max_id=0) -> Optional[dict]:
        """获取一页评论 (Mobile API fallback)"""
        params = {'id': str(mid), 'mid': str(mid), 'max_id': max_id, 'max_id_type': 0}
        headers = {
            'Referer': f'https://m.weibo.cn/detail/{mid}',
            'X-Requested-With': 'XMLHttpRequest',
        }
        resp = self._api_request(self.MOBILE_COMMENTS_URL, params=params,
                                 headers=headers, label=f'CM(max_id={max_id})')
        data = self._parse_json(resp, 'CM')
        if data and data.get('ok') == 1:
            inner = data.get('data', {})
            return {
                'ok': 1,
                'data': inner.get('data', []) or [],
                'max_id': inner.get('max_id', 0),
                'total_number': inner.get('total_number', 0) or inner.get('max', 0),
            }
        return None

    # ======================================================================
    # ★ API 端点自动发现
    # ======================================================================

    def discover_endpoint(self, mid: str) -> dict:
        """
        逐个测试所有已知评论 API 端点，找到当前可用的。

        Returns:
            {
                'endpoint_name': str,       # 端点名称
                'endpoint_url': str,        # 端点 URL
                'working': bool,
                'params_template': dict,    # 参数模板
                'response_parser': callable,# 响应解析器
                'total_test': int,          # 首页返回的 total_number
                'comments_retrieved': int,  # 首页返回的评论数
                'error': str or None,
            }
        """
        log.info("=" * 50)
        log.info("【API 发现】测试 %d 个已知端点...", len(self.KNOWN_COMMENT_ENDPOINTS))
        log.info("=" * 50)

        best = None

        for name, url_template, params_fn, parser_fn in self.KNOWN_COMMENT_ENDPOINTS:
            full_url = url_template
            params = params_fn(mid, 0, 5, 0)  # 测试：请求 5 条

            headers = {
                'Referer': 'https://weibo.com/',
                'X-Requested-With': 'XMLHttpRequest',
            }
            if 'm.weibo.cn' in full_url:
                headers['Referer'] = f'https://m.weibo.cn/detail/{mid}'

            log.info("  测试: %s", name)
            try:
                resp = self.session.get(full_url, params=params, headers=headers,
                                        timeout=CRAWLER_API_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('ok') == 1:
                        parsed = parser_fn(data)
                        comments = parsed.get('data', [])
                        total = parsed.get('total_number', 0)
                        log.info("    ✓ 可用! 返回 %d 条, 总数 %d",
                                 len(comments) if isinstance(comments, list) else 0, total)

                        if not best or (total > 0 and (not best.get('total_test') or total > best['total_test'])):
                            best = {
                                'endpoint_name': name,
                                'endpoint_url': full_url,
                                'working': True,
                                'params_fn': params_fn,
                                'parser_fn': parser_fn,
                                'total_test': total,
                                'comments_retrieved': len(comments) if isinstance(comments, list) else 0,
                                'error': None,
                            }
                    else:
                        log.info("    ✗ ok=%s errno=%s", data.get('ok'), data.get('errno'))
                else:
                    log.info("    ✗ HTTP %d", resp.status_code)
            except Exception as e:
                log.info("    ✗ 异常: %s", str(e)[:80])

        if best:
            log.info("【API 发现】推荐: %s (total=%d)", best['endpoint_name'], best['total_test'])
        else:
            log.warning("【API 发现】所有端点均不可用！需要登录或微博 API 已变更")
            best = {
                'endpoint_name': 'NONE',
                'endpoint_url': '',
                'working': False,
                'params_fn': None,
                'parser_fn': None,
                'total_test': 0,
                'comments_retrieved': 0,
                'error': '所有已知端点均失败',
            }

        return best

    # ======================================================================
    # ★ 优化后的自动分页 (核心)
    # ======================================================================

    def get_all_comments(self, mid: str, max_pages: int = None,
                         flow: int = 0, use_mobile: bool = False,
                         known_comment_ids: set[str] = None,
                         stop_after_known_pages: int = 2) -> tuple[list[str], dict]:
        """
        自动翻页抓取全部评论 (含二级回复)。

        优化:
          - 自动探测最优 count 参数 (20→60 条/页)
          - 自适应延迟 (0.08~5.0s, 按限流动态调整)
          - 递归提取嵌套 replies
          - 覆盖率追踪 + 截断检测

        Returns:
            (comments_text_list, report_dict)
            report = {
                'expected_total': API报告的评论总数,
                'actual_fetched': 实际抓到的纯文本数 (含replies),
                'coverage_pct': 覆盖率百分比,
                'pages': 请求页数,
                'count_per_page': 使用的每页条数,
                'truncated': 是否未覆盖标称评论总数,
                'truncated_by_pages': True/False,
                'nested_replies': 提取到的二级回复数,
                'api_calls': 本次 API 调用次数,
                'errors': 错误次数,
                'rate_limits': 触发限流次数,
            }
        """
        if max_pages is None:
            max_pages = CRAWLER_API_MAX_PAGES

        # ★ Step 0: 探测最优 count
        optimal_count = self._detect_max_count(mid)
        log.info("【API v2】mid=%s count=%d/页 max_pages=%s",
                 mid, optimal_count, max_pages or '∞')

        # 页容量由当前 API 会话决定；复用首次探测结果，避免逐帖重复请求。

        all_comments = []
        max_id = 0
        page = 0
        expected_total = 0
        seen_ids = set()
        seen_record_ids = set()
        comment_records = []
        known_comment_ids = set(known_comment_ids or [])
        consecutive_known_pages = 0
        known_records_seen = 0
        consecutive_errors = 0
        nested_count_before = self._stats['nested_replies']
        errors_before = self._stats['errors']
        calls_before = self._stats['api_calls']
        stop_reason = 'unknown'

        while True:
            # ── 截断检查 ──
            if max_pages and page >= max_pages:
                log.warning("  [TRUNCATED] 达到 MAX_PAGES=%d (共 %d 页)",
                            max_pages, page)
                stop_reason = 'max_pages'
                break

            # ── 连续错误检查 ──
            if consecutive_errors >= 4:
                log.warning("  -> 连续 %d 次错误，停止", consecutive_errors)
                stop_reason = 'consecutive_errors'
                break

            # ── 获取一页 ──
            if use_mobile:
                data = self.get_comments_page_mobile(mid, max_id=max_id)
            else:
                data = self.get_comments_page(mid, max_id=max_id,
                                              count=optimal_count, flow=flow)

            if not data and not use_mobile:
                log.info("  PC API 失败 → Mobile API")
                data = self.get_comments_page_mobile(mid, max_id=max_id)
                if data:
                    use_mobile = True

            if not data:
                consecutive_errors += 1
                self._adaptive_wait(rate_limited=False)
                continue

            # ── 限流检测 ──
            if data.get('ok') != 1 and data.get('errno') in ['100001', '100005']:
                log.warning("  API 限流信号 (errno=%s)", data.get('errno'))
                self._adaptive_wait(rate_limited=True)
                continue

            consecutive_errors = 0
            batch = data.get('data', [])

            if not batch:
                log.info("  -> 第 %d 页无数据，到末尾", page + 1)
                stop_reason = 'empty_page'
                break

            # ★ 递归提取评论 + 二级回复
            new_in_batch = 0
            known_in_batch = 0
            for comment in batch:
                cid = str(comment.get('id', ''))
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)

                # 递归：主评论 + 嵌套 replies
                records = self._extract_all_records(comment)
                for record in records:
                    record_id = record['comment_id']
                    if record_id in seen_record_ids:
                        continue
                    seen_record_ids.add(record_id)
                    if record_id in known_comment_ids:
                        known_records_seen += 1
                        known_in_batch += 1
                        continue
                    if record['text']:
                        comment_records.append(record)
                        all_comments.append(record['text'])
                        new_in_batch += 1
                        self._stats['comments_fetched'] += 1

            old_max_id = max_id
            max_id = data.get('max_id', 0)
            expected_total = data.get('total_number', expected_total)
            page += 1

            if known_comment_ids and new_in_batch == 0 and known_in_batch:
                consecutive_known_pages += 1
            else:
                consecutive_known_pages = 0

            # ★ 覆盖率日志
            actual = len(all_comments)
            pct = f"{actual/expected_total*100:.1f}%" if expected_total > 0 else "?"
            nested = self._stats['nested_replies'] - nested_count_before
            log.info("  P%d | +%d (+%d replies) | %d/%d (%s) | delay=%.2fs",
                     page, new_in_batch, nested if nested > 0 else 0,
                     actual, expected_total, pct, self._adaptive_delay)

            # ── 停止条件 ──
            if max_id == 0:
                log.info("  -> max_id=0，到末尾")
                stop_reason = 'max_id_zero'
                break
            if max_id == old_max_id:
                log.info("  -> max_id 不变 (%s)，到末尾", max_id)
                stop_reason = 'cursor_stalled'
                break
            if new_in_batch == 0 and page > 1:
                if known_comment_ids:
                    if consecutive_known_pages >= stop_after_known_pages:
                        log.info("  -> 已连续 %d 页命中历史断点，停止增量扫描",
                                 consecutive_known_pages)
                        stop_reason = 'checkpoint_reached'
                        break
                else:
                    log.info("  -> 连续无新评论，停止")
                    stop_reason = 'no_new_comments'
                    break

            # ★ 自适应速率控制 (0.08~0.5s 正常范围)
            self._adaptive_wait(rate_limited=False)

        # ── 构建报告 ──
        actual = len(all_comments)
        coverage = actual / expected_total * 100.0 if expected_total > 0 else 0.0
        truncated_by_pages = stop_reason == 'max_pages'
        incremental_checkpoint = bool(
            known_comment_ids and known_records_seen > 0 and actual == 0
            and stop_reason in {
                'empty_page', 'max_id_zero', 'cursor_stalled',
                'no_new_comments', 'checkpoint_reached',
            }
        )
        incomplete = bool(
            expected_total > 0 and actual < expected_total
            and not incremental_checkpoint
        )
        visible_window_limited = bool(
            incomplete and stop_reason in {
                'empty_page', 'max_id_zero', 'cursor_stalled', 'no_new_comments'
            }
        )
        nested_total = self._stats['nested_replies'] - nested_count_before
        api_calls_used = self._stats['api_calls'] - calls_before
        errors_used = self._stats['errors'] - errors_before

        report = {
            'expected_total': expected_total,
            'actual_fetched': actual,
            'coverage_pct': round(coverage, 1),
            'pages': page,
            'count_per_page': optimal_count,
            'truncated': incomplete,
            'truncated_by_pages': truncated_by_pages,
            'incomplete': incomplete,
            'stop_reason': stop_reason,
            'visible_window_limited': visible_window_limited,
            'nested_replies': nested_total,
            'api_calls': api_calls_used,
            'errors': errors_used,
            'rate_limits_hit': self._stats['rate_limits'],
            'use_mobile': use_mobile,
            'adaptive_delay': round(self._adaptive_delay, 3),
            'last_cursor': max_id,
            'comment_records': comment_records,
            'known_records_seen': known_records_seen,
            'checkpoint_reached': incremental_checkpoint,
            'request_succeeded': bool(page > 0 and errors_used == 0),
            'incremental_scan': bool(known_comment_ids),
            'new_fetched': actual,
        }

        log.info("=" * 55)
        log.info("【API v2 完成】mid=%s", mid)
        log.info("  预期评论: %d | 实际抓取: %d | 覆盖率: %.1f%%",
                 expected_total, actual, coverage)
        log.info("  二级回复: %d | 页数: %d (%d 条/页) | 截断: %s",
                 nested_total, page, optimal_count, incomplete)
        log.info("  API调用: %d | 错误: %d | 限流: %d | 终延迟: %.2fs",
                 api_calls_used, errors_used,
                 self._stats['rate_limits'], self._adaptive_delay)

        # ★ 截断警告
        if truncated_by_pages:
            gap = expected_total - actual
            log.warning("  ⚠ TRUNCATED: 因 MAX_PAGES 限制丢失约 %d 条 (%.1f%%)",
                        gap, 100.0 - coverage)

        # ★ 覆盖率警告
        if stop_reason == 'checkpoint_reached':
            log.info("  增量断点命中: 已见 %d 条历史评论，本轮新增 %d 条",
                     known_records_seen, actual)
        elif visible_window_limited:
            log.warning("  ⚠ API 可见窗口提前结束 (%s)，标称评论数不等于可访问评论数",
                        stop_reason)
        elif coverage < 80.0 and not truncated_by_pages and not known_comment_ids:
            log.warning("  ⚠ 覆盖率偏低 (%.1f%%)，可能存在删除评论、权限限制或过滤",
                        coverage)

        log.info("=" * 55)
        return all_comments, report

    # ======================================================================
    # 诊断
    # ======================================================================

    def test_connection(self) -> dict:
        """测试 API 连通性"""
        result = {
            'pc_api': False, 'mobile_api': False,
            'cookies_valid': False, 'details': '',
        }

        try:
            resp = self.session.get(
                'https://weibo.com/ajax/statuses/show?id=0', timeout=10,
                headers={'Referer': 'https://weibo.com/',
                         'X-Requested-With': 'XMLHttpRequest'},
            )
            result['pc_api'] = resp.status_code in [200, 201, 301, 302, 404]
            result['details'] = f"PC API: HTTP {resp.status_code}"
        except Exception as e:
            result['details'] = f"PC API 不可达: {e}"

        try:
            resp = self.session.get(
                'https://m.weibo.cn/statuses/show?id=0', timeout=10,
                headers={'Referer': 'https://m.weibo.cn/',
                         'X-Requested-With': 'XMLHttpRequest'},
            )
            result['mobile_api'] = resp.status_code in [200, 201, 301, 302, 404]
        except Exception:
            pass

        result['cookies_valid'] = bool(
            self.session.cookies.get('SUB', '') or
            self.session.cookies.get('XSRF-TOKEN', '')
        )

        log.info("【连通性】PC=%s Mobile=%s Cookie=%s",
                 result['pc_api'], result['mobile_api'], result['cookies_valid'])
        return result

    def get_stats(self) -> dict:
        """获取全局 API 统计"""
        return dict(self._stats)

    def reset_stats(self):
        """重置统计"""
        self._stats = {
            'api_calls': 0, 'comments_fetched': 0, 'errors': 0,
            'nested_replies': 0, 'rate_limits': 0,
        }
        self._adaptive_delay = self._DELAY_INITIAL
        self._detected_max_count = None
