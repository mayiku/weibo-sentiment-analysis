"""
DeepSeek API 客户端 — 统一的 LLM 调用封装

设计:
  - LLMClient 抽象基类 (预留 OpenAI/Claude 接口)
  - DeepSeekClient 具体实现
  - 自动重试 + 超时控制 + 详细日志

用法:
    client = DeepSeekClient()
    response = client.chat(prompt, system_prompt=..., temperature=0.3)
"""

import time
import json
from typing import Optional, Generator

import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_TIMEOUT,
    DEEPSEEK_MAX_RETRIES,
)
from src.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# LLMClient — 抽象基类 (预留扩展)
# ============================================================================

class LLMClient:
    """
    通用 LLM 客户端基类。

    子类化以支持不同后端:
      class OpenAIClient(LLMClient): ...
      class ClaudeClient(LLMClient): ...
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.last_error = None
        self.last_usage = {}
        self.last_latency_seconds = None

    def chat(self, prompt: str, system_prompt: str = None,
             temperature: float = None, max_tokens: int = None) -> Optional[str]:
        raise NotImplementedError

    def chat_stream(self, prompt: str, system_prompt: str = None,
                    temperature: float = None, max_tokens: int = None) -> Generator[str, None, None]:
        raise NotImplementedError


# ============================================================================
# DeepSeekClient
# ============================================================================

class DeepSeekClient(LLMClient):
    """
    DeepSeek Chat API 客户端 (OpenAI 兼容格式)。

    特性:
      - 自动重试 (指数退避)
      - 超时控制
      - 详细日志
      - 流式输出支持
    """

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        super().__init__(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
            model=model or DEEPSEEK_MODEL,
        )
        self._chat_url = f"{self.base_url}/chat/completions"
        self.session = requests.Session()
        self.last_usage = {}
        self.last_latency_seconds = None

    def chat(self, prompt: str, system_prompt: str = None,
             temperature: float = None, max_tokens: int = None) -> Optional[str]:
        """
        调用 DeepSeek Chat API — 非流式。

        Args:
            prompt: 用户消息
            system_prompt: 系统提示词
            temperature: 温度 (0.0-2.0)
            max_tokens: 最大输出 token 数

        Returns:
            AI 响应文本; 失败时返回 None
        """
        if temperature is None:
            temperature = DEEPSEEK_TEMPERATURE
        if max_tokens is None:
            max_tokens = DEEPSEEK_MAX_TOKENS

        if not self.api_key:
            self.last_error = "DeepSeek API Key 未配置"
            log.error("%s！请在 .env 中设置 DEEPSEEK_API_KEY", self.last_error)
            return None

        self.last_error = None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        log.info("【DeepSeek】调用 API: model=%s, prompt_len=%d, temp=%.1f",
                 self.model, len(prompt), temperature)

        last_error = None
        call_started = time.perf_counter()
        for attempt in range(DEEPSEEK_MAX_RETRIES):
            try:
                resp = self.session.post(
                    self._chat_url,
                    json=payload,
                    headers=headers,
                    timeout=DEEPSEEK_TIMEOUT,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    self.last_usage = dict(usage)
                    self.last_latency_seconds = round(
                        time.perf_counter() - call_started, 3
                    )
                    log.info("【DeepSeek】成功: tokens_in=%s, tokens_out=%s, total=%s, elapsed=%.2fs",
                             usage.get("prompt_tokens", "?"),
                             usage.get("completion_tokens", "?"),
                             usage.get("total_tokens", "?"),
                             self.last_latency_seconds)
                    return content

                elif resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    self.last_error = "DeepSeek 请求频率受限 (HTTP 429)"
                    if attempt < DEEPSEEK_MAX_RETRIES - 1:
                        log.warning("【DeepSeek】速率限制 (429)，等待 %ds...", wait)
                        time.sleep(wait)

                elif resp.status_code == 401:
                    self.last_error = "DeepSeek API Key 无效 (HTTP 401)"
                    log.error("【DeepSeek】API Key 无效 (401)。请检查 .env 中的 DEEPSEEK_API_KEY")
                    return None

                elif resp.status_code in (402, 403):
                    try:
                        error_data = resp.json()
                        message = error_data.get("error", {}).get("message") or error_data.get("message")
                    except ValueError:
                        message = None
                    self.last_error = message or f"DeepSeek 账户余额或权限异常 (HTTP {resp.status_code})"
                    log.error("【DeepSeek】%s", self.last_error)
                    return None

                elif resp.status_code >= 500:
                    wait = (attempt + 1) * 3
                    self.last_error = f"DeepSeek 服务暂时不可用 (HTTP {resp.status_code})"
                    if attempt < DEEPSEEK_MAX_RETRIES - 1:
                        log.warning("【DeepSeek】服务器错误 (%d)，%d/%d 重试，等待 %ds...",
                                    resp.status_code, attempt + 1, DEEPSEEK_MAX_RETRIES, wait)
                        time.sleep(wait)

                else:
                    log.error("【DeepSeek】HTTP %d: %s", resp.status_code, resp.text[:300])
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    self.last_error = last_error
                    break

            except requests.Timeout:
                self.last_error = f"DeepSeek 请求超时 ({DEEPSEEK_TIMEOUT}s)"
                log.warning("【DeepSeek】超时 (%ds)，%d/%d 重试...",
                            DEEPSEEK_TIMEOUT, attempt + 1, DEEPSEEK_MAX_RETRIES)
                if attempt < DEEPSEEK_MAX_RETRIES - 1:
                    time.sleep(3)

            except requests.RequestException as e:
                log.warning("【DeepSeek】网络异常: %s，%d/%d 重试...",
                            e, attempt + 1, DEEPSEEK_MAX_RETRIES)
                if attempt < DEEPSEEK_MAX_RETRIES - 1:
                    time.sleep(3)
                last_error = str(e)
                self.last_error = f"DeepSeek 网络异常: {e}"

        log.error(
            "【DeepSeek】所有重试均失败。最后错误: %s",
            self.last_error or last_error or "未知",
        )
        return None

    def chat_stream(self, prompt: str, system_prompt: str = None,
                    temperature: float = None, max_tokens: int = None) -> Generator[str, None, None]:
        """
        调用 DeepSeek Chat API — 流式输出。

        Yields:
            增量文本块
        """
        if temperature is None:
            temperature = DEEPSEEK_TEMPERATURE
        if max_tokens is None:
            max_tokens = DEEPSEEK_MAX_TOKENS

        if not self.api_key:
            log.error("DeepSeek API Key 未配置！")
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            resp = self.session.post(
                self._chat_url,
                json=payload,
                headers=headers,
                timeout=DEEPSEEK_TIMEOUT,
                stream=True,
            )

            if resp.status_code != 200:
                log.error("【DeepSeek】流式请求失败: HTTP %d %s",
                          resp.status_code, resp.text[:200])
                return

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        except requests.Timeout:
            log.error("【DeepSeek】流式请求超时")
        except Exception as e:
            log.error("【DeepSeek】流式请求异常: %s", e)


# ============================================================================
# 工厂函数 — 便于后续扩展
# ============================================================================

def create_client(provider: str = "deepseek", **kwargs) -> LLMClient:
    """
    LLM 客户端工厂 — 按 provider 创建对应的客户端。

    Args:
        provider: "deepseek" | "openai" | "claude" (预留)
        **kwargs: 传递给具体客户端的参数

    Returns:
        LLMClient 实例
    """
    providers = {
        "deepseek": DeepSeekClient,
    }

    cls = providers.get(provider.lower())
    if not cls:
        raise ValueError(f"不支持的 LLM provider: {provider}。可用: {list(providers.keys())}")

    return cls(**kwargs)
