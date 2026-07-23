"""
AI Provider 抽象层 — 支持多个 LLM 提供商 (SiliconFlow, DeepSeek, 等)
"""
import os
from abc import ABC, abstractmethod
from typing import Optional, Generator
import time
import requests
import json

import sys
from pathlib import Path

# 确保可以导入 config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    AI_PROVIDER,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)

from src.logger import get_logger

log = get_logger(__name__)


# 新增 SiliconFlow 配置 (从环境变量读取)
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3.1-Terminus")
SILICONFLOW_MAX_TOKENS = int(os.getenv("SILICONFLOW_MAX_TOKENS", "4096"))
SILICONFLOW_TEMPERATURE = float(os.getenv("SILICONFLOW_TEMPERATURE", "0.3"))
SILICONFLOW_TIMEOUT = int(os.getenv("SILICONFLOW_TIMEOUT", "120"))
SILICONFLOW_MAX_RETRIES = int(os.getenv("SILICONFLOW_MAX_RETRIES", "3"))

class AIProvider(ABC):
    """AI 提供商抽象基类"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.last_error = None

    @abstractmethod
    def chat(self, prompt: str, system_prompt: str = None,
             temperature: float = None, max_tokens: int = None) -> Optional[str]:
        """
        调用 AI API — 非流式

        Returns:
            AI 响应文本; 失败时返回 None
        """
        pass

    @abstractmethod
    def chat_stream(self, prompt: str, system_prompt: str = None,
                    temperature: float = None, max_tokens: int = None) -> Generator[str, None, None]:
        """
        调用 AI API — 流式输出

        Yields:
            增量文本块
        """
        pass


class SiliconFlowClient(AIProvider):
    """
    SiliconFlow AI 客户端 (OpenAI 兼容接口)

    支持模型示例:
      - deepseek-ai/DeepSeek-V3.1-Terminus
      - moonshotai/Kimi-K2-Instruct-0905
      - Qwen/Qwen3-Coder-480B-A35B-Instruct
      - zai-org/GLM-4.5
    """

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        super().__init__(
            api_key=api_key or SILICONFLOW_API_KEY,
            base_url=base_url or SILICONFLOW_BASE_URL,
            model=model or SILICONFLOW_MODEL,
        )
        self._chat_url = f"{self.base_url}/chat/completions"

    def chat(self, prompt: str, system_prompt: str = None,
             temperature: float = None, max_tokens: int = None) -> Optional[str]:
        """调用 SiliconFlow Chat API — 非流式"""
        if temperature is None:
            temperature = SILICONFLOW_TEMPERATURE
        if max_tokens is None:
            max_tokens = SILICONFLOW_MAX_TOKENS

        if not self.api_key:
            self.last_error = "SiliconFlow API Key 未配置"
            log.error("%s！请在 .env 中设置 SILICONFLOW_API_KEY", self.last_error)
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

        log.info("【SiliconFlow】调用 API: model=%s, prompt_len=%d, temp=%.1f",
                 self.model, len(prompt), temperature)

        last_error = None
        for attempt in range(SILICONFLOW_MAX_RETRIES):
            try:
                resp = requests.post(
                    self._chat_url,
                    json=payload,
                    headers=headers,
                    timeout=SILICONFLOW_TIMEOUT,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    log.info("【SiliconFlow】成功: tokens_in=%s, tokens_out=%s, total=%s",
                             usage.get("prompt_tokens", "?"),
                             usage.get("completion_tokens", "?"),
                             usage.get("total_tokens", "?"))
                    return content

                elif resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    log.warning("【SiliconFlow】速率限制 (429)，等待 %ds...", wait)
                    time.sleep(wait)

                elif resp.status_code == 402:
                    self.last_error = "SiliconFlow 账户余额不足 (HTTP 402)"
                    log.error("【SiliconFlow】%s。请充值或更换 API Key", self.last_error)
                    return None

                elif resp.status_code == 403:
                    try:
                        error_data = resp.json()
                    except ValueError:
                        error_data = {}
                    message = error_data.get("message", "访问被拒绝")
                    code = error_data.get("code")
                    if code == 30001 or "balance" in message.lower():
                        self.last_error = "SiliconFlow 账户余额不足 (HTTP 403, code 30001)"
                    else:
                        self.last_error = f"SiliconFlow 访问被拒绝 (HTTP 403): {message}"
                    log.error("【SiliconFlow】%s", self.last_error)
                    return None

                elif resp.status_code == 401:
                    self.last_error = "SiliconFlow API Key 无效 (HTTP 401)"
                    log.error("【SiliconFlow】API Key 无效 (401)。请检查 .env 中的 SILICONFLOW_API_KEY")
                    return None

                elif resp.status_code >= 500:
                    wait = (attempt + 1) * 3
                    log.warning("【SiliconFlow】服务器错误 (%d)，%d/%d 重试，等待 %ds...",
                                resp.status_code, attempt + 1, SILICONFLOW_MAX_RETRIES, wait)
                    time.sleep(wait)

                else:
                    log.error("【SiliconFlow】HTTP %d: %s", resp.status_code, resp.text[:300])
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    self.last_error = last_error
                    break

            except requests.Timeout:
                log.warning("【SiliconFlow】超时 (%ds)，%d/%d 重试...",
                            SILICONFLOW_TIMEOUT, attempt + 1, SILICONFLOW_MAX_RETRIES)
                time.sleep(3)

            except requests.RequestException as e:
                log.warning("【SiliconFlow】网络异常: %s，%d/%d 重试...",
                            e, attempt + 1, SILICONFLOW_MAX_RETRIES)
                time.sleep(3)
                last_error = str(e)
                self.last_error = f"SiliconFlow 网络异常: {e}"

        log.error("【SiliconFlow】所有重试均失败。最后错误: %s", last_error or "未知")
        return None

    def chat_stream(self, prompt: str, system_prompt: str = None,
                    temperature: float = None, max_tokens: int = None) -> Generator[str, None, None]:
        """调用 SiliconFlow Chat API — 流式输出"""
        if temperature is None:
            temperature = SILICONFLOW_TEMPERATURE
        if max_tokens is None:
            max_tokens = SILICONFLOW_MAX_TOKENS

        if not self.api_key:
            log.error("SiliconFlow API Key 未配置！")
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
            resp = requests.post(
                self._chat_url,
                json=payload,
                headers=headers,
                timeout=SILICONFLOW_TIMEOUT,
                stream=True,
            )

            if resp.status_code != 200:
                log.error("【SiliconFlow】流式请求失败: HTTP %d %s",
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
            log.error("【SiliconFlow】流式请求超时")
        except Exception as e:
            log.error("【SiliconFlow】流式请求异常: %s", e)


class DeepSeekClientWrapper(AIProvider):
    """DeepSeek 客户端包装器 (保持向后兼容)"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        from src.ai_agent.deepseek_client import DeepSeekClient
        self._client = DeepSeekClient(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
            model=model or DEEPSEEK_MODEL,
        )
        super().__init__(
            api_key=self._client.api_key,
            base_url=self._client.base_url,
            model=self._client.model,
        )

    def chat(self, prompt: str, system_prompt: str = None,
             temperature: float = None, max_tokens: int = None) -> Optional[str]:
        result = self._client.chat(prompt, system_prompt, temperature, max_tokens)
        self.last_error = self._client.last_error
        return result

    def chat_stream(self, prompt: str, system_prompt: str = None,
                    temperature: float = None, max_tokens: int = None) -> Generator[str, None, None]:
        return self._client.chat_stream(prompt, system_prompt, temperature, max_tokens)


def create_ai_client(provider: str = None) -> AIProvider:
    """
    创建 AI 客户端工厂函数

    Args:
        provider: "siliconflow" 或 "deepseek"，默认使用 AI_PROVIDER 环境变量

    Returns:
        AIProvider 实例
    """
    provider = provider or AI_PROVIDER

    providers = {
        "siliconflow": SiliconFlowClient,
        "deepseek": DeepSeekClientWrapper,
    }

    cls = providers.get(provider)
    if not cls:
        log.error("不支持的 AI Provider: %s，使用默认 siliconflow", provider)
        cls = SiliconFlowClient

    log.info("【AI Provider】创建客户端: %s", provider)
    return cls()


def test_connection(provider: str = None) -> dict:
    """测试 AI 提供商连接"""
    client = create_ai_client(provider)

    test_prompt = "请回复'连接成功'"

    try:
        response = client.chat(test_prompt, temperature=0.1, max_tokens=10)
        if response and '连接成功' in response:
            return {"success": True, "provider": provider, "response": response}
        else:
            return {"success": False, "provider": provider, "error": "响应内容异常"}
    except Exception as e:
        return {"success": False, "provider": provider, "error": str(e)}
