"""Ollama 调用封装（异步）。

为什么不用现成的 SDK：这里只需要 /api/chat、/api/tags 这几个接口，
用 httpx 直接写反而更透明，也方便加超时、错误提示和思考链剥离。
"""

from __future__ import annotations

import re
from typing import Any

import httpx

# DeepSeek-R1 这类推理模型会把思考过程包在  thinking...<｜end▁of▁thinking｜> 里，
# 用于 ReAct 解析时要把这段剔掉，否则很容易和 Thought/Action 格式串味
_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)


class OllamaError(RuntimeError):
    """Ollama 调用失败。"""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "deepseek-r1:7b",
        timeout: float = 300.0,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.temperature = temperature
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ #
    # 基础调用
    # ------------------------------------------------------------------ #

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature if temperature is None else temperature},
        }
        if stop:
            payload["stop"] = stop

        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"连不上 Ollama（{self.base_url}）。请确认已启动 ollama serve。"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaError(f"Ollama 响应超时（>{self._client.timeout.read}s）。") from exc

        if response.status_code == 404:
            raise OllamaError(
                f"模型 {payload['model']!r} 不存在。先执行：ollama pull {payload['model']}"
            )
        if response.status_code >= 400:
            raise OllamaError(f"Ollama 返回 {response.status_code}：{response.text[:300]}")

        data = response.json()
        message = data.get("message") or {}
        content = message.get("content") or ""

        # 有些版本会把思考内容放在 message.thinking 里，这里只取正式回答
        return strip_thinking(content)

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, model=model, temperature=temperature)

    # ------------------------------------------------------------------ #
    # 辅助接口
    # ------------------------------------------------------------------ #

    async def list_models(self) -> list[str]:
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"获取模型列表失败：{exc}") from exc
        return [m.get("name", "") for m in response.json().get("models", [])]

    async def is_available(self) -> bool:
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False


def strip_thinking(text: str) -> str:
    """去掉推理模型的思考块，只留正式回答。"""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    return cleaned.strip()