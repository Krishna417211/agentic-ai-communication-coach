"""OpenAI-compatible LLM provider (OpenAI, Groq, OpenRouter, Ollama, etc.)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    name = "openai-compatible"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        *,
        timeout: float = 45.0,
        max_retries: int = 2,
        default_temperature: float = 0.4,
        default_max_tokens: int = 2000,
        provider_name: str = "openai",
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._base_url = base_url.rstrip("/")
        self.name = provider_name
        self._max_retries = max_retries
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    @property
    def available(self) -> bool:
        return bool(self._api_key or "localhost" in self._base_url or "127.0.0.1" in self._base_url)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._default_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._default_max_tokens,
        }

        url = f"{self._base_url}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(url, headers=headers, json=payload)
                if response.status_code >= 400:
                    try:
                        err_msg = str(response.json().get("error", {}).get("message", ""))
                    except Exception:
                        err_msg = response.text[:200]
                    raise LLMError(
                        f"{self.name} request failed ({response.status_code}): {err_msg or response.text[:200]}"
                    )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError(f"{self.name} returned no completion choices.")
                content = str(choices[0].get("message", {}).get("content") or "").strip()
                if not content:
                    raise LLMError(f"{self.name} returned empty content.")
                return content
            except (httpx.HTTPError, KeyError, IndexError, ValueError, LLMError) as exc:
                last_error = exc
                if isinstance(exc, LLMError) and "40" in str(exc):
                    # Client errors (401, 404, 400) shouldn't retry uselessly
                    break
                if attempt < self._max_retries:
                    backoff = 1.5 * (2**attempt)
                    logger.warning(
                        "%s call failed (attempt %s/%s), retrying in %.1fs: %s",
                        self.name,
                        attempt + 1,
                        self._max_retries + 1,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)

        raise LLMError(f"{self.name} request failed: {last_error}") from last_error

    async def aclose(self) -> None:
        await self._client.aclose()
