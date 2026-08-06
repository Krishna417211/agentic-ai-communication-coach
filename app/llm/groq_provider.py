"""Groq provider (OpenAI-compatible chat completions endpoint)."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        *,
        timeout: float = 45.0,
        max_retries: int = 2,
        default_temperature: float = 0.4,
        default_max_tokens: int = 1600,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                self._default_temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self._default_max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self._base_url}/chat/completions", json=payload
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"groq returned {response.status_code}: {response.text[:200]}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    backoff = 1.5 * (2**attempt)
                    logger.warning(
                        "groq call failed (attempt %s/%s), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)

        raise LLMError(f"Groq request failed: {last_error}") from last_error

    async def aclose(self) -> None:
        await self._client.aclose()
