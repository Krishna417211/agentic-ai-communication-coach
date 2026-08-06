"""Google Gemini provider (generativelanguage REST API)."""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

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
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

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
        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": (
                    self._default_temperature if temperature is None else temperature
                ),
                "maxOutputTokens": max_tokens or self._default_max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{self._base_url}/models/{self.model}:generateContent"
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    url,
                    json=payload,
                    headers={"x-goog-api-key": self._api_key},
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"gemini returned {response.status_code}: {response.text[:200]}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return _first_text(response.json())
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    backoff = 1.5 * (2**attempt)
                    logger.warning(
                        "gemini call failed (attempt %s/%s), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)

        raise LLMError(f"Gemini request failed: {last_error}") from last_error

    async def aclose(self) -> None:
        await self._client.aclose()


def _first_text(data: dict) -> str:
    """Join the text parts of the first candidate."""
    candidates = data.get("candidates") or []
    if not candidates:
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        raise LLMError(f"Gemini returned no candidates (blockReason={blocked})")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    if not text:
        raise LLMError("Gemini returned an empty completion")
    return text
