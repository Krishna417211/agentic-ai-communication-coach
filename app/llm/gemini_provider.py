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
        thinking_level: str = "low",
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._thinking_level = thinking_level
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    #: Reasoning tokens are billed against maxOutputTokens, so a request must
    #: never be given a budget so small that reasoning consumes all of it —
    #: the model then returns truncated reasoning instead of an answer.
    MIN_OUTPUT_TOKENS = 1200

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
        budget = max(max_tokens or self._default_max_tokens, self.MIN_OUTPUT_TOKENS)
        generation_config: dict = {
            "temperature": (
                self._default_temperature if temperature is None else temperature
            ),
            "maxOutputTokens": budget,
        }
        if self._thinking_level:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": self._thinking_level
            }

        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{self._base_url}/models/{self.model}:generateContent"
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    url,
                    params={"key": self._api_key},
                    json=payload,
                    headers={"x-goog-api-key": self._api_key},
                )
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    try:
                        err_msg = response.json().get("error", {}).get("message")
                    except Exception:
                        err_msg = response.text[:200]
                    raise LLMError(
                        f"Gemini request failed ({response.status_code}): {err_msg or response.text[:200]}"
                    )
                if response.status_code == 429:
                    # Distinguish a burst rate-limit (worth a short retry) from
                    # an exhausted daily quota (not recoverable in-request —
                    # retrying just delays the fall back to the rule engine).
                    delay, quota_ids = _parse_quota_error(response)
                    if delay is None or delay > _MAX_RETRY_DELAY_SECONDS:
                        advice = (
                            "no retry delay advertised"
                            if delay is None
                            else f"suggested delay {delay:g}s"
                        )
                        raise LLMError(
                            f"Gemini quota exhausted ({', '.join(quota_ids) or '429'}); "
                            f"retry not viable in-request ({advice})."
                        )
                    raise httpx.HTTPStatusError(
                        f"gemini rate-limited, retry in {delay}s",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 500:
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


#: Above this, waiting inside the request costs the caller more than falling
#: back to the deterministic engine is worth.
_MAX_RETRY_DELAY_SECONDS = 10.0


def _parse_quota_error(response: httpx.Response) -> tuple[float | None, list[str]]:
    """Pull the suggested retry delay and quota ids out of a 429 body.

    Returns (delay_seconds or None if not advertised, quota ids). A per-day
    quota id means the limit will not clear during this request regardless of
    what retryDelay claims.
    """
    try:
        details = response.json().get("error", {}).get("details", [])
    except ValueError:
        return None, []

    delay: float | None = None
    quota_ids: list[str] = []
    for detail in details:
        type_url = str(detail.get("@type", ""))
        if "RetryInfo" in type_url:
            raw = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                delay = float(raw)
            except ValueError:
                delay = None
        elif "QuotaFailure" in type_url:
            quota_ids += [
                str(v.get("quotaId", "")) for v in detail.get("violations", [])
            ]

    if any("PerDay" in q for q in quota_ids):
        return None, quota_ids  # daily cap — do not retry
    return delay, quota_ids


def _first_text(data: dict) -> str:
    """Join the answer text of the first candidate.

    Gemini 2.5+ and 3.x reason before answering. Those reasoning tokens are
    billed against `maxOutputTokens` and, when the model is asked to expose
    them, arrive as parts flagged `thought: true` — which must not be
    concatenated into the answer.
    """
    candidates = data.get("candidates") or []
    if not candidates:
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        raise LLMError(f"Gemini returned no candidates (blockReason={blocked})")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(
        part.get("text", "") for part in parts if not part.get("thought")
    )

    if not text:
        reason = candidate.get("finishReason")
        if reason == "MAX_TOKENS":
            thoughts = (data.get("usageMetadata") or {}).get("thoughtsTokenCount", 0)
            raise LLMError(
                "Gemini hit maxOutputTokens before emitting an answer "
                f"({thoughts} tokens went to reasoning). Raise LLM_MAX_TOKENS."
            )
        raise LLMError(f"Gemini returned an empty completion (finishReason={reason})")
    return text
