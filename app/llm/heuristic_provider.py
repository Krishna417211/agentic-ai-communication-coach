"""Offline provider used when no API key is configured.

It deliberately does *not* fake open-ended generation. Instead it advertises
`supports_generation = False`, and every tool falls back to its deterministic
rule-based implementation. That keeps the app fully runnable (and demoable,
and testable) with zero credentials, without inventing model output.
"""

from __future__ import annotations

from app.llm.base import LLMError, LLMProvider


class HeuristicProvider(LLMProvider):
    name = "heuristic"
    supports_generation = False

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise LLMError(
            "No LLM configured. Set GEMINI_API_KEY to enable "
            "generative features; rule-based analysis is still available."
        )
