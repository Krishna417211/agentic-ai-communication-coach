"""Provider-agnostic LLM interface.

Every provider implements `complete()`. `complete_json()` is shared: it wraps
`complete()` with JSON-repair and a schema hint, because free-tier models are
inconsistent about honouring "reply with JSON only".
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a usable completion."""


class LLMProvider(ABC):
    """Minimal chat interface the agent depends on."""

    name: str = "base"
    #: False for the offline engine, which cannot do open-ended generation.
    supports_generation: bool = True

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text completion for `prompt`."""

    @property
    def available(self) -> bool:
        return True

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Complete and parse a JSON object, repairing common model slips."""
        system_json = (
            (system or "")
            + "\n\nRespond with a single valid JSON object and nothing else. "
            "No markdown fences, no prose before or after."
        ).strip()
        raw = await self.complete(
            prompt,
            system=system_json,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = extract_json(raw)
        if parsed is None:
            raise LLMError(f"{self.name} did not return parsable JSON: {raw[:280]!r}")
        return parsed

    async def aclose(self) -> None:  # pragma: no cover - overridden where needed
        return None


def extract_json(raw: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from a model response."""
    if not raw:
        return None
    candidates: list[str] = [raw.strip()]

    fenced = _FENCE_RE.search(raw)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    # Fall back to the outermost {...} span.
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        for attempt in (candidate, _strip_trailing_commas(candidate)):
            try:
                value = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    return None


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)
