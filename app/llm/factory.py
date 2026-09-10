"""Provider selection: explicit choice, or first configured provider."""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.heuristic_provider import HeuristicProvider

logger = logging.getLogger(__name__)


def build_provider(settings: Settings) -> LLMProvider:
    choice = settings.llm_provider

    if choice in ("auto", "gemini") and settings.gemini_api_key:
        logger.info("LLM provider: gemini (%s)", settings.gemini_model)
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            base_url=settings.gemini_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
            thinking_level=settings.gemini_thinking_level,
        )

    if choice == "gemini":
        logger.warning(
            "llm_provider=gemini requested but no GEMINI_API_KEY found; "
            "falling back to the offline rule-based engine."
        )
    else:
        logger.warning(
            "No GEMINI_API_KEY configured; running on the offline rule-based engine. "
            "Set GEMINI_API_KEY for generative coaching."
        )
    return HeuristicProvider()
