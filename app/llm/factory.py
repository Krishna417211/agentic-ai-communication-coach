"""Provider selection: explicit choice, or first configured provider."""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.gemini_provider import GeminiProvider
from app.llm.heuristic_provider import HeuristicProvider

logger = logging.getLogger(__name__)


def build_provider(settings: Settings) -> LLMProvider:
    choice = settings.llm_provider.lower()

    if choice == "gemini" or (choice == "auto" and settings.gemini_api_key):
        if settings.gemini_api_key:
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

    if choice == "groq" or (choice == "auto" and settings.groq_api_key):
        if settings.groq_api_key:
            from app.llm.openai_provider import OpenAICompatibleProvider
            logger.info("LLM provider: groq (%s)", settings.groq_model)
            return OpenAICompatibleProvider(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                base_url=settings.groq_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                default_temperature=settings.llm_temperature,
                default_max_tokens=settings.llm_max_tokens,
                provider_name="groq",
            )

    if choice == "openai" or (choice == "auto" and settings.openai_api_key):
        if settings.openai_api_key:
            from app.llm.openai_provider import OpenAICompatibleProvider
            logger.info("LLM provider: openai (%s)", settings.openai_model)
            return OpenAICompatibleProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                base_url=settings.openai_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                default_temperature=settings.llm_temperature,
                default_max_tokens=settings.llm_max_tokens,
                provider_name="openai",
            )

    if choice == "openrouter" or (choice == "auto" and settings.openrouter_api_key):
        if settings.openrouter_api_key:
            from app.llm.openai_provider import OpenAICompatibleProvider
            logger.info("LLM provider: openrouter (%s)", settings.openrouter_model)
            return OpenAICompatibleProvider(
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
                base_url=settings.openrouter_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
                default_temperature=settings.llm_temperature,
                default_max_tokens=settings.llm_max_tokens,
                provider_name="openrouter",
            )

    if choice == "ollama":
        from app.llm.openai_provider import OpenAICompatibleProvider
        logger.info("LLM provider: ollama (%s)", settings.ollama_model)
        return OpenAICompatibleProvider(
            api_key="",
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            default_temperature=settings.llm_temperature,
            default_max_tokens=settings.llm_max_tokens,
            provider_name="ollama",
        )

    logger.warning(
        "No working LLM API key configured (llm_provider=%s); running on the offline rule-based engine. "
        "Set GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY in .env.",
        choice
    )
    return HeuristicProvider()
