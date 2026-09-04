"""Application configuration, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["auto", "groq", "gemini", "heuristic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Agentic AI Communication Training Agent"
    version: str = "1.0.0"
    environment: str = "development"

    # --- LLM provider -------------------------------------------------
    # "auto" picks the first provider that has credentials, then falls back
    # to the offline heuristic engine so the app always runs.
    llm_provider: Provider = "auto"

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    # Gemini 3.x reasons before answering, and those tokens are billed against
    # max_tokens. "low" keeps the pipeline responsive (~3s vs ~12s per call)
    # and is ample for classification and structured rewriting. Use "high" for
    # more deliberation, or "" to send no thinkingConfig at all.
    gemini_thinking_level: str = "low"

    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.4
    llm_max_tokens: int = 3000

    # --- Memory -------------------------------------------------------
    memory_max_turns: int = 12
    memory_summary_after: int = 8
    session_ttl_seconds: int = 60 * 60 * 6

    # --- API ----------------------------------------------------------
    api_prefix: str = "/api/v1"
    cors_origins: str = "*"

    # --- Observability ------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    log_file: str | None = "logs/app.log"
    slow_request_ms: float = 3000.0

    # --- Bonus features ------------------------------------------------
    enable_rag: bool = True
    rag_top_k: int = 3
    max_resume_pages: int = 10

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
