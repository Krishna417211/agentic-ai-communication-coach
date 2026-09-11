"""FastAPI dependencies — everything resolves off `app.state`, set in the lifespan."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agent.orchestrator import CommunicationAgent
from app.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.memory.store import SessionStore


def get_agent(request: Request) -> CommunicationAgent:
    settings = get_settings()
    current_llm = getattr(request.app.state, "llm", None)
    from app.llm.factory import build_provider

    # Rebuild provider if not set or if config changed
    key_changed = False
    if current_llm is not None:
        if getattr(current_llm, "name", "") == "gemini":
            key_changed = (
                getattr(current_llm, "_api_key", None) != settings.gemini_api_key
                or getattr(current_llm, "model", None) != settings.gemini_model
            )
        elif getattr(current_llm, "name", "") in ("groq", "openai", "openrouter", "ollama"):
            expected_key = getattr(settings, f"{current_llm.name}_api_key", None)
            expected_model = getattr(settings, f"{current_llm.name}_model", None)
            key_changed = (
                getattr(current_llm, "_api_key", None) != expected_key
                or getattr(current_llm, "model", None) != expected_model
            )
        else:
            # Current is heuristic, check if any API key is now set
            if any([settings.gemini_api_key, settings.groq_api_key, settings.openai_api_key, settings.openrouter_api_key]):
                key_changed = True

    if current_llm is None or key_changed:
        new_llm = build_provider(settings)
        request.app.state.llm = new_llm
        request.app.state.agent = CommunicationAgent(
            llm=new_llm,
            registry=request.app.state.registry,
            memory=request.app.state.memory,
            settings=settings,
        )
    return request.app.state.agent


def get_memory(request: Request) -> SessionStore:
    return request.app.state.memory


def get_llm(request: Request) -> LLMProvider:
    return get_agent(request).llm


def get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "-")


AgentDep = Annotated[CommunicationAgent, Depends(get_agent)]
MemoryDep = Annotated[SessionStore, Depends(get_memory)]
LLMDep = Annotated[LLMProvider, Depends(get_llm)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
TraceDep = Annotated[str, Depends(get_trace_id)]
