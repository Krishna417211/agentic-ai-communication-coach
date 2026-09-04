"""FastAPI dependencies — everything resolves off `app.state`, set in the lifespan."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agent.orchestrator import CommunicationAgent
from app.config import Settings, get_settings
from app.llm.base import LLMProvider
from app.memory.profile import ProfileStore
from app.memory.store import SessionStore


def get_agent(request: Request) -> CommunicationAgent:
    return request.app.state.agent


def get_memory(request: Request) -> SessionStore:
    return request.app.state.memory


def get_profiles(request: Request) -> ProfileStore:
    return request.app.state.profiles


def get_llm(request: Request) -> LLMProvider:
    return request.app.state.llm


def get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "-")


AgentDep = Annotated[CommunicationAgent, Depends(get_agent)]
MemoryDep = Annotated[SessionStore, Depends(get_memory)]
ProfilesDep = Annotated[ProfileStore, Depends(get_profiles)]
LLMDep = Annotated[LLMProvider, Depends(get_llm)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
TraceDep = Annotated[str, Depends(get_trace_id)]
