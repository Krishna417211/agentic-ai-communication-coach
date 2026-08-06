"""Tool abstraction and registry.

Every tool exposes a description the planner sees, so adding a tool makes it
selectable without touching the planner or the router.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.llm.base import LLMError, LLMProvider
from app.memory.store import Session
from app.schemas import Intent, ToolName, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Everything a tool may need, assembled once per request."""

    user_message: str
    llm: LLMProvider
    intent: Intent = Intent.GENERAL_COACHING
    #: The text being coached — usually a quoted draft inside `user_message`.
    target_text: str = ""
    goal: str | None = None
    audience: str | None = None
    session: Session | None = None
    knowledge: list[dict[str, str]] = field(default_factory=list)
    #: Outputs of previously executed steps in this plan, keyed by tool name.
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """The text to operate on, falling back to the raw message."""
        return self.target_text or self.user_message

    def memory_brief(self, max_turns: int = 4) -> str:
        """A compact recap of the session for prompt grounding."""
        if not self.session or not self.session.turns:
            return "This is the first message of the session."
        lines: list[str] = []
        if self.session.summary:
            lines.append(f"Earlier context: {self.session.summary}")
        for turn in self.session.turns[-max_turns:]:
            speaker = "User" if turn.role == "user" else "Coach"
            lines.append(f"{speaker}: {turn.content[:220]}")
        recurring = self.session.recurring_issues()
        if recurring:
            lines.append(f"Recurring weaknesses: {'; '.join(recurring)}")
        return "\n".join(lines)

    def knowledge_brief(self) -> str:
        if not self.knowledge:
            return ""
        blocks = [f"[{k['title']}]\n{k['content']}" for k in self.knowledge]
        return "Reference material you may draw on:\n" + "\n\n".join(blocks)


class Tool(ABC):
    """Base class for every communication tool."""

    name: ToolName
    description: str = ""
    when_to_use: str = ""

    async def __call__(self, ctx: ToolContext) -> ToolResult:
        """Run the tool, timing it and containing its failures."""
        started = time.perf_counter()
        try:
            if ctx.llm.supports_generation:
                try:
                    result = await self.run_llm(ctx)
                except LLMError as exc:
                    logger.warning(
                        "tool=%s llm path failed, using rule-based fallback: %s",
                        self.name,
                        exc,
                    )
                    result = self.run_offline(ctx)
                    result.output["degraded"] = True
                    result.output["degraded_reason"] = str(exc)
            else:
                result = self.run_offline(ctx)
        except Exception as exc:  # noqa: BLE001 - one bad tool must not kill the turn
            logger.exception("tool=%s failed", self.name)
            result = ToolResult(
                tool=self.name,
                ok=False,
                summary=f"{self.name} could not complete.",
                error=str(exc),
            )
        result.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    @abstractmethod
    def run_offline(self, ctx: ToolContext) -> ToolResult:
        """Deterministic implementation; must never require network access."""

    async def run_llm(self, ctx: ToolContext) -> ToolResult:
        """LLM-backed implementation. Defaults to the deterministic one."""
        return self.run_offline(ctx)

    def spec(self) -> dict[str, str]:
        return {
            "name": self.name.value,
            "description": self.description,
            "when_to_use": self.when_to_use,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[ToolName, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: ToolName) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: ToolName) -> bool:
        return name in self._tools

    def names(self) -> list[ToolName]:
        return list(self._tools)

    def catalog(self) -> list[dict[str, str]]:
        return [tool.spec() for tool in self._tools.values()]

    def catalog_prompt(self) -> str:
        return "\n".join(
            f"- {t.name.value}: {t.description} Use when: {t.when_to_use}"
            for t in self._tools.values()
        )
