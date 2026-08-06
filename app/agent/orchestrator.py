"""The agent loop: intent -> plan -> tool selection -> execution -> feedback."""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from app.agent.intent import IntentDetector
from app.agent.planner import CommunicationPlanner
from app.agent.synthesis import FeedbackSynthesizer
from app.config import Settings
from app.llm.base import LLMProvider
from app.memory.store import SessionStore
from app.observability.logging_config import session_id_var
from app.rag.knowledge import get_knowledge_base
from app.schemas import (
    CoachingResponse,
    Intent,
    IntentResult,
    Plan,
    Role,
    ScoreBreakdown,
    ToolName,
    ToolResult,
)
from app.tools.base import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)


class CommunicationAgent:
    """Owns one pass of the pipeline for a single user message."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        registry: ToolRegistry,
        memory: SessionStore,
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.settings = settings
        self.intent_detector = IntentDetector(llm)
        self.planner = CommunicationPlanner(
            llm, registry, enable_rag=settings.enable_rag
        )
        self.synthesizer = FeedbackSynthesizer(llm)

    async def run(
        self,
        message: str,
        *,
        session_id: str | None = None,
        goal: str | None = None,
        audience: str | None = None,
        trace_id: str | None = None,
        options: dict[str, Any] | None = None,
        intent_override: Intent | None = None,
        plan_override: Plan | None = None,
    ) -> CoachingResponse:
        started = time.perf_counter()
        trace_id = trace_id or uuid.uuid4().hex[:12]
        session = await self.memory.get_or_create(session_id)
        # Every log line for the rest of this request carries the session id.
        session_id_var.set(session.session_id)

        # --- 1. Intent detection ---------------------------------------
        if intent_override is not None:
            intent = IntentResult(
                intent=intent_override,
                confidence=1.0,
                rationale="Intent supplied by the calling endpoint.",
                method="heuristic",
            )
            from app.agent.extract import extract_all

            intent.entities = extract_all(message)
        else:
            intent = await self.intent_detector.detect(message, session)

        entities = intent.entities or {}
        target_text = str(entities.get("target_text") or "")
        resolved_audience = audience or entities.get("audience")

        logger.info(
            "trace=%s session=%s intent=%s confidence=%.2f method=%s",
            trace_id, session.session_id, intent.intent.value,
            intent.confidence, intent.method,
        )

        # --- 2. Planning ------------------------------------------------
        has_draft = bool(target_text) or _looks_like_a_draft(message, intent.intent)
        plan = plan_override or await self.planner.plan(
            intent, message, has_draft=has_draft, session=session
        )
        logger.info(
            "trace=%s plan=%s method=%s",
            trace_id, [s.tool.value for s in plan.steps], plan.method,
        )

        # --- 3. Context assembly ----------------------------------------
        tool_options: dict[str, Any] = {
            "target_tone": entities.get("target_tone"),
            "role": entities.get("role"),
            "top_k": self.settings.rag_top_k,
        }
        tool_options.update({k: v for k, v in (options or {}).items() if v is not None})
        tool_options = {k: v for k, v in tool_options.items() if v is not None}

        ctx = ToolContext(
            user_message=message,
            llm=self.llm,
            intent=intent.intent,
            target_text=target_text or (message if has_draft else ""),
            goal=goal,
            audience=resolved_audience,
            session=session,
            options=tool_options,
        )

        # --- 4. Tool selection + execution -------------------------------
        results: list[ToolResult] = []
        for step in plan.steps:
            tool = self.registry.get(step.tool)
            if tool is None:
                logger.warning("trace=%s planned tool %s is not registered", trace_id, step.tool)
                continue

            result = await tool(ctx)
            results.append(result)
            logger.info(
                "trace=%s tool=%s ok=%s duration_ms=%.1f",
                trace_id, step.tool.value, result.ok, result.duration_ms,
            )

            if result.ok:
                # Later steps see earlier outputs.
                ctx.artifacts[step.tool.value] = result.output
                if step.tool is ToolName.KNOWLEDGE_LOOKUP:
                    ctx.knowledge = [
                        {"title": p["title"], "content": p["content"]}
                        for p in result.output.get("passages", [])
                    ]

        # --- 5. Feedback synthesis ---------------------------------------
        synthesis = await self.synthesizer.synthesize(
            ctx, intent=intent, plan=plan, results=results
        )

        score = synthesis.score
        overall = score.overall if score else None

        # --- 6. Memory update ---------------------------------------------
        await self.memory.append_turn(
            session.session_id,
            role=Role.USER,
            content=message,
            intent=intent.intent,
        )
        await self.memory.append_turn(
            session.session_id,
            role=Role.ASSISTANT,
            content=synthesis.coaching_message,
            intent=intent.intent,
            tools_used=[r.tool for r in results if r.ok],
            overall_score=overall,
            issues=synthesis.feedback,
        )

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "trace=%s completed tools=%s score=%s duration_ms=%.1f",
            trace_id, [r.tool.value for r in results], overall, duration_ms,
        )

        return CoachingResponse(
            session_id=session.session_id,
            trace_id=trace_id,
            intent=intent,
            plan=plan,
            tools_used=[r.tool for r in results if r.ok],
            coaching_message=synthesis.coaching_message,
            improved_response=synthesis.improved_response,
            feedback=synthesis.feedback,
            score=score,
            overall_score=overall,
            tool_results=results,
            knowledge_used=synthesis.citations,
            provider=self.llm.name,
            duration_ms=duration_ms,
        )

    async def run_single_tool(
        self,
        tool_name: ToolName,
        message: str,
        *,
        session_id: str | None = None,
        options: dict[str, Any] | None = None,
        audience: str | None = None,
        intent: Intent = Intent.GENERAL_COACHING,
        use_knowledge: bool = False,
    ) -> ToolResult:
        """Run one tool directly — used by the narrow, single-purpose endpoints."""
        tool = self.registry.get(tool_name)
        if tool is None:
            raise KeyError(f"tool {tool_name} is not registered")

        session = await self.memory.get_or_create(session_id) if session_id else None
        knowledge: list[dict[str, str]] = []
        if use_knowledge and self.settings.enable_rag:
            knowledge = [
                {"title": p["title"], "content": p["content"]}
                for p in get_knowledge_base().search(message, top_k=self.settings.rag_top_k)
            ]

        ctx = ToolContext(
            user_message=message,
            llm=self.llm,
            intent=intent,
            target_text=message,
            audience=audience,
            session=session,
            knowledge=knowledge,
            options=options or {},
        )
        return await tool(ctx)


_DRAFT_HINTS = (
    Intent.GRAMMAR_CORRECTION,
    Intent.TONE_IMPROVEMENT,
    Intent.CONFLICT_RESOLUTION,
    Intent.CUSTOMER_COMMUNICATION,
)

#: Someone asking the coach for advice is not handing over a draft. Rewriting
#: their question produces an "improved version" that is just their question
#: back, which is worse than useless.
_ADVICE_QUESTION_RE = re.compile(
    r"\b(?:how (?:do|should|can|would) i|what (?:should|do) i|"
    r"how (?:can|do) you|any (?:tips|advice|suggestions)|"
    r"what'?s the best way|should i|is it (?:ok|okay|better)|"
    r"can you (?:help|teach|explain|walk)|how to)\b",
    re.IGNORECASE,
)


def _looks_like_a_draft(message: str, intent: Intent) -> bool:
    """Decide whether the message body is itself the text to work on.

    Long messages, or messages for intents that presuppose existing text, are
    treated as drafts even when no explicit quoting was detected — unless the
    message is the user asking the coach a question about their situation.
    """
    if _ADVICE_QUESTION_RE.search(message):
        return False

    words = len(message.split())
    if words >= 45:
        return True
    return intent in _DRAFT_HINTS and words >= 15


def empty_score() -> ScoreBreakdown:
    return ScoreBreakdown(clarity=0, tone=0, grammar=0, structure=0, impact=0)
