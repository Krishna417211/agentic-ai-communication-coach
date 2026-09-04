"""Communication planner and tool selection.

The planner turns an intent into an ordered plan of tool calls. Rule-based
default plans encode what a coach would actually do for each intent; when an LLM
is available it may propose its own plan, which is then validated against the
tool registry — any hallucinated or unavailable tool is dropped, and an empty
plan falls back to the rule-based one. The registry is the authority on what can
run, never the model.
"""

from __future__ import annotations

import logging

from app.llm.base import LLMError, LLMProvider
from app.memory.store import Session
from app.schemas import Intent, IntentResult, Plan, PlanStep, ToolName
from app.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# What a coach does for each intent, in order.
#: Each entry is (tool, objective, depends_on_previous). ``False`` means the
#: step needs nothing the earlier steps produce, so the orchestrator may run it
#: concurrently with its neighbours. Retrieval and diagnosis both read only the
#: user's own draft; generation needs them; scoring needs what generation wrote.
_DEFAULT_PLANS: dict[Intent, list[tuple[ToolName, str, bool]]] = {
    Intent.EMAIL_WRITING: [
        (ToolName.KNOWLEDGE_LOOKUP, "Pull email best-practice guidance.", False),
        (ToolName.GRAMMAR_CORRECTION, "Clean up mechanics in the user's draft.", False),
        (ToolName.EMAIL_GENERATION, "Draft the email end to end.", True),
        (ToolName.COMMUNICATION_SCORING, "Score the draft and quantify the improvement.", True),
    ],
    Intent.INTERVIEW_PRACTICE: [
        (ToolName.KNOWLEDGE_LOOKUP, "Retrieve interview-answer principles.", False),
        (ToolName.INTERVIEW_COACHING, "Review the answer or generate practice questions.", True),
        (ToolName.COMMUNICATION_SCORING, "Score delivery and structure.", True),
    ],
    Intent.GRAMMAR_CORRECTION: [
        (ToolName.GRAMMAR_CORRECTION, "Find and fix every mechanical error.", False),
        (ToolName.COMMUNICATION_SCORING, "Score the text so the user sees the impact.", True),
    ],
    Intent.TONE_IMPROVEMENT: [
        (ToolName.TONE_ANALYSIS, "Diagnose how the current tone reads.", False),
        (ToolName.CONVERSATION_IMPROVEMENT, "Rewrite toward the target tone.", True),
        (ToolName.COMMUNICATION_SCORING, "Score before and after.", True),
    ],
    Intent.PUBLIC_SPEAKING: [
        (ToolName.KNOWLEDGE_LOOKUP, "Retrieve presentation and delivery guidance.", False),
        (ToolName.CONVERSATION_IMPROVEMENT, "Rewrite the script for the ear.", True),
        (ToolName.COMMUNICATION_SCORING, "Score clarity and pacing for spoken delivery.", True),
    ],
    Intent.CONFLICT_RESOLUTION: [
        (ToolName.KNOWLEDGE_LOOKUP, "Retrieve de-escalation principles.", False),
        (ToolName.TONE_ANALYSIS, "Identify phrasing that would escalate.", False),
        (ToolName.CONVERSATION_IMPROVEMENT, "Rewrite using non-blaming language.", True),
        (ToolName.COMMUNICATION_SCORING, "Score the rewritten message.", True),
    ],
    Intent.CUSTOMER_COMMUNICATION: [
        (ToolName.KNOWLEDGE_LOOKUP, "Retrieve customer-communication guidance.", False),
        (ToolName.TONE_ANALYSIS, "Check empathy and blame language.", False),
        (ToolName.CONVERSATION_IMPROVEMENT, "Rewrite for the customer.", True),
        (ToolName.COMMUNICATION_SCORING, "Score the response.", True),
    ],
    Intent.GENERAL_COACHING: [
        (ToolName.KNOWLEDGE_LOOKUP, "Find relevant coaching material.", False),
        (ToolName.COMMUNICATION_SCORING, "Assess whatever the user provided.", False),
    ],
}

_GOALS: dict[Intent, str] = {
    Intent.EMAIL_WRITING: "Produce an email the recipient will read and act on.",
    Intent.INTERVIEW_PRACTICE: "Turn the user's answer into one that would pass a real interview.",
    Intent.GRAMMAR_CORRECTION: "Return mechanically correct text and explain each fix.",
    Intent.TONE_IMPROVEMENT: "Shift the message to the tone the situation calls for.",
    Intent.PUBLIC_SPEAKING: "Make the material work when spoken aloud.",
    Intent.CONFLICT_RESOLUTION: "De-escalate and move toward a concrete resolution.",
    Intent.CUSTOMER_COMMUNICATION: "Protect the relationship while being straight with the customer.",
    Intent.GENERAL_COACHING: "Give the user something specific they can act on.",
}


class CommunicationPlanner:
    def __init__(self, llm: LLMProvider, registry: ToolRegistry, *, enable_rag: bool = True):
        self._llm = llm
        self._registry = registry
        self._enable_rag = enable_rag

    async def plan(
        self,
        intent: IntentResult,
        message: str,
        *,
        has_draft: bool,
        session: Session | None = None,
    ) -> Plan:
        baseline = self.rule_based_plan(intent.intent, has_draft=has_draft)

        # A confident rule-based plan for a clear intent needs no LLM round trip.
        if not self._llm.supports_generation or intent.confidence >= 0.85:
            return baseline

        try:
            return await self._plan_llm(intent, message, baseline, session)
        except (LLMError, ValueError) as exc:
            logger.warning("planner LLM call failed, using rule-based plan: %s", exc)
            return baseline

    def rule_based_plan(self, intent: Intent, *, has_draft: bool) -> Plan:
        steps: list[PlanStep] = []
        for tool_name, objective, depends in _DEFAULT_PLANS.get(
            intent, _DEFAULT_PLANS[Intent.GENERAL_COACHING]
        ):
            if tool_name is ToolName.KNOWLEDGE_LOOKUP and not self._enable_rag:
                continue
            # Analysis tools need a draft to analyse. Scoring is exempt: with no
            # draft it scores the user's own message, which is still real
            # feedback on how they communicate.
            if not has_draft and tool_name in (
                ToolName.GRAMMAR_CORRECTION,
                ToolName.TONE_ANALYSIS,
                ToolName.CONVERSATION_IMPROVEMENT,
            ):
                continue
            if not self._registry.has(tool_name):
                continue
            steps.append(
                PlanStep(
                    tool=tool_name,
                    objective=objective,
                    # The first surviving step can never depend on a predecessor,
                    # however the table declares it — earlier steps may have been
                    # filtered out above.
                    depends_on_previous=depends and bool(steps),
                )
            )

        if not steps:
            # No draft and nothing intent-specific: answer from knowledge, or
            # go straight to the intent's primary generator.
            fallback = _primary_tool(intent)
            if self._enable_rag and self._registry.has(ToolName.KNOWLEDGE_LOOKUP):
                steps.append(
                    PlanStep(
                        tool=ToolName.KNOWLEDGE_LOOKUP,
                        objective="Ground the advice in the coaching corpus.",
                    )
                )
            if self._registry.has(fallback):
                steps.append(
                    PlanStep(
                        tool=fallback,
                        objective=f"Handle the {intent.value.replace('_', ' ')} request.",
                        depends_on_previous=bool(steps),
                    )
                )

        return Plan(
            goal=_GOALS.get(intent, _GOALS[Intent.GENERAL_COACHING]),
            steps=steps,
            rationale=(
                f"Standard coaching workflow for {intent.value.replace('_', ' ')}"
                + ("." if has_draft else ", adapted because no draft was supplied.")
            ),
            method="rule-based",
        )

    async def _plan_llm(
        self,
        intent: IntentResult,
        message: str,
        baseline: Plan,
        session: Session | None,
    ) -> Plan:
        history = ""
        if session and session.turns:
            history = "\nSession so far: " + "; ".join(
                t.content[:120] for t in session.turns[-3:]
            )

        prompt = f"""You are the planning stage of a communication-coaching agent.
Choose which tools to run, in what order, for this request.

Detected intent: {intent.intent.value} (confidence {intent.confidence})
Extracted details: {intent.entities}{history}

AVAILABLE TOOLS (you may only use these ids):
{self._registry.catalog_prompt()}

Default plan for this intent: {[s.tool.value for s in baseline.steps]}

USER MESSAGE:
\"\"\"{message}\"\"\"

Pick 1-4 tools. Order matters: analyse before rewriting, rewrite before scoring.
Do not include a tool that has nothing to work on.

Return JSON:
{{
  "goal": "what a good outcome looks like for this user, one sentence",
  "steps": [{{"tool": "tool_id", "objective": "why this tool, for this request"}}],
  "rationale": "one sentence on why this order"
}}"""

        data = await self._llm.complete_json(
            prompt,
            system="You are a planning module. You output only valid JSON.",
            temperature=0.2,
            max_tokens=1500,
        )

        steps: list[PlanStep] = []
        seen: set[ToolName] = set()
        for raw in data.get("steps", [])[:4]:
            if not isinstance(raw, dict):
                continue
            try:
                tool = ToolName(str(raw.get("tool", "")).strip().lower())
            except ValueError:
                logger.info("planner proposed unknown tool %r, dropping", raw.get("tool"))
                continue
            if tool in seen or not self._registry.has(tool):
                continue
            if tool is ToolName.KNOWLEDGE_LOOKUP and not self._enable_rag:
                continue
            if tool is ToolName.RESUME_ANALYSIS:
                continue  # only reachable through the upload endpoint
            seen.add(tool)
            steps.append(
                PlanStep(
                    tool=tool,
                    objective=str(raw.get("objective", ""))[:200],
                    depends_on_previous=bool(steps),
                )
            )

        if not steps:
            logger.info("planner returned no usable steps; using rule-based plan")
            return baseline

        return Plan(
            goal=str(data.get("goal") or baseline.goal),
            steps=steps,
            rationale=str(data.get("rationale") or ""),
            method="llm",
        )


def _primary_tool(intent: Intent) -> ToolName:
    return {
        Intent.EMAIL_WRITING: ToolName.EMAIL_GENERATION,
        Intent.INTERVIEW_PRACTICE: ToolName.INTERVIEW_COACHING,
        Intent.GRAMMAR_CORRECTION: ToolName.GRAMMAR_CORRECTION,
        Intent.TONE_IMPROVEMENT: ToolName.TONE_ANALYSIS,
        Intent.PUBLIC_SPEAKING: ToolName.CONVERSATION_IMPROVEMENT,
        Intent.CONFLICT_RESOLUTION: ToolName.CONVERSATION_IMPROVEMENT,
        Intent.CUSTOMER_COMMUNICATION: ToolName.CONVERSATION_IMPROVEMENT,
    }.get(intent, ToolName.KNOWLEDGE_LOOKUP)
