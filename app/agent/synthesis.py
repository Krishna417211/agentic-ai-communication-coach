"""Feedback synthesis — the final stage of the pipeline.

Turns raw tool outputs into one coherent coaching reply. The improved text and
the scores always come from the tools, never from this stage: the synthesizer
narrates, it does not re-decide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.llm.base import LLMError, LLMProvider
from app.schemas import (
    IntentResult,
    Plan,
    ScoreBreakdown,
    ToolName,
    ToolResult,
)
from app.tools.base import ToolContext

logger = logging.getLogger(__name__)


@dataclass
class Synthesis:
    coaching_message: str
    improved_response: str | None = None
    feedback: list[str] = field(default_factory=list)
    score: ScoreBreakdown | None = None
    citations: list[str] = field(default_factory=list)


class FeedbackSynthesizer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def synthesize(
        self,
        ctx: ToolContext,
        *,
        intent: IntentResult,
        plan: Plan,
        results: list[ToolResult],
    ) -> Synthesis:
        facts = _collect(results)

        if self._llm.supports_generation:
            try:
                return await self._synthesize_llm(ctx, intent, plan, results, facts)
            except LLMError as exc:
                logger.warning("synthesis LLM call failed, composing offline: %s", exc)

        return self._synthesize_offline(ctx, intent, results, facts)

    # -- LLM path ---------------------------------------------------------

    async def _synthesize_llm(
        self,
        ctx: ToolContext,
        intent: IntentResult,
        plan: Plan,
        results: list[ToolResult],
        facts: "_Facts",
    ) -> Synthesis:
        findings = "\n".join(
            f"### {r.tool.value}\n{_render_for_prompt(r)}" for r in results if r.ok
        ) or "No tools produced output."

        history_note = ""
        if ctx.session and ctx.session.recurring_issues():
            history_note = (
                "\nThis user has repeated these habits across the session: "
                + "; ".join(ctx.session.recurring_issues())
                + ". Reference that pattern — do not treat this message as their first."
            )
        if ctx.session and len(ctx.session.scores) >= 2:
            history_note += f"\nTheir score trend: {ctx.session.score_trend()}."

        prompt = f"""You are a communication coach writing the reply the user sees.

Their request: \"\"\"{ctx.user_message}\"\"\"
Detected intent: {intent.intent.value}
Coaching goal: {plan.goal}{history_note}

TOOL FINDINGS (this is your evidence — do not contradict it, do not invent more):
{findings}

Write the coaching reply in markdown:
1. One or two sentences on what you see in their communication. Be specific and
   quote their words.
2. "**What to change**" — a short bulleted list, each bullet naming the problem
   and the fix. No generic advice.
3. If a tool produced improved text, present it under "**Improved version**" in a
   markdown block, reproduced EXACTLY as the tool wrote it.
4. One closing line: the single habit to focus on next.

Be direct and warm. No preamble, no "great question", no restating the request.

Return JSON:
{{
  "coaching_message": "the full markdown reply",
  "feedback": ["the 2-5 specific weaknesses you identified, one short line each"],
  "focus_next": "the one habit to work on"
}}"""

        data = await self._llm.complete_json(
            prompt,
            system=(
                "You are an experienced communication coach. You are specific, "
                "you quote the user's own words, and you never pad."
            ),
            temperature=0.5,
        )

        message = str(data.get("coaching_message") or "").strip()
        if not message:
            return self._synthesize_offline(ctx, intent, results, facts)

        feedback = [str(f) for f in data.get("feedback", []) if f][:5] or facts.weaknesses
        return Synthesis(
            coaching_message=message,
            improved_response=facts.improved,
            feedback=feedback,
            score=facts.score,
            citations=facts.citations,
        )

    # -- Offline path ------------------------------------------------------

    def _synthesize_offline(
        self,
        ctx: ToolContext,
        intent: IntentResult,
        results: list[ToolResult],
        facts: "_Facts",
    ) -> Synthesis:
        topic = intent.intent.value.replace("_", " ")
        target = ctx.target_text or ctx.user_message
        
        parts: list[str] = [
            f"### Coaching Analysis: {topic.title()}",
            f"**Request / Draft:** *\"{target[:200]}{'...' if len(target) > 200 else ''}\"*",
        ]

        if facts.score:
            parts.append(
                f"\n**Overall Communication Score: {facts.score.overall}/100**\n"
                f"- Clarity: {facts.score.clarity}/100\n"
                f"- Tone: {facts.score.tone}/100\n"
                f"- Grammar & Mechanics: {facts.score.grammar}/100\n"
                f"- Structure: {facts.score.structure}/100\n"
                f"- Impact & Persuasion: {facts.score.impact}/100"
            )
            if facts.score_after and facts.score_after.overall != facts.score.overall:
                delta = facts.score_after.overall - facts.score.overall
                parts.append(
                    f"\n*The improved version boosts your overall score to **{facts.score_after.overall}/100** ({delta:+d} points).* "
                )

        if facts.strengths:
            parts.append("\n**Key Strengths**")
            parts.extend(f"- {s}" for s in facts.strengths[:4])

        if facts.weaknesses:
            parts.append("\n**Areas for Improvement**")
            parts.extend(f"- {w}" for w in facts.weaknesses[:6])

        if facts.grammar_issues:
            parts.append("\n**Grammar & Sentence Corrections**")
            for issue in facts.grammar_issues[:6]:
                original = str(issue.get("original", ""))[:60]
                suggestion = str(issue.get("suggestion", ""))[:60]
                parts.append(f"- Original: `{original}` $\\rightarrow$ Suggested: `{suggestion}` ({issue.get('message', '')})")

        if facts.star_coverage:
            missing = [k.capitalize() for k, v in facts.star_coverage.items() if not v]
            covered = [k.capitalize() for k, v in facts.star_coverage.items() if v]
            parts.append(
                f"\n**STAR Interview Method Coverage:**\n"
                f"- Included components: {', '.join(covered) or 'None'}\n"
                + (f"- Missing components to add: **{', '.join(missing)}**" if missing else "- Complete STAR structure!")
            )

        if facts.questions:
            parts.append("\n**Recommended Practice Questions**")
            parts.extend(f"{i}. {q}" for i, q in enumerate(facts.questions[:6], 1))

        if facts.improved:
            label = "Suggested Email Draft" if facts.subject else "Improved Version"
            body = f"Subject: {facts.subject}\n\n{facts.improved}" if facts.subject else facts.improved
            parts.append(f"\n**{label}**\n\n```text\n{body}\n```")

        if facts.changes:
            parts.append("\n**Key Edits Made**")
            parts.extend(f"- {c}" for c in facts.changes[:6])

        if facts.knowledge_snippets:
            parts.append("\n**Best Practice Playbook**")
            for snippet in facts.knowledge_snippets[:2]:
                first_para = snippet["content"].strip().split("\n\n")[0]
                parts.append(f"- **{snippet['title']}**: {first_para[:280]}...")

        if ctx.session and ctx.session.recurring_issues():
            parts.append(
                "\n**Recurring Patterns in Session:** "
                + "; ".join(ctx.session.recurring_issues())
            )

        if not self._llm.supports_generation or getattr(self._llm, "name", "") == "heuristic":
            parts.append(
                "\n---\n*💡 **LLM Status**: Running offline rule-based fallback. Enter a valid API key (Gemini, Groq, OpenAI) in the sidebar under **⚙️ LLM Settings** for generative AI responses.*"
            )
        else:
            parts.append(
                "\n---\n*💡 **LLM Status**: Provider call used rule fallback due to rate-limit/quota. Check API key status in sidebar.*"
            )

        return Synthesis(
            coaching_message="\n".join(parts).strip(),
            improved_response=facts.improved,
            feedback=facts.weaknesses,
            score=facts.score,
            citations=facts.citations,
        )


@dataclass
class _Facts:
    """Everything worth reporting, pulled out of the raw tool outputs."""

    score: ScoreBreakdown | None = None
    score_after: ScoreBreakdown | None = None
    improved: str | None = None
    subject: str | None = None
    changes: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    grammar_issues: list[dict] = field(default_factory=list)
    star_coverage: dict[str, bool] = field(default_factory=dict)
    questions: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    knowledge_snippets: list[dict] = field(default_factory=list)


def _collect(results: list[ToolResult]) -> _Facts:
    facts = _Facts()

    for result in results:
        if not result.ok:
            continue
        out = result.output

        match result.tool:
            case ToolName.COMMUNICATION_SCORING:
                if out.get("score"):
                    facts.score = ScoreBreakdown(**out["score"])
                if out.get("score_after_improvement"):
                    facts.score_after = ScoreBreakdown(**out["score_after_improvement"])
                facts.strengths.extend(out.get("strengths", []))
                facts.weaknesses.extend(out.get("weaknesses", []))
                if out.get("biggest_lever"):
                    facts.weaknesses.insert(0, f"Biggest lever: {out['biggest_lever']}")

            case ToolName.GRAMMAR_CORRECTION:
                facts.grammar_issues.extend(out.get("issues", []))
                if out.get("corrected") and not facts.improved:
                    facts.improved = out["corrected"]

            case ToolName.TONE_ANALYSIS:
                tone = out.get("tone", {})
                facts.weaknesses.extend(tone.get("risks", []))
                if out.get("gap"):
                    facts.weaknesses.append(out["gap"])
                facts.changes.extend(out.get("quick_fixes", []))

            case ToolName.CONVERSATION_IMPROVEMENT:
                if out.get("improved"):
                    facts.improved = out["improved"]
                facts.changes.extend(out.get("changes", []))
                if out.get("watch_out"):
                    facts.weaknesses.append(out["watch_out"])

            case ToolName.EMAIL_GENERATION:
                if out.get("body"):
                    facts.improved = out["body"]
                    facts.subject = out.get("subject")
                facts.changes.extend(out.get("changes", []))
                facts.changes.extend(out.get("notes", []))

            case ToolName.INTERVIEW_COACHING:
                facts.star_coverage = out.get("star_coverage", {}) or {}
                facts.strengths.extend(out.get("strengths", []))
                facts.weaknesses.extend(out.get("feedback", []))
                if out.get("model_answer"):
                    facts.improved = out["model_answer"]
                facts.questions.extend(
                    q["question"] if isinstance(q, dict) else str(q)
                    for q in out.get("questions", [])
                )
                facts.questions.extend(
                    str(q) for q in out.get("followup_questions", [])
                )
                if out.get("score") and facts.score is None:
                    facts.score = ScoreBreakdown(**out["score"])

            case ToolName.KNOWLEDGE_LOOKUP:
                facts.citations.extend(out.get("citations", []))
                facts.knowledge_snippets.extend(out.get("passages", []))

            case ToolName.RESUME_ANALYSIS:
                facts.strengths.extend(out.get("strengths", []))
                facts.weaknesses.extend(out.get("weaknesses", []))
                facts.questions.extend(out.get("suggested_interview_questions", []))
                if out.get("score") and facts.score is None:
                    facts.score = ScoreBreakdown(**out["score"])

    facts.strengths = _dedupe(facts.strengths)
    facts.weaknesses = _dedupe(facts.weaknesses)
    facts.changes = _dedupe(facts.changes)
    facts.questions = _dedupe(facts.questions)
    facts.citations = _dedupe(facts.citations)
    return facts


def _render_for_prompt(result: ToolResult) -> str:
    """Compact, prompt-safe rendering of a tool's output."""
    out = result.output
    lines = [result.summary]

    for key in (
        "corrected", "improved", "body", "subject", "model_answer", "verdict",
        "biggest_lever", "gap", "reader_reaction", "kept", "watch_out", "summary",
    ):
        value = out.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value[:900]}")

    for key in (
        "strengths", "weaknesses", "feedback", "changes", "quick_fixes",
        "citations", "followup_questions", "missing_specifics",
        "suggested_interview_questions",
    ):
        value = out.get(key)
        if isinstance(value, list) and value:
            rendered = "; ".join(str(v)[:180] for v in value[:6])
            lines.append(f"{key}: {rendered}")

    if out.get("score"):
        lines.append(f"score: {out['score']} (overall {out.get('overall_score')})")
    if out.get("star_coverage"):
        lines.append(f"star_coverage: {out['star_coverage']}")
    if out.get("issues"):
        rendered = "; ".join(
            f"{i.get('original')} -> {i.get('suggestion')}" for i in out["issues"][:8]
        )
        lines.append(f"grammar_issues: {rendered}")
    if out.get("passages"):
        rendered = "; ".join(
            f"{p['title']}: {p['content'][:200]}" for p in out["passages"][:3]
        )
        lines.append(f"reference_material: {rendered}")
    if out.get("questions"):
        rendered = "; ".join(
            (q["question"] if isinstance(q, dict) else str(q)) for q in out["questions"][:6]
        )
        lines.append(f"questions: {rendered}")

    return "\n".join(lines)


def _dedupe(items: list) -> list:
    seen: set[str] = set()
    out: list = []
    for item in items:
        key = str(item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
