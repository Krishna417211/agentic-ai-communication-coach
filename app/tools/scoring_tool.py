"""Communication scoring tool."""

from __future__ import annotations

from app.nlp.grammar import check_grammar
from app.nlp.scoring import score_text
from app.nlp.textstats import analyze_text
from app.nlp.tone import analyze_tone
from app.schemas import Intent, ScoreBreakdown, ToolName, ToolResult
from app.tools.base import Tool, ToolContext

_CONTEXT_BY_INTENT = {
    Intent.INTERVIEW_PRACTICE: "interview",
    Intent.PUBLIC_SPEAKING: "public_speaking",
}


class CommunicationScoringTool(Tool):
    name = ToolName.COMMUNICATION_SCORING
    description = (
        "Scores a message 0-100 on clarity, tone, grammar, structure and impact, "
        "with the specific evidence behind each number."
    )
    when_to_use = (
        "the user asks how good their communication is, wants a rating, or at the "
        "end of any coaching turn to quantify the starting point."
    )

    def run_offline(self, ctx: ToolContext) -> ToolResult:
        text = ctx.text
        context = _CONTEXT_BY_INTENT.get(ctx.intent, "general")

        stats = analyze_text(text)
        tone = analyze_tone(text, stats)
        issues = check_grammar(text)
        assessment = score_text(stats, tone, issues, context=context)

        # If a rewrite already ran, score it too so the user sees the delta.
        after: ScoreBreakdown | None = None
        improved = _improved_text(ctx)
        if improved and improved.strip() != text.strip():
            i_stats = analyze_text(improved)
            i_tone = analyze_tone(improved, i_stats)
            i_issues = check_grammar(improved)
            after = score_text(i_stats, i_tone, i_issues, context=context).score

        return ToolResult(
            tool=self.name,
            summary=f"Overall communication score: {assessment.overall}/100.",
            output={
                "score": assessment.score.model_dump(),
                "overall_score": assessment.overall,
                "score_after_improvement": after.model_dump() if after else None,
                "overall_after_improvement": after.overall if after else None,
                "strengths": assessment.strengths,
                "weaknesses": assessment.weaknesses,
                "stats": stats.as_dict(),
                "tone": tone.as_dict(),
                "grammar_issue_count": len(issues),
                "context": context,
                "engine": "rules",
            },
        )

    async def run_llm(self, ctx: ToolContext) -> ToolResult:
        # Scores stay deterministic; the LLM only adds qualitative commentary
        # so the same text never scores differently on two runs.
        result = self.run_offline(ctx)
        if not result.ok:
            return result

        score = result.output["score"]
        prompt = f"""A rule-based scorer rated this message. Explain the result to
the writer in plain language and add anything the scorer could not see —
audience fit, missing context, unstated assumptions, whether the message
achieves its purpose at all.

Scores: {score} (overall {result.output['overall_score']}/100)
Detected strengths: {result.output['strengths']}
Detected weaknesses: {result.output['weaknesses']}

MESSAGE:
\"\"\"{ctx.text}\"\"\"

Return JSON:
{{
  "verdict": "two sentences: would this achieve its purpose, yes or no, and why",
  "strengths": ["at most 3"],
  "weaknesses": ["at most 4, each pointing at a specific phrase"],
  "biggest_lever": "the single change that would most improve this message"
}}"""

        data = await ctx.llm.complete_json(
            prompt,
            system="You are a communication coach who explains scores concretely.",
            temperature=0.3,
        )

        llm_strengths = [str(s) for s in data.get("strengths", []) if s]
        llm_weaknesses = [str(w) for w in data.get("weaknesses", []) if w]
        result.output.update(
            {
                "verdict": str(data.get("verdict", "")),
                "biggest_lever": str(data.get("biggest_lever", "")),
                "strengths": list(
                    dict.fromkeys(llm_strengths + result.output["strengths"])
                )[:5],
                "weaknesses": list(
                    dict.fromkeys(llm_weaknesses + result.output["weaknesses"])
                )[:6],
                "engine": ctx.llm.name,
            }
        )
        return result


def _improved_text(ctx: ToolContext) -> str:
    """Pull a rewritten version out of earlier steps, if one exists."""
    for tool_name in (
        ToolName.CONVERSATION_IMPROVEMENT.value,
        ToolName.EMAIL_GENERATION.value,
        ToolName.GRAMMAR_CORRECTION.value,
    ):
        artifact = ctx.artifacts.get(tool_name)
        if not artifact:
            continue
        for key in ("improved", "body", "corrected"):
            value = artifact.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""
