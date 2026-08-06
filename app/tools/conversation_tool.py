"""Conversation improvement tool — rewrites a message toward a target outcome."""

from __future__ import annotations

from app.nlp.rewrite import rewrite
from app.nlp.textstats import analyze_text
from app.nlp.tone import analyze_tone
from app.schemas import Intent, ToolName, ToolResult
from app.tools.base import Tool, ToolContext

# Intent-specific guidance injected into the rewrite prompt.
_PLAYBOOKS: dict[Intent, str] = {
    Intent.CONFLICT_RESOLUTION: (
        "Use a de-escalation structure: acknowledge the other person's position, "
        "state the impact using 'I' statements rather than 'you' accusations, "
        "separate the person from the problem, and propose one concrete next step. "
        "Strip every phrase that assigns blame."
    ),
    Intent.CUSTOMER_COMMUNICATION: (
        "Lead with acknowledgement of the customer's experience, then what you are "
        "doing about it, then when they will hear back. Never blame the customer, "
        "never hide behind policy language, and commit only to what is stated."
    ),
    Intent.PUBLIC_SPEAKING: (
        "Rewrite for the ear, not the eye: short sentences, concrete images, "
        "signposting between sections, and a memorable closing line. Mark natural "
        "pauses with a line break. Keep the speaker's own voice."
    ),
    Intent.TONE_IMPROVEMENT: (
        "Change only register and phrasing. Preserve every fact, request and "
        "commitment exactly as written."
    ),
    Intent.EMAIL_WRITING: (
        "Keep the email structure: subject, greeting, body, single clear ask, sign-off."
    ),
}


class ConversationImprovementTool(Tool):
    name = ToolName.CONVERSATION_IMPROVEMENT
    description = (
        "Rewrites a message or conversation turn into a stronger version and "
        "explains each change, adapting to conflict, customer, presentation or "
        "general professional contexts."
    )
    when_to_use = (
        "the user wants their wording improved, needs to handle a difficult or "
        "sensitive conversation, or asks 'how should I say this?'."
    )

    def run_offline(self, ctx: ToolContext) -> ToolResult:
        text = ctx.text
        target = str(ctx.options.get("target_tone") or _default_tone(ctx.intent))
        improved, changes = rewrite(text, target)

        before = analyze_text(text)
        after = analyze_text(improved)
        return ToolResult(
            tool=self.name,
            summary=f"Rewrote the message for a {target} tone.",
            output={
                "original": text,
                "improved": improved,
                "changes": changes,
                "target_tone": target,
                "before": before.as_dict(),
                "after": after.as_dict(),
                "engine": "rules",
            },
        )

    async def run_llm(self, ctx: ToolContext) -> ToolResult:
        text = ctx.text
        target = str(ctx.options.get("target_tone") or _default_tone(ctx.intent))
        audience = ctx.audience or "a professional colleague"
        stats = analyze_text(text)
        tone = analyze_tone(text, stats)
        playbook = _PLAYBOOKS.get(ctx.intent, "")

        tone_note = ""
        prior = ctx.artifacts.get(ToolName.TONE_ANALYSIS.value)
        if prior and prior.get("tone", {}).get("risks"):
            tone_note = "\nTone risks already identified: " + "; ".join(
                str(r) for r in prior["tone"]["risks"][:4]
            )

        prompt = f"""Rewrite the message below so it lands better with {audience}.

Target tone: {target}
{playbook}

Measured problems in the original (fix these):
- hedges: {stats.hedges or "none"}
- filler words: {stats.filler_words or "none"}
- passive constructions: {stats.passive_phrases or "none"}
- jargon: {stats.jargon or "none"}
- current tone: {tone.primary_tone} (politeness {tone.politeness}, assertiveness {tone.assertiveness})
- clear call to action present: {stats.has_call_to_action}{tone_note}

Session context:
{ctx.memory_brief()}
{ctx.knowledge_brief()}

ORIGINAL:
\"\"\"{text}\"\"\"

Rules: keep the writer's voice and every fact they stated. Do not invent names,
dates, numbers or commitments — use [brackets] for anything you don't know.

Return JSON:
{{
  "improved": "the rewritten message",
  "changes": ["what you changed and why, one line each, at most 6"],
  "kept": "what was already working and you deliberately preserved",
  "watch_out": "the one habit this writer should work on next"
}}"""

        data = await ctx.llm.complete_json(
            prompt,
            system=(
                "You are a communication coach. You improve what the writer wrote "
                "rather than replacing it with generic corporate prose."
            ),
            temperature=0.5,
        )

        improved = str(data.get("improved") or "").strip()
        if not improved:
            return self.run_offline(ctx)

        return ToolResult(
            tool=self.name,
            summary=f"Rewrote the message for a {target} tone.",
            output={
                "original": text,
                "improved": improved,
                "changes": [str(c) for c in data.get("changes", [])][:6],
                "kept": str(data.get("kept", "")),
                "watch_out": str(data.get("watch_out", "")),
                "target_tone": target,
                "before": stats.as_dict(),
                "after": analyze_text(improved).as_dict(),
                "engine": ctx.llm.name,
            },
        )


def _default_tone(intent: Intent) -> str:
    return {
        Intent.CONFLICT_RESOLUTION: "diplomatic",
        Intent.CUSTOMER_COMMUNICATION: "empathetic",
        Intent.PUBLIC_SPEAKING: "confident",
        Intent.INTERVIEW_PRACTICE: "confident",
    }.get(intent, "professional")
