"""Tone analysis tool."""

from __future__ import annotations

from app.nlp.textstats import analyze_text
from app.nlp.tone import TONE_LABELS, analyze_tone
from app.schemas import ToolName, ToolResult
from app.tools.base import Tool, ToolContext


class ToneAnalysisTool(Tool):
    name = ToolName.TONE_ANALYSIS
    description = (
        "Analyses the tone of a message across formality, politeness, warmth and "
        "assertiveness, names how it will land with the reader, and flags phrasing "
        "that could escalate or undermine the message."
    )
    when_to_use = (
        "the user asks how something sounds, wants a message to be more polite / "
        "firm / friendly, or is writing into a sensitive situation."
    )

    def run_offline(self, ctx: ToolContext) -> ToolResult:
        text = ctx.text
        stats = analyze_text(text)
        tone = analyze_tone(text, stats)
        target = str(ctx.options.get("target_tone") or "professional")

        return ToolResult(
            tool=self.name,
            summary=f"Tone reads as {tone.primary_tone} (target: {target}).",
            output={
                "tone": tone.as_dict(),
                "stats": stats.as_dict(),
                "target_tone": target,
                "gap": _describe_gap(tone.as_dict(), target),
                "engine": "rules",
            },
        )

    async def run_llm(self, ctx: ToolContext) -> ToolResult:
        text = ctx.text
        stats = analyze_text(text)
        tone = analyze_tone(text, stats)
        target = str(ctx.options.get("target_tone") or "professional")
        audience = ctx.audience or "a professional colleague"

        prompt = f"""Analyse how the message below will land with {audience}.

Measured signals (computed, trust these numbers):
- formality {tone.formality}/100, politeness {tone.politeness}/100,
  warmth {tone.warmth}/100, assertiveness {tone.assertiveness}/100
- hedges: {stats.hedges or "none"}
- filler words: {stats.filler_words or "none"}
- passive constructions: {len(stats.passive_phrases)}
- exclamation marks: {stats.exclamations}

Target tone: {target}

MESSAGE:
\"\"\"{text}\"\"\"

Return JSON:
{{
  "primary_tone": "one of {list(TONE_LABELS)}",
  "secondary_tones": ["..."],
  "reader_reaction": "one or two sentences on how the reader is likely to feel",
  "risks": ["specific phrases that could misfire, and why"],
  "gap": "what stands between the current tone and the target tone",
  "quick_fixes": ["concrete phrase-level swaps, at most 4"]
}}"""

        data = await ctx.llm.complete_json(
            prompt,
            system=(
                "You are a communication coach who reads tone precisely and "
                "always points at specific words rather than giving vague advice."
            ),
            temperature=0.3,
        )

        merged = tone.as_dict()
        if data.get("primary_tone") in TONE_LABELS:
            merged["primary_tone"] = data["primary_tone"]
        if isinstance(data.get("secondary_tones"), list):
            merged["secondary_tones"] = [str(t) for t in data["secondary_tones"]][:3]
        llm_risks = [str(r) for r in data.get("risks", []) if r]
        merged["risks"] = list(dict.fromkeys(tone.risks + llm_risks))[:6]

        return ToolResult(
            tool=self.name,
            summary=f"Tone reads as {merged['primary_tone']} (target: {target}).",
            output={
                "tone": merged,
                "stats": stats.as_dict(),
                "target_tone": target,
                "reader_reaction": str(data.get("reader_reaction", "")),
                "gap": str(data.get("gap") or _describe_gap(merged, target)),
                "quick_fixes": [str(f) for f in data.get("quick_fixes", [])][:4],
                "engine": ctx.llm.name,
            },
        )


_TARGET_BANDS: dict[str, dict[str, tuple[int, int]]] = {
    "professional": {"formality": (55, 85), "politeness": (60, 95), "assertiveness": (50, 80)},
    "formal": {"formality": (75, 100), "politeness": (65, 100), "assertiveness": (45, 75)},
    "friendly": {"formality": (30, 60), "warmth": (65, 100), "politeness": (60, 100)},
    "empathetic": {"warmth": (70, 100), "politeness": (70, 100)},
    "assertive": {"assertiveness": (70, 100), "politeness": (45, 85)},
    "concise": {"assertiveness": (60, 100)},
}


def _describe_gap(tone: dict, target: str) -> str:
    bands = _TARGET_BANDS.get(target.lower())
    if not bands:
        return f"No calibration data for target tone '{target}'."
    gaps: list[str] = []
    for dimension, (low, high) in bands.items():
        value = int(tone.get(dimension, 50))
        if value < low:
            gaps.append(f"{dimension} is {value}, below the {low}-{high} target band")
        elif value > high:
            gaps.append(f"{dimension} is {value}, above the {low}-{high} target band")
    if not gaps:
        return f"Tone already sits in the '{target}' range."
    return "; ".join(gaps) + "."
