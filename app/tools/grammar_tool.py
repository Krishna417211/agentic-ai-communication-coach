"""Grammar correction tool."""

from __future__ import annotations

from app.nlp.grammar import apply_corrections, check_grammar
from app.schemas import ToolName, ToolResult
from app.tools.base import Tool, ToolContext


class GrammarCorrectionTool(Tool):
    name = ToolName.GRAMMAR_CORRECTION
    description = (
        "Finds and fixes grammar, spelling, punctuation and word-choice errors, "
        "returning a corrected version plus an explanation of each change."
    )
    when_to_use = (
        "the user asks whether something is correct, asks for proofreading, or "
        "submits a draft that contains mechanical errors."
    )

    def run_offline(self, ctx: ToolContext) -> ToolResult:
        text = ctx.text
        issues = check_grammar(text)
        corrected = apply_corrections(text, issues)
        return ToolResult(
            tool=self.name,
            summary=(
                f"Found {len(issues)} issue(s)." if issues
                else "No grammar issues detected."
            ),
            output={
                "original": text,
                "corrected": corrected,
                "issues": [i.as_dict() for i in issues],
                "issue_count": len(issues),
                "major_count": sum(1 for i in issues if i.severity == "major"),
                "engine": "rules",
            },
        )

    async def run_llm(self, ctx: ToolContext) -> ToolResult:
        text = ctx.text
        # Rule findings are passed in as grounding so the model doesn't miss
        # mechanical errors, and are merged back in case it does.
        rule_issues = check_grammar(text)
        rule_hint = (
            "A rule-based checker flagged: "
            + "; ".join(f"{i.original!r} -> {i.suggestion!r} ({i.message})"
                        for i in rule_issues[:12])
            if rule_issues
            else "A rule-based checker found no obvious mechanical errors."
        )

        prompt = f"""Proofread the text below. Correct grammar, spelling, punctuation,
tense and word choice. Preserve the writer's voice, meaning and level of detail —
do not rewrite for style, and do not add new content.

{rule_hint}

TEXT:
\"\"\"{text}\"\"\"

Return JSON:
{{
  "corrected": "the fully corrected text",
  "issues": [
    {{"original": "...", "suggestion": "...", "type": "spelling|grammar|punctuation|word_choice",
      "message": "one-line explanation", "severity": "minor|major"}}
  ],
  "clean": true|false
}}"""

        data = await ctx.llm.complete_json(
            prompt,
            system="You are a meticulous proofreader for professional English.",
            temperature=0.1,
        )

        issues = [i for i in data.get("issues", []) if isinstance(i, dict)]
        # Merge any rule finding the model didn't mention.
        seen = {str(i.get("original", "")).lower() for i in issues}
        for rule_issue in rule_issues:
            if rule_issue.original.lower() not in seen:
                issues.append(rule_issue.as_dict())

        corrected = str(data.get("corrected") or "").strip() or apply_corrections(
            text, rule_issues
        )
        return ToolResult(
            tool=self.name,
            summary=(
                f"Found {len(issues)} issue(s)." if issues
                else "No grammar issues detected."
            ),
            output={
                "original": text,
                "corrected": corrected,
                "issues": issues,
                "issue_count": len(issues),
                "major_count": sum(
                    1 for i in issues if str(i.get("severity")) == "major"
                ),
                "engine": ctx.llm.name,
            },
        )
