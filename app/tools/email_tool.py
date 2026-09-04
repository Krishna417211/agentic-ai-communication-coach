"""Email generation tool."""

from __future__ import annotations

import re

from app.nlp.rewrite import rewrite
from app.nlp.textstats import analyze_text
from app.schemas import ToolName, ToolResult
from app.tools.base import Tool, ToolContext

_SUBJECT_STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "and", "i", "my", "me", "please",
    "want", "need", "write", "email", "help", "with", "about", "can", "you",
    "would", "like", "some", "that", "this", "is", "are", "it", "on", "in",
    "hi", "hey", "hello", "dear", "just", "so", "am", "was", "were", "be",
    "have", "has", "had", "do", "does", "did", "how", "what", "should",
    "could", "wondering", "maybe", "really", "very", "draft", "message",
    "if", "but", "dont", "don't", "im", "i'm", "seem", "look", "sound",
}


class EmailGenerationTool(Tool):
    name = ToolName.EMAIL_GENERATION
    description = (
        "Drafts or redrafts a complete professional email — subject line, greeting, "
        "body and sign-off — matched to the audience and the requested tone."
    )
    when_to_use = (
        "the user wants to write, send, reply to or rewrite an email or a formal "
        "written message."
    )

    def run_offline(self, ctx: ToolContext) -> ToolResult:
        tone = str(ctx.options.get("target_tone") or "professional")
        audience = ctx.audience or "the recipient"
        greeting = f"Hi {audience.split()[-1].title()}," if ctx.audience else "Hi there,"

        if ctx.target_text:
            # The user pasted a draft — transform it.
            body, changes = rewrite(ctx.target_text, tone)
            subject = _derive_subject(ctx.goal or ctx.target_text)
            note = (
                "Built by transforming your draft. Set an API key for a fully "
                "rewritten email."
            )
        else:
            # The user described what they want — a rewrite of the *request*
            # would be nonsense, so emit a structured scaffold instead.
            body, changes = _scaffold(ctx.user_message, greeting)
            subject = _derive_subject(ctx.goal or ctx.user_message)
            # This runs both when no key is set and when a configured provider
            # failed, so it must not assert a cause it cannot know. The reason
            # is reported separately via the response's `degraded_reason`.
            note = (
                "This is a fill-in scaffold, not a finished email — it was "
                "written without an LLM. Configure a working provider key for "
                "a complete draft."
            )

        if not analyze_text(body).has_greeting:
            body = f"{greeting}\n\n{body}"
        if not analyze_text(body).has_signoff:
            body = f"{body.rstrip()}\n\nThanks,\n[Your name]"

        return ToolResult(
            tool=self.name,
            summary=f"Drafted a {tone} email: '{subject}'.",
            output={
                "subject": subject,
                "body": body,
                "email": f"Subject: {subject}\n\n{body}",
                "target_tone": tone,
                "changes": changes,
                "notes": [note],
                "engine": "rules",
            },
        )

    async def run_llm(self, ctx: ToolContext) -> ToolResult:
        source = ctx.target_text or ctx.user_message
        tone = str(ctx.options.get("target_tone") or "professional")
        audience = ctx.audience or "a professional colleague"
        goal = ctx.goal or "achieve what the user described"

        grammar_note = ""
        prior = ctx.artifacts.get(ToolName.GRAMMAR_CORRECTION.value)
        if prior and prior.get("issues"):
            grammar_note = (
                "\nThe user's draft had these mechanical errors — do not repeat them: "
                + "; ".join(
                    f"{i.get('original')} -> {i.get('suggestion')}"
                    for i in prior["issues"][:6]
                )
            )

        prompt = f"""Write a complete email.

Recipient: {audience}
Goal: {goal}
Required tone: {tone}
Session context:
{ctx.memory_brief()}
{ctx.knowledge_brief()}{grammar_note}

What the user gave you (their request, or their rough draft):
\"\"\"{source}\"\"\"

Rules:
- Subject line under 60 characters, specific, no clickbait.
- Open with the ask or the news; do not bury it under pleasantries.
- Short paragraphs. No filler, no jargon, no hedging.
- End with one clear call to action, with a date if a deadline is implied.
- Use [square brackets] for details you genuinely don't have. Never invent
  names, dates, figures or commitments.

Return JSON:
{{
  "subject": "...",
  "body": "greeting through sign-off, with real line breaks",
  "rationale": "two sentences on the choices you made",
  "notes": ["anything the user must fill in or double-check"]
}}"""

        data = await ctx.llm.complete_json(
            prompt,
            system=(
                "You are an executive communication coach who writes emails busy "
                "people actually reply to."
            ),
            temperature=0.5,
        )

        subject = str(data.get("subject") or _derive_subject(source)).strip()
        body = str(data.get("body") or "").strip()
        if not body:
            return self.run_offline(ctx)

        return ToolResult(
            tool=self.name,
            summary=f"Drafted a {tone} email: '{subject}'.",
            output={
                "subject": subject,
                "body": body,
                "email": f"Subject: {subject}\n\n{body}",
                "target_tone": tone,
                "rationale": str(data.get("rationale", "")),
                "notes": [str(n) for n in data.get("notes", [])][:5],
                "engine": ctx.llm.name,
            },
        )


def _scaffold(request: str, greeting: str) -> tuple[str, list[str]]:
    """A four-part email skeleton the user fills in, following the corpus rules."""
    ask = _derive_ask(request)
    body = (
        f"{greeting}\n\n"
        f"{ask}\n\n"
        "[One or two sentences of context — why now, and what the recipient "
        "needs to know to decide.]\n\n"
        "[If relevant: what you have already done, or what happens if this "
        "doesn't move.]\n\n"
        "Could you let me know by [day] whether that works?"
    )
    changes = [
        "Structured as ask -> context -> next step, so the request is in the "
        "first two lines rather than buried.",
        "Added one explicit call to action with a date.",
        "Left [brackets] where only you have the facts — nothing was invented.",
    ]
    return body, changes


def _derive_ask(request: str) -> str:
    """Turn the user's description of what they want into an opening line."""
    # Most specific cue first — a bare "to"/"for" would match the user's framing
    # ("I need to write an email...") instead of the actual ask.
    for pattern in (
        r"\b(?:asking|ask)\s+(?:for|about)\s+(.{8,90})",
        r"\b(?:request(?:ing)?|regarding|apologi[sz]e for)\s+(.{8,90})",
        r"\bi (?:need|want)\s+(?:to\s+)?(?!write|send|draft|compose)(.{8,90})",
        r"\babout\s+(.{8,90})",
    ):
        match = re.search(pattern, request, re.IGNORECASE)
        if not match:
            continue
        fragment = re.split(r"[.!?\n]|\bbut\b|\bbecause\b|\bi don'?t\b", match.group(1))[0]
        fragment = fragment.strip().rstrip(",;")
        if len(fragment.split()) >= 3:
            return f"I'd like to ask about {fragment}."
    return "[State your ask in one sentence — what you need, and by when.]"


def _derive_subject(text: str) -> str:
    """Pick a serviceable subject line from the request text."""
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    words = [w for w in re.findall(r"[A-Za-z0-9'-]+", first_line)]
    keywords = [w for w in words if w.lower() not in _SUBJECT_STOPWORDS]
    if not keywords:
        return "Following up"
    subject = " ".join(keywords[:7])
    return subject[:58].strip().capitalize()
