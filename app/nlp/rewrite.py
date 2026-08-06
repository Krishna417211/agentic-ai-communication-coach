"""Deterministic rewriting used when no LLM is configured.

This is a transformation, not generation: it edits what the user actually
wrote — fixing mechanics, cutting hedges and fillers, de-escalating accusatory
phrasing, and adjusting register toward the requested tone.
"""

from __future__ import annotations

import re

from app.nlp.grammar import apply_corrections, check_grammar
from app.nlp.textstats import FILLER_WORDS, analyze_text

# Phrases replaced regardless of target tone.
# Replacements capture the following verb so the rewritten sentence stays
# grammatical — a bare phrase swap produces things like
# "I haven't yet seen send the numbers on time".
_DE_ESCALATION: list[tuple[str, str]] = [
    (r"\byou always\s+(\w+)", r"I've noticed you often \1"),
    (r"\byou never\s+(\w+)", r"I haven't reliably seen you \1"),
    (r"\byou failed to\s+(\w+)", r"I didn't see you \1"),
    (r"\byou should have\s+(\w+)", r"it would have helped to \1"),
    (r"\bclearly you\s+(\w+)", r"it looks like you \1"),
    (r"\bit'?s\s+(?:obviously\s+)?your fault\b", "this went wrong somewhere between us"),
    (r"\byour fault\b", "something that went wrong here"),
    (r"\bthis is unacceptable\b", "this isn't workable for us"),
    (r"\bthis is ridiculous\b", "this is a real problem for us"),
    (r"\bobviously\s+", ""),
    (r"\bi demand\b", "I'd like to request"),
    (r"\bhow many times\b", "I wanted to raise again"),
    (r"\byet again\b", "again"),
    (r"\buseless\b", "not working for us"),
    (r"\bincompetent\b", "not meeting the standard we expected"),
]

_HEDGE_REMOVALS: list[tuple[str, str]] = [
    (r"\bi just wanted to\b", "I'd like to"),
    (r"\bi was just wondering if\b", "could you"),
    (r"\bi was wondering if\b", "could you"),
    (r"\bjust wondering\b", ""),
    (r"\bsorry to bother you,?\s*but\b", ""),
    (r"\bsorry to bother you\b", ""),
    (r"\bi'?m not sure if\b", ""),
    (r"\bi think maybe\b", "I think"),
    (r"\bi guess\b", ""),
    (r"\bkind of\b", ""),
    (r"\bsort of\b", ""),
    (r"\bif that'?s ok\b", ""),
    (r"\bif that makes sense\b", ""),
    (r"\bhopefully\b", ""),
    (r"\bi could be wrong,? but\b", ""),
]

_JARGON_SWAPS: list[tuple[str, str]] = [
    (r"\bcircle back\b", "follow up"),
    (r"\btouch base\b", "check in"),
    (r"\bleverage\b", "use"),
    (r"\bsynergy\b", "shared benefit"),
    (r"\bbandwidth\b", "capacity"),
    (r"\bmoving forward\b", "from now on"),
    (r"\bgoing forward\b", "from now on"),
    (r"\bat the end of the day\b", "ultimately"),
    (r"\blow-hanging fruit\b", "the quick wins"),
    (r"\bdeep dive\b", "detailed review"),
    (r"\bmove the needle\b", "make a measurable difference"),
    (r"\bdo the needful\b", "take care of this"),
    (r"\brevert back\b", "reply"),
]

_FORMALISE: list[tuple[str, str]] = [
    (r"\bhey\b", "Hello"), (r"\byeah\b", "yes"), (r"\byep\b", "yes"),
    (r"\bnope\b", "no"), (r"\bgonna\b", "going to"), (r"\bwanna\b", "want to"),
    (r"\bgotta\b", "need to"), (r"\bkinda\b", "somewhat"),
    (r"\basap\b", "as soon as possible"), (r"\bthx\b", "thank you"),
    (r"\bguys\b", "everyone"), (r"\bstuff\b", "items"),
    (r"\bcan't\b", "cannot"), (r"\bwon't\b", "will not"),
    (r"\bdon't\b", "do not"), (r"\bdoesn't\b", "does not"),
    (r"\bdidn't\b", "did not"), (r"\bi'm\b", "I am"), (r"\bit's\b", "it is"),
    (r"\bwe've\b", "we have"), (r"\bi'd\b", "I would"), (r"\bi'll\b", "I will"),
]

_SOFTEN: list[tuple[str, str]] = [
    (r"\byou need to\b", "could you"),
    (r"\byou must\b", "it would help if you could"),
    (r"\bsend me\b", "could you send me"),
    (r"\bimmediately\b", "as soon as you're able"),
]

_STRENGTHEN: list[tuple[str, str]] = [
    (r"\bi think we should\b", "I recommend we"),
    (r"\bmaybe we could\b", "let's"),
    (r"\bi feel like\b", "I've found that"),
    (r"\bit might be good to\b", "we should"),
    (r"\bperhaps\b", ""),
]

_TONE_PROFILES: dict[str, list[list[tuple[str, str]]]] = {
    "professional": [_HEDGE_REMOVALS, _JARGON_SWAPS, _FORMALISE],
    "formal": [_HEDGE_REMOVALS, _JARGON_SWAPS, _FORMALISE],
    "friendly": [_HEDGE_REMOVALS, _JARGON_SWAPS],
    "empathetic": [_DE_ESCALATION, _JARGON_SWAPS, _SOFTEN],
    "diplomatic": [_DE_ESCALATION, _JARGON_SWAPS, _SOFTEN],
    "assertive": [_HEDGE_REMOVALS, _JARGON_SWAPS, _STRENGTHEN],
    "confident": [_HEDGE_REMOVALS, _JARGON_SWAPS, _STRENGTHEN],
    "concise": [_HEDGE_REMOVALS, _JARGON_SWAPS],
    "apologetic": [_DE_ESCALATION, _SOFTEN],
}


def rewrite(text: str, target_tone: str = "professional") -> tuple[str, list[str]]:
    """Rewrite `text` toward `target_tone`. Returns (improved, changes)."""
    changes: list[str] = []
    working = text

    issues = check_grammar(working)
    if issues:
        working = apply_corrections(working, issues)
        changes.append(f"Fixed {len(issues)} grammar/mechanics issue(s).")

    # De-escalation always runs — accusatory phrasing hurts in every register.
    working, hits = _apply(working, _DE_ESCALATION)
    if hits:
        changes.append(f"Replaced accusatory phrasing: {', '.join(hits[:3])}.")

    tone_key = target_tone.strip().lower()
    for table in _TONE_PROFILES.get(tone_key, [_HEDGE_REMOVALS, _JARGON_SWAPS]):
        working, hits = _apply(working, table)
        if not hits:
            continue
        if table is _HEDGE_REMOVALS:
            changes.append(f"Removed hedging: {', '.join(hits[:3])}.")
        elif table is _JARGON_SWAPS:
            changes.append(f"Replaced jargon: {', '.join(hits[:3])}.")
        elif table is _FORMALISE:
            changes.append("Raised register: expanded contractions and casual wording.")
        elif table is _SOFTEN:
            changes.append("Softened directives into requests.")
        elif table is _STRENGTHEN:
            changes.append("Turned tentative phrasing into direct recommendations.")

    working, removed = _strip_fillers(working)
    if removed:
        changes.append(f"Cut filler words: {', '.join(removed[:4])}.")

    working = _tidy(working)

    # The substitutions above can leave mechanical errors behind (a dropped
    # word, an "its" whose sentence changed shape), so sweep once more.
    residual = check_grammar(working)
    if residual:
        working = apply_corrections(working, residual)

    stats = analyze_text(working)
    if not stats.has_call_to_action and stats.word_count > 20:
        working = working.rstrip()
        working += "\n\nCould you let me know by end of day Thursday whether that works?"
        changes.append("Added an explicit call to action with a date.")

    if not stats.has_greeting and stats.word_count > 30 and "\n" in text:
        working = "Hi there,\n\n" + working
        changes.append("Added a greeting.")

    stats_after = analyze_text(working)
    if not stats_after.has_signoff and stats_after.word_count > 40:
        working = working.rstrip() + "\n\nThanks,\n"
        changes.append("Added a sign-off.")

    if not changes:
        changes.append("No mechanical changes needed — the text was already clean.")
    return working.strip(), changes


def _apply(text: str, table: list[tuple[str, str]]) -> tuple[str, list[str]]:
    hits: list[str] = []
    for pattern, replacement in table:
        compiled = re.compile(pattern, re.IGNORECASE)
        match = compiled.search(text)
        if not match:
            continue
        hits.append(match.group(0).strip())
        text = compiled.sub(replacement, text)
    return text, hits


_KEEP_FILLERS = {"things", "stuff", "kind", "sort"}  # handled by other tables


def _strip_fillers(text: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    for filler in sorted(FILLER_WORDS - _KEEP_FILLERS):
        pattern = re.compile(rf"\b{re.escape(filler)}\s+", re.IGNORECASE)
        if pattern.search(text):
            removed.append(filler)
            text = pattern.sub("", text, count=2)
    return text, removed


def _tidy(text: str) -> str:
    """Repair whitespace, punctuation and capitalisation after substitutions."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(^|[.!?]\s+|\n)\s*([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    return text.strip()
