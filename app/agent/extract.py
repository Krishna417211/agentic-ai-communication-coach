"""Entity extraction: pulling the draft, tone, audience and role out of a message.

Users mix instruction and content in one blob ("make this sound friendlier:
<draft>"). Separating them matters — the tools must operate on the draft, not on
the instruction wrapped around it.
"""

from __future__ import annotations

import re

_QUOTED_RE = re.compile(r"[\"“']{1,3}(.{25,}?)[\"”']{1,3}", re.DOTALL)
_LEAD_IN_RE = re.compile(
    r"(?:here'?s|this is|below is|check|review|fix|improve|rewrite|proofread|"
    r"look at|my draft|i wrote|i said|it says|feedback on)\b[^:\n]{0,60}[:\n]",
    re.IGNORECASE,
)

_TONE_RE = re.compile(
    r"\b(?:more|sound|make it|tone|feel|come across as|less)\s+(?:it\s+)?"
    r"(professional|formal|friendly|casual|polite|assertive|confident|concise|"
    r"empathetic|diplomatic|warm|direct|firm|apologetic)\b",
    re.IGNORECASE,
)
_TONE_WORD_RE = re.compile(
    r"\b(professional|formal|friendly|casual|polite|assertive|confident|concise|"
    r"empathetic|diplomatic|apologetic)\b",
    re.IGNORECASE,
)

_AUDIENCE_RE = re.compile(
    r"\b(?:to|for|with|at)\s+(my\s+(?:manager|boss|team|client|customer|colleague|"
    r"professor|landlord|coworker|co-worker|director|lead|mentor|recruiter|"
    r"supervisor|teammate)|the\s+(?:client|customer|team|board|ceo|cto|hiring\s+"
    r"manager|interviewer|panel)|a\s+(?:client|customer|recruiter|colleague))\b",
    re.IGNORECASE,
)

_ROLE_RE = re.compile(
    r"\b(?:interview(?:ing)?\s+for|applying\s+(?:for|to)|role\s+(?:of|as)|"
    r"position\s+(?:of|as)|as\s+an?)\s+([A-Za-z][A-Za-z /+\-]{2,40})",
    re.IGNORECASE,
)

_SYNONYM_TONE = {
    "warm": "friendly",
    "direct": "assertive",
    "firm": "assertive",
    "polite": "professional",
}


def extract_target_text(message: str) -> str:
    """Return the draft embedded in `message`, or "" if it is all instruction."""
    quoted = _QUOTED_RE.search(message)
    if quoted:
        candidate = quoted.group(1).strip()
        if len(candidate.split()) >= 5:
            return candidate

    lead_in = _LEAD_IN_RE.search(message)
    if lead_in:
        candidate = message[lead_in.end():].strip()
        if len(candidate.split()) >= 5:
            return candidate

    # A blank line usually separates instruction from pasted draft.
    if "\n\n" in message:
        head, _, tail = message.partition("\n\n")
        if len(head.split()) <= 30 and len(tail.split()) >= 8:
            return tail.strip()

    return ""


def extract_target_tone(message: str) -> str | None:
    match = _TONE_RE.search(message)
    if not match:
        # "rewrite this professionally" — a bare tone adjective near a verb.
        if re.search(r"\b(rewrite|improve|make|sound|redraft)\b", message, re.I):
            word = _TONE_WORD_RE.search(message)
            if word:
                tone = word.group(1).lower()
                return _SYNONYM_TONE.get(tone, tone)
        return None
    tone = match.group(1).lower()
    return _SYNONYM_TONE.get(tone, tone)


def extract_audience(message: str) -> str | None:
    match = _AUDIENCE_RE.search(message)
    return re.sub(r"\s+", " ", match.group(1)).strip().lower() if match else None


def extract_role(message: str) -> str | None:
    match = _ROLE_RE.search(message)
    if not match:
        return None
    role = re.sub(r"\s+", " ", match.group(1)).strip(" .,")
    # Trim trailing clause words the greedy pattern may have swallowed.
    role = re.split(r"\b(?:and|but|so|because|next|tomorrow|position|role)\b", role)[0]
    return role.strip() or None


def extract_all(message: str) -> dict[str, str | None]:
    return {
        "target_text": extract_target_text(message),
        "target_tone": extract_target_tone(message),
        "audience": extract_audience(message),
        "role": extract_role(message),
    }
