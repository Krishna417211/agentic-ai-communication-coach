"""Intent detection.

Two paths that back each other up:
  * a weighted keyword/pattern classifier that always runs, and
  * an LLM classifier that runs when a provider is configured.

The heuristic result is passed to the LLM as a prior and used as the fallback if
the LLM is unavailable or returns something unrecognised. Short follow-ups
("make it shorter", "what about the second one?") inherit the session's last
intent rather than being misclassified in isolation.
"""

from __future__ import annotations

import logging
import re

from app.agent.extract import extract_all
from app.llm.base import LLMError, LLMProvider
from app.memory.store import Session
from app.schemas import INTENT_DESCRIPTIONS, Intent, IntentResult

logger = logging.getLogger(__name__)

# (pattern, weight) per intent. Weights let a strong signal beat several weak ones.
_PATTERNS: dict[Intent, list[tuple[str, float]]] = {
    Intent.EMAIL_WRITING: [
        (r"\be-?mail\b", 3.0), (r"\bsubject line\b", 3.0),
        (r"\bdraft (?:an?|my|this)\b", 2.0), (r"\bwrite (?:an?|my|this)\b", 1.5),
        (r"\breply to\b", 1.5), (r"\bfollow[- ]up\b", 1.2), (r"\bcc\b", 1.0),
        (r"\bsend (?:an?|this|it) to\b", 1.5), (r"\bmemo\b", 1.5),
        (r"\bresignation\b", 2.0), (r"\bcover letter\b", 2.5),
    ],
    Intent.INTERVIEW_PRACTICE: [
        (r"\binterview\b", 3.5), (r"\bhiring manager\b", 2.0),
        (r"\brecruiter\b", 1.5), (r"\bstar (?:method|format|framework)\b", 3.0),
        (r"\btell me about (?:a time|yourself)\b", 3.0),
        (r"\bmock interview\b", 3.5), (r"\bpractice (?:my )?answer\b", 2.5),
        (r"\bapplying for\b", 1.5), (r"\bwhy (?:do you want|this role)\b", 2.0),
        (r"\bgreatest (?:weakness|strength)\b", 2.5), (r"\bresume\b", 1.2),
        (r"\bcv\b", 1.2),
    ],
    Intent.GRAMMAR_CORRECTION: [
        (r"\bgrammar\b", 3.5), (r"\bspelling\b", 3.0), (r"\bproofread\b", 3.5),
        (r"\bpunctuation\b", 3.0), (r"\btypos?\b", 2.5),
        (r"\bis this (?:correct|right|grammatically)\b", 3.0),
        (r"\bcorrect (?:this|my|the)\b", 2.0), (r"\bmistakes?\b", 1.5),
        (r"\bfix (?:the |my )?(?:sentence|wording|english)\b", 2.5),
    ],
    Intent.TONE_IMPROVEMENT: [
        (r"\btone\b", 3.5), (r"\bsound (?:more|less)\b", 3.0),
        (r"\brude\b", 2.5), (r"\bharsh\b", 2.5), (r"\bpolite\b", 2.0),
        (r"\btoo (?:formal|casual|blunt|aggressive|soft|harsh|rude|pushy|direct)\b", 3.0),
        (r"\bmore (?:professional|friendly|formal|confident|assertive)\b", 2.5),
        (r"\b(?:how |does |do )?(?:this|it) sounds?\b", 3.0), (r"\bcome across\b", 2.0),
        (r"\bpassive[- ]aggressive\b", 3.0),
    ],
    Intent.PUBLIC_SPEAKING: [
        (r"\bpresent(?:ation|ing)\b", 3.5), (r"\bspeech\b", 3.5),
        (r"\bpublic speaking\b", 4.0), (r"\baudience\b", 1.5),
        (r"\bpitch\b", 2.5), (r"\bslides?\b", 2.0), (r"\bkeynote\b", 2.5),
        (r"\bstage\b", 2.0), (r"\bnervous (?:about )?(?:speaking|presenting)\b", 3.0),
        (r"\btalk\b(?=.*\b(?:conference|audience|minutes)\b)", 2.0),
        (r"\bstand[- ]?up\b", 1.2), (r"\btoastmasters\b", 3.0),
    ],
    Intent.CONFLICT_RESOLUTION: [
        (r"\bconflict\b", 3.5), (r"\bargument\b", 3.0), (r"\bdisagree\w*\b", 3.0),
        (r"\bconfront\w*\b", 3.0), (r"\bupset\w*\b", 2.0), (r"\bangry\b", 2.5),
        (r"\bstart(?:ing)? a fight\b", 3.5), (r"\bdismissive\b", 2.5),
        (r"\bcondescending\b", 2.5), (r"\bdefensive\b", 2.0),
        (r"\bdifficult conversation\b", 3.5), (r"\btension\b", 2.5),
        (r"\bpush(?:ing)? back\b", 2.0), (r"\bmy (?:boss|manager) (?:is|keeps|won't)\b", 2.0),
        (r"\bblam\w+\b", 2.5), (r"\bfrustrated with\b", 2.5),
        (r"\bcall (?:him|her|them) out\b", 2.5),
        # Accusatory phrasing is itself the strongest signal that the user is
        # in a conflict, even when they never use the word.
        (r"\byou (?:never|always)\b", 3.0), (r"\byour fault\b", 3.0),
        (r"\bunacceptable\b", 2.0), (r"\btook credit\b", 3.0),
        (r"\bthrew me under\b", 3.0), (r"\bignor(?:ed|es|ing) me\b", 2.5),
        (r"\bkeeps? (?:interrupting|undermining|dismissing)\b", 3.0),
    ],
    Intent.CUSTOMER_COMMUNICATION: [
        (r"\bcustomer\b", 3.0), (r"\bclient\b", 2.5), (r"\bsupport ticket\b", 3.5),
        (r"\bcomplaint\b", 3.0), (r"\brefund\b", 3.0), (r"\bapolog\w+\b", 2.0),
        (r"\bservice\b", 1.2), (r"\bescalat\w+\b", 2.0),
        (r"\bangry (?:customer|client)\b", 4.0), (r"\bsla\b", 2.0),
        (r"\boutage\b", 2.5), (r"\bchurn\b", 1.5),
    ],
}

#: How much a signal inside a pasted draft counts relative to the same signal
#: in the user's own instruction.
_DRAFT_SIGNAL_WEIGHT = 0.5

_COMPILED: dict[Intent, list[tuple[re.Pattern[str], float]]] = {
    intent: [(re.compile(p, re.IGNORECASE), w) for p, w in patterns]
    for intent, patterns in _PATTERNS.items()
}

# Messages that only make sense relative to the previous turn.
_FOLLOWUP_RE = re.compile(
    r"^\s*(?:and |but |ok(?:ay)?[,. ]|thanks[,. ]|yes[,. ]|no[,. ])?"
    r"(?:make it|can you|could you|now |what about|try again|again|shorter|longer|"
    r"more |less |redo|another|do that|same but|instead|also)\b",
    re.IGNORECASE,
)


class IntentDetector:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def detect(self, message: str, session: Session | None = None) -> IntentResult:
        entities = extract_all(message)
        heuristic = classify_heuristic(message, entities)

        if _is_followup(message) and session and session.last_intent:
            return IntentResult(
                intent=session.last_intent,
                confidence=0.72,
                rationale=(
                    "Short follow-up with no new topic signal; continuing the "
                    f"previous intent ({session.last_intent.value})."
                ),
                entities=entities,
                method="memory-carryover",
            )

        if not self._llm.supports_generation:
            return heuristic

        try:
            return await self._detect_llm(message, session, heuristic, entities)
        except (LLMError, ValueError) as exc:
            logger.warning("intent LLM classification failed, using heuristic: %s", exc)
            return heuristic

    async def _detect_llm(
        self,
        message: str,
        session: Session | None,
        heuristic: IntentResult,
        entities: dict,
    ) -> IntentResult:
        catalog = "\n".join(
            f"- {intent.value}: {desc}" for intent, desc in INTENT_DESCRIPTIONS.items()
        )
        history = ""
        if session and session.turns:
            recent = session.turns[-4:]
            history = "\n".join(
                f"{'User' if t.role == 'user' else 'Coach'}: {t.content[:200]}"
                for t in recent
            )
            history = f"\nRecent conversation:\n{history}\n"

        prompt = f"""Classify the user's communication-coaching request.

Available intents:
{catalog}
{history}
A keyword classifier guessed: {heuristic.intent.value} (confidence {heuristic.confidence}).
Disagree with it if the message clearly says otherwise.

USER MESSAGE:
\"\"\"{message}\"\"\"

Return JSON:
{{
  "intent": "one of the intent ids above",
  "confidence": 0.0-1.0,
  "rationale": "one sentence naming the deciding signal",
  "target_tone": "requested tone, or null",
  "audience": "who the message is for, or null",
  "role": "job role if this is interview prep, or null"
}}"""

        data = await self._llm.complete_json(
            prompt,
            system="You are an intent classifier. You output only valid JSON.",
            temperature=0.0,
            max_tokens=1200,
        )

        raw_intent = str(data.get("intent", "")).strip().lower()
        try:
            intent = Intent(raw_intent)
        except ValueError:
            logger.warning("LLM returned unknown intent %r", raw_intent)
            return heuristic

        merged = dict(entities)
        for key in ("target_tone", "audience", "role"):
            value = data.get(key)
            if value and str(value).lower() not in ("null", "none", ""):
                merged[key] = str(value)

        confidence = data.get("confidence", 0.8)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.8

        # Agreement between the two classifiers is itself evidence.
        if intent == heuristic.intent:
            confidence = min(1.0, confidence + 0.1)

        return IntentResult(
            intent=intent,
            confidence=confidence,
            rationale=str(data.get("rationale", "")),
            entities=merged,
            method="llm",
        )


def classify_heuristic(message: str, entities: dict | None = None) -> IntentResult:
    """Weighted pattern classifier — the always-available baseline."""
    entities = entities if entities is not None else extract_all(message)

    # What the user *asks* outranks what their pasted draft happens to contain.
    # "Does this sound too harsh? 'You never deliver on time'" is a tone
    # question, even though the draft is full of conflict vocabulary.
    draft = str(entities.get("target_text") or "")
    instruction = message.replace(draft, " ") if draft else message

    scores: dict[Intent, float] = {}
    for intent, patterns in _COMPILED.items():
        total = _score_intent(instruction, patterns)
        if draft:
            total += _DRAFT_SIGNAL_WEIGHT * _score_intent(draft, patterns)
        if total:
            scores[intent] = total

    if not scores:
        return IntentResult(
            intent=Intent.GENERAL_COACHING,
            confidence=0.4,
            rationale="No strong intent signals; treating as general coaching.",
            entities=entities,
            method="heuristic",
        )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_intent, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    # Confidence rises with absolute score and with the margin over second place.
    margin = (top_score - runner_up) / top_score
    confidence = min(0.95, 0.45 + 0.1 * min(top_score, 4.0) + 0.2 * margin)

    return IntentResult(
        intent=top_intent,
        confidence=round(confidence, 2),
        rationale=(
            f"Keyword score {top_score:.1f} for {top_intent.value}"
            + (f", ahead of {ranked[1][0].value} at {runner_up:.1f}." if runner_up else ".")
        ),
        entities=entities,
        method="heuristic",
    )


def _score_intent(
    message: str, patterns: list[tuple[re.Pattern[str], float]]
) -> float:
    """Sum pattern weights, counting each span of the message only once.

    Two patterns for the same intent often match the same words (`\\bemail\\b`
    and `\\be-?mail\\b`). Summing both would double-count a single signal and
    let it outrank a genuinely stronger intent, so overlapping matches collapse
    to the highest-weighted one.
    """
    matches: list[tuple[int, int, float]] = []
    for pattern, weight in patterns:
        for match in pattern.finditer(message):
            matches.append((match.start(), match.end(), weight))

    matches.sort(key=lambda m: m[2], reverse=True)
    claimed: list[tuple[int, int]] = []
    total = 0.0
    for start, end, weight in matches:
        if any(start < c_end and end > c_start for c_start, c_end in claimed):
            continue
        claimed.append((start, end))
        total += weight
    return total


def _is_followup(message: str) -> bool:
    words = message.split()
    return len(words) <= 12 and bool(_FOLLOWUP_RE.match(message))
