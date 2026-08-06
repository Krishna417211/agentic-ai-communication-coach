"""Communication scoring: five dimensions, 0-100 each.

Scores are deterministic so the same input always yields the same numbers,
and so progress across a coaching session is comparable turn to turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.nlp.grammar import GrammarIssue
from app.nlp.textstats import TextStats
from app.nlp.tone import ToneProfile
from app.schemas import ScoreBreakdown


@dataclass
class Assessment:
    score: ScoreBreakdown
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    @property
    def overall(self) -> int:
        return self.score.overall


def _clamp(value: float) -> int:
    return int(max(0, min(100, round(value))))


def score_text(
    stats: TextStats,
    tone: ToneProfile,
    issues: list[GrammarIssue],
    *,
    context: str = "general",
) -> Assessment:
    """Score a piece of communication and explain the result."""
    strengths: list[str] = []
    weaknesses: list[str] = []

    # ---- Grammar ------------------------------------------------------
    major = sum(1 for i in issues if i.severity == "major")
    minor = len(issues) - major
    grammar = 100 - (9 * major + 3 * minor)
    if not issues:
        strengths.append("No grammar or mechanical errors detected.")
    elif major:
        weaknesses.append(
            f"{major} significant grammar issue(s), e.g. \"{issues[0].original}\"."
        )
    elif minor:
        weaknesses.append(f"{minor} minor mechanical issue(s) (spacing, punctuation).")

    # ---- Clarity ------------------------------------------------------
    clarity = 82.0
    if stats.avg_sentence_length > 25:
        clarity -= 18
        weaknesses.append(
            f"Sentences average {stats.avg_sentence_length:.0f} words — aim for 15-20."
        )
    elif stats.avg_sentence_length > 20:
        clarity -= 8
    elif 8 <= stats.avg_sentence_length <= 18:
        clarity += 8
        strengths.append("Sentence length is in the easy-to-read range.")

    if stats.longest_sentence_length > 40:
        clarity -= 10
        weaknesses.append(
            f"Longest sentence is {stats.longest_sentence_length} words — split it."
        )

    clarity -= 4 * len(stats.filler_words)
    clarity -= 5 * len(stats.jargon)
    clarity -= 3 * len(stats.passive_phrases)
    if stats.filler_words:
        weaknesses.append(
            f"Filler words weaken the message: {', '.join(stats.filler_words[:4])}."
        )
    if stats.passive_phrases:
        weaknesses.append(
            f"Passive voice in {len(stats.passive_phrases)} place(s) — "
            f"e.g. \"{stats.passive_phrases[0]}\"."
        )
    if 45 <= stats.flesch_reading_ease <= 75:
        clarity += 6
    elif stats.flesch_reading_ease < 30:
        clarity -= 10
        weaknesses.append("Reading ease is low; the text is dense for a busy reader.")

    # ---- Tone ---------------------------------------------------------
    tone_score = (tone.politeness * 0.4 + tone.warmth * 0.3 + tone.assertiveness * 0.3)
    tone_score -= 12 * len(tone.risks)
    if tone.primary_tone == "aggressive":
        tone_score -= 20
        weaknesses.append("Tone reads as confrontational.")
    if tone.politeness >= 70 and tone.assertiveness >= 55:
        strengths.append("Polite and clear about what you need — a strong combination.")
    if tone.primary_tone == "hesitant":
        weaknesses.append("Hedging makes the message sound uncertain.")

    # ---- Structure ----------------------------------------------------
    structure = 60.0
    if stats.has_greeting:
        structure += 10
    if stats.has_signoff:
        structure += 8
    if stats.has_call_to_action:
        structure += 14
        strengths.append("Includes a clear ask or next step.")
    elif context == "general":
        # An interview answer or a speech has no "call to action" to be missing.
        weaknesses.append("No explicit call to action — the reader won't know what to do.")
    else:
        structure += 7  # don't penalise a context where a CTA doesn't belong
    if stats.paragraph_count >= 2 and stats.word_count > 80:
        structure += 8
        strengths.append("Broken into paragraphs rather than one block of text.")
    elif stats.word_count > 120 and stats.paragraph_count == 1:
        structure -= 12
        weaknesses.append("A single long block of text — break it into paragraphs.")
    if stats.word_count < 15:
        structure -= 10

    # ---- Impact -------------------------------------------------------
    impact = 62.0
    if stats.has_call_to_action:
        impact += 12
    impact += min(len(tone.signals.get("assertive", [])) * 6, 18)
    impact -= 6 * len(stats.hedges)
    impact -= 4 * len(stats.filler_words)
    if 40 <= stats.word_count <= 200:
        impact += 10
    elif stats.word_count > 350:
        impact -= 12
        weaknesses.append(
            f"At {stats.word_count} words this is long; busy readers skim past 200."
        )
    elif stats.word_count < 15:
        impact -= 8
        weaknesses.append("Too short to carry context — the reader has to guess.")
    if tone.sentiment == "positive":
        impact += 5

    if context == "interview":
        # Interview answers live or die on specificity and structure.
        impact -= 10 if stats.word_count < 60 else 0
        structure += 6 if stats.paragraph_count > 1 else 0
    elif context == "public_speaking":
        if stats.avg_sentence_length > 20:
            clarity -= 8
            weaknesses.append("Spoken delivery needs shorter sentences than writing.")
        if 60 <= stats.estimated_speaking_seconds <= 300:
            structure += 5

    breakdown = ScoreBreakdown(
        clarity=_clamp(clarity),
        tone=_clamp(tone_score),
        grammar=_clamp(grammar),
        structure=_clamp(structure),
        impact=_clamp(impact),
    )
    return Assessment(
        score=breakdown,
        strengths=_dedupe(strengths)[:5],
        weaknesses=_dedupe(weaknesses)[:6],
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
