"""Lexicon-based tone and sentiment analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.nlp.textstats import TextStats

_FORMAL_MARKERS = {
    "regarding", "furthermore", "however", "therefore", "accordingly", "pursuant",
    "kindly", "sincerely", "respectfully", "please find", "i would appreciate",
    "with reference to", "as per", "hereby", "additionally", "consequently",
    "i am writing to", "should you", "at your earliest convenience",
}

_CASUAL_MARKERS = {
    "hey", "yeah", "yep", "nope", "gonna", "wanna", "gotta", "kinda", "sorta",
    "cool", "awesome", "stuff", "guys", "lol", "btw", "asap", "ok", "okay",
    "no worries", "sure thing", "cheers", "thanks a ton", "super",
}

_POSITIVE = {
    "thank", "thanks", "appreciate", "great", "excellent", "happy", "glad",
    "pleased", "wonderful", "delighted", "excited", "success", "helpful",
    "grateful", "impressive", "value", "welcome", "congratulations", "opportunity",
}

_NEGATIVE = {
    "unfortunately", "problem", "issue", "delay", "fail", "failed", "wrong",
    "bad", "poor", "disappointed", "frustrated", "unacceptable", "concern",
    "mistake", "error", "complaint", "unhappy", "sorry", "difficult", "unable",
}

_AGGRESSIVE = {
    "immediately", "unacceptable", "ridiculous", "incompetent", "obviously",
    "clearly you", "you failed", "you never", "you always", "you should have",
    "your fault", "i demand", "this is nonsense", "useless", "pathetic",
    "how many times", "yet again", "for the last time",
}

_EMPATHETIC = {
    "i understand", "i hear you", "that makes sense", "i can see why",
    "thank you for your patience", "i appreciate", "i'm sorry to hear",
    "i realise", "i realize", "let me help", "happy to help", "i know this",
    "we value", "your experience", "how can i", "i apologise", "i apologize",
}

_URGENT = {
    "urgent", "asap", "immediately", "right away", "critical", "deadline",
    "today", "eod", "time-sensitive", "blocking", "escalate",
}

_ASSERTIVE = {
    "i recommend", "i will", "we will", "i need", "let's", "i propose",
    "my recommendation", "the next step", "i've decided", "i suggest",
    "i'd like to", "i can", "here's what",
}

TONE_LABELS = (
    "formal", "casual", "positive", "negative", "aggressive",
    "empathetic", "urgent", "assertive", "hesitant", "neutral",
)


@dataclass
class ToneProfile:
    primary_tone: str = "neutral"
    secondary_tones: list[str] = field(default_factory=list)
    formality: int = 50       # 0 = very casual, 100 = very formal
    politeness: int = 50
    warmth: int = 50
    assertiveness: int = 50
    sentiment: str = "neutral"  # positive | neutral | negative
    signals: dict[str, list[str]] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "primary_tone": self.primary_tone,
            "secondary_tones": self.secondary_tones,
            "formality": self.formality,
            "politeness": self.politeness,
            "warmth": self.warmth,
            "assertiveness": self.assertiveness,
            "sentiment": self.sentiment,
            "signals": self.signals,
            "risks": self.risks,
        }


def _found(lowered: str, lexicon: set[str]) -> list[str]:
    hits = []
    for term in lexicon:
        pattern = r"\b" + re.escape(term) + r"\b" if " " not in term else re.escape(term)
        if re.search(pattern, lowered):
            hits.append(term)
    return sorted(hits)


def _clamp(value: float) -> int:
    return int(max(0, min(100, round(value))))


def analyze_tone(text: str, stats: TextStats) -> ToneProfile:
    """Score tone dimensions from lexical signals plus structural stats."""
    lowered = text.lower()
    profile = ToneProfile()

    signals = {
        "formal": _found(lowered, _FORMAL_MARKERS),
        "casual": _found(lowered, _CASUAL_MARKERS),
        "positive": _found(lowered, _POSITIVE),
        "negative": _found(lowered, _NEGATIVE),
        "aggressive": _found(lowered, _AGGRESSIVE),
        "empathetic": _found(lowered, _EMPATHETIC),
        "urgent": _found(lowered, _URGENT),
        "assertive": _found(lowered, _ASSERTIVE),
    }
    profile.signals = {k: v for k, v in signals.items() if v}

    # --- Formality ----------------------------------------------------
    formality = 50 + 9 * len(signals["formal"]) - 11 * len(signals["casual"])
    if stats.has_greeting and stats.has_signoff:
        formality += 8
    if re.search(r"\b(can't|won't|don't|i'm|it's|you're|we've)\b", lowered):
        formality -= 6
    if stats.exclamations > 1:
        formality -= 5 * min(stats.exclamations, 3)
    if stats.avg_sentence_length > 22:
        formality += 5
    profile.formality = _clamp(formality)

    # --- Politeness ---------------------------------------------------
    politeness = 55
    politeness += 6 * len(_found(lowered, {"please", "thank you", "thanks",
                                           "would you mind", "if you could",
                                           "i'd appreciate", "apologies"}))
    politeness += 5 * len(signals["empathetic"])
    politeness -= 14 * len(signals["aggressive"])
    politeness -= 4 * len(stats.all_caps_words)
    if stats.questions == 0 and len(signals["urgent"]) >= 2:
        politeness -= 8
    profile.politeness = _clamp(politeness)

    # --- Warmth -------------------------------------------------------
    warmth = 50 + 7 * len(signals["positive"]) + 8 * len(signals["empathetic"])
    warmth -= 8 * len(signals["negative"]) + 15 * len(signals["aggressive"])
    if stats.has_greeting:
        warmth += 6
    profile.warmth = _clamp(warmth)

    # --- Assertiveness -------------------------------------------------
    assertiveness = 50 + 9 * len(signals["assertive"])
    assertiveness -= 8 * len(stats.hedges)
    assertiveness -= 3 * len(stats.filler_words)
    assertiveness -= 5 * len(stats.passive_phrases)
    if stats.has_call_to_action:
        assertiveness += 8
    profile.assertiveness = _clamp(assertiveness)

    # --- Sentiment ------------------------------------------------------
    positive_score = len(signals["positive"]) + len(signals["empathetic"])
    negative_score = len(signals["negative"]) + 2 * len(signals["aggressive"])
    if positive_score > negative_score + 1:
        profile.sentiment = "positive"
    elif negative_score > positive_score + 1:
        profile.sentiment = "negative"

    # --- Labels ---------------------------------------------------------
    ranked: list[tuple[str, float]] = [
        ("aggressive", 3.0 * len(signals["aggressive"])),
        ("empathetic", 2.0 * len(signals["empathetic"])),
        ("urgent", 1.8 * len(signals["urgent"])),
        ("formal", 1.2 * len(signals["formal"]) + (profile.formality - 50) / 25),
        ("casual", 1.2 * len(signals["casual"]) + (50 - profile.formality) / 25),
        ("positive", 1.0 * len(signals["positive"])),
        ("negative", 1.0 * len(signals["negative"])),
        ("assertive", 1.0 * len(signals["assertive"]) + (profile.assertiveness - 50) / 25),
        ("hesitant", 1.5 * len(stats.hedges) + max(0, (50 - profile.assertiveness)) / 20),
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    strong = [label for label, score in ranked if score >= 1.0]
    profile.primary_tone = strong[0] if strong else "neutral"
    profile.secondary_tones = strong[1:3]

    # --- Risks ----------------------------------------------------------
    if signals["aggressive"]:
        profile.risks.append(
            "Contains accusatory phrasing that is likely to escalate the conversation."
        )
    if stats.all_caps_words:
        profile.risks.append("ALL-CAPS words read as shouting in written communication.")
    if stats.exclamations >= 3:
        profile.risks.append("Heavy exclamation use undercuts a professional tone.")
    if len(stats.hedges) >= 3:
        profile.risks.append(
            "Stacked hedging language makes the request easy to ignore."
        )
    if profile.politeness < 35:
        profile.risks.append("Politeness is low for a professional audience.")
    if stats.jargon:
        profile.risks.append(
            f"Business jargon ({', '.join(stats.jargon[:3])}) dilutes the message."
        )
    return profile
