"""Deterministic text statistics.

These metrics are computed in code rather than asked of the model: LLMs are
unreliable at counting, and the API should return stable numbers for the same
input. The LLM-backed tools use these stats as grounding facts in their prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")

FILLER_WORDS = {
    "just", "actually", "basically", "literally", "really", "very", "quite",
    "simply", "honestly", "obviously", "kind", "sort", "stuff", "things",
    "somewhat", "totally", "definitely", "absolutely",
}

HEDGES = {
    "maybe", "perhaps", "possibly", "probably", "might", "sort of", "kind of",
    "i think", "i guess", "i feel like", "i'm not sure", "hopefully",
    "if that's ok", "if possible", "just wondering", "i was wondering",
    "sorry to bother", "i could be wrong", "does that make sense",
}

WEAK_OPENERS = {
    "i just wanted to", "i was just", "sorry to bother you", "i hope this isn't",
    "i'm not sure if", "this might be a stupid",
}

BUSINESS_JARGON = {
    "synergy", "leverage", "circle back", "touch base", "moving forward",
    "at the end of the day", "low-hanging fruit", "bandwidth", "paradigm",
    "going forward", "deep dive", "boil the ocean", "move the needle",
}

_PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?(\w+(?:ed|en))\b",
    re.IGNORECASE,
)

# Common -ed/-en words that are adjectives, not passive participles.
_PASSIVE_FALSE_POSITIVES = {
    "interested", "excited", "pleased", "tired", "concerned", "committed",
    "based", "located", "involved", "related", "supposed", "used", "known",
}


@dataclass
class TextStats:
    text: str
    words: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_sentence_length: float = 0.0
    longest_sentence_length: int = 0
    flesch_reading_ease: float = 0.0
    reading_grade: float = 0.0
    filler_words: list[str] = field(default_factory=list)
    hedges: list[str] = field(default_factory=list)
    jargon: list[str] = field(default_factory=list)
    passive_phrases: list[str] = field(default_factory=list)
    exclamations: int = 0
    questions: int = 0
    all_caps_words: list[str] = field(default_factory=list)
    has_greeting: bool = False
    has_signoff: bool = False
    has_call_to_action: bool = False
    estimated_speaking_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "paragraph_count": self.paragraph_count,
            "avg_sentence_length": round(self.avg_sentence_length, 1),
            "longest_sentence_length": self.longest_sentence_length,
            "flesch_reading_ease": round(self.flesch_reading_ease, 1),
            "reading_grade": round(self.reading_grade, 1),
            "filler_words": self.filler_words,
            "hedges": self.hedges,
            "jargon": self.jargon,
            "passive_phrases": self.passive_phrases,
            "exclamations": self.exclamations,
            "questions": self.questions,
            "all_caps_words": self.all_caps_words,
            "has_greeting": self.has_greeting,
            "has_signoff": self.has_signoff,
            "has_call_to_action": self.has_call_to_action,
            "estimated_speaking_seconds": round(self.estimated_speaking_seconds),
        }


_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|dear|good\s+(morning|afternoon|evening)|greetings)\b",
    re.IGNORECASE,
)
_SIGNOFF_RE = re.compile(
    r"\b(regards|best|sincerely|thanks|thank you|cheers|respectfully|"
    r"looking forward)\b[\s,!.]*$",
    re.IGNORECASE,
)
_CTA_RE = re.compile(
    r"\b(could you|can you|please|let me know|would you|i'd appreciate|"
    r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|eod|"
    r"tomorrow|next week)|action item|next step)\b",
    re.IGNORECASE,
)


def count_syllables(word: str) -> int:
    word = word.lower().strip("'")
    if not word:
        return 0
    groups = _VOWEL_GROUP_RE.findall(word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def analyze_text(text: str) -> TextStats:
    """Compute all deterministic statistics for `text`."""
    stats = TextStats(text=text)
    lowered = text.lower()

    stats.words = _WORD_RE.findall(text)
    stats.sentences = split_sentences(text)
    stats.word_count = len(stats.words)
    stats.sentence_count = len(stats.sentences)
    stats.paragraph_count = len([p for p in text.split("\n\n") if p.strip()]) or 1

    sentence_lengths = [len(_WORD_RE.findall(s)) for s in stats.sentences]
    if sentence_lengths:
        stats.avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
        stats.longest_sentence_length = max(sentence_lengths)

    if stats.word_count and stats.sentence_count:
        syllables = sum(count_syllables(w) for w in stats.words)
        words_per_sentence = stats.word_count / stats.sentence_count
        syllables_per_word = syllables / stats.word_count
        stats.flesch_reading_ease = (
            206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        )
        stats.reading_grade = (
            0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
        )

    lowered_words = [w.lower() for w in stats.words]
    stats.filler_words = sorted({w for w in lowered_words if w in FILLER_WORDS})
    stats.hedges = sorted({h for h in HEDGES if h in lowered})
    stats.jargon = sorted({j for j in BUSINESS_JARGON if j in lowered})

    stats.passive_phrases = [
        match.group(0).strip()
        for match in _PASSIVE_RE.finditer(text)
        if match.group(1).lower() not in _PASSIVE_FALSE_POSITIVES
    ]

    stats.exclamations = text.count("!")
    stats.questions = text.count("?")
    stats.all_caps_words = [
        w for w in stats.words if len(w) > 2 and w.isupper() and w != "I"
    ]

    stats.has_greeting = bool(_GREETING_RE.search(text))
    stats.has_signoff = bool(_SIGNOFF_RE.search(text.strip()))
    stats.has_call_to_action = bool(_CTA_RE.search(text))

    # ~140 wpm is a comfortable presentation pace.
    stats.estimated_speaking_seconds = stats.word_count / 140 * 60

    return stats
