"""Rule-based grammar and mechanics checking.

Powers the offline path of the grammar tool, and cross-checks the LLM path so
that obvious mechanical errors are never silently missed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class GrammarIssue:
    type: str
    message: str
    original: str
    suggestion: str
    offset: int
    severity: str = "minor"  # minor | major

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Rule:
    name: str
    pattern: re.Pattern[str]
    message: str
    severity: str = "minor"
    #: Replacement template applied for the auto-corrected text. None = no autofix.
    replacement: str | None = None


_MISSPELLINGS: dict[str, str] = {
    "recieve": "receive", "recieved": "received", "seperate": "separate",
    "definately": "definitely", "occured": "occurred", "occuring": "occurring",
    "untill": "until", "wich": "which", "teh": "the", "adress": "address",
    "acheive": "achieve", "beleive": "believe", "buisness": "business",
    "calender": "calendar", "collegue": "colleague", "commited": "committed",
    "enviroment": "environment", "existance": "existence", "familar": "familiar",
    "goverment": "government", "immediatly": "immediately",
    "independant": "independent", "knowlege": "knowledge", "maintainance": "maintenance",
    "neccessary": "necessary", "noticable": "noticeable", "oppurtunity": "opportunity",
    "posible": "possible", "reccomend": "recommend", "refered": "referred",
    "relevent": "relevant", "responsibilty": "responsibility", "succesful": "successful",
    "sucessful": "successful", "tommorow": "tomorrow", "truely": "truly",
    "unfortunatly": "unfortunately", "wierd": "weird", "writting": "writing",
    "alot": "a lot", "aswell": "as well", "infront": "in front",
    "thankyou": "thank you", "everytime": "every time", "atleast": "at least",
    # Missing apostrophes. Only unambiguous forms — "cant", "wont", "ill",
    # "id" and "lets" are all real words in their own right.
    "dont": "don't", "doesnt": "doesn't", "didnt": "didn't", "isnt": "isn't",
    "arent": "aren't", "wasnt": "wasn't", "werent": "weren't",
    "havent": "haven't", "hasnt": "hasn't", "hadnt": "hadn't",
    "couldnt": "couldn't", "wouldnt": "wouldn't", "shouldnt": "shouldn't",
    "youre": "you're", "youve": "you've", "theyre": "they're",
    "weve": "we've", "im": "I'm", "ive": "I've", "thats": "that's",
    "whats": "what's", "theres": "there's", "wouldve": "would've",
}

_RULES: list[_Rule] = [
    _Rule(
        "subject_verb_agreement",
        re.compile(r"\b(he|she|it|this|that)\s+(are|were|have)\b", re.IGNORECASE),
        "Singular subject paired with a plural verb.",
        "major",
    ),
    _Rule(
        "subject_verb_agreement",
        re.compile(r"\b(they|we|you|these|those)\s+(is|was|has)\b", re.IGNORECASE),
        "Plural subject paired with a singular verb.",
        "major",
    ),
    _Rule(
        "subject_verb_agreement",
        re.compile(r"\bI\s+(is|are|has|were)\b"),
        "'I' takes 'am', 'have' or 'was'.",
        "major",
    ),
    _Rule(
        "double_negative",
        re.compile(r"\b(don't|doesn't|didn't|can't|won't)\s+\w*\s*(no|nothing|nobody|never)\b", re.IGNORECASE),
        "Double negative — this reverses your intended meaning.",
        "major",
    ),
    _Rule(
        "verb_form",
        re.compile(r"\b(did|does|do|didn't|doesn't|don't)\s+(\w+ed)\b", re.IGNORECASE),
        "After do/did, use the base form of the verb (e.g. 'did finish', not 'did finished').",
        "major",
    ),
    _Rule(
        "modal_verb_form",
        re.compile(r"\b(could|should|would|will|can|may|might|must)\s+(\w+ed|\w+ing)\b", re.IGNORECASE),
        "After a modal verb, use the base form (e.g. 'could send', not 'could sending').",
        "major",
    ),
    _Rule(
        "article_usage",
        re.compile(r"\ba\s+([aeiou]\w+)", re.IGNORECASE),
        "Use 'an' before a vowel sound.",
        "minor",
    ),
    _Rule(
        "repeated_word",
        re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE),
        "Repeated word.",
        "minor",
    ),
    _Rule(
        "spacing",
        re.compile(r"\S {2,}\S"),
        "Multiple consecutive spaces.",
        "minor",
    ),
    _Rule(
        "spacing",
        re.compile(r"\s+[,.;:!?]"),
        "Space before punctuation.",
        "minor",
    ),
    _Rule(
        "spacing",
        re.compile(r"[,;:](?=[A-Za-z])"),
        "Missing space after punctuation.",
        "minor",
    ),
    _Rule(
        "capitalization",
        re.compile(r"\bi\b(?!['\w])"),
        "The pronoun 'I' is always capitalised.",
        "major",
    ),
    _Rule(
        "their_theyre",
        re.compile(r"\b(?:their|there)\s+(\w+ing)\b", re.IGNORECASE),
        "'they're' is the contraction of 'they are'; 'their' shows possession "
        "and 'there' is a place.",
        "major",
    ),
    _Rule(
        "their_theyre",
        re.compile(r"\bthere\s+(own|team|manager|company|work|report|idea)\b", re.IGNORECASE),
        "Use 'their' to show possession.",
        "major",
    ),
    _Rule(
        "its_contraction",
        re.compile(
            r"\bits\s+(been|a|an|the|not|going|your|my|our|obviously|clear|because)\b",
            re.IGNORECASE,
        ),
        "'it's' is the contraction of 'it is/has'; 'its' is possessive.",
        "major",
    ),
    _Rule(
        "run_on",
        re.compile(r"\b\w+,\s+(?:however|therefore|moreover|furthermore)\s*,?\s+\w+"),
        "Comma splice — use a semicolon or start a new sentence before a conjunctive adverb.",
        "minor",
    ),
]

# word -> (correct alternative, explanation)
_CONFUSABLES: dict[str, tuple[str, str]] = {
    "your welcome": ("you're welcome", "'you're' is the contraction of 'you are'."),
    "your right": ("you're right", "'you're' is the contraction of 'you are'."),
    "its been": ("it's been", "'it's' is the contraction of 'it is/has'."),
    "its a ": ("it's a ", "'it's' is the contraction of 'it is'."),
    "could of": ("could have", "'could of' is never correct."),
    "would of": ("would have", "'would of' is never correct."),
    "should of": ("should have", "'should of' is never correct."),
    "must of": ("must have", "'must of' is never correct."),
    "there own": ("their own", "'their' shows possession."),
    "there is a lot of people": (
        "there are a lot of people",
        "'a lot of people' is plural.",
    ),
    "then me": ("than me", "'than' is used for comparisons."),
    "less people": ("fewer people", "Use 'fewer' for countable nouns."),
    "less errors": ("fewer errors", "Use 'fewer' for countable nouns."),
    "loose the": ("lose the", "'lose' is the verb; 'loose' means not tight."),
    "advice you": ("advise you", "'advise' is the verb, 'advice' the noun."),
    "affect on": ("effect on", "'effect' is the noun."),
    "irregardless": ("regardless", "'irregardless' is not standard English."),
    "revert back": ("reply", "'revert' already means 'go back'."),
    "kindly do the needful": (
        "please take care of this",
        "'do the needful' reads as dated and vague in most business contexts.",
    ),
}


def check_grammar(text: str) -> list[GrammarIssue]:
    """Return every grammar/mechanics issue found in `text`."""
    issues: list[GrammarIssue] = []
    lowered = text.lower()

    for match in re.finditer(r"[A-Za-z']+", text):
        word = match.group(0)
        correction = _MISSPELLINGS.get(word.lower())
        if correction:
            if word[0].isupper():
                correction = correction.capitalize()
            issues.append(
                GrammarIssue(
                    type="spelling",
                    message=f"'{word}' is misspelled.",
                    original=word,
                    suggestion=correction,
                    offset=match.start(),
                    severity="major",
                )
            )

    for rule in _RULES:
        for match in rule.pattern.finditer(text):
            issues.append(
                GrammarIssue(
                    type=rule.name,
                    message=rule.message,
                    original=match.group(0).strip(),
                    suggestion=_suggest(rule.name, match),
                    offset=match.start(),
                    severity=rule.severity,
                )
            )

    for phrase, (better, why) in _CONFUSABLES.items():
        start = lowered.find(phrase)
        if start != -1:
            issues.append(
                GrammarIssue(
                    type="word_choice",
                    message=why,
                    original=text[start : start + len(phrase)],
                    suggestion=better,
                    offset=start,
                    severity="major",
                )
            )

    issues.extend(_check_sentence_mechanics(text))
    issues.sort(key=lambda i: i.offset)
    return _dedupe(issues)


def _check_sentence_mechanics(text: str) -> list[GrammarIssue]:
    """Sentence-level checks: capitalisation and terminal punctuation."""
    from app.nlp.textstats import split_sentences

    issues: list[GrammarIssue] = []
    cursor = 0
    for sentence in split_sentences(text):
        offset = text.find(sentence, cursor)
        cursor = offset + len(sentence) if offset != -1 else cursor

        stripped_sentence = sentence.lstrip()
        first_word = stripped_sentence.split(" ")[0] if stripped_sentence else ""
        # Report the offending word, not a truncated slice of the sentence —
        # the slice is unreadable in the UI and breaks the autofix matcher.
        if first_word[:1].isalpha() and first_word[:1].islower():
            lead = len(sentence) - len(stripped_sentence)
            issues.append(
                GrammarIssue(
                    type="capitalization",
                    message="Sentence should start with a capital letter.",
                    original=first_word,
                    suggestion=first_word[0].upper() + first_word[1:],
                    offset=max(offset, 0) + lead,
                    severity="minor",
                )
            )

    stripped = text.strip()
    if stripped and stripped[-1] not in ".!?:\"')" and len(stripped.split()) > 3:
        tail = " ".join(stripped.split()[-4:])
        issues.append(
            GrammarIssue(
                type="punctuation",
                message="The text does not end with terminal punctuation.",
                original=tail,
                suggestion=tail + ".",
                offset=max(len(text.rstrip()) - len(tail), 0),
                severity="minor",
            )
        )
    return issues


def _suggest(rule_name: str, match: re.Match[str]) -> str:
    raw = match.group(0)
    match rule_name:
        case "repeated_word":
            return match.group(1)
        case "spacing":
            fixed = re.sub(r" {2,}", " ", raw)
            fixed = re.sub(r"\s+([,.;:!?])", r"\1", fixed)
            return re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", fixed)
        case "capitalization":
            return "I"
        case "article_usage":
            return "an " + match.group(1)
        case "its_contraction":
            return "it's " + match.group(1)
        case "their_theyre":
            word = match.group(1)
            return ("they're " if word.lower().endswith("ing") else "their ") + word
        case "subject_verb_agreement":
            return _fix_agreement(match)
        case _:
            return raw


_AGREEMENT_MAP = {
    ("he", "are"): "is", ("she", "are"): "is", ("it", "are"): "is",
    ("this", "are"): "is", ("that", "are"): "is",
    ("he", "were"): "was", ("she", "were"): "was", ("it", "were"): "was",
    ("this", "were"): "was", ("that", "were"): "was",
    ("he", "have"): "has", ("she", "have"): "has", ("it", "have"): "has",
    ("this", "have"): "has", ("that", "have"): "has",
    ("they", "is"): "are", ("we", "is"): "are", ("you", "is"): "are",
    ("these", "is"): "are", ("those", "is"): "are",
    ("they", "was"): "were", ("we", "was"): "were", ("you", "was"): "were",
    ("these", "was"): "were", ("those", "was"): "were",
    ("they", "has"): "have", ("we", "has"): "have", ("you", "has"): "have",
    ("these", "has"): "have", ("those", "has"): "have",
}


def _fix_agreement(match: re.Match[str]) -> str:
    groups = match.groups()
    if len(groups) < 2:
        # The "I is/are/has/were" rule has a single group.
        verb = (groups[0] or "").lower()
        return "I " + {"is": "am", "are": "am", "has": "have", "were": "was"}.get(verb, verb)
    subject, verb = groups[0], groups[1]
    fixed = _AGREEMENT_MAP.get((subject.lower(), verb.lower()), verb)
    return f"{subject} {fixed}"


def _dedupe(issues: list[GrammarIssue]) -> list[GrammarIssue]:
    seen: set[tuple[str, int, str]] = set()
    unique: list[GrammarIssue] = []
    for issue in issues:
        key = (issue.type, issue.offset, issue.original.lower())
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique


def apply_corrections(text: str, issues: list[GrammarIssue]) -> str:
    """Apply issues whose suggestion is a safe literal substitution."""
    autofixable = {"spelling", "capitalization", "word_choice", "repeated_word",
                   "spacing", "article_usage", "subject_verb_agreement",
                   "its_contraction", "their_theyre"}
    corrected = text
    # Apply right-to-left so earlier offsets stay valid.
    for issue in sorted(issues, key=lambda i: i.offset, reverse=True):
        if issue.type not in autofixable or not issue.suggestion:
            continue
        segment = corrected[issue.offset : issue.offset + len(issue.original)]
        if segment.lower() != issue.original.lower():
            continue
        corrected = (
            corrected[: issue.offset]
            + issue.suggestion
            + corrected[issue.offset + len(issue.original) :]
        )

    stripped = corrected.strip()
    if stripped and stripped[-1] not in ".!?:\"')" and len(stripped.split()) > 3:
        corrected = corrected.rstrip() + "."
    return corrected
