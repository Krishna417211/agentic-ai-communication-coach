"""PDF resume text extraction and structural analysis."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader

_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "summary": re.compile(r"^\s*(professional\s+)?(summary|profile|objective|about)\b", re.I | re.M),
    "experience": re.compile(r"^\s*(work\s+)?(experience|employment|career history)\b", re.I | re.M),
    "education": re.compile(r"^\s*(education|academic|qualifications)\b", re.I | re.M),
    "skills": re.compile(r"^\s*(technical\s+)?(skills|technologies|competencies|tech stack)\b", re.I | re.M),
    "projects": re.compile(r"^\s*(projects|portfolio|selected work)\b", re.I | re.M),
    "certifications": re.compile(r"^\s*(certifications?|licences?|licenses?|awards?)\b", re.I | re.M),
    "contact": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
}

_STRONG_VERBS = {
    "led", "built", "designed", "shipped", "launched", "reduced", "increased",
    "automated", "migrated", "architected", "owned", "delivered", "scaled",
    "negotiated", "founded", "rebuilt", "eliminated", "accelerated", "drove",
    "implemented", "created", "established", "improved", "streamlined",
}

_WEAK_OPENERS = {
    "responsible for", "worked on", "helped with", "assisted", "involved in",
    "participated in", "tasked with", "duties included", "part of a team",
    "familiar with", "exposure to", "worked with",
}

_BULLET_RE = re.compile(r"^\s*[•\-\*•●‣⁃o]\s+(.{10,})$", re.M)
_METRIC_RE = re.compile(r"\d+\s*(%|percent|x\b|k\b|m\b|hours?|days?|weeks?|users?|customers?|\$|€|£)", re.I)


@dataclass
class ResumeDocument:
    text: str
    pages: int
    sections: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    word_count: int = 0
    bullets_with_metrics: int = 0
    weak_bullets: list[str] = field(default_factory=list)
    strong_bullets: list[str] = field(default_factory=list)
    has_contact: bool = False


class ResumeParseError(ValueError):
    """Raised when a PDF cannot be read as a resume."""


def extract_text(data: bytes, *, max_pages: int = 10) -> tuple[str, int]:
    """Extract text from a PDF, capped at `max_pages`."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - pypdf raises many types
        raise ResumeParseError(f"Could not read the PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ResumeParseError("The PDF is password protected.") from exc

    total_pages = len(reader.pages)
    chunks = [page.extract_text() or "" for page in reader.pages[:max_pages]]
    text = "\n".join(chunks).strip()

    if len(text) < 80:
        raise ResumeParseError(
            "Almost no text could be extracted — the PDF is likely a scan. "
            "Export a text-based PDF and try again."
        )
    return text, total_pages


def parse_resume(data: bytes, *, max_pages: int = 10) -> ResumeDocument:
    text, pages = extract_text(data, max_pages=max_pages)
    doc = ResumeDocument(text=text, pages=pages)

    doc.sections = [
        name for name, pattern in _SECTION_PATTERNS.items()
        if name != "contact" and pattern.search(text)
    ]
    doc.has_contact = bool(_SECTION_PATTERNS["contact"].search(text))
    doc.word_count = len(re.findall(r"[A-Za-z']+", text))

    bullets = [b.strip() for b in _BULLET_RE.findall(text)]
    if not bullets:
        # Some resumes use plain lines rather than bullet glyphs.
        bullets = [
            line.strip()
            for line in text.splitlines()
            if 40 < len(line.strip()) < 300 and not line.strip().isupper()
        ]
    doc.bullets = bullets[:60]

    for bullet in doc.bullets:
        lowered = bullet.lower()
        if _METRIC_RE.search(bullet):
            doc.bullets_with_metrics += 1
        first_word = re.sub(r"[^a-z]", "", lowered.split(" ")[0] if lowered else "")
        if any(lowered.startswith(weak) for weak in _WEAK_OPENERS):
            doc.weak_bullets.append(bullet)
        elif first_word in _STRONG_VERBS:
            doc.strong_bullets.append(bullet)

    return doc
