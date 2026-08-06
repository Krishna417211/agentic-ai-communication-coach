"""RAG knowledge base over the bundled communication-coaching corpus.

Retrieval is BM25 implemented directly — the corpus is small and static, so an
embedding model and a vector database would add a heavyweight dependency and a
cold-start cost for no measurable gain in retrieval quality here.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).parent / "corpus"

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "is", "are", "was", "were", "be", "been", "it", "its", "that",
    "this", "these", "those", "you", "your", "they", "them", "their", "we",
    "our", "i", "my", "me", "at", "by", "from", "not", "no", "do", "does",
    "did", "so", "than", "then", "there", "what", "which", "who", "how", "when",
    "can", "will", "would", "should", "could", "have", "has", "had", "more",
    "most", "some", "any", "all", "one", "two", "up", "out", "about", "into",
}

_K1 = 1.5
_B = 0.75


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    content: str
    tokens: list[str]


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


class KnowledgeBase:
    """BM25 retrieval over markdown sections."""

    def __init__(self, corpus_dir: Path | None = None) -> None:
        self.documents: list[Document] = []
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0
        self._load(corpus_dir or CORPUS_DIR)

    @property
    def size(self) -> int:
        return len(self.documents)

    def _load(self, corpus_dir: Path) -> None:
        if not corpus_dir.exists():
            logger.warning("knowledge corpus missing at %s", corpus_dir)
            return

        for path in sorted(corpus_dir.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            for index, (title, body) in enumerate(_split_sections(raw)):
                content = body.strip()
                if not content:
                    continue
                tokens = tokenize(f"{title} {content}")
                self.documents.append(
                    Document(
                        doc_id=f"{path.stem}#{index}",
                        title=title,
                        source=path.name,
                        content=content,
                        tokens=tokens,
                    )
                )

        for doc in self.documents:
            self._df.update(set(doc.tokens))
        if self.documents:
            self._avg_len = sum(len(d.tokens) for d in self.documents) / len(self.documents)
        logger.info("knowledge base loaded: %s sections", len(self.documents))

    def search(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        """Return the top_k most relevant sections for `query`."""
        if not self.documents:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        total = len(self.documents)
        scored: list[tuple[float, Document]] = []
        for doc in self.documents:
            freqs = Counter(doc.tokens)
            length = len(doc.tokens) or 1
            score = 0.0
            for token in query_tokens:
                tf = freqs.get(token, 0)
                if not tf:
                    continue
                df = self._df.get(token, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = tf + _K1 * (1 - _B + _B * length / (self._avg_len or 1))
                score += idf * (tf * (_K1 + 1)) / denominator
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "id": doc.doc_id,
                "title": doc.title,
                "source": doc.source,
                "content": doc.content[:900],
                "score": round(score, 3),
            }
            for score, doc in scored[:top_k]
        ]


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown file into (heading, body) pairs on '##' headings."""
    sections: list[tuple[str, str]] = []
    current_title = "Overview"
    buffer: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("## "):
            if buffer:
                sections.append((current_title, "\n".join(buffer)))
                buffer = []
            current_title = line[3:].strip()
        elif line.startswith("# "):
            current_title = line[2:].strip()
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_title, "\n".join(buffer)))
    return sections


_kb: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
