"""Short-term conversational memory.

Sessions live in-process with a TTL. Each session keeps a bounded window of
turns, a rolling summary of what fell out of that window, a running score
history, and a tally of recurring weaknesses that lets the coach say
"you've hedged in three of your last four messages" instead of treating every
turn as the first.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.schemas import Intent, SessionSnapshot, Role, ToolName, Turn


@dataclass
class Session:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    scores: list[int] = field(default_factory=list)
    issue_counts: Counter[str] = field(default_factory=Counter)
    #: Free-form slots for multi-turn flows (e.g. an interview in progress).
    state: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_access: float = field(default_factory=time.monotonic)
    total_turns: int = 0

    @property
    def last_intent(self) -> Intent | None:
        for turn in reversed(self.turns):
            if turn.intent is not None:
                return turn.intent
        return None

    @property
    def last_user_message(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.role == Role.USER:
                return turn.content
        return None

    @property
    def average_score(self) -> float | None:
        return round(sum(self.scores) / len(self.scores), 1) if self.scores else None

    def recurring_issues(self, min_count: int = 2) -> list[str]:
        return [
            f"{issue} (seen {count}x)"
            for issue, count in self.issue_counts.most_common(5)
            if count >= min_count
        ]

    def score_trend(self) -> str:
        if len(self.scores) < 2:
            return "not enough data yet"
        delta = self.scores[-1] - self.scores[0]
        if delta >= 5:
            return f"improving (+{delta} since the first message)"
        if delta <= -5:
            return f"slipping ({delta} since the first message)"
        return "holding steady"


class SessionStore:
    """Async-safe in-memory session store with TTL eviction.

    Swappable for Redis behind the same three methods if the deployment ever
    needs to run more than one worker process.
    """

    def __init__(self, *, max_turns: int = 12, ttl_seconds: int = 21600) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._max_turns = max_turns
        self._ttl = ttl_seconds

    async def get_or_create(self, session_id: str | None) -> Session:
        async with self._lock:
            self._evict_expired()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_access = time.monotonic()
                return session
            new_id = session_id or uuid.uuid4().hex[:16]
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            return session

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session:
                session.last_access = time.monotonic()
            return session

    async def append_turn(
        self,
        session_id: str,
        *,
        role: Role,
        content: str,
        intent: Intent | None = None,
        tools_used: list[ToolName] | None = None,
        overall_score: int | None = None,
        issues: list[str] | None = None,
    ) -> Session:
        async with self._lock:
            session = self._sessions.setdefault(session_id, Session(session_id))
            session.turns.append(
                Turn(
                    role=role,
                    content=content,
                    intent=intent,
                    tools_used=tools_used or [],
                    overall_score=overall_score,
                )
            )
            session.total_turns += 1
            session.updated_at = datetime.now(timezone.utc)
            session.last_access = time.monotonic()

            if overall_score is not None:
                session.scores.append(overall_score)
            for issue in issues or []:
                session.issue_counts[normalise_issue(issue)] += 1

            self._trim(session)
            return session

    async def set_state(self, session_id: str, key: str, value) -> None:
        async with self._lock:
            session = self._sessions.setdefault(session_id, Session(session_id))
            session.state[key] = value
            session.last_access = time.monotonic()

    async def snapshot(self, session_id: str) -> SessionSnapshot | None:
        session = await self.get(session_id)
        if session is None:
            return None
        return SessionSnapshot(
            session_id=session.session_id,
            turns=session.turns,
            summary=session.summary,
            recurring_issues=session.recurring_issues(),
            average_score=session.average_score,
            turn_count=session.total_turns,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    async def clear(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def active_count(self) -> int:
        async with self._lock:
            self._evict_expired()
            return len(self._sessions)

    # -- internals -------------------------------------------------------

    def _trim(self, session: Session) -> None:
        """Fold overflow turns into the rolling summary."""
        overflow = len(session.turns) - self._max_turns
        if overflow <= 0:
            return
        dropped, session.turns = session.turns[:overflow], session.turns[overflow:]
        topics = [t.intent.value.replace("_", " ") for t in dropped if t.intent]
        parts = [p for p in [session.summary] if p]
        if topics:
            unique = list(dict.fromkeys(topics))
            parts.append(
                f"Earlier in this session the user worked on: {', '.join(unique)}."
            )
        first_user = next((t.content for t in dropped if t.role == Role.USER), None)
        if first_user and not session.summary:
            parts.append(f'They opened with: "{first_user[:160]}".')
        session.summary = " ".join(parts)[:1200]

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            sid for sid, s in self._sessions.items() if now - s.last_access > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]


def normalise_issue(issue: str) -> str:
    """Collapse specific findings into a stable label we can count."""
    lowered = issue.lower()
    buckets = [
        ("hedg", "hedging language"),
        ("filler", "filler words"),
        ("passive", "passive voice"),
        ("call to action", "missing call to action"),
        ("grammar", "grammar errors"),
        ("long", "overly long sentences"),
        ("jargon", "business jargon"),
        ("paragraph", "wall-of-text formatting"),
        ("confrontational", "confrontational tone"),
        ("aggressive", "confrontational tone"),
        ("vague", "vague specifics"),
        ("short", "insufficient detail"),
    ]
    for needle, label in buckets:
        if needle in lowered:
            return label
    return issue[:60]
