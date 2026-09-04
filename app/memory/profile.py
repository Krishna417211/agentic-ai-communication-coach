"""Long-term coaching memory.

`SessionStore` is short-term working memory: what was said in *this*
conversation, evicted after a few hours. That is the wrong lifetime for the
thing that makes a coach a coach — whether you are actually getting better.

This module keeps the longitudinal signal in SQLite, keyed by a handle the
user chooses. It survives restarts, so a returning user is recognised and the
coach can point at a habit fading across weeks rather than meeting a stranger
every time.

Only derived coaching signal is stored — scores, intents and normalised issue
labels. Message content stays in the ephemeral session store and is never
written to disk.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    handle        TEXT    NOT NULL,
    recorded_at   TEXT    NOT NULL,
    intent        TEXT,
    overall_score INTEGER,
    issues        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_handle ON events(handle, id);
"""

#: Handles are storage keys and are shown back to the user, so keep them
#: boring: this prevents path surprises and keeps the value display-safe.
_HANDLE_RE = re.compile(r"[^a-z0-9_-]+")

#: Below this many turns, splitting the history in half to detect a fading
#: habit gives each half too little to say anything honest.
_MIN_TURNS_FOR_TREND = 6


def normalise_handle(handle: str) -> str:
    cleaned = _HANDLE_RE.sub("-", handle.strip().lower()).strip("-")
    return cleaned[:32]


class ProfileStore:
    """Durable per-user coaching history.

    sqlite3 is blocking, so every call hops to a worker thread. The volume is a
    couple of rows per coaching turn, which does not justify an async driver.
    """

    def __init__(self, db_path: str = "data/coach.db") -> None:
        self._path = Path(db_path)
        self._lock = asyncio.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sync(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    async def _ensure(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if not self._ready:
                await asyncio.to_thread(self._init_sync)
                self._ready = True

    # -- writes ----------------------------------------------------------

    async def record(
        self,
        handle: str,
        *,
        intent: str | None = None,
        overall_score: int | None = None,
        issues: list[str] | None = None,
    ) -> bool:
        """Append one coaching turn to a profile.

        Returns whether it was stored. A profile write is a side effect of
        coaching, not the point of it, so a locked or unwritable database must
        never cost the user their answer.
        """
        handle = normalise_handle(handle)
        if not handle:
            return False
        payload = json.dumps(issues or [])
        recorded_at = datetime.now(timezone.utc).isoformat()

        def _write() -> None:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO events"
                    " (handle, recorded_at, intent, overall_score, issues)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (handle, recorded_at, intent, overall_score, payload),
                )

        try:
            await self._ensure()
            await asyncio.to_thread(_write)
            return True
        except (sqlite3.Error, OSError):
            return False

    async def forget(self, handle: str) -> int:
        """Delete everything stored for a handle. Returns rows removed."""
        handle = normalise_handle(handle)
        if not handle:
            return 0

        def _delete() -> int:
            with self._connect() as conn:
                return conn.execute(
                    "DELETE FROM events WHERE handle = ?", (handle,)
                ).rowcount

        try:
            await self._ensure()
            return await asyncio.to_thread(_delete)
        except (sqlite3.Error, OSError):
            return 0

    # -- reads -----------------------------------------------------------

    async def progress(self, handle: str) -> dict:
        """Everything the progress view needs, in one round trip."""
        handle = normalise_handle(handle)
        if not handle:
            return _empty_progress(handle)

        def _read() -> list[sqlite3.Row]:
            with self._connect() as conn:
                return conn.execute(
                    "SELECT recorded_at, intent, overall_score, issues"
                    " FROM events WHERE handle = ? ORDER BY id",
                    (handle,),
                ).fetchall()

        try:
            await self._ensure()
            rows = await asyncio.to_thread(_read)
        except (sqlite3.Error, OSError):
            return _empty_progress(handle)
        return _summarise(handle, rows)


def _empty_progress(handle: str) -> dict:
    return {
        "handle": handle,
        "turns": 0,
        "scores": [],
        "first_score": None,
        "latest_score": None,
        "best_score": None,
        "average_score": None,
        "delta": None,
        "trend": "not enough data yet",
        "top_habits": [],
        "fixed_habits": [],
        "intents": [],
        "last_seen": None,
    }


def _summarise(handle: str, rows: list) -> dict:
    out = _empty_progress(handle)
    if not rows:
        return out

    scores = [r["overall_score"] for r in rows if r["overall_score"] is not None]
    out["turns"] = len(rows)
    out["scores"] = scores
    out["last_seen"] = rows[-1]["recorded_at"]
    out["intents"] = [
        {"intent": name, "count": count}
        for name, count in Counter(
            r["intent"] for r in rows if r["intent"]
        ).most_common(5)
    ]

    if scores:
        out["first_score"] = scores[0]
        out["latest_score"] = scores[-1]
        out["best_score"] = max(scores)
        out["average_score"] = round(sum(scores) / len(scores), 1)
        if len(scores) >= 2:
            delta = scores[-1] - scores[0]
            out["delta"] = delta
            if delta >= 5:
                out["trend"] = f"improving (+{delta} since you started)"
            elif delta <= -5:
                out["trend"] = f"slipping ({delta} since you started)"
            else:
                out["trend"] = "holding steady"

    per_turn = [_load_issues(r["issues"]) for r in rows]
    totals = Counter(issue for turn in per_turn for issue in turn)
    out["top_habits"] = [
        {"habit": habit, "count": count}
        for habit, count in totals.most_common(5)
        if count >= 2
    ]

    # A habit counts as fading when it was common early and has gone quieter
    # since — that is the sentence a coach earns the right to say.
    if len(rows) >= _MIN_TURNS_FOR_TREND:
        midpoint = len(per_turn) // 2
        early = Counter(i for turn in per_turn[:midpoint] for i in turn)
        late = Counter(i for turn in per_turn[midpoint:] for i in turn)
        out["fixed_habits"] = [
            {"habit": habit, "was": count, "now": late.get(habit, 0)}
            for habit, count in early.most_common()
            if count >= 2 and late.get(habit, 0) < count
        ][:5]

    return out


def _load_issues(raw: str | None) -> list[str]:
    """Tolerate anything an older or half-written row might hold."""
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []
