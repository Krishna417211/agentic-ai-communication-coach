"""Tests for long-term coaching memory (`app.memory.profile`)."""

from __future__ import annotations

import pytest

from app.memory.profile import ProfileStore, normalise_handle


@pytest.fixture
def store(tmp_path) -> ProfileStore:
    return ProfileStore(str(tmp_path / "coach.db"))


class TestHandleNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  Krishna Priya! ", "krishna-priya"),
            ("UPPER_case-99", "upper_case-99"),
            ("../../etc/passwd", "etc-passwd"),
            ("!!!", ""),
            ("", ""),
        ],
    )
    def test_normalises(self, raw, expected):
        assert normalise_handle(raw) == expected

    def test_is_length_capped(self):
        assert len(normalise_handle("x" * 200)) == 32

    def test_lookup_is_handle_insensitive(self, store):
        # "Krishna Priya" and "krishna-priya" must be the same person.
        import asyncio

        async def go():
            await store.record("Krishna Priya", overall_score=70)
            return await store.progress("krishna-priya")

        assert asyncio.run(go())["turns"] == 1


@pytest.mark.asyncio
class TestProgress:
    async def test_unknown_handle_returns_an_empty_shape(self, store):
        progress = await store.progress("nobody")
        assert progress["turns"] == 0
        assert progress["scores"] == []
        assert progress["trend"] == "not enough data yet"

    async def test_blank_handle_is_never_stored(self, store):
        assert await store.record("   ", overall_score=80) is False
        assert (await store.progress("   "))["turns"] == 0

    async def test_single_turn_has_no_trend(self, store):
        await store.record("kp", overall_score=64)
        progress = await store.progress("kp")
        assert progress["latest_score"] == 64
        assert progress["delta"] is None
        assert progress["trend"] == "not enough data yet"

    async def test_reports_improvement(self, store):
        for score in (50, 60, 72):
            await store.record("kp", overall_score=score)
        progress = await store.progress("kp")
        assert progress["delta"] == 22
        assert progress["trend"] == "improving (+22 since you started)"
        assert progress["best_score"] == 72
        assert progress["average_score"] == pytest.approx(60.7)

    async def test_reports_slipping(self, store):
        for score in (80, 60):
            await store.record("kp", overall_score=score)
        assert (await store.progress("kp"))["trend"].startswith("slipping")

    async def test_small_moves_are_not_called_progress(self, store):
        for score in (70, 72):
            await store.record("kp", overall_score=score)
        assert (await store.progress("kp"))["trend"] == "holding steady"

    async def test_turns_without_a_score_still_count(self, store):
        await store.record("kp", intent="email_writing")
        progress = await store.progress("kp")
        assert progress["turns"] == 1
        assert progress["scores"] == []
        assert progress["latest_score"] is None

    async def test_profiles_are_isolated(self, store):
        await store.record("alice", overall_score=90)
        await store.record("bob", overall_score=10)
        assert (await store.progress("alice"))["latest_score"] == 90
        assert (await store.progress("bob"))["latest_score"] == 10


@pytest.mark.asyncio
class TestHabits:
    async def test_one_off_issues_are_not_habits(self, store):
        await store.record("kp", issues=["hedging language"])
        assert (await store.progress("kp"))["top_habits"] == []

    async def test_repeated_issues_become_habits(self, store):
        for _ in range(3):
            await store.record("kp", issues=["hedging language", "filler words"])
        habits = {h["habit"]: h["count"] for h in (await store.progress("kp"))["top_habits"]}
        assert habits == {"hedging language": 3, "filler words": 3}

    async def test_a_fading_habit_is_reported_as_fixed(self, store):
        for _ in range(3):
            await store.record("kp", overall_score=50, issues=["hedging language"])
        for _ in range(3):
            await store.record("kp", overall_score=75, issues=["passive voice"])
        fixed = (await store.progress("kp"))["fixed_habits"]
        assert [f["habit"] for f in fixed] == ["hedging language"]
        assert fixed[0] == {"habit": "hedging language", "was": 3, "now": 0}

    async def test_a_persistent_habit_is_not_reported_as_fixed(self, store):
        for _ in range(8):
            await store.record("kp", issues=["hedging language"])
        assert (await store.progress("kp"))["fixed_habits"] == []

    async def test_too_few_turns_to_claim_a_habit_is_fading(self, store):
        # Splitting four turns in half cannot support the claim honestly.
        for _ in range(2):
            await store.record("kp", issues=["hedging language"])
        for _ in range(2):
            await store.record("kp", issues=["passive voice"])
        assert (await store.progress("kp"))["fixed_habits"] == []


@pytest.mark.asyncio
class TestForget:
    async def test_forget_removes_everything_for_one_handle(self, store):
        await store.record("alice", overall_score=70)
        await store.record("alice", overall_score=75)
        await store.record("bob", overall_score=60)

        assert await store.forget("alice") == 2
        assert (await store.progress("alice"))["turns"] == 0
        assert (await store.progress("bob"))["turns"] == 1

    async def test_forgetting_an_unknown_handle_is_harmless(self, store):
        assert await store.forget("ghost") == 0


@pytest.mark.asyncio
class TestDurability:
    async def test_history_survives_a_new_store_on_the_same_file(self, tmp_path):
        path = str(tmp_path / "coach.db")
        first = ProfileStore(path)
        await first.record("kp", overall_score=61, issues=["filler words"])

        # A fresh process pointed at the same file must recognise the user.
        reopened = ProfileStore(path)
        progress = await reopened.progress("kp")
        assert progress["turns"] == 1
        assert progress["latest_score"] == 61

    async def test_an_unwritable_database_does_not_raise(self, tmp_path):
        # The write is a side effect of coaching; it must never cost an answer.
        # A regular file where the parent directory should be makes the whole
        # path unusable, which is the shape a read-only deploy fails in.
        wall = tmp_path / "wall"
        wall.write_text("not a directory")

        blocked = ProfileStore(str(wall / "coach.db"))
        assert await blocked.record("kp", overall_score=70) is False
        assert (await blocked.progress("kp"))["turns"] == 0
