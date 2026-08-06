"""Tests for intent detection, planning, memory and the orchestrator."""

from __future__ import annotations

import pytest

from app.agent.extract import extract_all, extract_target_text, extract_target_tone
from app.agent.intent import IntentDetector, classify_heuristic
from app.agent.orchestrator import CommunicationAgent
from app.agent.planner import CommunicationPlanner
from app.config import Settings
from app.llm.base import extract_json
from app.llm.heuristic_provider import HeuristicProvider
from app.memory.store import SessionStore
from app.schemas import Intent, Role, ToolName
from app.tools import build_registry
from tests.conftest import FakeLLM


class TestIntentHeuristic:
    @pytest.mark.parametrize(
        "message,expected",
        [
            ("Help me write an email to my manager", Intent.EMAIL_WRITING),
            ("I have an interview tomorrow, can we do a mock interview?", Intent.INTERVIEW_PRACTICE),
            ("Can you proofread this for grammar mistakes?", Intent.GRAMMAR_CORRECTION),
            ("Does this sound too harsh? I want a softer tone.", Intent.TONE_IMPROVEMENT),
            ("I'm nervous about my presentation to the whole company", Intent.PUBLIC_SPEAKING),
            ("My colleague and I had an argument and I need to confront him", Intent.CONFLICT_RESOLUTION),
            ("An angry customer is demanding a refund after the outage", Intent.CUSTOMER_COMMUNICATION),
        ],
    )
    def test_routes_correctly(self, message, expected):
        assert classify_heuristic(message).intent == expected

    def test_unknown_falls_back_to_general(self):
        result = classify_heuristic("hello there")
        assert result.intent == Intent.GENERAL_COACHING
        assert result.confidence < 0.6

    def test_instruction_outranks_the_pasted_draft(self):
        # The draft is full of conflict vocabulary, but the user asked a tone
        # question — the question is what they want answered.
        message = (
            'Does this sound too harsh? "You never send the numbers on time. '
            'This is unacceptable and it is your fault."'
        )
        assert classify_heuristic(message).intent == Intent.TONE_IMPROVEMENT

    def test_overlapping_patterns_do_not_double_count(self):
        # "email" matches two EMAIL_WRITING patterns; that single signal must
        # not outweigh the two distinct GRAMMAR_CORRECTION signals.
        result = classify_heuristic("Please proofread my email for typos")
        assert result.intent == Intent.GRAMMAR_CORRECTION

    def test_confidence_is_bounded(self):
        for message in ("email interview grammar tone presentation conflict customer", "x"):
            assert 0.0 <= classify_heuristic(message).confidence <= 1.0


class TestExtraction:
    def test_pulls_quoted_draft(self):
        message = 'Can you fix this: "Their going to the meeting tommorow and its late"'
        assert "Their going" in extract_target_text(message)

    def test_pulls_draft_after_blank_line(self):
        message = "Make this better\n\nHi Sam, I just wanted to check on the report you sent last week."
        assert extract_target_text(message).startswith("Hi Sam")

    def test_no_draft_in_a_pure_request(self):
        assert extract_target_text("Help me write an email to my boss") == ""

    @pytest.mark.parametrize(
        "message,tone",
        [
            ("make it sound more professional", "professional"),
            ("can you make it friendlier and more polite", "professional"),
            ("rewrite this to be more assertive", "assertive"),
        ],
    )
    def test_target_tone(self, message, tone):
        assert extract_target_tone(message) is not None

    def test_extracts_audience_and_role(self):
        entities = extract_all("I'm interviewing for a Backend Engineer role and writing to my manager")
        assert entities["audience"] == "my manager"
        assert entities["role"] and "Backend Engineer" in entities["role"]


@pytest.mark.asyncio
class TestIntentDetector:
    async def test_offline_uses_heuristic(self):
        detector = IntentDetector(HeuristicProvider())
        result = await detector.detect("Please proofread my email for typos")
        assert result.method == "heuristic"
        assert result.intent == Intent.GRAMMAR_CORRECTION

    async def test_llm_result_is_used(self):
        llm = FakeLLM({"Classify the user's": {
            "intent": "conflict_resolution", "confidence": 0.9, "rationale": "test"
        }})
        result = await IntentDetector(llm).detect("something ambiguous")
        assert result.intent == Intent.CONFLICT_RESOLUTION
        assert result.method == "llm"

    async def test_unknown_llm_intent_falls_back(self):
        llm = FakeLLM({"Classify": {"intent": "make_coffee", "confidence": 0.99}})
        result = await IntentDetector(llm).detect("Please proofread this email")
        assert result.intent == Intent.GRAMMAR_CORRECTION
        assert result.method == "heuristic"

    async def test_provider_failure_falls_back(self):
        result = await IntentDetector(FakeLLM(fail=True)).detect("proofread my email")
        assert result.method == "heuristic"

    async def test_followup_inherits_previous_intent(self):
        store = SessionStore()
        session = await store.get_or_create(None)
        await store.append_turn(
            session.session_id, role=Role.USER, content="write an email",
            intent=Intent.EMAIL_WRITING,
        )
        result = await IntentDetector(HeuristicProvider()).detect("make it shorter", session)
        assert result.intent == Intent.EMAIL_WRITING
        assert result.method == "memory-carryover"


@pytest.mark.asyncio
class TestPlanner:
    async def test_rule_based_plan_per_intent(self):
        planner = CommunicationPlanner(HeuristicProvider(), build_registry())
        plan = planner.rule_based_plan(Intent.CONFLICT_RESOLUTION, has_draft=True)
        tools = [s.tool for s in plan.steps]
        assert ToolName.TONE_ANALYSIS in tools
        assert ToolName.CONVERSATION_IMPROVEMENT in tools
        assert tools.index(ToolName.TONE_ANALYSIS) < tools.index(
            ToolName.CONVERSATION_IMPROVEMENT
        ), "analysis must run before rewriting"

    async def test_no_draft_skips_analysis_tools_but_keeps_scoring(self):
        planner = CommunicationPlanner(HeuristicProvider(), build_registry())
        tools = [s.tool for s in planner.rule_based_plan(Intent.EMAIL_WRITING, has_draft=False).steps]
        assert ToolName.GRAMMAR_CORRECTION not in tools
        assert ToolName.COMMUNICATION_SCORING in tools

    async def test_plan_is_never_empty(self):
        planner = CommunicationPlanner(HeuristicProvider(), build_registry())
        for intent in Intent:
            for has_draft in (True, False):
                assert planner.rule_based_plan(intent, has_draft=has_draft).steps

    async def test_hallucinated_tools_are_dropped(self):
        llm = FakeLLM({"planning stage": {
            "goal": "test",
            "steps": [
                {"tool": "send_email_for_real", "objective": "nope"},
                {"tool": "grammar_correction", "objective": "yes"},
            ],
        }})
        planner = CommunicationPlanner(llm, build_registry())
        from app.schemas import IntentResult

        intent = IntentResult(intent=Intent.GRAMMAR_CORRECTION, confidence=0.5)
        plan = await planner.plan(intent, "fix this", has_draft=True)
        assert [s.tool for s in plan.steps] == [ToolName.GRAMMAR_CORRECTION]

    async def test_empty_llm_plan_falls_back_to_rules(self):
        llm = FakeLLM({"planning stage": {"goal": "x", "steps": []}})
        planner = CommunicationPlanner(llm, build_registry())
        from app.schemas import IntentResult

        intent = IntentResult(intent=Intent.EMAIL_WRITING, confidence=0.5)
        plan = await planner.plan(intent, "write an email", has_draft=False)
        assert plan.method == "rule-based"
        assert plan.steps


@pytest.mark.asyncio
class TestMemory:
    async def test_creates_and_reuses_sessions(self):
        store = SessionStore()
        first = await store.get_or_create(None)
        again = await store.get_or_create(first.session_id)
        assert first.session_id == again.session_id

    async def test_trims_and_summarises(self):
        store = SessionStore(max_turns=4)
        session = await store.get_or_create(None)
        for i in range(10):
            await store.append_turn(
                session.session_id, role=Role.USER, content=f"message {i}",
                intent=Intent.EMAIL_WRITING,
            )
        snapshot = await store.snapshot(session.session_id)
        assert len(snapshot.turns) == 4
        assert snapshot.turn_count == 10
        assert "email writing" in snapshot.summary

    async def test_tracks_recurring_issues(self):
        store = SessionStore()
        session = await store.get_or_create(None)
        for _ in range(3):
            await store.append_turn(
                session.session_id, role=Role.ASSISTANT, content="feedback",
                issues=["Hedging makes the message sound uncertain."],
                overall_score=60,
            )
        assert any("hedging" in i.lower() for i in session.recurring_issues())

    async def test_score_trend(self):
        store = SessionStore()
        session = await store.get_or_create(None)
        for score in (50, 60, 75):
            await store.append_turn(
                session.session_id, role=Role.ASSISTANT, content="x", overall_score=score
            )
        assert "improving" in session.score_trend()
        assert session.average_score == pytest.approx(61.7, abs=0.1)

    async def test_clear(self):
        store = SessionStore()
        session = await store.get_or_create(None)
        assert await store.clear(session.session_id) is True
        assert await store.get(session.session_id) is None


@pytest.mark.asyncio
class TestOrchestrator:
    async def _agent(self, llm=None) -> CommunicationAgent:
        settings = Settings(log_file=None)
        return CommunicationAgent(
            llm=llm or HeuristicProvider(),
            registry=build_registry(),
            memory=SessionStore(),
            settings=settings,
        )

    async def test_full_pipeline_offline(self, hostile_message):
        agent = await self._agent()
        response = await agent.run(hostile_message)
        assert response.intent.intent == Intent.CONFLICT_RESOLUTION
        assert response.tools_used
        assert response.overall_score is not None
        assert response.improved_response
        assert "you never" not in response.improved_response.lower()

    async def test_memory_persists_across_turns(self):
        agent = await self._agent()
        first = await agent.run("Help me write an email to my manager about a raise")
        second = await agent.run("make it shorter", session_id=first.session_id)
        assert second.session_id == first.session_id
        assert second.intent.intent == Intent.EMAIL_WRITING
        snapshot = await agent.memory.snapshot(first.session_id)
        assert snapshot.turn_count == 4

    async def test_tool_failure_does_not_break_the_turn(self, monkeypatch):
        agent = await self._agent()
        tool = agent.registry.get(ToolName.COMMUNICATION_SCORING)

        def boom(ctx):
            raise RuntimeError("tool exploded")

        monkeypatch.setattr(tool, "run_offline", boom)
        response = await agent.run("Please proofread this note for me, it is short.")
        assert response.coaching_message
        failed = [r for r in response.tool_results if not r.ok]
        assert failed and failed[0].error

    async def test_llm_failure_degrades_to_rules(self, hostile_message):
        agent = await self._agent(FakeLLM(fail=True))
        response = await agent.run(hostile_message)
        assert response.coaching_message
        assert response.overall_score is not None

    async def test_scores_are_reproducible(self, sample_email):
        agent = await self._agent()
        first = await agent.run(sample_email)
        second = await agent.run(sample_email)
        assert first.overall_score == second.overall_score

    async def test_every_intent_completes(self):
        agent = await self._agent()
        messages = [
            "write an email to my client about the delay",
            "help me prepare for my interview next Tuesday",
            "proofread this: their going to be late tommorow",
            "does this sound rude? you need to fix this now",
            "I have to give a presentation to 100 people",
            "my coworker keeps interrupting me in meetings and I'm frustrated",
            "a customer is angry about a refund delay",
            "how do I communicate better at work",
        ]
        for message in messages:
            response = await agent.run(message)
            assert response.coaching_message, message
            assert response.plan.steps, message


class TestJsonRepair:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"a": 1}',
            '```json\n{"a": 1}\n```',
            'Sure! Here is the JSON:\n{"a": 1}',
            '{"a": 1,}',
            '```\n{"a": 1}\n```',
        ],
    )
    def test_extracts_json(self, raw):
        assert extract_json(raw) == {"a": 1}

    def test_returns_none_for_garbage(self):
        assert extract_json("no json here at all") is None
        assert extract_json("") is None
