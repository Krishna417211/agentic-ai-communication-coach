"""REST API routes."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.api.deps import AgentDep, MemoryDep, ProfilesDep, SettingsDep, TraceDep
from app.memory.store import normalise_issue
from app.nlp.grammar import check_grammar
from app.nlp.scoring import score_text
from app.nlp.textstats import analyze_text
from app.nlp.tone import analyze_tone
from app.observability.metrics import metrics
from app.rag.knowledge import get_knowledge_base
from app.resume.parser import ResumeParseError, parse_resume
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CoachRequest,
    CoachingResponse,
    HistoryResponse,
    ImproveRequest,
    ImproveResponse,
    Intent,
    InterviewAnswerRequest,
    InterviewAnswerResponse,
    InterviewQuestion,
    InterviewStartRequest,
    InterviewStartResponse,
    ResumeAnalysisResponse,
    ScoreBreakdown,
    ToolName,
)

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_RESUME_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Core required endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/coach",
    response_model=CoachingResponse,
    summary="Coaching response",
    description=(
        "Runs the full agentic pipeline: intent detection, planning, tool "
        "selection, execution and feedback synthesis, with session memory."
    ),
)
async def coach(
    payload: CoachRequest, agent: AgentDep, profiles: ProfilesDep, trace_id: TraceDep
) -> CoachingResponse:
    response = await agent.run(
        payload.message,
        session_id=payload.session_id,
        goal=payload.goal,
        audience=payload.audience,
        trace_id=trace_id,
    )
    if payload.handle:
        # Long-term memory: only the derived signal, never the message itself.
        await profiles.record(
            payload.handle,
            intent=response.intent.intent.value,
            overall_score=response.overall_score,
            issues=[normalise_issue(f) for f in response.feedback],
        )
    metrics.record_intent(response.intent.intent.value)
    metrics.record_llm_call(response.provider)
    for result in response.tool_results:
        metrics.record_tool(result.tool.value, result.ok, result.duration_ms)
    return response


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Communication analysis",
    description="Scores and diagnoses a piece of text without rewriting it.",
)
async def analyze(
    payload: AnalyzeRequest, agent: AgentDep, trace_id: TraceDep
) -> AnalyzeResponse:
    started = time.perf_counter()

    from app.agent.intent import classify_heuristic

    intent = classify_heuristic(payload.text)

    scoring = await agent.run_single_tool(
        ToolName.COMMUNICATION_SCORING,
        payload.text,
        session_id=payload.session_id,
        intent=intent.intent,
        options={"context": payload.context},
    )
    grammar = await agent.run_single_tool(
        ToolName.GRAMMAR_CORRECTION, payload.text, session_id=payload.session_id
    )
    tone = await agent.run_single_tool(
        ToolName.TONE_ANALYSIS,
        payload.text,
        session_id=payload.session_id,
        intent=intent.intent,
    )

    for result in (scoring, grammar, tone):
        metrics.record_tool(result.tool.value, result.ok, result.duration_ms)
    metrics.record_intent(intent.intent.value)

    if not scoring.ok:
        raise HTTPException(status_code=502, detail=scoring.error or "scoring failed")

    score = ScoreBreakdown(**scoring.output["score"])
    return AnalyzeResponse(
        trace_id=trace_id,
        intent=intent,
        score=score,
        overall_score=score.overall,
        grammar_issues=grammar.output.get("issues", []) if grammar.ok else [],
        tone=tone.output.get("tone", {}) if tone.ok else {},
        strengths=scoring.output.get("strengths", []),
        weaknesses=scoring.output.get("weaknesses", []),
        provider=agent.llm.name,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.post(
    "/improve",
    response_model=ImproveResponse,
    summary="Communication improvement",
    description="Rewrites text toward a target tone and reports the score delta.",
)
async def improve(
    payload: ImproveRequest, agent: AgentDep, trace_id: TraceDep
) -> ImproveResponse:
    started = time.perf_counter()

    result = await agent.run_single_tool(
        ToolName.CONVERSATION_IMPROVEMENT,
        payload.text,
        session_id=payload.session_id,
        audience=payload.audience,
        intent=Intent.TONE_IMPROVEMENT,
        options={"target_tone": payload.target_tone},
        use_knowledge=True,
    )
    metrics.record_tool(result.tool.value, result.ok, result.duration_ms)

    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error or "rewrite failed")

    improved = result.output.get("improved", payload.text)
    return ImproveResponse(
        trace_id=trace_id,
        original=payload.text,
        improved=improved,
        changes=result.output.get("changes", []),
        score_before=_score_of(payload.text),
        score_after=_score_of(improved),
        provider=agent.llm.name,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@router.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    summary="Chat history",
)
async def history(session_id: str, memory: MemoryDep) -> HistoryResponse:
    snapshot = await memory.snapshot(session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'.")
    return HistoryResponse(
        session_id=snapshot.session_id,
        turn_count=snapshot.turn_count,
        summary=snapshot.summary,
        recurring_issues=snapshot.recurring_issues,
        average_score=snapshot.average_score,
        turns=snapshot.turns,
    )


@router.get(
    "/progress/{handle}",
    summary="Long-term progress for a profile",
    description=(
        "Score trend, recurring habits and habits that are fading, aggregated "
        "across every session this handle has had."
    ),
)
async def progress(handle: str, profiles: ProfilesDep) -> dict:
    return await profiles.progress(handle)


@router.delete("/progress/{handle}", summary="Delete a profile's history")
async def forget_progress(handle: str, profiles: ProfilesDep) -> dict:
    removed = await profiles.forget(handle)
    return {"handle": handle, "removed": removed}


@router.delete("/history/{session_id}", summary="Clear a session")
async def clear_history(session_id: str, memory: MemoryDep) -> dict:
    cleared = await memory.clear(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'.")
    return {"session_id": session_id, "cleared": True}


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


@router.get("/tools", summary="List available tools")
async def list_tools(agent: AgentDep) -> dict:
    return {"tools": agent.registry.catalog()}


@router.get("/intents", summary="List recognised intents")
async def list_intents() -> dict:
    from app.schemas import INTENT_DESCRIPTIONS

    return {
        "intents": [
            {"id": intent.value, "description": desc}
            for intent, desc in INTENT_DESCRIPTIONS.items()
        ]
    }


@router.get("/sessions", summary="Active session count")
async def sessions(memory: MemoryDep) -> dict:
    return {"active_sessions": await memory.active_count()}


# ---------------------------------------------------------------------------
# Bonus: knowledge base, resume analysis, interview simulation
# ---------------------------------------------------------------------------


@router.get("/knowledge/search", summary="Search the coaching knowledge base")
async def knowledge_search(
    q: str = Query(min_length=2, max_length=400),
    top_k: int = Query(default=3, ge=1, le=10),
) -> dict:
    return {"query": q, "results": get_knowledge_base().search(q, top_k=top_k)}


@router.post(
    "/resume/analyze",
    response_model=ResumeAnalysisResponse,
    summary="PDF resume analysis",
)
async def analyze_resume(
    agent: AgentDep,
    settings: SettingsDep,
    trace_id: TraceDep,
    file: UploadFile = File(..., description="A text-based PDF resume."),
    session_id: str | None = Form(default=None),
) -> ResumeAnalysisResponse:
    filename = file.filename or "resume.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than {MAX_RESUME_BYTES // (1024 * 1024)}MB.",
        )

    try:
        document = parse_resume(data, max_pages=settings.max_resume_pages)
    except ResumeParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = await agent.run_single_tool(
        ToolName.RESUME_ANALYSIS,
        document.text[:6000],
        session_id=session_id,
        intent=Intent.INTERVIEW_PRACTICE,
        options={"resume": document},
    )
    metrics.record_tool(result.tool.value, result.ok, result.duration_ms)

    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error or "analysis failed")

    out = result.output
    return ResumeAnalysisResponse(
        trace_id=trace_id,
        filename=filename,
        pages=out["pages"],
        detected_sections=out["detected_sections"],
        strengths=out["strengths"],
        weaknesses=out["weaknesses"],
        suggested_interview_questions=out["suggested_interview_questions"],
        communication_score=ScoreBreakdown(**out["score"]),
        summary=out["summary"],
        provider=agent.llm.name,
    )


@router.post(
    "/interview/start",
    response_model=InterviewStartResponse,
    summary="Start an interview simulation",
)
async def interview_start(
    payload: InterviewStartRequest, agent: AgentDep, memory: MemoryDep
) -> InterviewStartResponse:
    session = await memory.get_or_create(payload.session_id)

    result = await agent.run_single_tool(
        ToolName.INTERVIEW_COACHING,
        f"Generate interview questions for a {payload.role}.",
        session_id=session.session_id,
        intent=Intent.INTERVIEW_PRACTICE,
        options={
            "role": payload.role,
            "difficulty": payload.difficulty,
            "question_count": payload.question_count,
        },
    )
    metrics.record_tool(result.tool.value, result.ok, result.duration_ms)

    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error or "generation failed")

    questions = [
        InterviewQuestion(
            index=i,
            question=q["question"] if isinstance(q, dict) else str(q),
            competency=(q.get("competency", "general") if isinstance(q, dict) else "general"),
        )
        for i, q in enumerate(result.output.get("questions", []), start=1)
    ]
    await memory.set_state(
        session.session_id,
        "interview",
        {
            "role": payload.role,
            "difficulty": payload.difficulty,
            "questions": [q.model_dump() for q in questions],
        },
    )
    return InterviewStartResponse(
        session_id=session.session_id,
        role=payload.role,
        difficulty=payload.difficulty,
        questions=questions,
    )


@router.post(
    "/interview/answer",
    response_model=InterviewAnswerResponse,
    summary="Submit an interview answer for review",
)
async def interview_answer(
    payload: InterviewAnswerRequest,
    agent: AgentDep,
    memory: MemoryDep,
    trace_id: TraceDep,
) -> InterviewAnswerResponse:
    session = await memory.get(payload.session_id)
    role = "the role"
    if session and isinstance(session.state.get("interview"), dict):
        role = session.state["interview"].get("role", role)

    result = await agent.run_single_tool(
        ToolName.INTERVIEW_COACHING,
        payload.answer,
        session_id=payload.session_id,
        intent=Intent.INTERVIEW_PRACTICE,
        options={"question": payload.question, "answer": payload.answer, "role": role},
        use_knowledge=True,
    )
    metrics.record_tool(result.tool.value, result.ok, result.duration_ms)

    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error or "review failed")

    out = result.output
    score = ScoreBreakdown(**out["score"])

    from app.schemas import Role as TurnRole

    await memory.append_turn(
        payload.session_id,
        role=TurnRole.USER,
        content=f"[Interview answer] {payload.answer}",
        intent=Intent.INTERVIEW_PRACTICE,
    )
    await memory.append_turn(
        payload.session_id,
        role=TurnRole.ASSISTANT,
        content="; ".join(out.get("feedback", []))[:2000],
        intent=Intent.INTERVIEW_PRACTICE,
        tools_used=[ToolName.INTERVIEW_COACHING],
        overall_score=score.overall,
        issues=out.get("feedback", []),
    )

    return InterviewAnswerResponse(
        trace_id=trace_id,
        session_id=payload.session_id,
        question=payload.question,
        score=score,
        overall_score=score.overall,
        star_coverage=out.get("star_coverage", {}),
        feedback=out.get("feedback", []),
        model_answer=out.get("model_answer", ""),
        provider=agent.llm.name,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _score_of(text: str) -> ScoreBreakdown:
    stats = analyze_text(text)
    tone = analyze_tone(text, stats)
    return score_text(stats, tone, check_grammar(text)).score
