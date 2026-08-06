"""Pydantic models shared by the agent core and the REST layer."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Intent(StrEnum):
    EMAIL_WRITING = "email_writing"
    INTERVIEW_PRACTICE = "interview_practice"
    GRAMMAR_CORRECTION = "grammar_correction"
    TONE_IMPROVEMENT = "tone_improvement"
    PUBLIC_SPEAKING = "public_speaking"
    CONFLICT_RESOLUTION = "conflict_resolution"
    CUSTOMER_COMMUNICATION = "customer_communication"
    GENERAL_COACHING = "general_coaching"


INTENT_DESCRIPTIONS: dict[Intent, str] = {
    Intent.EMAIL_WRITING: "User wants to write, draft or rewrite a professional email or message.",
    Intent.INTERVIEW_PRACTICE: "User wants to practise interview answers or prepare for an interview.",
    Intent.GRAMMAR_CORRECTION: "User wants grammar, spelling or sentence-structure corrections.",
    Intent.TONE_IMPROVEMENT: "User wants the tone, politeness or formality of a message adjusted.",
    Intent.PUBLIC_SPEAKING: "User wants help with presentations, speeches, pitches or stage delivery.",
    Intent.CONFLICT_RESOLUTION: "User needs to handle a disagreement, complaint or difficult conversation.",
    Intent.CUSTOMER_COMMUNICATION: "User is writing to a customer or client: support replies, updates, apologies.",
    Intent.GENERAL_COACHING: "General communication advice that does not fit a more specific intent.",
}


class ToolName(StrEnum):
    GRAMMAR_CORRECTION = "grammar_correction"
    TONE_ANALYSIS = "tone_analysis"
    EMAIL_GENERATION = "email_generation"
    INTERVIEW_COACHING = "interview_coaching"
    CONVERSATION_IMPROVEMENT = "conversation_improvement"
    COMMUNICATION_SCORING = "communication_scoring"
    KNOWLEDGE_LOOKUP = "knowledge_lookup"
    RESUME_ANALYSIS = "resume_analysis"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


# --------------------------------------------------------------------------
# Agent pipeline models
# --------------------------------------------------------------------------


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    entities: dict[str, Any] = Field(default_factory=dict)
    method: Literal["llm", "heuristic", "memory-carryover"] = "heuristic"


class PlanStep(BaseModel):
    tool: ToolName
    objective: str
    depends_on_previous: bool = False


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]
    rationale: str = ""
    method: Literal["llm", "rule-based"] = "rule-based"


class ToolResult(BaseModel):
    tool: ToolName
    ok: bool = True
    summary: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None


class ScoreBreakdown(BaseModel):
    clarity: int = Field(ge=0, le=100)
    tone: int = Field(ge=0, le=100)
    grammar: int = Field(ge=0, le=100)
    structure: int = Field(ge=0, le=100)
    impact: int = Field(ge=0, le=100)

    @property
    def overall(self) -> int:
        values = [self.clarity, self.tone, self.grammar, self.structure, self.impact]
        return round(sum(values) / len(values))


class CoachingResponse(BaseModel):
    session_id: str
    trace_id: str
    intent: IntentResult
    plan: Plan
    tools_used: list[ToolName]
    coaching_message: str
    improved_response: str | None = None
    feedback: list[str] = Field(default_factory=list)
    score: ScoreBreakdown | None = None
    overall_score: int | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    knowledge_used: list[str] = Field(default_factory=list)
    provider: str = "heuristic"
    duration_ms: float = 0.0
    created_at: datetime = Field(default_factory=_now)


class Turn(BaseModel):
    role: Role
    content: str
    intent: Intent | None = None
    tools_used: list[ToolName] = Field(default_factory=list)
    overall_score: int | None = None
    created_at: datetime = Field(default_factory=_now)


class SessionSnapshot(BaseModel):
    session_id: str
    turns: list[Turn]
    summary: str = ""
    recurring_issues: list[str] = Field(default_factory=list)
    average_score: float | None = None
    turn_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------
# API request / response models
# --------------------------------------------------------------------------


class CoachRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(
        default=None, description="Omit to start a new coaching session."
    )
    goal: str | None = Field(
        default=None, description="Optional explicit communication goal."
    )
    audience: str | None = Field(
        default=None, description="Who the message is for, e.g. 'my manager'."
    )


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    context: str | None = None
    session_id: str | None = None


class AnalyzeResponse(BaseModel):
    trace_id: str
    intent: IntentResult
    score: ScoreBreakdown
    overall_score: int
    grammar_issues: list[dict[str, Any]] = Field(default_factory=list)
    tone: dict[str, Any] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    provider: str
    duration_ms: float


class ImproveRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    target_tone: str = Field(
        default="professional",
        description="e.g. professional, friendly, assertive, empathetic, concise.",
    )
    audience: str | None = None
    session_id: str | None = None


class ImproveResponse(BaseModel):
    trace_id: str
    original: str
    improved: str
    changes: list[str] = Field(default_factory=list)
    score_before: ScoreBreakdown | None = None
    score_after: ScoreBreakdown | None = None
    provider: str
    duration_ms: float


class HistoryResponse(BaseModel):
    session_id: str
    turn_count: int
    summary: str
    recurring_issues: list[str]
    average_score: float | None
    turns: list[Turn]


class ResumeAnalysisResponse(BaseModel):
    trace_id: str
    filename: str
    pages: int
    detected_sections: list[str]
    strengths: list[str]
    weaknesses: list[str]
    suggested_interview_questions: list[str]
    communication_score: ScoreBreakdown
    summary: str
    provider: str


class InterviewStartRequest(BaseModel):
    role: str = Field(default="Software Engineer", max_length=200)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question_count: int = Field(default=5, ge=1, le=15)
    session_id: str | None = None


class InterviewQuestion(BaseModel):
    index: int
    question: str
    competency: str = "general"


class InterviewStartResponse(BaseModel):
    session_id: str
    role: str
    difficulty: str
    questions: list[InterviewQuestion]


class InterviewAnswerRequest(BaseModel):
    session_id: str
    question: str
    answer: str = Field(min_length=1, max_length=8000)


class InterviewAnswerResponse(BaseModel):
    trace_id: str
    session_id: str
    question: str
    score: ScoreBreakdown
    overall_score: int
    star_coverage: dict[str, bool]
    feedback: list[str]
    model_answer: str
    provider: str


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    llm_available: bool
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
    trace_id: str
