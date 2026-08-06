"""Interview coaching tool: question generation and STAR-based answer review."""

from __future__ import annotations

import re

from app.nlp.scoring import score_text
from app.nlp.textstats import analyze_text
from app.nlp.tone import analyze_tone
from app.schemas import ToolName, ToolResult
from app.tools.base import Tool, ToolContext

# Cues that a STAR component is actually present in the answer.
_STAR_CUES: dict[str, tuple[str, ...]] = {
    "situation": (
        "at my", "when i was", "in my role", "we were", "the team was", "at the time",
        "our company", "during", "last year", "the project was", "context",
    ),
    "task": (
        "i was responsible", "my job was", "i had to", "i was asked", "my task",
        "the goal was", "i needed to", "i was tasked", "my responsibility",
    ),
    "action": (
        "i built", "i led", "i wrote", "i designed", "i proposed", "i organised",
        "i organized", "i implemented", "i decided", "i set up", "i created",
        "i ran", "i introduced", "i negotiated", "i rewrote", "i migrated",
        "i automated", "i coordinated", "so i ", "i started",
    ),
    "result": (
        "as a result", "which meant", "we reduced", "we increased", "we shipped",
        "the outcome", "resulted in", "improved by", "saved", "grew", "%",
        "cut the", "went from", "we delivered", "led to",
    ),
}

_QUESTION_BANK: dict[str, list[tuple[str, str]]] = {
    "behavioural": [
        ("Tell me about a time you disagreed with a teammate. How did you handle it?", "conflict"),
        ("Describe a project that failed. What did you take from it?", "ownership"),
        ("Tell me about a time you had to deliver under a tight deadline.", "delivery"),
        ("Give an example of feedback that was hard to hear.", "coachability"),
        ("Describe a time you had to influence someone without authority.", "influence"),
        ("Tell me about a decision you made with incomplete information.", "judgement"),
        ("Describe a time you had to explain something technical to a non-technical audience.", "communication"),
    ],
    "general": [
        ("Walk me through your background.", "self-presentation"),
        ("Why this role, and why now?", "motivation"),
        ("What are you looking for in your next team?", "fit"),
        ("Where do you see the biggest gap in your current skills?", "self-awareness"),
        ("What questions do you have for us?", "engagement"),
    ],
    "technical": [
        ("Walk me through a technical decision you're proud of.", "depth"),
        ("How do you approach debugging a problem you've never seen before?", "problem-solving"),
        ("How do you decide when something is good enough to ship?", "pragmatism"),
    ],
}

_HARD_FOLLOWUPS = [
    "What would you do differently if you ran that again?",
    "What was the hardest trade-off in that decision?",
    "How did you measure whether it actually worked?",
    "Who disagreed with you, and what was their argument?",
]


class InterviewCoachingTool(Tool):
    name = ToolName.INTERVIEW_COACHING
    description = (
        "Generates role-specific interview questions, and reviews answers against "
        "the STAR framework with a score, targeted feedback and a model answer."
    )
    when_to_use = (
        "the user is preparing for an interview, practising an answer, or asks how "
        "to talk about their experience."
    )

    def run_offline(self, ctx: ToolContext) -> ToolResult:
        question = str(ctx.options.get("question") or "").strip()
        answer = str(ctx.options.get("answer") or "").strip() or (
            ctx.target_text if question else ""
        )
        role = str(ctx.options.get("role") or "the role")

        if not answer:
            questions = self._generate_questions(
                role, int(ctx.options.get("question_count", 5))
            )
            return ToolResult(
                tool=self.name,
                summary=f"Generated {len(questions)} practice questions for {role}.",
                output={"mode": "questions", "role": role, "questions": questions,
                        "engine": "rules"},
            )

        coverage = detect_star(answer)
        stats = analyze_text(answer)
        tone = analyze_tone(answer, stats)
        assessment = score_text(stats, tone, [], context="interview")

        feedback = self._star_feedback(coverage, stats)
        feedback.extend(assessment.weaknesses[:3])

        return ToolResult(
            tool=self.name,
            summary=(
                f"Answer covers {sum(coverage.values())}/4 STAR components; "
                f"score {assessment.overall}/100."
            ),
            output={
                "mode": "review",
                "question": question,
                "answer": answer,
                "star_coverage": coverage,
                "score": assessment.score.model_dump(),
                "overall_score": assessment.overall,
                "strengths": assessment.strengths,
                "feedback": feedback[:6],
                "followup_questions": _HARD_FOLLOWUPS[:2],
                "model_answer": "",
                "engine": "rules",
            },
        )

    async def run_llm(self, ctx: ToolContext) -> ToolResult:
        question = str(ctx.options.get("question") or "").strip()
        answer = str(ctx.options.get("answer") or "").strip() or (
            ctx.target_text if question else ""
        )
        role = str(ctx.options.get("role") or "the role the user mentioned")

        if not answer:
            return await self._questions_llm(ctx, role)
        return await self._review_llm(ctx, role, question, answer)

    # -- LLM paths ---------------------------------------------------------

    async def _questions_llm(self, ctx: ToolContext, role: str) -> ToolResult:
        count = int(ctx.options.get("question_count", 5))
        difficulty = str(ctx.options.get("difficulty", "medium"))
        prompt = f"""Generate {count} interview questions for a {role} candidate at
{difficulty} difficulty. Mix behavioural, situational and role-specific questions.
Avoid generic filler like "what is your greatest weakness".

Session context:
{ctx.memory_brief()}

Return JSON:
{{"questions": [{{"question": "...", "competency": "what it tests"}}]}}"""

        data = await ctx.llm.complete_json(
            prompt,
            system="You are an experienced hiring manager.",
            temperature=0.7,
        )
        questions = [
            {"question": str(q.get("question", "")), "competency": str(q.get("competency", "general"))}
            for q in data.get("questions", [])
            if isinstance(q, dict) and q.get("question")
        ]
        if not questions:
            questions = self._generate_questions(role, count)
        return ToolResult(
            tool=self.name,
            summary=f"Generated {len(questions)} practice questions for {role}.",
            output={"mode": "questions", "role": role, "questions": questions,
                    "engine": ctx.llm.name},
        )

    async def _review_llm(
        self, ctx: ToolContext, role: str, question: str, answer: str
    ) -> ToolResult:
        coverage = detect_star(answer)
        stats = analyze_text(answer)
        tone = analyze_tone(answer, stats)
        assessment = score_text(stats, tone, [], context="interview")

        prompt = f"""Review this interview answer as the hiring manager for {role}.

QUESTION: {question or "(the user did not specify the question)"}

ANSWER:
\"\"\"{answer}\"\"\"

Measured signals (trust these):
- STAR coverage detected: {coverage}
- {stats.word_count} words, ~{stats.estimated_speaking_seconds:.0f}s spoken
- hedges: {stats.hedges or "none"}; filler: {stats.filler_words or "none"}
- computed score: {assessment.overall}/100

Session context:
{ctx.memory_brief()}

Be specific and honest — quote the candidate's own words when you critique them.

Return JSON:
{{
  "star_coverage": {{"situation": true, "task": true, "action": true, "result": true}},
  "strengths": ["..."],
  "feedback": ["specific, actionable critiques, at most 5"],
  "missing_specifics": ["what numbers or details the answer needs"],
  "model_answer": "a stronger version of THIS candidate's answer, using only facts they gave; mark invented detail as [placeholder]",
  "followup_questions": ["what an interviewer would probe next"]
}}"""

        data = await ctx.llm.complete_json(
            prompt,
            system=(
                "You are a hiring manager who gives blunt, useful interview feedback. "
                "You never invent achievements the candidate did not claim."
            ),
            temperature=0.4,
        )

        llm_coverage = data.get("star_coverage")
        if isinstance(llm_coverage, dict):
            coverage = {
                key: bool(llm_coverage.get(key, coverage[key])) for key in coverage
            }

        feedback = [str(f) for f in data.get("feedback", []) if f]
        feedback.extend(
            f"Add specifics: {s}" for s in data.get("missing_specifics", [])[:2]
        )
        if not feedback:
            feedback = self._star_feedback(coverage, stats)

        return ToolResult(
            tool=self.name,
            summary=(
                f"Answer covers {sum(coverage.values())}/4 STAR components; "
                f"score {assessment.overall}/100."
            ),
            output={
                "mode": "review",
                "question": question,
                "answer": answer,
                "star_coverage": coverage,
                "score": assessment.score.model_dump(),
                "overall_score": assessment.overall,
                "strengths": [str(s) for s in data.get("strengths", [])][:4]
                or assessment.strengths,
                "feedback": feedback[:6],
                "model_answer": str(data.get("model_answer", "")),
                "followup_questions": [
                    str(q) for q in data.get("followup_questions", [])
                ][:3]
                or _HARD_FOLLOWUPS[:2],
                "engine": ctx.llm.name,
            },
        )

    # -- helpers -----------------------------------------------------------

    def _generate_questions(self, role: str, count: int) -> list[dict[str, str]]:
        pool = (
            _QUESTION_BANK["behavioural"]
            + _QUESTION_BANK["general"]
            + _QUESTION_BANK["technical"]
        )
        return [
            {"question": q, "competency": c} for q, c in pool[: max(1, min(count, len(pool)))]
        ]

    def _star_feedback(self, coverage: dict[str, bool], stats) -> list[str]:
        feedback: list[str] = []
        labels = {
            "situation": "Set the scene in one sentence — where, when, who was involved.",
            "task": "State what you specifically were responsible for.",
            "action": "Describe what YOU did, in first person and in sequence.",
            "result": "Close with a measurable outcome — a number, a delta, a decision.",
        }
        for part, present in coverage.items():
            if not present:
                feedback.append(f"Missing the {part.upper()} of STAR. {labels[part]}")
        if stats.word_count < 60:
            feedback.append(
                f"At {stats.word_count} words this is too thin — aim for 150-250 "
                "words (about 60-90 seconds spoken)."
            )
        elif stats.estimated_speaking_seconds > 150:
            feedback.append(
                f"At ~{stats.estimated_speaking_seconds:.0f}s spoken this runs long; "
                "interviewers start tuning out after 2 minutes."
            )
        if stats.hedges:
            feedback.append(
                f"Hedging undercuts your credibility: {', '.join(stats.hedges[:3])}."
            )
        if not re.search(r"\d", " ".join(stats.words)) and stats.word_count > 40:
            feedback.append(
                "No numbers anywhere — quantified results are what interviewers remember."
            )
        return feedback


def detect_star(answer: str) -> dict[str, bool]:
    """Detect which STAR components appear in an answer."""
    lowered = answer.lower()
    coverage = {
        part: any(cue in lowered for cue in cues) for part, cues in _STAR_CUES.items()
    }
    # A percentage or a "from X to Y" phrasing is strong evidence of a result.
    if re.search(r"\d+\s*%|\bfrom \d+.{0,15}to \d+", lowered):
        coverage["result"] = True
    return coverage
