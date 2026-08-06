"""Resume analysis tool (bonus feature).

The parsed resume is placed on `ctx.options["resume"]` by the API layer, so the
tool never touches the filesystem or the upload itself.
"""

from __future__ import annotations

from app.nlp.scoring import score_text
from app.nlp.textstats import analyze_text
from app.nlp.tone import analyze_tone
from app.resume.parser import ResumeDocument
from app.schemas import ToolName, ToolResult
from app.tools.base import Tool, ToolContext

_EXPECTED_SECTIONS = ("summary", "experience", "education", "skills")


class ResumeAnalysisTool(Tool):
    name = ToolName.RESUME_ANALYSIS
    description = (
        "Analyses an uploaded PDF resume for structure, impact and evidence, and "
        "generates interview questions an interviewer would ask from it."
    )
    when_to_use = "the user uploads a resume or asks for CV feedback."

    def run_offline(self, ctx: ToolContext) -> ToolResult:
        doc = ctx.options.get("resume")
        if not isinstance(doc, ResumeDocument):
            return ToolResult(
                tool=self.name,
                ok=False,
                summary="No resume was supplied.",
                error="resume_missing",
            )

        strengths, weaknesses = self._structural_findings(doc)
        stats = analyze_text(doc.text)
        tone = analyze_tone(doc.text, stats)
        assessment = score_text(stats, tone, [], context="general")

        return ToolResult(
            tool=self.name,
            summary=(
                f"Analysed a {doc.pages}-page resume: "
                f"{len(doc.bullets)} bullets, {doc.bullets_with_metrics} quantified."
            ),
            output={
                "pages": doc.pages,
                "word_count": doc.word_count,
                "detected_sections": doc.sections,
                "bullet_count": len(doc.bullets),
                "bullets_with_metrics": doc.bullets_with_metrics,
                "weak_bullets": doc.weak_bullets[:6],
                "strengths": strengths + assessment.strengths[:2],
                "weaknesses": weaknesses + assessment.weaknesses[:2],
                "score": assessment.score.model_dump(),
                "overall_score": assessment.overall,
                "suggested_interview_questions": self._questions_from_resume(doc),
                "summary": self._offline_summary(doc),
                "engine": "rules",
            },
        )

    async def run_llm(self, ctx: ToolContext) -> ToolResult:
        result = self.run_offline(ctx)
        if not result.ok:
            return result

        doc: ResumeDocument = ctx.options["resume"]
        sample = "\n".join(f"- {b}" for b in doc.bullets[:25]) or doc.text[:2500]

        prompt = f"""Review this resume as a hiring manager.

Detected sections: {doc.sections}
Bullets: {len(doc.bullets)}, of which {doc.bullets_with_metrics} contain a metric.
Bullets opening with weak phrasing: {doc.weak_bullets[:5]}

RESUME CONTENT:
\"\"\"{sample}\"\"\"

Return JSON:
{{
  "summary": "three sentences: what this candidate is, what stands out, what is missing",
  "strengths": ["at most 4, specific to this resume"],
  "weaknesses": ["at most 5, each naming the bullet or section at fault"],
  "rewritten_bullets": [{{"before": "...", "after": "stronger version using only their facts"}}],
  "interview_questions": ["5 questions an interviewer would ask from THIS resume"]
}}

Never invent achievements, employers or numbers the resume does not contain."""

        data = await ctx.llm.complete_json(
            prompt,
            system=(
                "You are a hiring manager reviewing resumes. You are specific and "
                "you never fabricate candidate experience."
            ),
            temperature=0.4,
        )

        questions = [str(q) for q in data.get("interview_questions", []) if q]
        result.output.update(
            {
                "summary": str(data.get("summary")) or result.output["summary"],
                "strengths": [str(s) for s in data.get("strengths", [])][:4]
                or result.output["strengths"],
                "weaknesses": [str(w) for w in data.get("weaknesses", [])][:5]
                or result.output["weaknesses"],
                "rewritten_bullets": [
                    b for b in data.get("rewritten_bullets", []) if isinstance(b, dict)
                ][:5],
                "suggested_interview_questions": questions[:6]
                or result.output["suggested_interview_questions"],
                "engine": ctx.llm.name,
            }
        )
        return result

    # -- helpers -----------------------------------------------------------

    def _structural_findings(self, doc: ResumeDocument) -> tuple[list[str], list[str]]:
        strengths: list[str] = []
        weaknesses: list[str] = []

        missing = [s for s in _EXPECTED_SECTIONS if s not in doc.sections]
        if missing:
            weaknesses.append(f"Missing standard section(s): {', '.join(missing)}.")
        else:
            strengths.append("All standard resume sections are present.")

        if not doc.has_contact:
            weaknesses.append("No email address found — recruiters cannot reach you.")

        if doc.bullets:
            ratio = doc.bullets_with_metrics / len(doc.bullets)
            if ratio < 0.3:
                weaknesses.append(
                    f"Only {doc.bullets_with_metrics} of {len(doc.bullets)} bullets "
                    "contain a number. Quantified impact is what gets read."
                )
            else:
                strengths.append(
                    f"{doc.bullets_with_metrics} bullets carry measurable results."
                )

        if doc.weak_bullets:
            weaknesses.append(
                f"{len(doc.weak_bullets)} bullet(s) open with passive phrasing "
                f'like "{doc.weak_bullets[0][:60]}" — start with an action verb.'
            )
        if doc.strong_bullets:
            strengths.append(
                f"{len(doc.strong_bullets)} bullet(s) open with a strong action verb."
            )

        if doc.pages > 2:
            weaknesses.append(
                f"{doc.pages} pages is long; most reviewers read the first page only."
            )
        if doc.word_count > 1200:
            weaknesses.append(
                f"{doc.word_count} words is dense — cut to the strongest evidence."
            )
        return strengths, weaknesses

    def _questions_from_resume(self, doc: ResumeDocument) -> list[str]:
        questions = [
            "Walk me through your background in two minutes.",
            "Which of the projects on here are you proudest of, and what was your "
            "specific contribution?",
        ]
        for bullet in doc.bullets[:3]:
            questions.append(
                f'You wrote: "{bullet[:110]}". What was the measurable outcome, '
                "and what would you do differently?"
            )
        if not doc.bullets_with_metrics:
            questions.append(
                "None of your bullets carry numbers — can you quantify the impact "
                "of your last role?"
            )
        return questions[:6]

    def _offline_summary(self, doc: ResumeDocument) -> str:
        return (
            f"{doc.pages}-page resume, {doc.word_count} words, "
            f"{len(doc.bullets)} bullet points, {doc.bullets_with_metrics} of them "
            f"quantified. Sections found: {', '.join(doc.sections) or 'none detected'}."
        )
