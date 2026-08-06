"""Tool registry assembly."""

from __future__ import annotations

from app.tools.base import Tool, ToolContext, ToolRegistry
from app.tools.conversation_tool import ConversationImprovementTool
from app.tools.email_tool import EmailGenerationTool
from app.tools.grammar_tool import GrammarCorrectionTool
from app.tools.interview_tool import InterviewCoachingTool
from app.tools.knowledge_tool import KnowledgeLookupTool
from app.tools.resume_tool import ResumeAnalysisTool
from app.tools.scoring_tool import CommunicationScoringTool
from app.tools.tone_tool import ToneAnalysisTool


def build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            GrammarCorrectionTool(),
            ToneAnalysisTool(),
            EmailGenerationTool(),
            InterviewCoachingTool(),
            ConversationImprovementTool(),
            CommunicationScoringTool(),
            KnowledgeLookupTool(),
            ResumeAnalysisTool(),
        ]
    )


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "build_registry",
    "CommunicationScoringTool",
    "ConversationImprovementTool",
    "EmailGenerationTool",
    "GrammarCorrectionTool",
    "InterviewCoachingTool",
    "KnowledgeLookupTool",
    "ResumeAnalysisTool",
    "ToneAnalysisTool",
]
