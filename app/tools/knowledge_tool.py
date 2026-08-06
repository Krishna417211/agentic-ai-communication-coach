"""Knowledge lookup tool (RAG over the bundled coaching corpus)."""

from __future__ import annotations

from app.rag.knowledge import get_knowledge_base
from app.schemas import ToolName, ToolResult
from app.tools.base import Tool, ToolContext


class KnowledgeLookupTool(Tool):
    name = ToolName.KNOWLEDGE_LOOKUP
    description = (
        "Retrieves relevant guidance from a curated communication-coaching "
        "knowledge base so advice cites an established principle rather than "
        "being improvised."
    )
    when_to_use = (
        "the user asks how or why something works, wants best practice, or the "
        "coaching would benefit from a named framework."
    )

    def run_offline(self, ctx: ToolContext) -> ToolResult:
        query = " ".join(
            part for part in (ctx.goal, ctx.user_message, ctx.intent.value) if part
        )
        top_k = int(ctx.options.get("top_k", 3))
        passages = get_knowledge_base().search(query, top_k=top_k)

        return ToolResult(
            tool=self.name,
            summary=(
                f"Retrieved {len(passages)} reference passage(s)."
                if passages
                else "No relevant reference material found."
            ),
            output={
                "query": query[:300],
                "passages": passages,
                "citations": [f"{p['source']} — {p['title']}" for p in passages],
                "engine": "bm25",
            },
        )
