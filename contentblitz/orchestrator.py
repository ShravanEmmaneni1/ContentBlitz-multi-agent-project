"""Coordinates Research → vector store → Blog, LinkedIn, and Image agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ingest_se401k import ensure_se401k_brochure_ingested
from rag import self_employed_401k_rag_query
from vector_store import ContentVectorStore
from config import MCP_ENABLED

try:
    from fastmcp import Client

    from mcp_agent_server import _run_full_pipeline_impl, mcp as mcp_server

    _HAVE_FASTMCP = True
except Exception:
    Client = None  # type: ignore[assignment]
    mcp_server = None  # type: ignore[assignment]
    _run_full_pipeline_impl = None  # type: ignore[assignment]
    _HAVE_FASTMCP = False

if TYPE_CHECKING:
    from agents.blog_agent import BlogOutput, BlogWritingAgent
    from agents.image_agent import ImageCreationAgent, ImageResult
    from agents.linkedin_agent import LinkedInAgent, LinkedInOutput
    from agents.research_agent import ResearchAgent, ResearchResult


@dataclass
class PipelineResult:
    topic: str
    research: "ResearchResult"
    blog: "BlogOutput"
    linkedin: "LinkedInOutput"
    image: "ImageResult"
    vector_doc_ids: list[str]


class ContentBlitzOrchestrator:
    """Lazy-loads agents so modes only require the API keys they use."""

    def __init__(self, store: ContentVectorStore | None = None) -> None:
        self.store = store or ContentVectorStore()
        self._research: Any = None
        self._blog: Any = None
        self._linkedin: Any = None
        self._image: Any = None
        self.last_trace: list[dict[str, str]] = []

    def _trace(self, event_type: str, name: str, detail: str = "") -> None:
        self.last_trace.append({"type": event_type, "name": name, "detail": detail})

    def _call_mcp_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        self._trace("tool", tool_name, "attempt")
        if not MCP_ENABLED or not _HAVE_FASTMCP or Client is None or mcp_server is None:
            self._trace("tool", tool_name, "skipped (mcp unavailable)")
            return None

        import asyncio

        async def _inner() -> dict[str, Any]:
            async with Client(mcp_server) as client:
                res = await client.call_tool(tool_name, args)
                return res.data  # type: ignore[no-any-return]

        try:
            out = asyncio.run(_inner())
            self._trace("tool", tool_name, "ok")
            return out
        except RuntimeError:
            # If an event loop is already running, fall back to direct implementation.
            self._trace("tool", tool_name, "fallback (runtime loop)")
            if tool_name == "run_full_pipeline" and _run_full_pipeline_impl is not None:
                return _run_full_pipeline_impl(**args)
            return None

    @property
    def research_agent(self) -> "ResearchAgent":
        if self._research is None:
            from agents.research_agent import ResearchAgent

            self._research = ResearchAgent()
        return self._research

    @property
    def blog_agent(self) -> "BlogWritingAgent":
        if self._blog is None:
            from agents.blog_agent import BlogWritingAgent

            self._blog = BlogWritingAgent()
        return self._blog

    @property
    def linkedin_agent(self) -> "LinkedInAgent":
        if self._linkedin is None:
            from agents.linkedin_agent import LinkedInAgent

            self._linkedin = LinkedInAgent()
        return self._linkedin

    @property
    def image_agent(self) -> "ImageCreationAgent":
        if self._image is None:
            from agents.image_agent import ImageCreationAgent

            self._image = ImageCreationAgent()
        return self._image

    def run_full_pipeline(
        self,
        topic: str,
        *,
        audience: str | None = None,
        research_depth: str = "balanced",
        blog_tone: str = "professional",
        target_words: int = 1200,
        image_style: str = "modern, clean, professional",
        linkedin_cta: str | None = None,
    ) -> PipelineResult:
        self.last_trace = []
        self._trace("orchestration", "run_full_pipeline", "start")
        mcp_result = self._call_mcp_tool(
            "run_full_pipeline",
            {
                "topic": topic,
                "audience": audience or "",
                "research_depth": research_depth,
                "blog_tone": blog_tone,
                "target_words": target_words,
                "image_style": image_style,
                "linkedin_cta": linkedin_cta or "",
            },
        )
        if mcp_result is not None:
            from agents.blog_agent import BlogOutput
            from agents.image_agent import ImageResult
            from agents.linkedin_agent import LinkedInOutput
            from agents.research_agent import ResearchResult

            self._trace("orchestration", "run_full_pipeline", "completed via MCP")
            mcp_trace = list(mcp_result.get("trace") or [])
            if mcp_trace:
                self.last_trace = mcp_trace
            return PipelineResult(
                topic=mcp_result["topic"],
                research=ResearchResult(**mcp_result["research"]),
                blog=BlogOutput(**mcp_result["blog"]),
                linkedin=LinkedInOutput(**mcp_result["linkedin"]),
                image=ImageResult(**mcp_result["image"]),
                vector_doc_ids=list(mcp_result.get("vector_doc_ids") or []),
            )

        self._trace("orchestration", "ingest_brochure", "start")
        ensure_se401k_brochure_ingested(self.store)
        self._trace("orchestration", "ingest_brochure", "done")

        self._trace("agent", "research_agent", "start")
        research = self.research_agent.research(
            topic, depth=research_depth, audience=audience
        )
        self._trace("agent", "research_agent", f"done ({research.model})")
        doc_ids: list[str] = []
        doc_ids.append(
            self.store.add_document(
                research.summary,
                "research_summary",
                metadata={"topic": topic},
            )
        )
        self._trace("tool", "vector_store.add_document", "research_summary")

        rag_q = self_employed_401k_rag_query(topic)
        self._trace("tool", "vector_store.query_context", "self-employed 401k query")
        vec_snippets = self.store.query_context(rag_q, n_results=8)
        vector_context = "\n".join(vec_snippets)
        rag_for_social = "\n".join(vec_snippets[:6])
        rag_for_image = "\n".join(vec_snippets[:4])

        self._trace("agent", "blog_agent", "start")
        blog = self.blog_agent.write(
            topic=topic,
            research_summary=research.summary,
            vector_context=vector_context,
            tone=blog_tone,
            target_words=target_words,
        )
        self._trace("agent", "blog_agent", "done")
        doc_ids.append(
            self.store.add_document(
                blog.body_markdown[:8000],
                "blog_draft",
                metadata={"title": blog.title},
            )
        )
        self._trace("tool", "vector_store.add_document", "blog_draft")

        source_for_li = f"{blog.title}\n\n{blog.body_markdown}"
        self._trace("agent", "linkedin_agent", "start")
        linkedin = self.linkedin_agent.generate(
            topic=topic,
            source_text=source_for_li,
            cta=linkedin_cta,
            rag_context=rag_for_social,
        )
        self._trace("agent", "linkedin_agent", "done")

        self._trace("agent", "image_agent", "start")
        image = self.image_agent.create(
            topic=topic,
            style_hints=image_style,
            rag_context=rag_for_image,
        )
        self._trace("agent", "image_agent", "done")
        self._trace("orchestration", "run_full_pipeline", "completed")

        return PipelineResult(
            topic=topic,
            research=research,
            blog=blog,
            linkedin=linkedin,
            image=image,
            vector_doc_ids=doc_ids,
        )

    def research_only(
        self, topic: str, audience: str | None = None, depth: str = "balanced"
    ) -> "ResearchResult":
        self.last_trace = []
        self._trace("orchestration", "research_only", "start")
        mcp_result = self._call_mcp_tool(
            "run_research_only",
            {"topic": topic, "audience": audience or "", "depth": depth},
        )
        if mcp_result is not None:
            from agents.research_agent import ResearchResult

            self._trace("orchestration", "research_only", "completed via MCP")
            mcp_trace = list(mcp_result.get("trace") or [])
            if mcp_trace:
                self.last_trace = mcp_trace
            payload = dict(mcp_result)
            payload.pop("trace", None)
            return ResearchResult(**payload)
        self._trace("agent", "research_agent", "start")
        out = self.research_agent.research(topic, depth=depth, audience=audience)
        self._trace("agent", "research_agent", f"done ({out.model})")
        self._trace("orchestration", "research_only", "completed")
        return out

    def image_only(self, topic: str, image_style: str) -> "ImageResult":
        self.last_trace = []
        self._trace("orchestration", "image_only", "start")
        mcp_result = self._call_mcp_tool(
            "run_image_only",
            {"topic": topic, "image_style": image_style},
        )
        if mcp_result is not None:
            from agents.image_agent import ImageResult

            self._trace("orchestration", "image_only", "completed via MCP")
            mcp_trace = list(mcp_result.get("trace") or [])
            if mcp_trace:
                self.last_trace = mcp_trace
            payload = dict(mcp_result)
            payload.pop("trace", None)
            return ImageResult(**payload)

        # Fallback: direct implementation.
        self._trace("orchestration", "ingest_brochure", "start")
        ensure_se401k_brochure_ingested(self.store)
        self._trace("orchestration", "ingest_brochure", "done")
        rag_q = self_employed_401k_rag_query(topic)
        self._trace("tool", "vector_store.query_context", "image query")
        vec_snippets = self.store.query_context(rag_q, n_results=4)
        rag_for_image = "\n".join(vec_snippets[:4])
        self._trace("agent", "image_agent", "start")
        out = self.image_agent.create(
            topic=topic,
            style_hints=image_style,
            rag_context=rag_for_image,
        )
        self._trace("agent", "image_agent", "done")
        self._trace("orchestration", "image_only", "completed")
        return out

    def store_payload(self, text: str, doc_type: str, metadata: dict[str, Any] | None = None) -> str:
        return self.store.add_document(text, doc_type, metadata=metadata)
