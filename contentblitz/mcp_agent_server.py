"""Local MCP server exposing agent and RAG tools.

Goal:
- allow orchestrator (and later, agents) to call shared tools via MCP
- keep persistence (vector store + brochure ingestion) centralized
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastmcp import FastMCP

from agents.blog_agent import BlogWritingAgent
from agents.image_agent import ImageCreationAgent
from agents.linkedin_agent import LinkedInAgent
from agents.research_agent import ResearchAgent
from ingest_se401k import ensure_se401k_brochure_ingested
from rag import self_employed_401k_rag_query
from vector_store import ContentVectorStore

mcp = FastMCP("contentblitz")


def _run_full_pipeline_impl(
    topic: str,
    audience: str = "",
    research_depth: str = "balanced",
    blog_tone: str = "professional",
    target_words: int = 1200,
    image_style: str = "modern, clean, professional",
    linkedin_cta: str = "",
) -> dict[str, Any]:
    trace: list[dict[str, str]] = []
    trace.append({"type": "orchestration", "name": "run_full_pipeline", "detail": "start (MCP server)"})

    store = ContentVectorStore()
    trace.append({"type": "orchestration", "name": "ingest_brochure", "detail": "start"})
    ensure_se401k_brochure_ingested(store)
    trace.append({"type": "orchestration", "name": "ingest_brochure", "detail": "done"})

    trace.append({"type": "agent", "name": "research_agent", "detail": "start"})
    research = ResearchAgent().research(
        topic, depth=research_depth, audience=(audience or None)
    )
    trace.append({"type": "agent", "name": "research_agent", "detail": f"done ({research.model})"})

    # Store research + generated content for future retrieval.
    doc_ids: list[str] = []
    doc_ids.append(
        store.add_document(
            research.summary,
            "research_summary",
            metadata={"topic": topic},
        )
    )
    trace.append({"type": "tool", "name": "vector_store.add_document", "detail": "research_summary"})

    rag_q = self_employed_401k_rag_query(topic)
    trace.append({"type": "tool", "name": "vector_store.query_context", "detail": "self-employed 401k query"})
    vec_snippets = store.query_context(rag_q, n_results=8)
    vector_context = "\n".join(vec_snippets)
    rag_for_social = "\n".join(vec_snippets[:6])
    rag_for_image = "\n".join(vec_snippets[:4])

    trace.append({"type": "agent", "name": "blog_agent", "detail": "start"})
    blog = BlogWritingAgent().write(
        topic=topic,
        research_summary=research.summary,
        vector_context=vector_context,
        tone=blog_tone,
        target_words=target_words,
    )
    trace.append({"type": "agent", "name": "blog_agent", "detail": "done"})
    doc_ids.append(
        store.add_document(
            blog.body_markdown[:8000],
            "blog_draft",
            metadata={"title": blog.title},
        )
    )
    trace.append({"type": "tool", "name": "vector_store.add_document", "detail": "blog_draft"})

    source_for_li = f"{blog.title}\n\n{blog.body_markdown}"
    trace.append({"type": "agent", "name": "linkedin_agent", "detail": "start"})
    linkedin = LinkedInAgent().generate(
        topic=topic,
        source_text=source_for_li,
        cta=(linkedin_cta or None),
        rag_context=rag_for_social,
    )
    trace.append({"type": "agent", "name": "linkedin_agent", "detail": "done"})
    trace.append({"type": "agent", "name": "image_agent", "detail": "start"})
    image = ImageCreationAgent().create(
        topic=topic,
        style_hints=image_style,
        rag_context=rag_for_image,
    )
    trace.append({"type": "agent", "name": "image_agent", "detail": "done"})
    trace.append({"type": "orchestration", "name": "run_full_pipeline", "detail": "completed (MCP server)"})

    return {
        "topic": topic,
        "research": asdict(research),
        "blog": asdict(blog),
        "linkedin": asdict(linkedin),
        "image": asdict(image),
        "vector_doc_ids": doc_ids,
        "trace": trace,
    }


@mcp.tool()
def run_full_pipeline(
    topic: str,
    audience: str = "",
    research_depth: str = "balanced",
    blog_tone: str = "professional",
    target_words: int = 1200,
    image_style: str = "modern, clean, professional",
    linkedin_cta: str = "",
) -> dict[str, Any]:
    """Run the whole pipeline (research → blog → LinkedIn → image)."""

    return _run_full_pipeline_impl(
        topic=topic,
        audience=audience,
        research_depth=research_depth,
        blog_tone=blog_tone,
        target_words=target_words,
        image_style=image_style,
        linkedin_cta=linkedin_cta,
    )


@mcp.tool()
def run_research_only(
    topic: str,
    audience: str = "",
    depth: str = "balanced",
) -> dict[str, Any]:
    """Run research only and return the research summary."""

    trace: list[dict[str, str]] = []
    trace.append({"type": "orchestration", "name": "research_only", "detail": "start (MCP server)"})
    trace.append({"type": "agent", "name": "research_agent", "detail": "start"})
    r = ResearchAgent().research(topic, depth=depth, audience=(audience or None))
    trace.append({"type": "agent", "name": "research_agent", "detail": f"done ({r.model})"})
    trace.append({"type": "orchestration", "name": "research_only", "detail": "completed (MCP server)"})
    out = asdict(r)
    out["trace"] = trace
    return out


@mcp.tool()
def run_image_only(
    topic: str,
    image_style: str = "modern, clean, professional",
) -> dict[str, Any]:
    """Retrieve brochure context and generate an image for the topic."""
    trace: list[dict[str, str]] = []
    trace.append({"type": "orchestration", "name": "image_only", "detail": "start (MCP server)"})

    store = ContentVectorStore()
    trace.append({"type": "orchestration", "name": "ingest_brochure", "detail": "start"})
    ensure_se401k_brochure_ingested(store)
    trace.append({"type": "orchestration", "name": "ingest_brochure", "detail": "done"})

    rag_q = self_employed_401k_rag_query(topic)
    trace.append({"type": "tool", "name": "vector_store.query_context", "detail": "image query"})
    vec_snippets = store.query_context(rag_q, n_results=4)
    rag_for_image = "\n".join(vec_snippets[:4])

    trace.append({"type": "agent", "name": "image_agent", "detail": "start"})
    image = ImageCreationAgent().create(
        topic=topic,
        style_hints=image_style,
        rag_context=rag_for_image,
    )
    trace.append({"type": "agent", "name": "image_agent", "detail": "done"})
    trace.append({"type": "orchestration", "name": "image_only", "detail": "completed (MCP server)"})
    out = asdict(image)
    out["trace"] = trace
    return out

