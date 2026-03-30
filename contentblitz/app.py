"""
ContentBlitz — Streamlit UI for the multi-agent content pipeline.
"""

from __future__ import annotations

import base64
import io

import streamlit as st

from ingest_se401k import ensure_se401k_brochure_ingested
from orchestrator import ContentBlitzOrchestrator
from rag import self_employed_401k_rag_query


def _image_to_bytes(image_result) -> bytes | None:
    if image_result.b64_json:
        return base64.b64decode(image_result.b64_json)
    return None


def _render_trace_panel(trace: list[dict[str, str]]) -> None:
    st.divider()
    st.subheader("Agent Orchestration")
    if not trace:
        st.caption("Run a mode to see orchestration and agent calls.")
        return

    orchestration = [t for t in trace if t.get("type") != "tool"]
    if not orchestration:
        st.caption("No orchestration or agent calls recorded.")
    for t in orchestration:
        st.code(f"[{t.get('type', '')}] {t.get('name', '')} — {t.get('detail', '')}")


def main() -> None:
    st.set_page_config(
        page_title="ContentBlitz",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ContentBlitz")
    st.caption(
        "Research, blog, LinkedIn, and visuals — orchestrated with GPT, "
        "SerpAPI, and a local FAISS vector index (OpenAI embeddings)."
    )

    with st.sidebar:
        st.subheader("Pipeline")
        mode = st.radio(
            "Mode",
            [
                "Full pipeline",
                "Research only",
                "Blog only",
                "LinkedIn only",
                "Image only",
            ],
            horizontal=False,
        )
        st.subheader("Inputs")
        topic = st.text_input("Topic / working title", placeholder="e.g. Edge AI in retail")
        audience = st.text_input("Audience (optional)", placeholder="Marketing leaders")
        research_depth = st.select_slider(
            "Research depth",
            options=["quick", "balanced", "deep"],
            value="balanced",
        )
        blog_tone = st.selectbox(
            "Blog tone",
            ["professional", "conversational", "technical", "friendly"],
        )
        target_words = st.slider("Target blog words", 600, 2500, 1200, step=100)
        image_style = st.text_input(
            "Image style hints",
            value="modern, clean, professional, subtle gradients",
        )
        linkedin_cta = st.text_input("LinkedIn CTA (optional)", placeholder="Ask a question…")

    tab_run, tab_help = st.tabs(["Run", "About"])

    with tab_run:
        if mode == "Full pipeline":
            if st.button("Run full pipeline", type="primary"):
                if not topic.strip():
                    st.warning("Enter a topic.")
                else:
                    with st.spinner("Running agents (research → blog → LinkedIn → image)…"):
                        try:
                            orch = ContentBlitzOrchestrator()
                            result = orch.run_full_pipeline(
                                topic.strip(),
                                audience=audience or None,
                                research_depth=research_depth,
                                blog_tone=blog_tone,
                                target_words=target_words,
                                image_style=image_style,
                                linkedin_cta=linkedin_cta or None,
                            )
                        except Exception as e:
                            st.error(str(e))
                            st.stop()
                    st.session_state["last_trace"] = orch.last_trace

                    st.success("Done. Vector index updated with research and blog snippets.")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Research summary")
                        st.markdown(result.research.summary)
                    with c2:
                        st.subheader("SEO blog")
                        st.markdown(f"**{result.blog.title}**")
                        st.caption(result.blog.meta_description)
                        if result.blog.keywords:
                            st.caption("Keywords: " + ", ".join(result.blog.keywords))
                        st.markdown(result.blog.body_markdown)

                    st.subheader("LinkedIn")
                    st.text_area("Post", result.linkedin.post_text, height=220)
                    if result.linkedin.hook_variants:
                        st.caption("Hook variants")
                        for h in result.linkedin.hook_variants:
                            st.code(h)
                    if result.linkedin.hashtags:
                        st.caption(" ".join("#" + t.lstrip("#") for t in result.linkedin.hashtags))

                    st.subheader("Image")
                    img_bytes = _image_to_bytes(result.image)
                    if result.image.image_url:
                        st.image(result.image.image_url, use_container_width=True)
                    elif img_bytes:
                        st.image(io.BytesIO(img_bytes), use_container_width=True)
                    else:
                        st.info("No image URL returned; check API response or model settings.")

        elif mode == "Research only":
            if st.button("Run research", type="primary"):
                if not topic.strip():
                    st.warning("Enter a topic.")
                else:
                    with st.spinner("Searching via SerpAPI…"):
                        try:
                            orch = ContentBlitzOrchestrator()
                            r = orch.research_only(
                                topic.strip(),
                                audience=audience or None,
                                depth=research_depth,
                            )
                            orch.store.add_document(
                                r.summary,
                                "research_summary",
                                metadata={"topic": topic.strip()},
                            )
                        except Exception as e:
                            st.error(str(e))
                            st.stop()
                    st.session_state["last_trace"] = orch.last_trace
                    st.markdown(r.summary)
                    st.caption(f"Model: {r.model}")

        elif mode == "Image only":
            if st.button("Generate image", type="primary"):
                if not topic.strip():
                    st.warning("Enter a topic.")
                else:
                    with st.spinner("Generating image with brochure RAG…"):
                        try:
                            orch = ContentBlitzOrchestrator()
                            image = orch.image_only(topic.strip(), image_style=image_style)
                        except Exception as e:
                            st.error(str(e))
                            st.stop()
                    st.session_state["last_trace"] = orch.last_trace

                    st.subheader("Image")
                    img_bytes = _image_to_bytes(image)
                    if image.image_url:
                        st.image(image.image_url, use_container_width=True)
                    elif img_bytes:
                        st.image(io.BytesIO(img_bytes), use_container_width=True)
                    else:
                        st.info("No image URL returned; check API response or model settings.")

        elif mode == "Blog only":
            pasted = st.text_area(
                "Paste research summary (optional)",
                height=200,
                placeholder="Leave blank to auto-generate research via SerpAPI",
            )
            if st.button("Generate blog", type="primary"):
                if not topic.strip():
                    st.warning("Topic is required.")
                else:
                    from agents.blog_agent import BlogWritingAgent
                    from vector_store import ContentVectorStore

                    with st.spinner("Generating blog…"):
                        try:
                            local_trace: list[dict[str, str]] = []
                            local_trace.append(
                                {"type": "orchestration", "name": "blog_only", "detail": "start"}
                            )
                            store = ContentVectorStore()
                            local_trace.append({"type": "orchestration", "name": "ingest_brochure", "detail": "start"})
                            ensure_se401k_brochure_ingested(store)
                            local_trace.append({"type": "orchestration", "name": "ingest_brochure", "detail": "done"})
                            rq = self_employed_401k_rag_query(topic.strip())
                            local_trace.append(
                                {"type": "tool", "name": "vector_store.query_context", "detail": "self-employed 401k query"}
                            )
                            vec = store.query_context(rq, n_results=8)
                            vc = "\n".join(vec)
                            blog_agent = BlogWritingAgent()
                            research_text = pasted.strip()
                            if not research_text:
                                local_trace.append({"type": "agent", "name": "research_agent", "detail": "start"})
                                orch = ContentBlitzOrchestrator(store=store)
                                rr = orch.research_only(
                                    topic.strip(),
                                    audience=audience or None,
                                    depth=research_depth,
                                )
                                research_text = rr.summary
                                local_trace.append({"type": "agent", "name": "research_agent", "detail": f"done ({rr.model})"})
                            local_trace.append({"type": "agent", "name": "blog_agent", "detail": "start"})
                            blog = blog_agent.write(
                                topic=topic.strip(),
                                research_summary=research_text,
                                vector_context=vc,
                                tone=blog_tone,
                                target_words=target_words,
                            )
                            local_trace.append({"type": "agent", "name": "blog_agent", "detail": "done"})
                            local_trace.append({"type": "orchestration", "name": "blog_only", "detail": "completed"})
                        except Exception as e:
                            st.error(str(e))
                            st.stop()
                    st.session_state["last_trace"] = local_trace

                    st.subheader("Blog")
                    st.markdown(f"**{blog.title}**")
                    st.caption(blog.meta_description)
                    st.markdown(blog.body_markdown)

        elif mode == "LinkedIn only":
            pasted = st.text_area(
                "Paste research summary (optional)",
                height=200,
                placeholder="Leave blank to auto-generate research via SerpAPI",
            )
            if st.button("Generate LinkedIn", type="primary"):
                if not topic.strip():
                    st.warning("Topic is required.")
                else:
                    from agents.linkedin_agent import LinkedInAgent
                    from vector_store import ContentVectorStore

                    with st.spinner("Generating LinkedIn…"):
                        try:
                            local_trace = []
                            local_trace.append({"type": "agent", "name": "linkedin_agent", "detail": "start"})
                            local_trace.append({"type": "orchestration", "name": "linkedin_only", "detail": "start"})
                            store = ContentVectorStore()
                            local_trace.append({"type": "orchestration", "name": "ingest_brochure", "detail": "start"})
                            ensure_se401k_brochure_ingested(store)
                            local_trace.append({"type": "orchestration", "name": "ingest_brochure", "detail": "done"})
                            rq = self_employed_401k_rag_query(topic.strip())
                            local_trace.append(
                                {"type": "tool", "name": "vector_store.query_context", "detail": "self-employed 401k query"}
                            )
                            vec = store.query_context(rq, n_results=8)
                            rag_li = "\n".join(vec[:6])
                            li_agent = LinkedInAgent()
                            research_text = pasted.strip()
                            if not research_text:
                                local_trace.append({"type": "agent", "name": "research_agent", "detail": "start"})
                                orch = ContentBlitzOrchestrator(store=store)
                                rr = orch.research_only(
                                    topic.strip(),
                                    audience=audience or None,
                                    depth=research_depth,
                                )
                                research_text = rr.summary
                                local_trace.append({"type": "agent", "name": "research_agent", "detail": f"done ({rr.model})"})
                            li = li_agent.generate(
                                topic=topic.strip(),
                                source_text=research_text,
                                cta=linkedin_cta or None,
                                rag_context=rag_li,
                            )
                            local_trace.append({"type": "agent", "name": "linkedin_agent", "detail": "done"})
                            local_trace.append(
                                {"type": "orchestration", "name": "linkedin_only", "detail": "completed"}
                            )
                        except Exception as e:
                            st.error(str(e))
                            st.stop()
                    st.session_state["last_trace"] = local_trace

                    st.subheader("LinkedIn")
                    st.text_area("Post", li.post_text, height=200)

        _render_trace_panel(st.session_state.get("last_trace", []))

    with tab_help:
        st.markdown(
            """
            **Agents**
            - **Research** — Web search via SerpAPI (then summarized with OpenAI).
            - **Blog writing** — GPT JSON output with title, meta description, markdown body, keywords.
            - **LinkedIn** — GPT for hooks, post body, limited hashtags.
            - **Image** — DALL·E 3 hero/social visuals from the topic.

            **Stack:** OpenAI Python SDK (GPT + images + embeddings), local FAISS/NumPy index under `./data/faiss`,
            Streamlit UI + MCP (tool-based orchestration).

            Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `SERPAPI_API_KEY` (Perplexity is no longer used).
            """
        )


if __name__ == "__main__":
    main()
