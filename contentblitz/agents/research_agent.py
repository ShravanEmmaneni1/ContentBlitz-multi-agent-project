"""Research agent: SerpAPI + OpenAI summarization for grounded research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, SERPAPI_API_KEY, SERPAPI_ENABLED, SERPAPI_ENGINE


@dataclass
class ResearchResult:
    summary: str
    raw_response: str
    model: str


class ResearchAgent:
    """Searches the web via SerpAPI, then summarizes via OpenAI."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self._model = OPENAI_MODEL

        if not SERPAPI_ENABLED:
            self._serp_enabled = False
            self._serp_disabled_reason = "SERPAPI_ENABLED is off"
            return

        if not SERPAPI_API_KEY.strip():
            # Keep the app running even if the key isn't configured yet.
            self._serp_enabled = False
            self._serp_disabled_reason = "SERPAPI_API_KEY is missing"
            return
        self._serp_enabled = True
        self._serp_disabled_reason = ""
        self._engine = SERPAPI_ENGINE

    def research(
        self,
        topic: str,
        depth: str = "balanced",
        audience: str | None = None,
    ) -> ResearchResult:
        if not self._serp_enabled:
            audience_line = f"\n**Audience:** {audience}" if audience else ""
            extra = (
                f"\nReason: {self._serp_disabled_reason}."
                if self._serp_disabled_reason
                else ""
            )
            stub = (
                "## Research (SerpAPI disabled)\n\n"
                "SerpAPI is turned off (set `SERPAPI_ENABLED=1` in `.env` to enable).\n\n"
                f"**Topic:** {topic}{audience_line}{extra}\n\n"
                "_No web API call was made._"
            )
            return ResearchResult(summary=stub, raw_response=stub, model="serpapi-disabled")

        # Import lazily so the app can still run without serpapi installed (e.g. in minimal envs).
        import serpapi

        depth_hint = {
            "quick": "Brief scan; key facts only.",
            "balanced": "Balanced depth with trends and definitions.",
            "deep": "Comprehensive: subtopics, counterpoints, and recent developments.",
        }.get(depth, "balanced")

        audience_line = f"Audience: {audience}." if audience else ""
        query = f"{topic} {audience or ''}".strip()

        params: dict[str, Any] = {
            "engine": self._engine,
            "q": query,
            # Keep output small but useful for the summarizer.
            "num": 7,
        }
        client = serpapi.Client(api_key=SERPAPI_API_KEY)
        try:
            results = client.search(params) or {}
        except Exception as e:  # noqa: BLE001
            # Keep the app running even if the web API is blocked/unreachable.
            err = str(e)
            stub = (
                "## Research (SerpAPI error)\n\n"
                "The web search request failed, so I could not retrieve external sources.\n\n"
                f"Error: {err}\n\n"
                f"**Topic:** {topic}\n"
                f"{audience_line}\n\n"
                "_No web API call was made._"
            )
            return ResearchResult(summary=stub, raw_response=stub, model="serpapi-error")

        organic = results.get("organic_results") or []
        sources: list[dict[str, str]] = []
        for r in organic[:6]:
            title = str(r.get("title") or "").strip()
            link = str(r.get("link") or "").strip()
            snippet = str(r.get("snippet") or r.get("description") or "").strip()
            if title or link or snippet:
                sources.append({"title": title, "link": link, "snippet": snippet})

        sources_block = "\n".join(
            [f"- {s['title']} ({s['link']})\n  {s['snippet']}" for s in sources if s.get("link")]
            or "- (No organic sources found.)"
        )

        system = (
            "You are a research analyst. Using the provided web search snippets, produce a structured "
            "research summary optimized for SEO blog drafting and social snippets. "
            "Be factual; when uncertain, say so. Include a section 'Suggested angles'. "
            "Add short bullet points with citations as plain URLs."
        )
        user = (
            f"Topic: {topic}\n"
            f"{audience_line}\n"
            f"Depth: {depth_hint}\n\n"
            "Web search snippets:\n"
            f"{sources_block}\n"
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        return ResearchResult(summary=text, raw_response=text, model=f"{self._model}+serpapi")
