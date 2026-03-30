"""Blog writing agent: GPT for long-form, SEO-oriented articles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL


@dataclass
class BlogOutput:
    title: str
    meta_description: str
    body_markdown: str
    keywords: list[str]


class BlogWritingAgent:
    """Produces platform-ready blog drafts with SEO metadata."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is required for blog generation.")
        self._client = OpenAI(api_key=key)
        self._model = OPENAI_MODEL

    def write(
        self,
        topic: str,
        research_summary: str,
        vector_context: str = "",
        tone: str = "professional",
        target_words: int = 1200,
    ) -> BlogOutput:
        context_block = ""
        if vector_context.strip():
            context_block = (
                "\n\nAdditional context (retrieved from SE 401(k) brochure / knowledge base — "
                "prefer facts about self-employed, SEP, and solo 401(k) where relevant):\n"
                f"{vector_context}\n"
            )

        system = (
            "You are an expert SEO content writer. Output valid JSON only with keys: "
            'title, meta_description, body_markdown, keywords (array of strings). '
            "body_markdown must use ## and ### headings, bullet lists where useful, "
            "and a short FAQ section. meta_description <= 160 characters. "
            "Include primary keyword in title and first paragraph naturally. "
            "Respect the requested length range."
        )
        user = (
            f"Topic: {topic}\n"
            f"Tone: {tone}\n"
            f"Target length: {int(target_words * 0.9)} to {int(target_words * 1.1)} words.\n\n"
            f"Research summary:\n{research_summary}"
            f"{context_block}"
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        data = json.loads(raw)
        body = str(data.get("body_markdown", ""))
        words = len(re.findall(r"\b[\w'-]+\b", body))

        # If the first draft misses target badly, run one expansion pass.
        if words < int(target_words * 0.8):
            expand_system = (
                "You are improving an existing SEO blog draft. Return JSON with keys: "
                "title, meta_description, body_markdown, keywords. "
                "Expand depth and practical details without fluff. Keep markdown structure."
            )
            expand_user = (
                f"Topic: {topic}\n"
                f"Tone: {tone}\n"
                f"Target length: {int(target_words * 0.9)} to {int(target_words * 1.1)} words.\n\n"
                f"Current draft ({words} words):\n{body}\n\n"
                f"Research summary:\n{research_summary}"
                f"{context_block}"
            )
            expand_resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": expand_system},
                    {"role": "user", "content": expand_user},
                ],
                temperature=0.6,
                response_format={"type": "json_object"},
            )
            expand_raw = (expand_resp.choices[0].message.content or "{}").strip()
            data = json.loads(expand_raw)

        return BlogOutput(
            title=str(data.get("title", topic)),
            meta_description=str(data.get("meta_description", ""))[:160],
            body_markdown=str(data.get("body_markdown", "")),
            keywords=list(data.get("keywords") or []),
        )
