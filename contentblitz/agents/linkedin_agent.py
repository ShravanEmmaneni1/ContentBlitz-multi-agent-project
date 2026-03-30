"""LinkedIn post agent: GPT for platform-specific professional social copy."""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL


@dataclass
class LinkedInOutput:
    post_text: str
    hook_variants: list[str]
    hashtags: list[str]


class LinkedInAgent:
    """Short-form LinkedIn-optimized posts with hooks and hashtag discipline."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is required for LinkedIn generation.")
        self._client = OpenAI(api_key=key)
        self._model = OPENAI_MODEL

    def generate(
        self,
        topic: str,
        source_text: str,
        cta: str | None = None,
        rag_context: str = "",
    ) -> LinkedInOutput:
        cta_line = f"Preferred CTA: {cta}" if cta else "Include a soft CTA (comment / follow / read more)."

        rag_block = ""
        if rag_context.strip():
            rag_block = (
                "\n\nRetrieved reference (401(k) / self-employed / SEP rules — use only facts supported here):\n"
                f"{rag_context[:8000]}\n"
            )

        system = (
            "You are a LinkedIn content strategist. Output JSON only with keys: "
            "post_text (under 3000 chars, line breaks for readability), "
            "hook_variants (array of 3 short opening lines), "
            "hashtags (array, max 5 relevant professional tags, no stuffing). "
            "Voice: credible, specific, no clickbait. Use 1-2 emoji max if any."
        )
        user = (
            f"Topic: {topic}\n{cta_line}\n\n"
            f"Source material (blog or research):\n{source_text[:12000]}"
            f"{rag_block}"
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.75,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        data = json.loads(raw)
        post = str(data.get("post_text", ""))[:3000]
        hooks = list(data.get("hook_variants") or [])[:3]
        tags = list(data.get("hashtags") or [])[:5]
        return LinkedInOutput(post_text=post, hook_variants=hooks, hashtags=tags)
