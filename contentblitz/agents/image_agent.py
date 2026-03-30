"""Image creation agent: OpenAI Images API (DALL·E) for campaign visuals."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from config import IMAGE_MODEL, OPENAI_API_KEY


@dataclass
class ImageResult:
    revised_prompt: str
    image_url: str | None
    b64_json: str | None


class ImageCreationAgent:
    """Generates platform-ready hero or social images from a content brief."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is required for image generation.")
        self._client = OpenAI(api_key=key)
        self._model = IMAGE_MODEL

    def create(
        self,
        topic: str,
        style_hints: str = "modern, clean, professional",
        size: str = "1024x1024",
        rag_context: str = "",
    ) -> ImageResult:
        visual_brief = ""
        if rag_context.strip():
            visual_brief = (
                " Visual themes suggested by the reference material (abstract, no text): "
                f"{rag_context[:1200].replace(chr(10), ' ')}"
            )
        prompt = (
            f"Editorial illustration for an article about: {topic}. "
            f"Style: {style_hints}. No text overlays, no logos, no watermarks. "
            "Suitable for blog hero and LinkedIn."
            f"{visual_brief}"
        )

        resp = self._client.images.generate(
            model=self._model,
            prompt=prompt[:4000],
            size=size,  # type: ignore[arg-type]
            quality="standard",
            n=1,
        )
        item = resp.data[0]
        return ImageResult(
            revised_prompt=getattr(item, "revised_prompt", None) or prompt,
            image_url=getattr(item, "url", None),
            b64_json=getattr(item, "b64_json", None),
        )
