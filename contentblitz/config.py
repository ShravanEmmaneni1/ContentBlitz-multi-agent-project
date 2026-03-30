"""Application configuration from environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = "k-proj-WPODLzTA8TpZOzbLx2cX7PcnvX75YnphcoLBkGdfyq3C47SYic49M3M_tMEBPY6B-xYJWN66M7T3BlbkFJ8UaM3mAOESOoDs_k9EffOD88BJTT3AoOQjHEke3JqzR8fxh7E3O0IOTx-6vJAiY_3TqZ8jvsEA"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_ENABLED = os.getenv("PERPLEXITY_ENABLED", "0").lower() in ("1", "true", "yes")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "dall-e-3")

SERPAPI_API_KEY = "4db203d87f37bdedb9885f0e8185e18fbeef81fbe9f45885ab8f4085c5c06218"
SERPAPI_ENABLED = os.getenv("SERPAPI_ENABLED", "1").lower() in ("1", "true", "yes")
SERPAPI_ENGINE = os.getenv("SERPAPI_ENGINE", "google")

MCP_ENABLED = os.getenv("MCP_ENABLED", "1").lower() in ("1", "true", "yes")

VECTOR_STORE_PATH = Path(
    os.getenv("VECTOR_STORE_PATH") or os.getenv("CHROMA_PATH") or os.getenv("FAISS_PATH") or "./data/faiss"
)
COLLECTION_NAME = "contentblitz_context"

# Default: project-root se401kbrochure.pdf (same directory as this package when running from repo).
_PKG_ROOT = Path(__file__).resolve().parent
SE401K_BROCHURE_PATH = Path(
    os.getenv("SE401K_BROCHURE_PATH", str(_PKG_ROOT / "se401kbrochure.pdf"))
)

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
