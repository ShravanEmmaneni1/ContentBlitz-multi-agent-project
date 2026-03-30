"""Load se401kbrochure.pdf into the vector store (chunked). Idempotent via manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import SE401K_BROCHURE_PATH, VECTOR_STORE_PATH

if TYPE_CHECKING:
    from vector_store import ContentVectorStore

_MANIFEST_NAME = "se401k_ingest.json"


def _pdf_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(parts)


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> list[str]:
    text = " ".join(text.split())
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [text.strip()]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _manifest_path() -> Path:
    return VECTOR_STORE_PATH / _MANIFEST_NAME


def brochure_already_ingested(pdf_path: Path) -> bool:
    if not pdf_path.is_file():
        return False
    mp = _manifest_path()
    if not mp.is_file():
        return False
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("pdf_sha256") == _pdf_sha256(pdf_path)


def ingest_se401k_brochure(
    store: ContentVectorStore,
    pdf_path: Path | None = None,
    *,
    force: bool = False,
) -> int:
    """
    Chunk and embed the SE 401(k) brochure. Returns number of chunks added.
    Skips if manifest matches file hash unless force=True.
    """
    path = pdf_path or SE401K_BROCHURE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Brochure PDF not found: {path}")

    if not force and brochure_already_ingested(path):
        return 0

    raw = extract_pdf_text(path)
    chunks = chunk_text(raw)
    if not chunks:
        raise ValueError(f"No extractable text from {path}")

    items: list[tuple[str, str, dict[str, Any] | None]] = []
    for i, chunk in enumerate(chunks):
        items.append(
            (
                chunk,
                "se401k_brochure",
                {
                    "source": path.name,
                    "chunk_index": str(i),
                    "focus": "self-employed 401k SEP solo",
                },
            )
        )

    store.add_documents_batch(items)

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(
        json.dumps(
            {
                "pdf_path": str(path.resolve()),
                "pdf_sha256": _pdf_sha256(path),
                "chunk_count": len(chunks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(chunks)


def ensure_se401k_brochure_ingested(store: ContentVectorStore) -> int:
    """Ingest brochure if present on disk and not yet in manifest. Returns chunks added (0 if skipped)."""
    if not SE401K_BROCHURE_PATH.is_file():
        return 0
    return ingest_se401k_brochure(store, SE401K_BROCHURE_PATH, force=False)


if __name__ == "__main__":
    import argparse

    from vector_store import ContentVectorStore

    ap = argparse.ArgumentParser(description="Ingest se401kbrochure.pdf into the vector store.")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if manifest matches (may duplicate chunks unless store was cleared).",
    )
    args = ap.parse_args()
    n = ingest_se401k_brochure(ContentVectorStore(), force=args.force)
    if n == 0:
        print(
            "No new chunks added (PDF missing, or brochure already ingested — "
            "manifest matches file). Use --force to re-ingest."
        )
    else:
        print(f"Ingested {n} chunks from {SE401K_BROCHURE_PATH}")
