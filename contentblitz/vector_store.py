"""Local vector index: FAISS (flat inner product on normalized embeddings) with NumPy fallback."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from config import COLLECTION_NAME, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, VECTOR_STORE_PATH

try:
    import faiss

    _HAVE_FAISS = True
except ImportError:
    _HAVE_FAISS = False

_INDEX_NAME = "index.faiss"
_VECTORS_NAME = "embeddings.npy"
_STATE_NAME = "state.pkl"


def _doc_id_to_i64(doc_id: str) -> np.int64:
    h = hashlib.sha256(doc_id.encode()).digest()[:8]
    return np.int64(int.from_bytes(h, "little", signed=True))


class ContentVectorStore:
    """Stores and retrieves text chunks for RAG-style context."""

    def __init__(self, persist_path: str | None = None) -> None:
        base = Path(persist_path) if persist_path else VECTOR_STORE_PATH
        base.mkdir(parents=True, exist_ok=True)
        self._dir = base
        self._embedding_model = OPENAI_EMBEDDING_MODEL
        self._client: OpenAI | None = None
        self._dim: int | None = None
        self._texts: dict[str, str] = {}
        self._metas: dict[str, dict[str, Any]] = {}
        self._use_faiss = _HAVE_FAISS

        self._index: Any = None
        self._i64_to_doc: dict[int, str] = {}
        self._vectors: np.ndarray | None = None
        self._doc_order: list[str] = []
        self._doc_row: dict[str, int] = {}

        self._load_or_init()

    def _openai(self) -> OpenAI:
        if not OPENAI_API_KEY.strip():
            raise RuntimeError(
                "OPENAI_API_KEY is required for embeddings (vector store). "
                "Set it in your environment or .env file."
            )
        if self._client is None:
            self._client = OpenAI(api_key=OPENAI_API_KEY)
        return self._client

    def _load_or_init(self) -> None:
        index_path = self._dir / _INDEX_NAME
        vec_path = self._dir / _VECTORS_NAME
        state_path = self._dir / _STATE_NAME

        if not state_path.is_file():
            return

        with open(state_path, "rb") as f:
            state = pickle.load(f)
        self._texts = state["texts"]
        self._metas = state["metas"]
        self._embedding_model = state.get("embedding_model", self._embedding_model)
        self._dim = state.get("dim")
        backend = state.get("backend")
        if backend is None and index_path.is_file():
            backend = "faiss"
        if backend is None:
            backend = "numpy"

        if backend == "faiss":
            if not self._use_faiss:
                raise RuntimeError(
                    "This vector store was saved with FAISS. Install faiss-cpu (recommended: Python 3.12–3.13) "
                    "or delete the store directory to recreate with the NumPy backend."
                )
            if not index_path.is_file():
                raise RuntimeError("FAISS vector store is missing index.faiss")
            self._index = faiss.read_index(str(index_path))
            self._i64_to_doc = {int(k): v for k, v in state["i64_to_doc"].items()}
        elif vec_path.is_file():
            self._use_faiss = False
            self._vectors = np.load(str(vec_path))
            self._doc_order = state["doc_order"]
            self._doc_row = {did: i for i, did in enumerate(self._doc_order)}
            if self._dim is None and self._vectors is not None:
                self._dim = int(self._vectors.shape[1])

    def _save(self) -> None:
        state_path = self._dir / _STATE_NAME
        if self._use_faiss and self._index is not None:
            faiss.write_index(self._index, str(self._dir / _INDEX_NAME))
            with open(state_path, "wb") as f:
                pickle.dump(
                    {
                        "texts": self._texts,
                        "metas": self._metas,
                        "i64_to_doc": self._i64_to_doc,
                        "embedding_model": self._embedding_model,
                        "dim": self._dim,
                        "collection": COLLECTION_NAME,
                        "backend": "faiss",
                    },
                    f,
                )
        elif self._vectors is not None:
            np.save(str(self._dir / _VECTORS_NAME), self._vectors)
            with open(state_path, "wb") as f:
                pickle.dump(
                    {
                        "doc_order": self._doc_order,
                        "texts": self._texts,
                        "metas": self._metas,
                        "embedding_model": self._embedding_model,
                        "dim": self._dim,
                        "collection": COLLECTION_NAME,
                        "backend": "numpy",
                    },
                    f,
                )

    def _embed(self, text: str) -> np.ndarray:
        text_in = text.strip() or " "
        client = self._openai()
        resp = client.embeddings.create(
            model=self._embedding_model,
            input=text_in[:32000],
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        d = vec.shape[1]
        if self._dim is None:
            self._dim = d
        elif d != self._dim:
            raise RuntimeError(
                f"Embedding dimension {d} does not match index ({self._dim}). "
                "Use a fresh VECTOR_STORE_PATH or the same embedding model as before."
            )
        if _HAVE_FAISS:
            faiss.normalize_L2(vec)
        else:
            nrm = float(np.linalg.norm(vec))
            if nrm > 0:
                vec = vec / nrm
        return vec

    def _ensure_faiss_index(self, vec: np.ndarray) -> None:
        if self._index is None:
            self._dim = vec.shape[1]
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))

    def add_document(
        self,
        text: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        doc_id = hashlib.sha256(f"{doc_type}:{text[:500]}".encode()).hexdigest()[:24]
        meta: dict[str, Any] = {"doc_type": doc_type}
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items() if v is not None})

        vec = self._embed(text)
        i64 = _doc_id_to_i64(doc_id)

        if self._use_faiss:
            self._ensure_faiss_index(vec)
            assert self._index is not None
            if doc_id in self._texts:
                self._index.remove_ids(np.array([i64], dtype=np.int64))
                self._i64_to_doc.pop(int(i64), None)
            self._index.add_with_ids(vec, np.array([i64], dtype=np.int64))
            self._i64_to_doc[int(i64)] = doc_id
        else:
            if self._vectors is None:
                self._vectors = vec.copy()
                self._doc_order = [doc_id]
                self._doc_row = {doc_id: 0}
            elif doc_id in self._doc_row:
                r = self._doc_row[doc_id]
                self._vectors[r] = vec.reshape(-1)
            else:
                r = len(self._doc_order)
                self._vectors = np.vstack([self._vectors, vec])
                self._doc_order.append(doc_id)
                self._doc_row[doc_id] = r

        self._texts[doc_id] = text
        self._metas[doc_id] = meta
        self._save()
        return doc_id

    def add_documents_batch(
        self,
        items: list[tuple[str, str, dict[str, Any] | None]],
    ) -> list[str]:
        """Add many chunks; one disk save at the end. items: (text, doc_type, metadata)."""
        out_ids: list[str] = []
        for i, (text, doc_type, metadata) in enumerate(items):
            doc_id = hashlib.sha256(
                f"{doc_type}:{i}:{text[:2000]}".encode()
            ).hexdigest()[:24]
            meta: dict[str, Any] = {"doc_type": doc_type}
            if metadata:
                meta.update({k: str(v) for k, v in metadata.items() if v is not None})

            vec = self._embed(text)
            i64 = _doc_id_to_i64(doc_id)

            if self._use_faiss:
                self._ensure_faiss_index(vec)
                assert self._index is not None
                if doc_id in self._texts:
                    self._index.remove_ids(np.array([i64], dtype=np.int64))
                    self._i64_to_doc.pop(int(i64), None)
                self._index.add_with_ids(vec, np.array([i64], dtype=np.int64))
                self._i64_to_doc[int(i64)] = doc_id
            else:
                if self._vectors is None:
                    self._vectors = vec.copy()
                    self._doc_order = [doc_id]
                    self._doc_row = {doc_id: 0}
                elif doc_id in self._doc_row:
                    r = self._doc_row[doc_id]
                    self._vectors[r] = vec.reshape(-1)
                else:
                    r = len(self._doc_order)
                    self._vectors = np.vstack([self._vectors, vec])
                    self._doc_order.append(doc_id)
                    self._doc_row[doc_id] = r

            self._texts[doc_id] = text
            self._metas[doc_id] = meta
            out_ids.append(doc_id)

        self._save()
        return out_ids

    def query_context(self, query: str, n_results: int = 6) -> list[str]:
        if not query.strip():
            return []
        q = self._embed(query)

        if self._use_faiss and self._index is not None and self._index.ntotal > 0:
            k = min(n_results, self._index.ntotal)
            _distances, labels = self._index.search(q, k)
            row = labels[0]
            out: list[str] = []
            for lab in row:
                if lab == -1:
                    continue
                did = self._i64_to_doc.get(int(lab))
                if did and did in self._texts:
                    out.append(self._texts[did])
            return out

        if self._vectors is None or len(self._doc_order) == 0:
            return []
        n = self._vectors.shape[0]
        k = min(n_results, n)
        sims = (self._vectors @ q.reshape(-1, 1)).ravel()
        if k >= n:
            order = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k - 1)[:k]
            order = part[np.argsort(-sims[part])]
        return [self._texts[self._doc_order[int(i)]] for i in order[:k] if self._doc_order[int(i)] in self._texts]

    def count(self) -> int:
        if self._use_faiss:
            if self._index is None:
                return 0
            return int(self._index.ntotal)
        if self._vectors is None:
            return 0
        return int(self._vectors.shape[0])
