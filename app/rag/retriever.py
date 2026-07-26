"""
FAISS-backed retriever with exact error-code shortcut.

Retrieval flow (PRD Section 5):
  1. If the query contains an exact error-code pattern (e.g. 0x0003),
     skip embedding and do a direct dict lookup → instant result.
  2. Otherwise, embed the query, search FAISS for top_k candidates,
     filter by confidence_threshold, return top return_n chunks.
  3. If nothing clears the threshold, return [] — the pipeline must
     return the not-found message without invoking the LLM.

Multi-index support:
  If the general-document index exists at data/index/general/, it is
  loaded alongside the fault-code index.  Semantic searches query both
  indexes and merge results by score.  Exact code lookups only search
  the fault-code index (general documents don't have error codes).
  The general index is optional — its absence is silently ignored.
"""
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.ingestion.chunker import load_chunks
from app.logger import get_logger
from app.settings import (
    CHUNKS_STORE_PATH,
    CONFIDENCE_THRESHOLD,
    ERROR_CODE_PATTERN,
    FAISS_INDEX_PATH,
    RETURN_N,
    TOP_K,
)

log = get_logger(__name__)

_lock = threading.Lock()
_retriever_instance = None

# General-document index paths (mirrors ingest_general.py constants)
_GENERAL_INDEX_DIR = FAISS_INDEX_PATH.parent / "general"
_GENERAL_FAISS_PATH = _GENERAL_INDEX_DIR / "faiss.index"
_GENERAL_CHUNKS_PATH = _GENERAL_INDEX_DIR / "chunks.jsonl"


@dataclass
class RetrievedChunk:
    chunk: dict
    score: float   # cosine similarity 0–1


class Retriever:
    """Loads and queries the FAISS index + chunk store."""

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        chunks_path: Path = CHUNKS_STORE_PATH,
    ) -> None:
        import faiss  # type: ignore

        self._lock = threading.Lock()

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run the ingestion script first: python -m app.ingestion.ingest"
            )
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks store not found at {chunks_path}. "
                "Run the ingestion script first."
            )

        # ── Primary (fault-code) index ──
        log.info("Loading FAISS index from %s …", index_path)
        self._index = faiss.read_index(str(index_path))
        log.info("FAISS index loaded: %d vectors", self._index.ntotal)

        self._chunks = load_chunks(chunks_path)
        log.info("Loaded %d chunks from %s", len(self._chunks), chunks_path)

        # Assert len(self._chunks) == self._index.ntotal after every load/reload
        if len(self._chunks) != self._index.ntotal:
            raise ValueError(
                f"Fidelity violation: index has {self._index.ntotal} vectors but chunks store has "
                f"{len(self._chunks)} chunks."
            )

        # Build error-code lookup for O(1) exact matches
        self._code_index: dict[str, list[dict]] = {}
        for chunk in self._chunks:
            code = (chunk.get("error_code") or "").lower()
            if code:
                self._code_index.setdefault(code, []).append(chunk)

        self._code_pattern = re.compile(ERROR_CODE_PATTERN, re.IGNORECASE)

        # ── General-document index (optional) ──
        self._general_index = None
        self._general_chunks: list[dict] = []
        self._load_general_index()

    def _load_general_index(self) -> None:
        """Load the general-document index if it exists. Silent no-op if not."""
        import faiss  # type: ignore

        if not _GENERAL_FAISS_PATH.exists() or not _GENERAL_CHUNKS_PATH.exists():
            log.info(
                "General-document index not found at %s — skipping.",
                _GENERAL_INDEX_DIR,
            )
            return

        try:
            self._general_index = faiss.read_index(str(_GENERAL_FAISS_PATH))
            self._general_chunks = load_chunks(_GENERAL_CHUNKS_PATH)

            if len(self._general_chunks) != self._general_index.ntotal:
                log.warning(
                    "General index fidelity violation: %d vectors vs %d chunks — "
                    "disabling general index.",
                    self._general_index.ntotal, len(self._general_chunks),
                )
                self._general_index = None
                self._general_chunks = []
                return

            log.info(
                "General-document index loaded: %d vectors",
                self._general_index.ntotal,
            )
        except Exception as e:
            log.warning("Failed to load general index (%s) — skipping.", e)
            self._general_index = None
            self._general_chunks = []

    # ── Exact code lookup ────────────────────────────────────────────────

    def _exact_lookup(self, query: str) -> list[RetrievedChunk]:
        """Find all error codes mentioned in the query and return their chunks."""
        matches = self._code_pattern.findall(query)
        results = []
        for raw_code in matches:
            code = raw_code.lower()
            chunks = self._code_index.get(code, [])
            for c in chunks:
                results.append(RetrievedChunk(chunk=c, score=1.0))
        return results

    # ── Semantic search ──────────────────────────────────────────────────

    def _semantic_search(self, query: str) -> list[RetrievedChunk]:
        from app.rag.embedder import get_embedder

        embedder = get_embedder()
        q_vec = np.array([embedder.embed_query(query)], dtype=np.float32)

        import faiss  # type: ignore
        faiss.normalize_L2(q_vec)

        # Search primary (fault-code) index
        results = self._search_index(
            self._index, self._chunks, q_vec, TOP_K,
        )

        # Search general-document index if loaded
        if self._general_index is not None:
            general_results = self._search_index(
                self._general_index, self._general_chunks, q_vec, TOP_K,
            )
            results.extend(general_results)

        # Merge, sort descending by score, take top return_n
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:RETURN_N]

    def _search_index(
        self,
        index: Any,
        chunks: list[dict],
        q_vec: "np.ndarray",
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Search a single FAISS index and return results above threshold."""
        k = min(top_k, index.ntotal)
        scores, indices = index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            similarity = float(score)
            if similarity < CONFIDENCE_THRESHOLD:
                continue
            results.append(RetrievedChunk(chunk=chunks[idx], score=similarity))
        return results

    # ── Public API ───────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        Returns an empty list if nothing clears the confidence threshold.
        """
        with self._lock:
            # Try exact code match first (FR-3)
            exact = self._exact_lookup(query)
            if exact:
                log.info("Exact code match for query '%s': %d chunk(s)", query[:60], len(exact))
                return exact

            # Semantic search (FR-2)
            semantic = self._semantic_search(query)
            log.info(
                "Semantic search for '%s': %d chunk(s) above threshold %.2f",
                query[:60], len(semantic), CONFIDENCE_THRESHOLD,
            )
            return semantic

    def reload(self) -> None:
        """Hot-reload the index and chunk store (after re-ingestion)."""
        import faiss  # type: ignore

        with self._lock:
            new_index = faiss.read_index(str(FAISS_INDEX_PATH))
            new_chunks = load_chunks(CHUNKS_STORE_PATH)

            if len(new_chunks) != new_index.ntotal:
                raise ValueError(
                    f"Fidelity violation: index has {new_index.ntotal} vectors but chunks store has "
                    f"{len(new_chunks)} chunks."
                )

            self._index = new_index
            self._chunks = new_chunks

            self._code_index = {}
            for chunk in self._chunks:
                code = (chunk.get("error_code") or "").lower()
                if code:
                    self._code_index.setdefault(code, []).append(chunk)

            # Also reload general index if it exists
            self._load_general_index()

            total = self._index.ntotal + (
                self._general_index.ntotal if self._general_index else 0
            )
            log.info("Retriever reloaded: %d total vectors", total)


def get_retriever() -> Retriever:
    """Return singleton Retriever, initialised on first call."""
    global _retriever_instance
    if _retriever_instance is None:
        with _lock:
            if _retriever_instance is None:
                _retriever_instance = Retriever()
    return _retriever_instance
