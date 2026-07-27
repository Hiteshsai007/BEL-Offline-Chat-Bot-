"""
FAISS-backed retriever with BM25 hybrid search and exact error-code shortcut.

Retrieval flow (PRD Section 5):
  1. If the query contains an exact error-code pattern (e.g. 0x0003),
     skip embedding and do a direct dict lookup -> instant result.
  2. Otherwise, run hybrid search:
     a. Dense: embed the query, search FAISS for top candidates
     b. Sparse: BM25 keyword search for top candidates
     c. Fuse results using Reciprocal Rank Fusion (RRF)
     d. Deduplicate, filter by confidence_threshold, return top return_n
  3. If nothing clears the threshold, return [] -- the pipeline must
     return the not-found message without invoking the LLM.

Multi-index support:
  If the general-document index exists at data/index/general/, it is
  loaded alongside the fault-code index.  Hybrid searches query both
  indexes and merge results by fused score.  Exact code lookups only
  search the fault-code index (general documents don't have error codes).
  The general index is optional -- its absence is silently ignored.

BM25 hybrid search:
  When bm25_enabled is true (default), a BM25Okapi index is built from
  chunk texts alongside the FAISS index.  Queries are scored by both
  dense (cosine similarity) and sparse (BM25) retrievers, then combined
  using Reciprocal Rank Fusion (RRF).  This dramatically improves recall
  for keyword-heavy queries that dense embeddings alone may miss.
  If rank-bm25 is not installed, falls back to dense-only retrieval.
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
    BM25_ENABLED,
    CHUNKS_STORE_PATH,
    CONFIDENCE_THRESHOLD,
    ERROR_CODE_PATTERN,
    FAISS_INDEX_PATH,
    RETRIEVAL_CANDIDATES,
    RETURN_N,
    RRF_K,
    TOP_K,
)

log = get_logger(__name__)

_lock = threading.Lock()
_retriever_instance = None

# General-document index paths (mirrors ingest_general.py constants)
_GENERAL_INDEX_DIR = FAISS_INDEX_PATH.parent / "general"
_GENERAL_FAISS_PATH = _GENERAL_INDEX_DIR / "faiss.index"
_GENERAL_CHUNKS_PATH = _GENERAL_INDEX_DIR / "chunks.jsonl"

# Try to import BM25; graceful fallback if not installed
_bm25_available = False
if BM25_ENABLED:
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        _bm25_available = True
    except ImportError:
        log.warning(
            "rank-bm25 not installed -- falling back to dense-only retrieval. "
            "Install with: pip install rank-bm25"
        )


def _tokenize(text: str) -> list[str]:
    """Clean word tokenizer for BM25 (strips punctuation)."""
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
class RetrievedChunk:
    chunk: dict
    score: float   # cosine similarity 0-1 (dense) or fused RRF score (hybrid)


class Retriever:
    """Loads and queries the FAISS index + chunk store with optional BM25 hybrid search."""

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

        # -- Primary (fault-code) index --
        log.info("Loading FAISS index from %s ...", index_path)
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

        # -- BM25 index for primary chunks --
        self._bm25 = self._build_bm25_index(self._chunks)

        # -- General-document index (optional) --
        self._general_index = None
        self._general_chunks: list[dict] = []
        self._general_bm25 = None
        self._load_general_index()

    @staticmethod
    def _build_bm25_index(chunks: list[dict], cache_path: Path | None = None):
        """Build or load a cached BM25 index from chunk texts."""
        import pickle

        if not _bm25_available or not chunks:
            return None

        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    bm25 = pickle.load(f)
                log.info("BM25 index loaded from cache: %s", cache_path)
                return bm25
            except Exception as e:
                log.warning("Failed to load cached BM25 index (%s) -- rebuilding.", e)

        corpus = [_tokenize(c.get("chunk_text", "")) for c in chunks]
        try:
            bm25 = BM25Okapi(corpus)
            log.info("BM25 index built: %d documents", len(corpus))
            if cache_path:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "wb") as f:
                        pickle.dump(bm25, f)
                    log.info("Saved BM25 index cache -> %s", cache_path)
                except Exception as e:
                    log.warning("Could not save BM25 index cache: %s", e)
            return bm25
        except Exception as e:
            log.warning("Failed to build BM25 index: %s -- using dense-only.", e)
            return None

    def _load_general_index(self) -> None:
        """Load the general-document index if it exists. Silent no-op if not."""
        import faiss  # type: ignore

        if not _GENERAL_FAISS_PATH.exists() or not _GENERAL_CHUNKS_PATH.exists():
            log.info(
                "General-document index not found at %s -- skipping.",
                _GENERAL_INDEX_DIR,
            )
            return

        try:
            self._general_index = faiss.read_index(str(_GENERAL_FAISS_PATH))
            self._general_chunks = load_chunks(_GENERAL_CHUNKS_PATH)

            if len(self._general_chunks) != self._general_index.ntotal:
                log.warning(
                    "General index fidelity violation: %d vectors vs %d chunks -- "
                    "disabling general index.",
                    self._general_index.ntotal, len(self._general_chunks),
                )
                self._general_index = None
                self._general_chunks = []
                self._general_bm25 = None
                return

            gen_bm25_cache = _GENERAL_INDEX_DIR / "bm25.pkl"
            self._general_bm25 = self._build_bm25_index(self._general_chunks, gen_bm25_cache)

            log.info(
                "General-document index loaded: %d vectors",
                self._general_index.ntotal,
            )
        except Exception as e:
            log.warning("Failed to load general index (%s) -- skipping.", e)
            self._general_index = None
            self._general_chunks = []
            self._general_bm25 = None

    # -- Exact code lookup ------------------------------------------------

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

    # -- Dense (FAISS) search ---------------------------------------------

    def _dense_search(
        self,
        index: Any,
        chunks: list[dict],
        q_vec: "np.ndarray",
        n_candidates: int,
    ) -> list[tuple[int, float]]:
        """
        Search a single FAISS index.
        Returns list of (chunk_index, cosine_similarity) tuples.
        """
        k = min(n_candidates, index.ntotal)
        scores, indices = index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            similarity = float(score)
            if similarity < CONFIDENCE_THRESHOLD:
                continue
            results.append((int(idx), similarity))
        return results

    # -- BM25 (sparse) search ---------------------------------------------

    def _sparse_search(
        self,
        bm25,
        chunks: list[dict],
        query: str,
        n_candidates: int,
    ) -> list[tuple[int, float]]:
        """
        Search a BM25 index with stop-word filtering and term-coverage boosting.
        Returns list of (chunk_index, bm25_score) tuples, sorted descending.
        """
        if bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []

        stop_words = {
            "what", "is", "the", "a", "an", "for", "in", "of", "to", "on",
            "should", "i", "use", "my", "how", "do", "does", "where", "can",
            "which", "are", "with", "about", "be", "this", "that", "it",
        }
        keywords = [t for t in tokens if t not in stop_words] or tokens

        scores = bm25.get_scores(keywords)

        # Apply term-coverage boost for chunks that contain ALL keywords
        boosted_scores = np.copy(scores)
        for idx in range(len(chunks)):
            if boosted_scores[idx] <= 0:
                continue
            chunk_tokens = set(_tokenize(chunks[idx].get("chunk_text", "")))
            matches = sum(1 for k in keywords if k in chunk_tokens)
            coverage = matches / len(keywords)
            # Up to 2.5x boost for 100% keyword coverage
            boosted_scores[idx] *= (1.0 + 1.5 * coverage)

        # Get top-N indices sorted by boosted score descending
        top_indices = np.argsort(boosted_scores)[::-1][:n_candidates]
        results = []
        for idx in top_indices:
            s = float(boosted_scores[idx])
            if s <= 0.0:
                break
            results.append((int(idx), s))
        return results

    # -- Reciprocal Rank Fusion -------------------------------------------

    def _rrf_fuse(
        self,
        dense_results: list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
        chunks: list[dict],
        rrf_k: int = RRF_K,
    ) -> list[RetrievedChunk]:
        """
        Fuse dense and sparse results using Reciprocal Rank Fusion.
        Formula: RRF_score(d) = sum(1 / (k + rank_i)) for each ranker i.
        """
        fused_scores: dict[int, float] = {}

        for rank, (idx, _score) in enumerate(dense_results):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, (idx, _score) in enumerate(sparse_results):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Sort by fused score descending
        sorted_indices = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)

        results = []
        for idx in sorted_indices:
            results.append(RetrievedChunk(
                chunk=chunks[idx],
                score=fused_scores[idx],
            ))

        return results

    # -- Hybrid search (main retrieval path) ------------------------------

    def _hybrid_search(self, query: str) -> list[RetrievedChunk]:
        """Run hybrid (dense + sparse) search across all loaded indexes."""
        from app.rag.embedder import get_embedder
        from app.settings import RERANKER_ENABLED
        import faiss  # type: ignore

        embedder = get_embedder()
        q_vec = np.array([embedder.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        n_candidates = RETRIEVAL_CANDIDATES if _bm25_available else TOP_K

        # Search primary (fault-code) index
        all_results = self._search_single_index(
            self._index, self._chunks, self._bm25, q_vec, query, n_candidates,
        )

        # Search general-document index if loaded
        if self._general_index is not None:
            general_results = self._search_single_index(
                self._general_index, self._general_chunks, self._general_bm25,
                q_vec, query, n_candidates,
            )
            all_results.extend(general_results)

        # Deduplicate by chunk_id across indexes
        seen_ids: set[str] = set()
        deduped: list[RetrievedChunk] = []
        for rc in sorted(all_results, key=lambda r: r.score, reverse=True):
            cid = rc.chunk.get("chunk_id", "")
            if cid and cid in seen_ids:
                continue
            if cid:
                seen_ids.add(cid)
            deduped.append(rc)

        # Re-rank top candidates using Cross-Encoder if enabled
        if RERANKER_ENABLED and deduped:
            from app.rag.reranker import get_reranker
            reranker = get_reranker()
            deduped = reranker.rerank(query, deduped[:n_candidates])

        return deduped[:RETURN_N]

    def _search_single_index(
        self,
        faiss_index: Any,
        chunks: list[dict],
        bm25_index,
        q_vec: "np.ndarray",
        query: str,
        n_candidates: int,
    ) -> list[RetrievedChunk]:
        """Search a single index pair (FAISS + BM25) and return fused results."""
        dense_results = self._dense_search(faiss_index, chunks, q_vec, n_candidates)

        if bm25_index is not None and _bm25_available:
            sparse_results = self._sparse_search(bm25_index, chunks, query, n_candidates)
            return self._rrf_fuse(dense_results, sparse_results, chunks)
        else:
            # Dense-only fallback
            results = []
            for idx, score in dense_results:
                results.append(RetrievedChunk(chunk=chunks[idx], score=score))
            results.sort(key=lambda r: r.score, reverse=True)
            return results

    # -- Public API -------------------------------------------------------

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

            # Hybrid search (FR-2)
            hybrid = self._hybrid_search(query)
            mode = "hybrid (dense+BM25)" if _bm25_available else "dense-only"
            log.info(
                "Search [%s] for '%s': %d chunk(s) returned",
                mode, query[:60], len(hybrid),
            )
            return hybrid

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

            # Rebuild BM25 index
            self._bm25 = self._build_bm25_index(self._chunks)

            # Also reload general index if it exists
            self._general_bm25 = None
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
