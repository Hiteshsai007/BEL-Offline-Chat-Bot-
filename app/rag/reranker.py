"""
Cross-Encoder Reranker — re-scores query-passage pairs for high precision.

Uses sentence-transformers CrossEncoder (e.g. BAAI/bge-reranker-base).
Runs CPU-only, offline, with thread-safe singleton pattern.
"""
import os
import threading
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.logger import get_logger  # noqa: E402
from app.settings import EMBED_DEVICE, RERANKER_MODEL  # noqa: E402

log = get_logger(__name__)

_lock = threading.Lock()
_reranker_instance = None


class Reranker:
    """Wrapper around sentence-transformers CrossEncoder for joint re-scoring."""

    def __init__(self, model_name: str = RERANKER_MODEL, device: str = EMBED_DEVICE) -> None:
        log.info("Loading cross-encoder reranker '%s' on %s ...", model_name, device)
        from sentence_transformers import CrossEncoder  # type: ignore

        try:
            self._model = CrossEncoder(model_name, max_length=512, device=device)
            self._available = True
            log.info("Cross-encoder reranker loaded successfully.")
        except Exception as e:
            log.warning("Failed to load reranker model '%s': %s -- disabling reranker.", model_name, e)
            self._model = None
            self._available = False

    def rerank(self, query: str, candidate_chunks: list[Any]) -> list[Any]:
        """
        Re-score and re-order candidate chunks using the Cross-Encoder.
        Returns candidate_chunks sorted by joint relevance score descending.
        """
        if not self._available or not self._model or not candidate_chunks:
            return candidate_chunks

        # Build (query, passage) pairs
        pairs = [(query, c.chunk.get("chunk_text", "")) for c in candidate_chunks]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)

            # Update scores on RetrievedChunk items and sort
            for c, s in zip(candidate_chunks, scores):
                c.score = float(s)

            candidate_chunks.sort(key=lambda r: r.score, reverse=True)
            top_score = candidate_chunks[0].score if candidate_chunks else 0.0
            log.info(
                "Reranked %d candidates with CrossEncoder (top score: %.4f)",
                len(candidate_chunks),
                top_score,
            )
            return candidate_chunks
        except Exception as e:
            log.warning("Reranking prediction failed: %s -- returning original candidate ordering.", e)
            return candidate_chunks


def get_reranker() -> Reranker:
    """Return singleton Reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        with _lock:
            if _reranker_instance is None:
                _reranker_instance = Reranker()
    return _reranker_instance
