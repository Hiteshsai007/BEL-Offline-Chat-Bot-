"""
BGE embedding wrapper — CPU-only, offline, singleton pattern.

Uses BAAI/bge-small-en-v1.5 via sentence-transformers.
GPU is reserved entirely for Ollama (PRD Section 7 + Section 11).

Environment variables set at process start (setup.ps1 / Launch*.bat / launch*.sh):
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    SENTENCE_TRANSFORMERS_HOME=<local cache dir>
"""
import os
import threading
from typing import List

# Force offline mode — no runtime network calls (PRD Section 12).
#
# HF_HUB_OFFLINE is the variable modern huggingface_hub (which
# sentence-transformers >=3.x uses for all resolution) actually keys off.
# TRANSFORMERS_OFFLINE alone does NOT stop a hub lookup on a cache miss or a
# revision check, so without this the "no runtime network calls" guarantee was
# asserted in this docstring but never enforced (finding S-5).
#
# setdefault (not assignment) is deliberate: the setup scripts intentionally
# export these as "0" for the one-time model download, and that must keep
# working.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# E402: these imports must follow the os.environ.setdefault calls above --
# app.settings/app.logger pull in modules that can trigger HF resolution, so
# the offline switches have to be armed first. Suppression is deliberate.
from app.logger import get_logger  # noqa: E402
from app.settings import EMBED_DEVICE, EMBED_MODEL, QUERY_PREFIX  # noqa: E402

log = get_logger(__name__)

_lock = threading.Lock()
_embedder_instance = None


class BGEEmbedder:
    """Thin wrapper around sentence-transformers for BGE models."""

    def __init__(self, model_name: str, device: str) -> None:
        log.info("Loading embedding model '%s' on %s …", model_name, device)
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name, device=device)
        self._device = device
        log.info("Embedding model loaded.")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string with the BGE query prefix."""
        prefixed = QUERY_PREFIX + text
        vec = self._model.encode(
            [prefixed],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks (no prefix for documents)."""
        # Use smaller batch size on memory-constrained systems
        _batch_size = int(os.environ.get("BEL_EMBED_BATCH_SIZE", "32"))
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 20,
            batch_size=_batch_size,
        )
        return vecs.tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


def get_embedder() -> BGEEmbedder:
    """Return the singleton BGEEmbedder, loading it on first call."""
    global _embedder_instance
    if _embedder_instance is None:
        with _lock:
            if _embedder_instance is None:
                _embedder_instance = BGEEmbedder(EMBED_MODEL, EMBED_DEVICE)
    return _embedder_instance
