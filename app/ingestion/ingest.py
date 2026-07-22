"""
Main ingestion CLI — orchestrates the full pipeline:
    parse → chunk → embed → build FAISS index → validate → atomic swap

Usage:
    python -m app.ingestion.ingest
    python -m app.ingestion.ingest --pdf "IRL Fault Codes.pdf"
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from app.ingestion.chunker import build_chunks, save_chunks
from app.ingestion.parser import parse_pdf
from app.ingestion.validator import validate_chunks
from app.logger import get_logger
from app.settings import (
    CHUNKS_STORE_PATH,
    FAISS_INDEX_PATH,
    SOURCE_PDF_PATH,
)

log = get_logger(__name__)


def _embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Embed all chunk texts using the BGE model. Returns (N, D) float32 array."""
    # Import here so the embedder is not loaded unless ingestion is running
    from app.rag.embedder import get_embedder

    embedder = get_embedder()
    texts = [c["chunk_text"] for c in chunks]
    log.info("Embedding %d chunks …", len(texts))
    t0 = time.perf_counter()
    vectors = embedder.embed_documents(texts)
    elapsed = time.perf_counter() - t0
    log.info("Embedding done in %.2fs", elapsed)
    return np.array(vectors, dtype=np.float32)


from typing import Any

def _build_faiss_index(vectors: np.ndarray) -> Any:
    import faiss  # type: ignore

    dim = vectors.shape[1]
    n   = vectors.shape[0]

    if n < 100:
        # Small corpus: flat L2 index (exact search, no quantisation needed)
        index = faiss.IndexFlatIP(dim)          # inner-product = cosine on normalised vecs
    else:
        # Larger corpus: HNSW for fast approximate search
        index = faiss.IndexHNSWFlat(dim, 32)    # M=32 neighbours
        index.hnsw.efConstruction = 200

    # Normalise vectors so inner product == cosine similarity
    faiss.normalize_L2(vectors)
    index.add(vectors)
    log.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


def _save_faiss_index(index, path: Path) -> None:
    import faiss  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then atomic rename
    tmp = path.with_suffix(".tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(path)
    log.info("FAISS index saved → %s", path)


def run_ingestion(pdf_path: Path) -> bool:
    """
    Full ingestion pipeline. Returns True on success, False on failure.
    Index files are only swapped into place after validation passes.
    """
    t_start = time.perf_counter()
    log.info("═" * 60)
    log.info("Starting ingestion: %s", pdf_path)

    # ── 1. Parse ──────────────────────────────────────────────────
    try:
        parse_result = parse_pdf(pdf_path)
    except Exception as e:
        log.error("Parse failed: %s", e)
        return False

    document_name = pdf_path.name

    # ── 2. Chunk ──────────────────────────────────────────────────
    chunks = build_chunks(parse_result, document_name)
    if not chunks:
        log.error("No chunks produced — aborting.")
        return False

    # ── 3. Validate fidelity (pre-embed) ─────────────────────────
    errors = validate_chunks(chunks, parse_result)
    if errors:
        log.error("Fidelity validation FAILED — aborting index build.")
        log.error("Fix the issues above before re-running ingestion.")
        return False

    # ── 4. Embed ──────────────────────────────────────────────────
    try:
        vectors = _embed_chunks(chunks)
    except Exception as e:
        log.error("Embedding failed: %s", e)
        return False

    # ── 5. Build FAISS index ──────────────────────────────────────
    try:
        index = _build_faiss_index(vectors)
    except Exception as e:
        log.error("FAISS index build failed: %s", e)
        return False

    # ── 6. Atomic save (chunks + index written to tmp, then swapped) ──
    tmp_chunks = CHUNKS_STORE_PATH.with_suffix(".tmp.jsonl")
    tmp_index  = FAISS_INDEX_PATH.with_suffix(".tmp")

    try:
        save_chunks(chunks, tmp_chunks)
        _save_faiss_index(index, tmp_index)

        # Atomic swap
        CHUNKS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_chunks), str(CHUNKS_STORE_PATH))
        shutil.move(str(tmp_index),  str(FAISS_INDEX_PATH))
    except Exception as e:
        log.error("Save/swap failed: %s", e)
        for f in [tmp_chunks, tmp_index]:
            if f.exists():
                f.unlink()
        return False

    elapsed = time.perf_counter() - t_start
    log.info("═" * 60)
    log.info(
        "Ingestion complete in %.1fs — %d chunks indexed.",
        elapsed, len(chunks),
    )
    log.info("Source hash: %s", parse_result["source_hash"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BEL Offline AI — knowledge base ingestion script"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=SOURCE_PDF_PATH,
        help="Path to the source PDF (default: config paths.source_pdf)",
    )
    args = parser.parse_args()

    success = run_ingestion(args.pdf)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
