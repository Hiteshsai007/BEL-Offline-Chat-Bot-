"""
Ingestion CLI for general (unstructured) documents.

Mirrors the existing ingest.py structure and atomic-swap-safety pattern
(C-6 fix) but writes to a SEPARATE index location:
    data/index/general/faiss.index
    data/index/general/chunks.jsonl

This does NOT touch the existing fault-code index at:
    data/index/faiss.index
    data/index/chunks.jsonl

Usage:
    python -m app.ingestion.ingest_general --pdf path/to/manual.pdf
    python -m app.ingestion.ingest_general --md  path/to/manual.md
"""
import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.ingestion.general_chunker import (
    build_general_chunks,
    deduplicate_boilerplate_chunks,
    save_general_chunks,
)
from app.ingestion.general_parser import parse_general_pdf
from app.ingestion.markdown_parser import parse_markdown_file
from app.ingestion.general_validator import validate_general_chunks
from app.logger import get_logger
from app.settings import FAISS_INDEX_PATH

log = get_logger(__name__)

# Separate index location -- never touches the fault-code index.
GENERAL_INDEX_DIR = FAISS_INDEX_PATH.parent / "general"
GENERAL_FAISS_PATH = GENERAL_INDEX_DIR / "faiss.index"
GENERAL_CHUNKS_PATH = GENERAL_INDEX_DIR / "chunks.jsonl"


def _embed_chunks(chunks: list[dict]) -> np.ndarray:
    """Embed all chunk texts using the BGE model. Returns (N, D) float32."""
    from app.rag.embedder import get_embedder

    embedder = get_embedder()
    texts = [c["chunk_text"] for c in chunks]
    log.info("Embedding %d chunks ...", len(texts))
    t0 = time.perf_counter()
    vectors = embedder.embed_documents(texts)
    elapsed = time.perf_counter() - t0
    log.info("Embedding done in %.2fs", elapsed)
    return np.array(vectors, dtype=np.float32)


def _build_faiss_index(vectors: np.ndarray) -> Any:
    """Build a FAISS index using the same logic as the existing ingest.py."""
    import faiss  # type: ignore

    dim = vectors.shape[1]
    n = vectors.shape[0]

    if n < 100:
        # Small corpus: flat inner-product index (exact search)
        index = faiss.IndexFlatIP(dim)
    else:
        # Larger corpus: HNSW for fast approximate search (C-3 fix)
        index = faiss.IndexHNSWFlat(
            dim, 32, faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = 200

    faiss.normalize_L2(vectors)
    index.add(vectors)
    log.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


def _save_faiss_index(index: Any, path: Path) -> None:
    """Write FAISS index atomically via temp file + rename."""
    import faiss  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(path)
    log.info("FAISS index saved -> %s", path)


def run_general_ingestion(pdf_path: Path) -> bool:
    """
    Full ingestion pipeline for a general document.

    Hybrid PDF + Markdown Flow:
      1. Parse PDF directly (parse_general_pdf).
      2. Save original PDF and extracted images to data/index/general/.
      3. Generate Markdown representation (generated.md) with explicit
         <!-- PAGE:N --> markers.
      4. Chunk and index Markdown representation while retaining PDF page
         metadata.
      5. Failsafe: Fall back seamlessly to direct PDF chunking if Markdown
         generation fails.
      6. Run 10-point automated validation phase before atomic index
         replacement.

    Returns True on success, False on failure.
    Index files are only swapped into place after validation passes.
    """
    t_start = time.perf_counter()
    log.info("=" * 60)
    log.info("Starting general document ingestion: %s", pdf_path)

    # -- 1. Parse PDF ---------------------------------------------------
    try:
        image_dir = GENERAL_INDEX_DIR / "extracted_images"
        parse_result = parse_general_pdf(
            pdf_path, image_output_dir=image_dir,
        )
    except Exception as e:
        log.error("Parse failed: %s", e)
        return False

    # Save original PDF into general index directory
    try:
        GENERAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        orig_pdf_dest = GENERAL_INDEX_DIR / "original.pdf"
        shutil.copy2(str(pdf_path), str(orig_pdf_dest))
    except Exception as e:
        log.warning(
            "Could not copy original PDF to index directory (non-fatal): %s", e
        )

    # -- 2. Generate Markdown & Chunk (with Failsafe Rollback) -----------
    chunks = None
    md_success = False
    try:
        from app.ingestion.pdf_to_markdown import (
            generate_markdown_from_pdf_parse,
        )
        from app.ingestion.markdown_parser import parse_markdown_file

        md_content = generate_markdown_from_pdf_parse(parse_result)
        md_path = GENERAL_INDEX_DIR / "generated.md"
        md_path.write_text(md_content, encoding="utf-8")
        log.info("Generated Markdown representation -> %s", md_path)

        md_parse_result = parse_markdown_file(md_path)
        md_parse_result["source_document"] = parse_result["source_document"]
        md_parse_result["source_hash"] = parse_result["source_hash"]
        image_items = [
            it for it in parse_result.get("items", [])
            if it.get("type") == "image"
        ]
        if image_items:
            md_parse_result["items"].extend(image_items)
        chunks = build_general_chunks(md_parse_result)
        if chunks:
            md_success = True
            log.info(
                "Built %d chunks from Markdown representation", len(chunks)
            )
    except Exception as e:
        log.warning(
            "Markdown conversion/parsing failed (%s) -- "
            "triggering PDF fallback", e
        )

    if not md_success or not chunks:
        log.info("Failsafe: Using direct PDF chunking fallback")
        chunks = build_general_chunks(parse_result)

    if not chunks:
        log.error("No chunks produced -- aborting.")
        return False

    # -- 2b. Boilerplate dedup -------------------------------------------
    chunks_before = len(chunks)
    chunks = deduplicate_boilerplate_chunks(chunks)
    if len(chunks) < chunks_before:
        log.info(
            "Boilerplate dedup: %d -> %d chunks (-%d)",
            chunks_before, len(chunks), chunks_before - len(chunks),
        )

    # -- 3. Validate (pre-embed) ----------------------------------------
    warnings, errors = validate_general_chunks(chunks)
    if warnings:
        for w in warnings:
            log.warning("Validation warning: %s", w)
    if errors:
        log.error("Validation FAILED -- aborting index build.")
        return False

    # -- 4. Embed --------------------------------------------------------
    try:
        vectors = _embed_chunks(chunks)
    except Exception as e:
        log.error("Embedding failed: %s", e)
        return False

    # -- 5. Build FAISS index -------------------------------------------
    try:
        index = _build_faiss_index(vectors)
    except Exception as e:
        log.error("FAISS index build failed: %s", e)
        return False

    # -- 6. Atomic save (same C-6 pattern as existing ingest.py) ---------
    tmp_chunks = GENERAL_CHUNKS_PATH.with_suffix(".tmp.jsonl")
    tmp_index = GENERAL_FAISS_PATH.with_suffix(".tmp")

    bak_chunks = GENERAL_CHUNKS_PATH.with_suffix(".bak")
    bak_index = GENERAL_FAISS_PATH.with_suffix(".bak")

    chunks_backed_up = False
    index_backed_up = False
    chunks_moved = False
    index_moved = False

    try:
        save_general_chunks(chunks, tmp_chunks)
        _save_faiss_index(index, tmp_index)

        # Back up existing files if present
        if GENERAL_CHUNKS_PATH.exists():
            shutil.move(str(GENERAL_CHUNKS_PATH), str(bak_chunks))
            chunks_backed_up = True
        if GENERAL_FAISS_PATH.exists():
            shutil.move(str(GENERAL_FAISS_PATH), str(bak_index))
            index_backed_up = True

        GENERAL_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        GENERAL_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Move new files into place
        shutil.move(str(tmp_chunks), str(GENERAL_CHUNKS_PATH))
        chunks_moved = True
        shutil.move(str(tmp_index), str(GENERAL_FAISS_PATH))
        index_moved = True

        # Clean up backups on success
        if chunks_backed_up and bak_chunks.exists():
            bak_chunks.unlink()
        if index_backed_up and bak_index.exists():
            bak_index.unlink()

    except Exception as e:
        log.error("Save/swap failed, rolling back: %s", e)
        if chunks_moved and GENERAL_CHUNKS_PATH.exists():
            GENERAL_CHUNKS_PATH.unlink()
        if chunks_backed_up:
            shutil.move(str(bak_chunks), str(GENERAL_CHUNKS_PATH))
        if index_moved and GENERAL_FAISS_PATH.exists():
            GENERAL_FAISS_PATH.unlink()
        if index_backed_up:
            shutil.move(str(bak_index), str(GENERAL_FAISS_PATH))
        for f in [tmp_chunks, tmp_index]:
            if f.exists():
                f.unlink()
        return False

    elapsed = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info(
        "General ingestion complete in %.1fs -- %d chunks indexed.",
        elapsed, len(chunks),
    )
    log.info("Index: %s", GENERAL_FAISS_PATH)
    log.info("Chunks: %s", GENERAL_CHUNKS_PATH)
    log.info("Source hash: %s", parse_result["source_hash"])
    return True


def run_general_ingestion_md(md_path: Path) -> bool:
    """
    Full ingestion pipeline for a Markdown document.

    Mirrors run_general_ingestion() exactly, replacing only the parse step:
    instead of calling parse_general_pdf(), it calls parse_markdown_file().
    All downstream steps (chunking, deduplication, validation, embedding,
    FAISS index build, atomic save) are identical to the PDF pipeline.

    Returns True on success, False on failure.
    NEVER touches data/index/faiss.index or data/index/chunks.jsonl.
    """
    t_start = time.perf_counter()
    log.info("=" * 60)
    log.info("Starting Markdown document ingestion: %s", md_path)

    # -- 1. Parse --------------------------------------------------------
    try:
        parse_result = parse_markdown_file(md_path)
    except Exception as e:
        log.error("Markdown parse failed: %s", e)
        return False

    # -- 2. Chunk --------------------------------------------------------
    chunks = build_general_chunks(parse_result)
    if not chunks:
        log.error("No chunks produced -- aborting.")
        return False

    # -- 2b. Boilerplate dedup -------------------------------------------
    chunks_before = len(chunks)
    chunks = deduplicate_boilerplate_chunks(chunks)
    if len(chunks) < chunks_before:
        log.info(
            "Boilerplate dedup: %d -> %d chunks (-%d)",
            chunks_before, len(chunks), chunks_before - len(chunks),
        )

    # -- 3. Validate (pre-embed) -----------------------------------------
    warnings, errors = validate_general_chunks(chunks)
    if warnings:
        for w in warnings:
            log.warning("Validation warning: %s", w)
    if errors:
        log.error("Validation FAILED -- aborting index build.")
        return False

    # -- 4. Embed --------------------------------------------------------
    try:
        vectors = _embed_chunks(chunks)
    except Exception as e:
        log.error("Embedding failed: %s", e)
        return False

    # -- 5. Build FAISS index -------------------------------------------
    try:
        index = _build_faiss_index(vectors)
    except Exception as e:
        log.error("FAISS index build failed: %s", e)
        return False

    # -- 6. Atomic save (same C-6 pattern as existing ingest.py) ---------
    tmp_chunks = GENERAL_CHUNKS_PATH.with_suffix(".tmp.jsonl")
    tmp_index = GENERAL_FAISS_PATH.with_suffix(".tmp")

    bak_chunks = GENERAL_CHUNKS_PATH.with_suffix(".bak")
    bak_index = GENERAL_FAISS_PATH.with_suffix(".bak")

    chunks_backed_up = False
    index_backed_up = False
    chunks_moved = False
    index_moved = False

    try:
        save_general_chunks(chunks, tmp_chunks)
        _save_faiss_index(index, tmp_index)

        if GENERAL_CHUNKS_PATH.exists():
            shutil.move(str(GENERAL_CHUNKS_PATH), str(bak_chunks))
            chunks_backed_up = True
        if GENERAL_FAISS_PATH.exists():
            shutil.move(str(GENERAL_FAISS_PATH), str(bak_index))
            index_backed_up = True

        GENERAL_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        GENERAL_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(tmp_chunks), str(GENERAL_CHUNKS_PATH))
        chunks_moved = True
        shutil.move(str(tmp_index), str(GENERAL_FAISS_PATH))
        index_moved = True

        if chunks_backed_up and bak_chunks.exists():
            bak_chunks.unlink()
        if index_backed_up and bak_index.exists():
            bak_index.unlink()

    except Exception as e:
        log.error("Save/swap failed, rolling back: %s", e)
        if chunks_moved and GENERAL_CHUNKS_PATH.exists():
            GENERAL_CHUNKS_PATH.unlink()
        if chunks_backed_up:
            shutil.move(str(bak_chunks), str(GENERAL_CHUNKS_PATH))
        if index_moved and GENERAL_FAISS_PATH.exists():
            GENERAL_FAISS_PATH.unlink()
        if index_backed_up:
            shutil.move(str(bak_index), str(GENERAL_FAISS_PATH))
        for f in [tmp_chunks, tmp_index]:
            if f.exists():
                f.unlink()
        return False

    elapsed = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info(
        "Markdown ingestion complete in %.1fs -- %d chunks indexed.",
        elapsed, len(chunks),
    )
    log.info("Index: %s", GENERAL_FAISS_PATH)
    log.info("Chunks: %s", GENERAL_CHUNKS_PATH)
    log.info("Source hash: %s", parse_result["source_hash"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "BEL Offline AI -- general document ingestion script"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pdf",
        type=Path,
        metavar="PDF",
        help="Path to the source PDF file.",
    )
    group.add_argument(
        "--md",
        type=Path,
        metavar="MD",
        help="Path to a Markdown (.md) file.",
    )
    args = parser.parse_args()

    if args.pdf:
        success = run_general_ingestion(args.pdf)
    else:
        success = run_general_ingestion_md(args.md)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
