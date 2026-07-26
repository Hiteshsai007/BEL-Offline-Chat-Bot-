#!/usr/bin/env python3
"""
Standalone retrieval sanity-check tool for the general-document FAISS index.

Loads data/index/general/faiss.index and chunks.jsonl directly (not wired
into the chat UI), embeds a query with BGE-small-en-v1.5, and prints the
top-K results with similarity scores, page numbers, and image references.

Usage:
    python scripts/query_general_index.py "how do I check tire pressure"
    python scripts/query_general_index.py --top 10 "what oil should I use"

Requires:
    - The general index built (run python -m app.ingestion.ingest_general first)
    - The embedding model cached locally (run setup.sh / bootstrap.py first)

Exit codes:
    0 — query completed successfully
    1 — index files missing or model load failure (clear error message)
"""
import argparse
import os
import sys
from pathlib import Path

# Ensure we can import the app package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Match the offline flags used by the app at runtime
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _check_index_exists() -> tuple[Path, Path]:
    """Return paths to the general index files, or exit with a clear error."""
    from app.settings import FAISS_INDEX_PATH

    index_dir = FAISS_INDEX_PATH.parent / "general"
    faiss_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.jsonl"

    if not faiss_path.exists():
        print(f"ERROR: FAISS index not found at {faiss_path}")
        print("Run ingestion first: python -m app.ingestion.ingest_general "
              "--pdf path/to/document.pdf")
        sys.exit(1)
    if not chunks_path.exists():
        print(f"ERROR: Chunks file not found at {chunks_path}")
        print("Run ingestion first: python -m app.ingestion.ingest_general "
              "--pdf path/to/document.pdf")
        sys.exit(1)
    return faiss_path, chunks_path


def _load_chunks(path: Path) -> list[dict]:
    """Load chunks from a JSONL file."""
    import json
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _truncate(text: str, max_len: int = 300) -> str:
    """Truncate text to max_len characters, adding ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query the general-document FAISS index",
    )
    parser.add_argument(
        "query",
        help="Search query string",
    )
    parser.add_argument(
        "--top", "-k",
        type=int,
        default=5,
        help="Number of results to show (default: 5)",
    )
    args = parser.parse_args()

    faiss_path, chunks_path = _check_index_exists()

    # Load index
    try:
        import faiss  # type: ignore
        index = faiss.read_index(str(faiss_path))
    except Exception as e:
        print(f"ERROR: Failed to load FAISS index: {e}")
        return 1

    # Load chunks
    try:
        chunks = _load_chunks(chunks_path)
    except Exception as e:
        print(f"ERROR: Failed to load chunks: {e}")
        return 1

    if index.ntotal != len(chunks):
        print(f"WARNING: Index has {index.ntotal} vectors but chunks file "
              f"has {len(chunks)} entries.")

    # Load embedder
    try:
        from app.rag.embedder import get_embedder
        embedder = get_embedder()
    except Exception as e:
        print(f"ERROR: Cannot load embedding model: {e}")
        print("Ensure the model is cached locally (run setup.sh first).")
        return 1

    import numpy as np

    # Embed query
    q_vec = np.array(
        [embedder.embed_query(args.query)], dtype=np.float32,
    )

    # Search
    import faiss as _faiss
    _faiss.normalize_L2(q_vec)
    k = min(args.top, index.ntotal)
    scores, indices = index.search(q_vec, k)

    # Display results
    print("=" * 72)
    print(f"  Query: \"{args.query}\"")
    print(f"  Index: {index.ntotal} vectors x {index.d}d")
    print(f"  Top {k} results:")
    print("=" * 72)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1,
    ):
        if idx < 0:
            continue
        chunk = chunks[idx]
        text = chunk.get("chunk_text", "")
        page = chunk.get("page_number", "?")
        heading = chunk.get("section_heading", "")
        ctype = chunk.get("chunk_type", "prose")
        img_path = chunk.get("image_file_path", "")
        fig_refs = chunk.get("figure_references", [])

        print(f"\n--- Result {rank} (score: {score:.4f}) ---")
        print(f"  Page: {page}  |  Type: {ctype}")
        if heading:
            print(f"  Section: {heading}")
        print(f"  Text: {_truncate(text)}")
        if img_path:
            print(f"  Image: {img_path}")
        if fig_refs:
            print(f"  Figure refs: {', '.join(fig_refs)}")

    print("\n" + "=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
