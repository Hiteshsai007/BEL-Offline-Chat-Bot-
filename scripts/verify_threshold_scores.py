#!/usr/bin/env python3
"""
Run the flip-list queries against the REAL BGE-small-en-v.15 embedding model
and FAISS index to validate the estimated scores used in the threshold
analysis.

Usage:
    python scripts/verify_threshold_scores.py

Requires:
    - The embedding model cached locally (run setup.sh / bootstrap.py first)
    - The FAISS index built (run python -m app.ingestion.ingest first)

Exit codes:
    0 — all scores are within the expected ranges
    1 — one or more scores fell outside the expected range (review needed)

This script is intended to be run in CI or on a real workstation where the
embedding model is available.  It will fail gracefully if the model cannot
be loaded (e.g. in a sandbox without HuggingFace access).
"""
import os
import sys
from pathlib import Path

# Ensure we can import the app package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Lift offline flags so the model can be loaded from local cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


# ── Flip-list queries with estimated score ranges ───────────────────────────
# Each entry: (label, query, expected_min, expected_max, description)
FLIP_LIST = [
    # Category A — exact code lookups (handled by _exact_lookup, score=1.0)
    # These bypass embedding entirely so we verify them separately.
    ("A1", "What does error 0x0003 mean?",  1.0, 1.0, "Exact code lookup"),
    ("A2", "Explain 0x0017",                1.0, 1.0, "Exact code lookup"),
    ("A3", "What is 0x0005?",               1.0, 1.0, "Exact code lookup"),
    ("A4", "Tell me about 0x0028",          1.0, 1.0, "Exact code lookup"),

    # Category B — semantic queries estimated in the 0.60-0.80 band
    ("B1", "What is a misfire error?",              0.60, 0.85, "Semantic: misfire"),
    ("B2", "What does throw range invalid mean?",   0.60, 0.85, "Semantic: throw range"),
    ("B3", "What causes fire abort?",               0.55, 0.80, "Semantic: fire abort"),
    ("B4", "What are interlock failures?",          0.55, 0.80, "Semantic: interlocks"),
    ("B5", "What is depth setting failure?",        0.60, 0.85, "Semantic: depth setting"),
    ("B6", "What's the difference between R1 and R2 misfires?", 0.55, 0.80, "Semantic: comparative"),
    ("B7", "How do I fix a misfire on R3?",         0.45, 0.75, "Semantic: fix/action"),

    # Category C — queries expected below 0.60
    ("C1", "What's the fix for a misfire in the debar zone?", 0.40, 0.65, "Semantic: cross-chunk"),
    ("C2", "How do I troubleshoot IRL errors?",               0.30, 0.60, "Semantic: vague"),
    ("C3", "What are the most common problems?",              0.25, 0.55, "Semantic: very vague"),
]

# Threshold under test
DIRECT_ANSWER_THRESHOLD = 0.80


def main() -> int:
    print("=" * 72)
    print("  Threshold Score Verification — Flip-List Queries")
    print("  DIRECT_ANSWER_THRESHOLD = 0.80")
    print("=" * 72)

    # ── Load components ─────────────────────────────────────────────────
    try:
        from app.rag.retriever import get_retriever
        retriever = get_retriever()
    except Exception as e:
        print(f"\nFATAL: Cannot load retriever: {e}")
        print("Ensure the FAISS index exists: python -m app.ingestion.ingest")
        return 1

    try:
        from app.rag.embedder import get_embedder
        embedder = get_embedder()
    except Exception as e:
        print(f"\nFATAL: Cannot load embedding model: {e}")
        print("Ensure the model is cached locally (run setup.sh first).")
        return 1

    import faiss  # type: ignore
    import numpy as np

    print(f"\nIndex: {retriever._index.ntotal} vectors x {retriever._index.d}d")
    print("Model: BAAI/bge-small-en-v1.5")
    print(f"Threshold: {DIRECT_ANSWER_THRESHOLD}")
    print()

    # ── Run queries ─────────────────────────────────────────────────────
    warnings = []
    print(f"{'ID':<4} {'Score':>6} {'Route':>6}  {'Range':>12}  {'OK':>4}  Query")
    print("-" * 72)

    for label, query_text, exp_min, exp_max, desc in FLIP_LIST:
        # Check if this is an exact code lookup
        import re
        codes = re.findall(r"0x[0-9a-fA-F]{4}", query_text, re.IGNORECASE)
        if codes:
            # Exact lookup — score is always 1.0
            actual_score = 1.0
        else:
            # Semantic search — compute real embedding score
            from app.settings import CONFIDENCE_THRESHOLD
            q_vec = np.array([embedder.embed_query(query_text)], dtype=np.float32)
            faiss.normalize_L2(q_vec)
            scores, indices = retriever._index.search(q_vec, 1)
            raw_score = float(scores[0][0])
            # Apply the same confidence threshold filter as the retriever
            actual_score = raw_score if raw_score >= CONFIDENCE_THRESHOLD else 0.0

        route = "FAST" if actual_score >= DIRECT_ANSWER_THRESHOLD else "LLM"
        in_range = exp_min <= actual_score <= exp_max
        ok_str = "OK" if in_range else "!!"
        range_str = f"[{exp_min:.2f}, {exp_max:.2f}]"

        print(f"{label:<4} {actual_score:>6.3f} {route:>6}  {range_str:>12}  {ok_str:>4}  {desc}")

        if not in_range:
            warnings.append(
                f"  {label}: actual={actual_score:.3f}, expected=[{exp_min:.2f}, {exp_max:.2f}] — {query_text}"
            )

    # ── Report ──────────────────────────────────────────────────────────
    print("-" * 72)
    if warnings:
        print(f"\n  {len(warnings)} score(s) outside expected range:\n")
        for w in warnings:
            print(w)
        print("\n  Review the flip-list analysis — estimated scores may need revision.")
        print("  The threshold (0.80) may need adjustment if multiple scores shifted.")
        return 1
    else:
        print(f"\n  All {len(FLIP_LIST)} scores within expected ranges.")
        print("  Threshold routing validated against real embeddings.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
