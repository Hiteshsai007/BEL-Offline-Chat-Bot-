"""
Post-ingestion validator for general (unstructured) documents.

This validator performs sanity checks appropriate for unstructured text --
it does NOT do the verbatim row-level fidelity checking that the fault-code
validator (validator.py) does.  The reason is fundamental: the fault-code
validator compares each chunk against a known-structured source row
(error_code / error_description / error_remarks), so it can verify that
every chunk is a faithful copy.  General documents have no such ground-truth
structure -- a prose paragraph may be legally chunked in many different ways,
and there is no single "correct" representation to compare against.

Instead, this validator checks:
  - No empty chunks (would produce meaningless embeddings)
  - No chunk exceeds a reasonable max length (flags chunking failures)
  - Every chunk has required metadata (source_document, page_number)
  - Duplicate chunk detection (PDF extraction artifacts can repeat text)
  - Every chunk has a valid chunk_type

Warnings are logged but do not block ingestion.  Errors do block.
"""
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)

# A chunk longer than this is likely a chunking failure (missed split).
MAX_CHUNK_TOKENS = 3000

# Similarity threshold for duplicate detection (Jaccard on word sets).
_DUPLICATE_JACCARD_THRESHOLD = 0.85


def validate_general_chunks(
    chunks: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Validate general-document chunks.

    Returns:
        (warnings, errors)
        Empty errors list means validation passed.
    """
    warnings: list[str] = []
    errors: list[str] = []

    if not chunks:
        errors.append("CRITICAL: No chunks produced from ingestion.")
        return warnings, errors

    seen_word_sets: list[tuple[int, set[str]]] = []

    for chunk in chunks:
        cid = chunk.get("chunk_id", "?")
        text = chunk.get("chunk_text", "")
        tokens = chunk.get("token_count", 0)
        doc = chunk.get("document_name", "")
        page = chunk.get("page_number")
        ctype = chunk.get("chunk_type", "")

        # -- Empty chunk --
        if not text or not text.strip():
            errors.append(f"EMPTY: Chunk {cid} has no text content.")
            continue

        # -- Missing metadata --
        if not doc:
            errors.append(
                f"MISSING_DOC: Chunk {cid} has no source_document."
            )
        if page is None:
            warnings.append(
                f"NO_PAGE: Chunk {cid} has no page_number."
            )

        # -- Invalid chunk type --
        if ctype not in ("prose", "table", "image"):
            errors.append(
                f"BAD_TYPE: Chunk {cid} has invalid chunk_type "
                f"'{ctype}'."
            )

        # -- Oversized chunk --
        if tokens > MAX_CHUNK_TOKENS:
            warnings.append(
                f"OVERSIZE: Chunk {cid} has {tokens} tokens "
                f"(max {MAX_CHUNK_TOKENS}). May indicate a "
                f"chunking failure."
            )

        # -- Duplicate detection --
        words = set(text.lower().split())
        for prev_id, prev_words in seen_word_sets:
            if not words or not prev_words:
                continue
            jaccard = len(words & prev_words) / len(words | prev_words)
            if jaccard > _DUPLICATE_JACCARD_THRESHOLD:
                warnings.append(
                    f"DUPLICATE: Chunk {cid} is {jaccard:.0%} similar "
                    f"to chunk {prev_id}. Possible PDF extraction "
                    f"artifact."
                )
                break
        seen_word_sets.append((cid, words))

    if errors:
        log.error("General validation found %d error(s):", len(errors))
        for e in errors:
            log.error("  - %s", e)
    if warnings:
        log.warning("General validation found %d warning(s):", len(warnings))
        for w in warnings:
            log.warning("  - %s", w)
    if not errors and not warnings:
        log.info(
            "General validation passed: %d chunks OK.", len(chunks),
        )

    return warnings, errors
