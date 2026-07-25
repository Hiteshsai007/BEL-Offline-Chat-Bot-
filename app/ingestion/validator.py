"""
Post-ingestion validator — enforces the PRD Section 9 fidelity requirement.

Spot-checks that every ingested chunk's text is traceable back to the source
PDF content. Flags discrepancies and blocks the index swap if hard errors are
found; warnings are logged but do not abort ingestion.
"""
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)


def validate_chunks(
    chunks: list[dict[str, Any]],
    parse_result: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """
    Compare ingested chunks against the raw parse result.

    Returns:
        (warnings, errors)
        Empty errors list means validation passed and ingestion may proceed.
        Warnings are informational and do not block the pipeline.
    """
    warnings: list[str] = []
    errors: list[str] = []
    source_rows = parse_result["rows"]

    # Build a lookup: error_code -> list of source rows with that code
    # Using a list preserves duplicate codes in the source PDF so every
    # occurrence can be validated against (finding H-3).
    code_to_sources: dict[str, list[dict]] = {}
    for row in source_rows:
        code = (row.get("error_code") or "").strip()
        if code:
            code_to_sources.setdefault(code.lower(), []).append(row)

    table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]

    if not table_chunks:
        errors.append("CRITICAL: No table chunks produced from ingestion.")
        return warnings, errors

    if len(table_chunks) < len(source_rows):
        warnings.append(
            f"Fewer chunks ({len(table_chunks)}) than source rows "
            f"({len(source_rows)}) — some rows may have been dropped."
        )

    for chunk in table_chunks:
        code = (chunk.get("error_code") or "").lower()
        if not code:
            continue

        sources = code_to_sources.get(code)
        if not sources:
            errors.append(
                f"FABRICATION: Chunk {chunk['chunk_id']} has error code {code} "
                f"that does not exist in the source PDF."
            )
            continue

        # Verify description matches verbatim against at least one source row
        chunk_desc = (chunk.get("error_description") or "").strip()
        source_descs = [
            (s.get("error_description") or "").strip() for s in sources
        ]
        if chunk_desc and source_descs and chunk_desc not in source_descs:
            errors.append(
                f"MISMATCH: Code {code} — chunk desc '{chunk_desc}' ≠ "
                f"any source desc {source_descs}"
            )

        # Verify remarks match verbatim against at least one source row
        chunk_rem = (chunk.get("error_remarks") or "").strip()
        source_rems = [
            (s.get("error_remarks") or "").strip() for s in sources
        ]
        if chunk_rem and source_rems and chunk_rem not in source_rems:
            errors.append(
                f"MISMATCH: Code {code} — chunk remarks '{chunk_rem}' ≠ "
                f"any source remarks {source_rems}"
            )

    if errors:
        log.error("Validation found %d error(s):", len(errors))
        for e in errors:
            log.error("  • %s", e)
    if warnings:
        log.warning("Validation found %d warning(s):", len(warnings))
        for w in warnings:
            log.warning("  • %s", w)
    if not errors and not warnings:
        log.info(
            "Validation passed: %d chunks verified against %d source rows",
            len(table_chunks), len(source_rows),
        )

    return warnings, errors
