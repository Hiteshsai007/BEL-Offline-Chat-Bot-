"""
Post-ingestion validator — enforces the PRD Section 9 fidelity requirement.

Spot-checks that every ingested chunk's text is traceable back to the source
PDF content. Flags discrepancies and blocks the index swap if any are found.
"""
from pathlib import Path
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)


def validate_chunks(
    chunks: list[dict[str, Any]],
    parse_result: dict[str, Any],
) -> list[str]:
    """
    Compare ingested chunks against the raw parse result.

    Returns a list of error strings. Empty list = valid.
    """
    errors: list[str] = []
    source_rows = parse_result["rows"]

    # Build a lookup: error_code → source row
    code_to_source: dict[str, dict] = {}
    for row in source_rows:
        code = (row.get("error_code") or "").strip()
        if code:
            code_to_source[code.lower()] = row

    table_chunks = [c for c in chunks if c.get("chunk_type") == "table"]

    if not table_chunks:
        errors.append("CRITICAL: No table chunks produced from ingestion.")
        return errors

    if len(table_chunks) < len(source_rows):
        errors.append(
            f"WARNING: Fewer chunks ({len(table_chunks)}) than source rows "
            f"({len(source_rows)}) — some rows may have been dropped."
        )

    for chunk in table_chunks:
        code = (chunk.get("error_code") or "").lower()
        if not code:
            continue

        source = code_to_source.get(code)
        if source is None:
            errors.append(
                f"FABRICATION: Chunk {chunk['chunk_id']} has error code {code} "
                f"that does not exist in the source PDF."
            )
            continue

        # Verify description matches verbatim
        chunk_desc = (chunk.get("error_description") or "").strip()
        source_desc = (source.get("error_description") or "").strip()
        if chunk_desc and source_desc and chunk_desc != source_desc:
            errors.append(
                f"MISMATCH: Code {code} — chunk desc '{chunk_desc}' ≠ "
                f"source desc '{source_desc}'"
            )

        # Verify remarks match verbatim
        chunk_rem = (chunk.get("error_remarks") or "").strip()
        source_rem = (source.get("error_remarks") or "").strip()
        if chunk_rem and source_rem and chunk_rem != source_rem:
            errors.append(
                f"MISMATCH: Code {code} — chunk remarks '{chunk_rem}' ≠ "
                f"source remarks '{source_rem}'"
            )

    if errors:
        log.error("Validation found %d issue(s):", len(errors))
        for e in errors:
            log.error("  • %s", e)
    else:
        log.info(
            "Validation passed: %d chunks verified against %d source rows",
            len(table_chunks), len(source_rows),
        )

    return errors
