"""
Chunker — converts parsed PDF rows into embedding-ready chunk dicts.

Each fault-code row becomes exactly one chunk. Metadata follows the
schema in PRD Section 9. Phase 1 fields that are unknown from the
source are explicitly set to null (not guessed).
"""
import json
import uuid
from pathlib import Path
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)

# Phase 1: these fields are unavailable from IRL_Fault_Codes.pdf
_NULL_PHASE1 = {
    "document_version": None,
    "system_name": None,
    "subsystem": None,
    "chapter": None,
    "section": None,
}


def _make_chunk(
    row: dict[str, Any],
    document_name: str,
    source_hash: str,
) -> dict[str, Any]:
    """
    Build one chunk dict from a parsed table row.

    The chunk text is constructed verbatim from source fields —
    no summarisation or paraphrasing (PRD Section 9 fidelity).
    """
    error_code = row.get("error_code", "") or ""
    desc = row.get("error_description", "") or ""
    remarks = row.get("error_remarks", "") or ""
    sl_no = row.get("sl_no", "") or ""
    page_number = row.get("page_number", None)

    # Verbatim text representation used for embedding and display
    text_parts = []
    if error_code:
        text_parts.append(f"Error Code: {error_code}")
    if desc:
        text_parts.append(f"Error Description: {desc}")
    if remarks:
        text_parts.append(f"Error Remarks: {remarks}")
    chunk_text = " | ".join(text_parts)

    token_count = len(chunk_text.split())  # approximate

    return {
        "chunk_id": str(uuid.uuid4()),
        "chunk_text": chunk_text,
        # --- PRD Section 9 metadata schema ---
        "document_name": document_name,
        **_NULL_PHASE1,
        "page_number": page_number,
        "error_code": error_code if error_code else None,
        "chunk_type": "table",
        "token_count": token_count,
        "source_hash": source_hash,
        # --- Phase 1 convenience fields (not in schema, useful for UI) ---
        "sl_no": sl_no,
        "error_description": desc,
        "error_remarks": remarks,
    }


def _make_footnote_chunk(
    footnote_text: str,
    document_name: str,
    source_hash: str,
    index: int,
) -> dict[str, Any]:
    """
    Preserve footnotes (e.g. 'Note 6') as separate chunks.
    Kept verbatim from source — may clarify throw-range entries in Phase 2.
    """
    return {
        "chunk_id": str(uuid.uuid4()),
        "chunk_text": footnote_text,
        "document_name": document_name,
        **_NULL_PHASE1,
        "page_number": None,
        "error_code": None,
        "chunk_type": "prose",
        "token_count": len(footnote_text.split()),
        "source_hash": source_hash,
        "sl_no": None,
        "error_description": f"Footnote {index}",
        "error_remarks": footnote_text,
    }


def build_chunks(
    parse_result: dict[str, Any],
    document_name: str,
) -> list[dict[str, Any]]:
    """
    Convert parse_result (from parser.parse_pdf) into a list of chunk dicts.
    Also includes footnote chunks if any were found.
    """
    rows: list[dict] = parse_result["rows"]
    footnotes: list[str] = parse_result.get("footnotes", [])
    source_hash: str = parse_result["source_hash"]

    chunks = []
    for row in rows:
        chunk = _make_chunk(row, document_name, source_hash)
        chunks.append(chunk)

    for i, fn in enumerate(footnotes, start=1):
        chunk = _make_footnote_chunk(fn, document_name, source_hash, i)
        chunks.append(chunk)

    log.info(
        "Built %d chunks (%d table rows + %d footnotes)",
        len(chunks), len(rows), len(footnotes),
    )
    return chunks


def save_chunks(chunks: list[dict[str, Any]], path: Path) -> None:
    """Persist chunks to a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Saved %d chunks → %s", len(chunks), path)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
