"""
Tests for app/ingestion/chunker.py — chunk building, save, and load.
"""
from pathlib import Path

from app.ingestion.chunker import build_chunks, load_chunks, save_chunks


def test_build_chunks() -> None:
    """build_chunks must create table chunks from rows and prose chunks from footnotes."""
    parse_result = {
        "rows": [
            {
                "sl_no": "1",
                "error_description": "Fire aborted",
                "error_remarks": "Operator abort",
                "error_code": "0x0003",
                "page_number": 1,
            },
        ],
        "footnotes": ["Note 1: throw range is advisory only."],
        "source_hash": "abc123",
    }

    chunks = build_chunks(parse_result, "IRL Fault Codes.pdf")

    assert len(chunks) == 2  # 1 row + 1 footnote
    assert chunks[0]["error_code"] == "0x0003"
    assert chunks[0]["chunk_type"] == "table"
    assert "Fire aborted" in chunks[0]["chunk_text"]
    assert chunks[0]["document_name"] == "IRL Fault Codes.pdf"
    assert chunks[0]["source_hash"] == "abc123"
    assert chunks[1]["chunk_type"] == "prose"
    assert "Note 1" in chunks[1]["chunk_text"]


def test_save_and_load_chunks(tmp_path: Path) -> None:
    """save_chunks and load_chunks must round-trip correctly."""
    chunks = [
        {"chunk_id": "1", "chunk_text": "Test chunk"},
        {"chunk_id": "2", "chunk_text": "Another chunk"},
    ]
    path = tmp_path / "chunks.jsonl"

    save_chunks(chunks, path)
    loaded = load_chunks(path)

    assert len(loaded) == 2
    assert loaded[0]["chunk_id"] == "1"
    assert loaded[1]["chunk_text"] == "Another chunk"
