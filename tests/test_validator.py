"""
Tests for app/ingestion/validator.py — fidelity validation logic.

These cover the basic validation paths.  H-2+H-3 will add tests for the
warning-vs-error separation and duplicate-code handling.
"""
from app.ingestion.validator import validate_chunks


def test_validate_chunks_pass() -> None:
    """All chunks match source rows -> no errors."""
    parse_result = {
        "rows": [
            {
                "error_code": "0x0003",
                "error_description": "Fire aborted",
                "error_remarks": "Operator abort",
            },
        ],
    }
    chunks = [
        {
            "chunk_id": "c1",
            "chunk_type": "table",
            "error_code": "0x0003",
            "error_description": "Fire aborted",
            "error_remarks": "Operator abort",
        },
    ]

    errors = validate_chunks(chunks, parse_result)
    assert errors == []


def test_validate_chunks_fabrication() -> None:
    """Chunk with a code not in the source -> FABRICATION error."""
    parse_result = {
        "rows": [
            {
                "error_code": "0x0003",
                "error_description": "Fire aborted",
                "error_remarks": "",
            },
        ],
    }
    chunks = [
        {
            "chunk_id": "c1",
            "chunk_type": "table",
            "error_code": "0x9999",
            "error_description": "Unknown",
            "error_remarks": "",
        },
    ]

    errors = validate_chunks(chunks, parse_result)
    assert any("FABRICATION" in e for e in errors)


def test_validate_chunks_mismatch() -> None:
    """Chunk description differing from source -> MISMATCH error."""
    parse_result = {
        "rows": [
            {
                "error_code": "0x0003",
                "error_description": "Fire aborted",
                "error_remarks": "",
            },
        ],
    }
    chunks = [
        {
            "chunk_id": "c1",
            "chunk_type": "table",
            "error_code": "0x0003",
            "error_description": "Wrong description",
            "error_remarks": "",
        },
    ]

    errors = validate_chunks(chunks, parse_result)
    assert any("MISMATCH" in e for e in errors)
