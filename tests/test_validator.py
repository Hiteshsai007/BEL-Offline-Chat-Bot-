"""
Tests for app/ingestion/validator.py — fidelity validation logic.

Covers H-2 (warnings vs hard errors) and H-3 (duplicate error codes).
"""
from app.ingestion.validator import validate_chunks


def test_validate_chunks_pass() -> None:
    """All chunks match source rows -> no warnings, no errors."""
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

    warnings, errors = validate_chunks(chunks, parse_result)
    assert warnings == []
    assert errors == []


def test_validate_chunks_fabrication() -> None:
    """Chunk with a code not in the source -> error (not warning)."""
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

    warnings, errors = validate_chunks(chunks, parse_result)
    assert any("FABRICATION" in e for e in errors)


def test_validate_chunks_mismatch() -> None:
    """Chunk description differing from source -> error."""
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

    warnings, errors = validate_chunks(chunks, parse_result)
    assert any("MISMATCH" in e for e in errors)


def test_validate_chunks_row_count_warning_only() -> None:
    """
    H-2: A benign row-count difference must produce a warning, not an error,
    so ingestion is not aborted.
    """
    parse_result = {
        "rows": [
            {"error_code": "0x0003", "error_description": "Fire aborted", "error_remarks": ""},
            {"error_code": "0x0004", "error_description": "Misfire", "error_remarks": ""},
        ],
    }
    # Only one chunk produced — fewer than source rows
    chunks = [
        {
            "chunk_id": "c1",
            "chunk_type": "table",
            "error_code": "0x0003",
            "error_description": "Fire aborted",
            "error_remarks": "",
        },
    ]

    warnings, errors = validate_chunks(chunks, parse_result)
    assert len(warnings) == 1
    assert "Fewer chunks" in warnings[0]
    assert errors == []


def test_validate_chunks_duplicate_code_validates() -> None:
    """
    H-3: Duplicate error codes across two source rows must validate correctly
    against both occurrences instead of failing on the first.
    """
    parse_result = {
        "rows": [
            {
                "error_code": "0x0003",
                "error_description": "Fire aborted",
                "error_remarks": "Operator abort",
            },
            {
                "error_code": "0x0003",
                "error_description": "Fire aborted (duplicate)",
                "error_remarks": "Duplicate entry",
            },
        ],
    }
    # Two chunks matching the two different source rows with the same code
    chunks = [
        {
            "chunk_id": "c1",
            "chunk_type": "table",
            "error_code": "0x0003",
            "error_description": "Fire aborted",
            "error_remarks": "Operator abort",
        },
        {
            "chunk_id": "c2",
            "chunk_type": "table",
            "error_code": "0x0003",
            "error_description": "Fire aborted (duplicate)",
            "error_remarks": "Duplicate entry",
        },
    ]

    warnings, errors = validate_chunks(chunks, parse_result)
    assert warnings == []
    assert errors == []


def test_validate_chunks_critical_no_table_chunks() -> None:
    """No table chunks at all -> critical error."""
    parse_result = {
        "rows": [
            {"error_code": "0x0003", "error_description": "Fire aborted", "error_remarks": ""},
        ],
    }
    chunks = [
        {"chunk_id": "c1", "chunk_type": "prose", "chunk_text": "Note 1"},
    ]

    warnings, errors = validate_chunks(chunks, parse_result)
    assert any("CRITICAL" in e for e in errors)
