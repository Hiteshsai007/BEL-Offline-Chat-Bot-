"""
Tests for app/ingestion/parser.py — primary pdfplumber path and PyMuPDF fallback.

Mocks the heavy PDF dependencies via sys.modules so the suite runs without them.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.parser import parse_pdf


def _make_mock_pdfplumber(rows: list, footnotes: list) -> MagicMock:
    """Build a mock pdfplumber module that yields the given rows/footnotes."""
    mock_page = MagicMock()
    # Header + data rows
    mock_table = [["SL", "Error Description", "Remarks", "Error Code"]] + rows
    mock_page.extract_tables.return_value = [mock_table]
    mock_page.extract_text.return_value = "\n".join(footnotes)

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]

    mock_pdfplumber = MagicMock()
    mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)
    return mock_pdfplumber


def _make_mock_fitz(rows: list, footnotes: list) -> MagicMock:
    """Build a mock fitz (PyMuPDF) module that yields the given rows/footnotes."""
    text = "\n".join(
        [f"{r[0]} {r[1]} {r[3]}" for r in rows] + footnotes
    )
    mock_page = MagicMock()
    mock_page.get_text.return_value = text

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.page_count = 1

    mock_fitz = MagicMock()
    mock_fitz.open.return_value.__enter__ = MagicMock(return_value=mock_doc)
    mock_fitz.open.return_value.__exit__ = MagicMock(return_value=False)
    return mock_fitz


def test_parse_pdf_pdfplumber_success(tmp_path: Path) -> None:
    """Primary path: pdfplumber extracts rows successfully."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy", encoding="utf-8")

    mock_pdfplumber = _make_mock_pdfplumber(
        rows=[["1", "Fire aborted", "Operator abort", "0x0003"]],
        footnotes=[],
    )
    mock_fitz = _make_mock_fitz(rows=[], footnotes=[])

    with patch.dict(sys.modules, {"pdfplumber": mock_pdfplumber, "fitz": mock_fitz}):
        result = parse_pdf(pdf_path)

    assert len(result["rows"]) == 1
    assert result["rows"][0]["error_code"] == "0x0003"
    assert result["rows"][0]["error_description"] == "Fire aborted"
    assert result["source_hash"] is not None
    assert result["page_count"] == 1


def test_parse_pdf_pymupdf_fallback(tmp_path: Path) -> None:
    """Fallback path: pdfplumber fails, PyMuPDF succeeds."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy", encoding="utf-8")

    mock_pdfplumber = MagicMock()
    mock_pdfplumber.open.side_effect = Exception("pdfplumber failed")

    mock_fitz = _make_mock_fitz(
        rows=[["1", "Fire aborted", "", "0x0003"]],
        footnotes=[],
    )

    with patch.dict(sys.modules, {"pdfplumber": mock_pdfplumber, "fitz": mock_fitz}):
        result = parse_pdf(pdf_path)

    assert len(result["rows"]) >= 1
    assert any(r["error_code"] == "0x0003" for r in result["rows"])


def test_parse_pdf_both_fail(tmp_path: Path) -> None:
    """Both extraction methods fail -> RuntimeError."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy", encoding="utf-8")

    mock_pdfplumber = MagicMock()
    mock_pdfplumber.open.side_effect = Exception("fail")

    mock_fitz = MagicMock()
    mock_fitz.open.side_effect = Exception("fail")

    with patch.dict(sys.modules, {"pdfplumber": mock_pdfplumber, "fitz": mock_fitz}):
        with pytest.raises(RuntimeError, match="No rows extracted"):
            parse_pdf(pdf_path)


def test_parse_pdf_not_found() -> None:
    """Missing PDF -> FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_pdf(Path("/nonexistent/file.pdf"))
