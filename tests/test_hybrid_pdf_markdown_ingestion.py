"""
Unit tests for Hybrid PDF + Markdown Ingestion Pipeline.

Verifies:
  - PDF -> Markdown conversion with <!-- PAGE:N --> markers
  - Page marker preservation through chunking
  - Image extraction and metadata mapping (page_54_img_0)
  - Table conversion to Markdown pipe format and fallback logic
  - Citation page mapping ([Document Name, page N])
  - Rollback/failsafe behavior (PDF fallback on Markdown error)
  - Document isolation and fault-code compatibility
"""
from pathlib import Path
from unittest.mock import patch

from app.ingestion.general_chunker import build_general_chunks
from app.ingestion.ingest_general import run_general_ingestion
from app.ingestion.markdown_parser import parse_markdown_file
from app.ingestion.pdf_to_markdown import (
    _format_table_as_markdown,
    generate_markdown_from_pdf_parse,
)

SAMPLE_PDF = Path(
    "data/manuals/969205485-Ninja-Zx-10r-Se-2018-Owners-Manual.pdf"
)


def test_pdf_to_markdown_conversion_with_page_markers() -> None:
    """PDF parse result -> Markdown containing <!-- PAGE:N --> markers."""
    parse_result = {
        "source_document": "Test-Manual.pdf",
        "source_hash": "dummyhash123",
        "page_count": 2,
        "items": [
            {
                "type": "prose",
                "page_number": 1,
                "text": "Coolant Temperature Warning",
                "section_heading": "Coolant Temperature Warning",
            },
            {
                "type": "table",
                "page_number": 1,
                "data": [["Code", "Description"], ["0x0003", "Fire Aborted"]],
            },
            {
                "type": "image",
                "page_number": 2,
                "image_id": "page_2_img_0",
                "caption": "Warning Indicator Light",
                "image_file_path": "data/images/page_2_0.png",
            },
        ],
    }

    md_output = generate_markdown_from_pdf_parse(parse_result)
    assert "# Test-Manual.pdf" in md_output
    assert "<!-- PAGE:1 -->" in md_output
    assert "<!-- PAGE:2 -->" in md_output
    assert "| Code | Description |" in md_output
    assert "| 0x0003 | Fire Aborted |" in md_output
    assert "[Image: page_2_img_0]" in md_output
    assert "Warning Indicator Light" in md_output


def test_page_marker_preservation_in_markdown_parser(tmp_path: Path) -> None:
    """Markdown parser extracts PDF page numbers from <!-- PAGE:N -->."""
    md_file = tmp_path / "test.md"
    md_content = """# Test Document

<!-- PAGE:54 -->

## Coolant Temperature Meter

If the coolant temperature rises, the numerical value starts blinking.

<!-- PAGE:171 -->

## Tire Pressure

Check tire pressure when cold.
"""
    md_file.write_text(md_content, encoding="utf-8")

    parsed = parse_markdown_file(md_file)
    items = parsed["items"]
    assert len(items) >= 2

    # Check page numbers
    page_numbers = {item["page_number"] for item in items}
    assert 54 in page_numbers
    assert 171 in page_numbers


def test_table_conversion_to_markdown_pipe_format_and_fallback() -> None:
    """Table data converts to GFM pipe table format; handles empty inputs."""
    table_data = [
        ["Header 1", "Header 2"],
        ["Val 1", "Val 2"],
    ]
    md_table = _format_table_as_markdown(table_data)
    assert "| Header 1 | Header 2 |" in md_table
    assert "| --- | --- |" in md_table
    assert "| Val 1 | Val 2 |" in md_table

    # Fallback / Empty
    assert _format_table_as_markdown([]) == ""


def test_citation_page_mapping_preservation(tmp_path: Path) -> None:
    """Chunks from Markdown retain original PDF page numbers & doc names."""
    md_file = tmp_path / "test_doc.md"
    md_file.write_text(
        "<!-- PAGE:54 -->\n\nBlinking pattern details.", encoding="utf-8"
    )

    md_parsed = parse_markdown_file(md_file)
    md_parsed["source_document"] = "Ninja-Zx-10r.pdf"
    md_parsed["source_hash"] = "abc123hash"

    chunks = build_general_chunks(md_parsed)
    assert len(chunks) > 0
    c = chunks[0]
    assert c["document_name"] == "Ninja-Zx-10r.pdf"
    assert c["page_number"] == 54


def test_failsafe_rollback_on_markdown_failure(tmp_path: Path) -> None:
    """If Markdown generation fails, falls back to direct PDF chunking."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("fake pdf content", encoding="utf-8")

    mock_parse = {
        "source_document": "fake.pdf",
        "source_hash": "fakehash",
        "page_count": 1,
        "items": [
            {"type": "prose", "page_number": 1, "text": "Prose content"}
        ],
    }

    with patch(
        "app.ingestion.ingest_general.parse_general_pdf",
        return_value=mock_parse,
    ):
        with patch(
            "app.ingestion.pdf_to_markdown.generate_markdown_from_pdf_parse",
            side_effect=RuntimeError("MD Error"),
        ):
            with patch(
                "app.ingestion.ingest_general._embed_chunks",
                return_value=[[0.1] * 384],
            ):
                with patch("app.ingestion.ingest_general._build_faiss_index"):
                    with patch(
                        "app.ingestion.ingest_general.save_general_chunks"
                    ):
                        with patch(
                            "app.ingestion.ingest_general._save_faiss_index"
                        ):
                            with patch("shutil.move"):
                                success = run_general_ingestion(fake_pdf)
                                assert success is True


def test_multi_document_isolation_and_fault_code_compatibility() -> None:
    """General PDF ingestion uses data/index/general and isolates indices."""
    from app.ingestion.ingest_general import GENERAL_INDEX_DIR
    from app.settings import FAISS_INDEX_PATH

    assert "general" in str(GENERAL_INDEX_DIR)
    assert FAISS_INDEX_PATH != GENERAL_INDEX_DIR / "faiss.index"
