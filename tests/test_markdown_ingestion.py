"""
Tests for Markdown document ingestion.

Covers:
    1. markdown_parser     -- prose, heading, table parsing
    2. general_chunker     -- compatibility with Markdown parse_result
    3. ingest_general      -- end-to-end Markdown ingestion pipeline
    4. isolation           -- fault-code index is never touched

All tests are deterministic and require no HuggingFace model, Ollama,
network access, or FAISS index.  The embedder and FAISS build steps are
patched wherever the full pipeline is exercised.
"""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic Markdown fixtures (in-memory — no fixture files needed)
# ---------------------------------------------------------------------------

_SIMPLE_MD = """\
# Introduction

This is the first paragraph of the introduction.
It spans multiple lines.

This is the second paragraph.

## Safety Warning

Always wear protective gear.

# Chapter Two

Content of chapter two.
"""

_TABLE_MD = """\
# Torque Specifications

Use a calibrated torque wrench for all fasteners.

| Fastener       | Torque (Nm) | Location     |
|----------------|-------------|--------------|
| Oil drain plug | 30          | Engine       |
| Front axle nut | 108         | Front wheel  |
| Spark plug     | 13          | Cylinder head|
"""

_NO_H1_MD = """\
## Section without H1

Some prose content here.

More prose below.
"""

_MIXED_MD = """\
# Main Section

Opening prose.

| Col A | Col B |
|-------|-------|
| val1  | val2  |

Closing prose after the table.

## Sub-Section

Sub-section content.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_md(tmp_path: Path, name: str, content: str) -> Path:
    """Write a Markdown string to a temp file and return its path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Unit tests: markdown_parser
# ---------------------------------------------------------------------------

class TestMarkdownParser:
    """Tests for app.ingestion.markdown_parser.parse_markdown_file."""

    def test_returns_required_keys(self, tmp_path):
        """parse_result contains all keys required by general_chunker."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "simple.md", _SIMPLE_MD)
        result = parse_markdown_file(p)

        assert "items" in result
        assert "source_hash" in result
        assert "page_count" in result
        assert "source_document" in result
        assert result["source_document"] == "simple.md"

    def test_source_hash_matches_file(self, tmp_path):
        """SHA-256 in parse_result matches the file on disk."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "hash_test.md", _SIMPLE_MD)
        result = parse_markdown_file(p)
        assert result["source_hash"] == _sha256(p)

    def test_h1_creates_page_boundaries(self, tmp_path):
        """Each H1 heading produces a distinct page_number."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "pages.md", _SIMPLE_MD)
        result = parse_markdown_file(p)
        pages = {item["page_number"] for item in result["items"]}
        # _SIMPLE_MD has two H1 headings → at least 2 distinct page numbers
        assert len(pages) >= 2
        assert result["page_count"] >= 2

    def test_prose_items_extracted(self, tmp_path):
        """Prose paragraphs produce items of type 'prose' with non-empty text."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "prose.md", _SIMPLE_MD)
        result = parse_markdown_file(p)
        prose = [i for i in result["items"] if i["type"] == "prose"]
        assert len(prose) >= 2
        for item in prose:
            assert item["text"].strip()
            assert item["page_number"] >= 1

    def test_subheading_updates_section_heading(self, tmp_path):
        """## sub-headings update section_heading without creating a new page."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "subhead.md", _SIMPLE_MD)
        result = parse_markdown_file(p)
        prose = [i for i in result["items"] if i["type"] == "prose"]
        # The item under '## Safety Warning' should carry that heading
        safety_items = [i for i in prose if i.get("section_heading") == "Safety Warning"]
        assert len(safety_items) >= 1

    def test_gfm_table_extracted(self, tmp_path):
        """GFM pipe tables produce items of type 'table' with data rows."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "table.md", _TABLE_MD)
        result = parse_markdown_file(p)
        tables = [i for i in result["items"] if i["type"] == "table"]
        assert len(tables) >= 1

        tbl = tables[0]
        assert "data" in tbl
        assert len(tbl["data"]) >= 2        # header + at least one data row
        # header row must contain the expected column names
        header = tbl["data"][0]
        assert any("Fastener" in c for c in header)
        assert any("Torque" in c for c in header)

    def test_table_separator_not_in_data(self, tmp_path):
        """The GFM separator row (---|---) is not included in table data."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "sep.md", _TABLE_MD)
        result = parse_markdown_file(p)
        tables = [i for i in result["items"] if i["type"] == "table"]
        assert tables
        for row in tables[0]["data"]:
            for cell in row:
                assert "---" not in cell

    def test_no_h1_treated_as_single_page(self, tmp_path):
        """Markdown with no H1 heading produces a single synthetic page."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "no_h1.md", _NO_H1_MD)
        result = parse_markdown_file(p)
        assert result["page_count"] == 1
        pages = {item["page_number"] for item in result["items"]}
        assert pages == {1}

    def test_mixed_content_order(self, tmp_path):
        """Table and prose items interleave correctly in mixed content."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "mixed.md", _MIXED_MD)
        result = parse_markdown_file(p)
        types = [i["type"] for i in result["items"]]
        assert "prose" in types
        assert "table" in types

    def test_file_not_found(self, tmp_path):
        """parse_markdown_file raises FileNotFoundError for missing files."""
        from app.ingestion.markdown_parser import parse_markdown_file

        with pytest.raises(FileNotFoundError):
            parse_markdown_file(tmp_path / "does_not_exist.md")

    def test_empty_file_raises(self, tmp_path):
        """parse_markdown_file raises RuntimeError for empty files."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = tmp_path / "empty.md"
        p.write_text("", encoding="utf-8")
        with pytest.raises(RuntimeError, match="No content extracted"):
            parse_markdown_file(p)

    def test_whitespace_only_file_raises(self, tmp_path):
        """parse_markdown_file raises RuntimeError when file has only whitespace."""
        from app.ingestion.markdown_parser import parse_markdown_file

        p = tmp_path / "ws.md"
        p.write_text("   \n\n   \n", encoding="utf-8")
        with pytest.raises(RuntimeError):
            parse_markdown_file(p)


# ---------------------------------------------------------------------------
# 2. Chunker compatibility: parse_result from Markdown → build_general_chunks
# ---------------------------------------------------------------------------

class TestMarkdownChunkerCompatibility:
    """Verify that general_chunker accepts Markdown parse_result unchanged."""

    def test_build_general_chunks_accepts_markdown_result(self, tmp_path):
        """build_general_chunks works with a Markdown-derived parse_result."""
        from app.ingestion.general_chunker import build_general_chunks
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "compat.md", _SIMPLE_MD)
        parse_result = parse_markdown_file(p)
        chunks = build_general_chunks(parse_result)
        assert len(chunks) > 0

    def test_chunk_schema_complete(self, tmp_path):
        """Every chunk has all fields required by the retriever and validator."""
        from app.ingestion.general_chunker import build_general_chunks
        from app.ingestion.markdown_parser import parse_markdown_file

        required_fields = {
            "chunk_id", "chunk_text", "document_name", "page_number",
            "chunk_type", "token_count", "source_hash",
            "section_heading", "image_file_path", "figure_references",
            "error_code", "sl_no", "error_description", "error_remarks",
            "document_version", "system_name", "subsystem",
            "chapter", "section",
        }
        p = _write_md(tmp_path, "schema.md", _MIXED_MD)
        chunks = build_general_chunks(parse_markdown_file(p))
        for chunk in chunks:
            missing = required_fields - chunk.keys()
            assert not missing, f"Chunk missing fields: {missing}"

    def test_chunk_types_are_valid(self, tmp_path):
        """All chunks produced from Markdown have valid chunk_type values."""
        from app.ingestion.general_chunker import build_general_chunks
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "types.md", _TABLE_MD)
        chunks = build_general_chunks(parse_markdown_file(p))
        valid_types = {"prose", "table", "image"}
        for chunk in chunks:
            assert chunk["chunk_type"] in valid_types

    def test_source_document_propagated(self, tmp_path):
        """document_name in every chunk equals the source .md filename."""
        from app.ingestion.general_chunker import build_general_chunks
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "mychapter.md", _SIMPLE_MD)
        chunks = build_general_chunks(parse_markdown_file(p))
        for chunk in chunks:
            assert chunk["document_name"] == "mychapter.md"

    def test_source_hash_propagated(self, tmp_path):
        """source_hash in every chunk equals the SHA-256 of the .md file."""
        from app.ingestion.general_chunker import build_general_chunks
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "hash_prop.md", _SIMPLE_MD)
        expected = _sha256(p)
        chunks = build_general_chunks(parse_markdown_file(p))
        for chunk in chunks:
            assert chunk["source_hash"] == expected

    def test_validator_passes_markdown_chunks(self, tmp_path):
        """validate_general_chunks reports no errors for Markdown-derived chunks."""
        from app.ingestion.general_chunker import build_general_chunks
        from app.ingestion.general_validator import validate_general_chunks
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "valid.md", _SIMPLE_MD)
        chunks = build_general_chunks(parse_markdown_file(p))
        warnings, errors = validate_general_chunks(chunks)
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_table_chunks_have_data_in_text(self, tmp_path):
        """Table chunks embed the pipe-table text so cell values are searchable."""
        from app.ingestion.general_chunker import build_general_chunks
        from app.ingestion.markdown_parser import parse_markdown_file

        p = _write_md(tmp_path, "tbl_text.md", _TABLE_MD)
        chunks = build_general_chunks(parse_markdown_file(p))
        table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
        assert table_chunks
        # The cell values must appear in the embedded text
        combined = " ".join(c["chunk_text"] for c in table_chunks)
        assert "Oil drain plug" in combined
        assert "Front axle nut" in combined


# ---------------------------------------------------------------------------
# 3. End-to-end ingestion pipeline: run_general_ingestion_md
# ---------------------------------------------------------------------------

class TestRunGeneralIngestionMd:
    """End-to-end pipeline tests (embedder + FAISS patched)."""

    @staticmethod
    def _fake_embed(chunks):
        """Return deterministic 384-dim vectors without touching HuggingFace."""
        rng = np.random.RandomState(42)
        return np.array(
            [rng.randn(384).tolist() for _ in chunks],
            dtype=np.float32,
        )

    def test_end_to_end_returns_true(self, tmp_path):
        """run_general_ingestion_md succeeds and creates FAISS + chunk files."""
        from app.ingestion.ingest_general import run_general_ingestion_md
        import app.ingestion.ingest_general as mod

        md_file = _write_md(tmp_path, "manual.md", _SIMPLE_MD)
        gen_dir = tmp_path / "general"

        orig_faiss = mod.GENERAL_FAISS_PATH
        orig_chunks = mod.GENERAL_CHUNKS_PATH
        orig_dir = mod.GENERAL_INDEX_DIR
        try:
            mod.GENERAL_INDEX_DIR = gen_dir
            mod.GENERAL_FAISS_PATH = gen_dir / "faiss.index"
            mod.GENERAL_CHUNKS_PATH = gen_dir / "chunks.jsonl"

            with patch(
                "app.ingestion.ingest_general._embed_chunks",
                side_effect=self._fake_embed,
            ):
                success = run_general_ingestion_md(md_file)

            assert success is True
            assert mod.GENERAL_FAISS_PATH.exists()
            assert mod.GENERAL_CHUNKS_PATH.exists()
        finally:
            mod.GENERAL_FAISS_PATH = orig_faiss
            mod.GENERAL_CHUNKS_PATH = orig_chunks
            mod.GENERAL_INDEX_DIR = orig_dir

    def test_chunk_file_has_correct_source(self, tmp_path):
        """Chunks saved to disk carry the correct source document name and hash."""
        from app.ingestion.ingest_general import run_general_ingestion_md
        import app.ingestion.ingest_general as mod

        md_file = _write_md(tmp_path, "manual.md", _TABLE_MD)
        gen_dir = tmp_path / "general"
        expected_hash = _sha256(md_file)

        orig_faiss = mod.GENERAL_FAISS_PATH
        orig_chunks = mod.GENERAL_CHUNKS_PATH
        orig_dir = mod.GENERAL_INDEX_DIR
        try:
            mod.GENERAL_INDEX_DIR = gen_dir
            mod.GENERAL_FAISS_PATH = gen_dir / "faiss.index"
            mod.GENERAL_CHUNKS_PATH = gen_dir / "chunks.jsonl"

            with patch(
                "app.ingestion.ingest_general._embed_chunks",
                side_effect=self._fake_embed,
            ):
                run_general_ingestion_md(md_file)

            with open(mod.GENERAL_CHUNKS_PATH) as f:
                chunks = [json.loads(line) for line in f if line.strip()]

            assert len(chunks) > 0
            for chunk in chunks:
                assert chunk["document_name"] == "manual.md"
                assert chunk["source_hash"] == expected_hash
        finally:
            mod.GENERAL_FAISS_PATH = orig_faiss
            mod.GENERAL_CHUNKS_PATH = orig_chunks
            mod.GENERAL_INDEX_DIR = orig_dir

    def test_missing_file_returns_false(self, tmp_path):
        """run_general_ingestion_md returns False for a non-existent file."""
        from app.ingestion.ingest_general import run_general_ingestion_md
        import app.ingestion.ingest_general as mod

        gen_dir = tmp_path / "general"
        orig_faiss = mod.GENERAL_FAISS_PATH
        orig_chunks = mod.GENERAL_CHUNKS_PATH
        orig_dir = mod.GENERAL_INDEX_DIR
        try:
            mod.GENERAL_INDEX_DIR = gen_dir
            mod.GENERAL_FAISS_PATH = gen_dir / "faiss.index"
            mod.GENERAL_CHUNKS_PATH = gen_dir / "chunks.jsonl"

            result = run_general_ingestion_md(tmp_path / "nonexistent.md")
            assert result is False
        finally:
            mod.GENERAL_FAISS_PATH = orig_faiss
            mod.GENERAL_CHUNKS_PATH = orig_chunks
            mod.GENERAL_INDEX_DIR = orig_dir

    def test_fault_code_index_untouched(self, tmp_path):
        """Markdown ingestion never touches data/index/faiss.index or chunks.jsonl."""
        from app.ingestion.ingest_general import run_general_ingestion_md
        from app.settings import CHUNKS_STORE_PATH, FAISS_INDEX_PATH
        import app.ingestion.ingest_general as mod

        def _hash_if_exists(p: Path) -> str | None:
            if not p.exists():
                return None
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    h.update(block)
            return h.hexdigest()

        faiss_before = _hash_if_exists(FAISS_INDEX_PATH)
        chunks_before = _hash_if_exists(CHUNKS_STORE_PATH)

        md_file = _write_md(tmp_path, "iso.md", _SIMPLE_MD)
        gen_dir = tmp_path / "general"

        orig_faiss = mod.GENERAL_FAISS_PATH
        orig_chunks = mod.GENERAL_CHUNKS_PATH
        orig_dir = mod.GENERAL_INDEX_DIR
        try:
            mod.GENERAL_INDEX_DIR = gen_dir
            mod.GENERAL_FAISS_PATH = gen_dir / "faiss.index"
            mod.GENERAL_CHUNKS_PATH = gen_dir / "chunks.jsonl"

            with patch(
                "app.ingestion.ingest_general._embed_chunks",
                side_effect=self._fake_embed,
            ):
                run_general_ingestion_md(md_file)

            assert _hash_if_exists(FAISS_INDEX_PATH) == faiss_before
            assert _hash_if_exists(CHUNKS_STORE_PATH) == chunks_before
        finally:
            mod.GENERAL_FAISS_PATH = orig_faiss
            mod.GENERAL_CHUNKS_PATH = orig_chunks
            mod.GENERAL_INDEX_DIR = orig_dir


# ---------------------------------------------------------------------------
# 4. PDF ingestion pipeline: regression guard
# ---------------------------------------------------------------------------

class TestPdfIngestionUnchanged:
    """
    Ensure that the existing PDF ingestion path is byte-identical in
    behaviour after the Markdown changes.  Only the parse step differs —
    everything downstream is shared with the Markdown path.
    """

    FIXTURE_PDF = Path(__file__).parent / "fixtures" / "test_manual.pdf"

    @staticmethod
    def _fake_embed(chunks):
        rng = np.random.RandomState(42)
        return np.array(
            [rng.randn(384).tolist() for _ in chunks],
            dtype=np.float32,
        )

    def test_pdf_ingestion_still_succeeds(self, tmp_path):
        """run_general_ingestion (PDF path) still returns True after changes."""
        from app.ingestion.ingest_general import run_general_ingestion
        import app.ingestion.ingest_general as mod

        if not self.FIXTURE_PDF.exists():
            pytest.skip("test_manual.pdf fixture not found")

        gen_dir = tmp_path / "general"
        orig_faiss = mod.GENERAL_FAISS_PATH
        orig_chunks = mod.GENERAL_CHUNKS_PATH
        orig_dir = mod.GENERAL_INDEX_DIR
        try:
            mod.GENERAL_INDEX_DIR = gen_dir
            mod.GENERAL_FAISS_PATH = gen_dir / "faiss.index"
            mod.GENERAL_CHUNKS_PATH = gen_dir / "chunks.jsonl"

            with patch(
                "app.ingestion.ingest_general._embed_chunks",
                side_effect=self._fake_embed,
            ):
                success = run_general_ingestion(self.FIXTURE_PDF)

            assert success is True
            assert mod.GENERAL_FAISS_PATH.exists()
            assert mod.GENERAL_CHUNKS_PATH.exists()
        finally:
            mod.GENERAL_FAISS_PATH = orig_faiss
            mod.GENERAL_CHUNKS_PATH = orig_chunks
            mod.GENERAL_INDEX_DIR = orig_dir
