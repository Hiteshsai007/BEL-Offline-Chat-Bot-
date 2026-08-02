"""
Tests for document-aware retrieval filtering in app.rag.retriever.

Validates that:
  1. Comparison queries (containing keywords like 'compare', 'difference', 'versus')
     retain multi-document results.
  2. Non-comparison queries containing fault codes or fault-code chunks filter out
     all general-document chunks and retain only fault-code chunks.
  3. Non-comparison general queries group results by document_name and retain only
     chunks from the single highest-scoring document.
"""
from app.rag.retriever import (
    RetrievedChunk,
    _apply_document_filtering,
    _is_comparison_query,
)


def _make_chunk(doc_name: str, text: str, error_code: str | None = None) -> RetrievedChunk:
    """Helper to create a RetrievedChunk with specified metadata and dummy score."""
    return RetrievedChunk(
        chunk={
            "chunk_id": f"id-{hash(text)}",
            "chunk_text": text,
            "document_name": doc_name,
            "error_code": error_code,
            "page_number": 1,
        },
        score=1.0,
    )


class TestDocumentAwareRetrievalFiltering:
    """Unit tests for document filtering logic in retriever.py."""

    def test_comparison_query_detection(self):
        """Verify comparison query detection works for common patterns."""
        assert _is_comparison_query("Compare oil capacity in Ninja and Radar manuals") is True
        assert _is_comparison_query("What is the difference between fault code 0x0001 and 0x0002?") is True
        assert _is_comparison_query("Ninja manual versus Radar manual") is True
        assert _is_comparison_query("Show information across all manuals") is True
        assert _is_comparison_query("List specs for both documents") is True

        # Non-comparison queries
        assert _is_comparison_query("What is depth setting failed?") is False
        assert _is_comparison_query("How do I start the motorcycle?") is False
        assert _is_comparison_query("Explain error 0x0003") is False

    def test_fault_code_match_excludes_general_documents(self):
        """
        When a fault code match exists in candidates, general document chunks are excluded.
        """
        results = [
            RetrievedChunk(
                chunk={
                    "chunk_id": "fc-1",
                    "chunk_text": "Depth setting failed error description",
                    "document_name": "IRL Fault Codes.pdf",
                    "error_code": "0x0003",
                },
                score=0.90,
            ),
            RetrievedChunk(
                chunk={
                    "chunk_id": "gen-1",
                    "chunk_text": "Ninja manual page 169 depth info",
                    "document_name": "Ninja ZX-10R Manual.pdf",
                    "error_code": None,
                },
                score=0.75,
            ),
            RetrievedChunk(
                chunk={
                    "chunk_id": "gen-2",
                    "chunk_text": "Ninja manual page 171 depth info",
                    "document_name": "Ninja ZX-10R Manual.pdf",
                    "error_code": None,
                },
                score=0.70,
            ),
        ]

        filtered = _apply_document_filtering("What is depth setting failed?", results)
        assert len(filtered) == 1
        assert filtered[0].chunk["error_code"] == "0x0003"
        assert filtered[0].chunk["document_name"] == "IRL Fault Codes.pdf"

    def test_general_document_grouping_selects_highest_scoring_document(self):
        """
        For general queries, candidates from lower-scoring documents are filtered out,
        keeping only chunks from the single highest-scoring document.
        """
        results = [
            RetrievedChunk(
                chunk={
                    "chunk_id": "ninja-1",
                    "chunk_text": "Ninja oil capacity is 3.7L",
                    "document_name": "Ninja ZX-10R Manual.pdf",
                    "error_code": None,
                },
                score=0.88,
            ),
            RetrievedChunk(
                chunk={
                    "chunk_id": "radar-1",
                    "chunk_text": "Radar oil spec",
                    "document_name": "BEL Radar Manual.md",
                    "error_code": None,
                },
                score=0.82,
            ),
            RetrievedChunk(
                chunk={
                    "chunk_id": "ninja-2",
                    "chunk_text": "Ninja engine oil drain plug torque",
                    "document_name": "Ninja ZX-10R Manual.pdf",
                    "error_code": None,
                },
                score=0.78,
            ),
        ]

        filtered = _apply_document_filtering("What is the engine oil capacity?", results)
        assert len(filtered) == 2
        for rc in filtered:
            assert rc.chunk["document_name"] == "Ninja ZX-10R Manual.pdf"

    def test_comparison_query_bypasses_document_filtering(self):
        """
        Comparison queries retain chunks from multiple documents.
        """
        results = [
            RetrievedChunk(
                chunk={
                    "chunk_id": "ninja-1",
                    "chunk_text": "Ninja oil capacity is 3.7L",
                    "document_name": "Ninja ZX-10R Manual.pdf",
                    "error_code": None,
                },
                score=0.88,
            ),
            RetrievedChunk(
                chunk={
                    "chunk_id": "radar-1",
                    "chunk_text": "Radar oil spec is 2.0L",
                    "document_name": "BEL Radar Manual.md",
                    "error_code": None,
                },
                score=0.82,
            ),
        ]

        query = "What is the difference between oil capacity in Ninja manual and Radar manual?"
        filtered = _apply_document_filtering(query, results)
        assert len(filtered) == 2
        docs = {rc.chunk["document_name"] for rc in filtered}
        assert docs == {"Ninja ZX-10R Manual.pdf", "BEL Radar Manual.md"}
