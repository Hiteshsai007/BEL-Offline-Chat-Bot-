"""
Tests for source isolation, retrieval routing, and strict citation grounding.

Validates that:
  1. Fault-code queries search ONLY the fault-code index and never return general-document chunks.
  2. General-document queries search ONLY the general-document index and never return fault-code chunks.
  3. Queries like 'What is a misfire error?' return only documented facts without hallucinating engine concepts.
  4. Generated answers contain no information or terms absent from retrieved chunks.
"""
import re
import threading
from unittest.mock import patch

import numpy as np

from app.rag.generator import _extract_ungrounded_terms, generate
from app.rag.retriever import RetrievedChunk, Retriever
from app.settings import ERROR_CODE_PATTERN


def _make_fault_chunk(code: str, desc: str, remarks: str = "") -> dict:
    return {
        "chunk_id": f"fc-{code}",
        "chunk_text": f"Error Code: {code} | Description: {desc} | Remarks: {remarks}",
        "document_name": "IRL Fault Codes.pdf",
        "error_code": code,
        "error_description": desc,
        "error_remarks": remarks,
        "page_number": 1,
        "chunk_type": "table",
    }


def _make_general_chunk(doc: str, text: str, page: int = 1) -> dict:
    return {
        "chunk_id": f"gen-{hash(text)}",
        "chunk_text": text,
        "document_name": doc,
        "error_code": None,
        "page_number": page,
        "chunk_type": "prose",
    }


class TestSourceIsolationAndGrounding:

    def test_fault_code_query_never_retrieves_general_chunks(self, tmp_path):
        """Fault-code query routes strictly to fault-code index and returns zero general chunks."""
        import faiss

        fc_chunks = [_make_fault_chunk("0x0003", "Depth setting failed")]
        gen_chunks = [_make_general_chunk("Ninja ZX-10R Manual.pdf", "Instrument cluster layout")]

        # Build synthetic indexes with normalized vector (similarity = 1.0)
        dim = 384
        vec = np.zeros((1, dim), dtype=np.float32)
        vec[0, 0] = 1.0

        fc_idx = faiss.IndexFlatIP(dim)
        fc_idx.add(vec)

        gen_idx = faiss.IndexFlatIP(dim)
        gen_idx.add(vec)

        retriever = Retriever.__new__(Retriever)
        retriever._lock = threading.Lock()
        retriever._index = fc_idx
        retriever._chunks = fc_chunks
        retriever._bm25 = None
        retriever._general_index = gen_idx
        retriever._general_chunks = gen_chunks
        retriever._general_bm25 = None
        retriever._code_pattern = re.compile(ERROR_CODE_PATTERN, re.IGNORECASE)
        retriever._code_index = {"0x0003": [fc_chunks[0]]}

        with patch("app.rag.embedder.get_embedder") as mock_emb:
            mock_emb.return_value.embed_query.return_value = vec[0].tolist()
            results = retriever._hybrid_search("What is error code 0x0003?")

        assert retriever.last_selected_index == "fault_code"
        assert len(results) > 0
        for rc in results:
            assert rc.chunk["document_name"] == "IRL Fault Codes.pdf"
            assert rc.chunk["error_code"] == "0x0003"

    def test_general_document_query_never_retrieves_fault_code_chunks(self, tmp_path):
        """General document query routes strictly to general index and returns zero fault-code chunks."""
        import faiss

        fc_chunks = [_make_fault_chunk("0x0003", "Depth setting failed")]
        gen_chunks = [_make_general_chunk("Ninja ZX-10R Manual.pdf", "To start the bike, turn key ON")]

        dim = 384
        vec = np.zeros((1, dim), dtype=np.float32)
        vec[0, 0] = 1.0

        fc_idx = faiss.IndexFlatIP(dim)
        fc_idx.add(vec)

        gen_idx = faiss.IndexFlatIP(dim)
        gen_idx.add(vec)

        retriever = Retriever.__new__(Retriever)
        retriever._lock = threading.Lock()
        retriever._index = fc_idx
        retriever._chunks = fc_chunks
        retriever._bm25 = None
        retriever._general_index = gen_idx
        retriever._general_chunks = gen_chunks
        retriever._general_bm25 = None

        with patch("app.rag.embedder.get_embedder") as mock_emb:
            mock_emb.return_value.embed_query.return_value = vec[0].tolist()
            results = retriever._hybrid_search("How to start a bike?")

        assert retriever.last_selected_index == "general_document"
        assert len(results) > 0
        for rc in results:
            assert rc.chunk["document_name"] == "Ninja ZX-10R Manual.pdf"
            assert rc.chunk.get("error_code") is None

    def test_misfire_query_returns_only_documented_facts(self):
        """'What is a misfire error?' produces an answer containing only documented facts."""
        misfire_chunks = [
            RetrievedChunk(
                chunk=_make_fault_chunk("0x0013", "R9 Misfired", "Fired but still rocket present"),
                score=0.90,
            )
        ]

        # Mock Ollama returning a hallucinated explanation
        hallucinated_llm_response = (
            "A misfire error occurs in internal combustion engines when the ignition "
            "system fails to ignite fuel in a cylinder [IRL Fault Codes.pdf, 0x0013]."
        )

        with patch("app.rag.generator._call_ollama", return_value=(hallucinated_llm_response, 800.0)):
            res = generate("What is a misfire error?", misfire_chunks)

        # Grounding check must fail on hallucinated terms and trigger fallback
        answer = res["answer"]
        assert "combustion" not in answer
        assert "engine" not in answer
        assert "ignition" not in answer
        assert "cylinder" not in answer
        assert "this condition indicates" in answer or "Fired but still rocket present" in answer

    def test_no_answer_may_contain_unsupported_terms(self):
        """Ungrounded terms extraction catches non-contextual vocabulary."""
        retrieved = [
            RetrievedChunk(
                chunk=_make_fault_chunk("0x0010", "Rocket 10 Invalid", "Rocket 10 not ready"),
                score=0.95,
            )
        ]

        bad_answer = "This error indicates that electronic parameter threshold limits were exceeded."
        ungrounded = _extract_ungrounded_terms(bad_answer, retrieved)
        assert "electronic" in ungrounded
        assert "parameter" in ungrounded
        assert "threshold" in ungrounded
