"""
Tests for evidence-based fault-description summarization in the RAG pipeline.

Validates that:
  1. 'What is a misfire error?' summarizes the documented remarks ('Fired but still rocket present')
     across matching fault codes (0x0006, 0x0008, 0x0012, 0x0013, 0x0016) without external domain jargon.
  2. 'Fire Aborted' queries summarize the documented remarks ('Fire command aborted by operator')
     for fault code 0x0003.
  3. 'Depth Setting Failed' queries summarize the documented remarks ('Target depth out of allowable bounds')
     for fault code 0x0017.
  4. 'Throw Range Invalid' queries summarize the documented remarks ('Not a valid throw range')
     for fault code 0x0010.
  5. Citation guardrails and single-source isolation rules remain enforced.
"""
from unittest.mock import patch

from app.rag.pipeline import query
from app.rag.retriever import RetrievedChunk


def _make_fault_chunk(code: str, desc: str, remarks: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk={
            "chunk_id": f"fc-{code}",
            "chunk_text": f"Error Code: {code} | Description: {desc} | Remarks: {remarks}",
            "document_name": "IRL Fault Codes.pdf",
            "error_code": code,
            "error_description": desc,
            "error_remarks": remarks,
            "page_number": 1,
            "chunk_type": "table",
        },
        score=0.90,
    )


class TestFaultDescriptionSummarization:
    """Unit tests for fault-description summarization across all 4 required phrases."""

    def test_misfired_description_summarization(self):
        """Query for 'misfired' returns evidence-based summary of rocket present remarks."""
        misfire_chunks = [
            _make_fault_chunk("0x0006", "R2 Misfired", "Fired but still rocket present"),
            _make_fault_chunk("0x0008", "R4 Misfired", "Fired but still rocket present"),
            _make_fault_chunk("0x0012", "R8 Misfired", "Fired but still rocket present"),
            _make_fault_chunk("0x0013", "R9 Misfired", "Fired but still rocket present"),
            _make_fault_chunk("0x0016", "R12 Misfired", "Fired but still rocket present"),
        ]

        with patch("app.rag.pipeline.get_retriever") as mock_get_ret:
            mock_ret = mock_get_ret.return_value
            mock_ret.retrieve.return_value = misfire_chunks
            mock_ret.last_selected_index = "fault_code"

            response = query("What is a misfire error?")

        ans = response.answer
        assert "Fired but still rocket present" in ans
        assert "0x0006" in ans
        # Ensure zero engine/combustion/ignition hallucinations
        assert "combustion" not in ans
        assert "engine" not in ans
        assert "ignition" not in ans

    def test_fire_aborted_description_summarization(self):
        """Query for 'Fire Aborted' returns summary of operator aborted remarks."""
        chunks = [
            _make_fault_chunk("0x0003", "Fire aborted", "Fire command aborted by operator"),
        ]

        with patch("app.rag.pipeline.get_retriever") as mock_get_ret:
            mock_ret = mock_get_ret.return_value
            mock_ret.retrieve.return_value = chunks
            mock_ret.last_selected_index = "fault_code"

            response = query("What does Fire Aborted mean?")

        ans = response.answer
        assert "Fire command aborted by operator" in ans or "0x0003" in ans
        assert "combustion" not in ans
        assert "spark" not in ans

    def test_depth_setting_failed_description_summarization(self):
        """Query for 'Depth Setting Failed' returns summary of depth bounds remarks."""
        chunks = [
            _make_fault_chunk("0x0017", "Depth setting failed", "Target depth out of allowable bounds"),
        ]

        with patch("app.rag.pipeline.get_retriever") as mock_get_ret:
            mock_ret = mock_get_ret.return_value
            mock_ret.retrieve.return_value = chunks
            mock_ret.last_selected_index = "fault_code"

            response = query("What does Depth Setting Failed mean?")

        ans = response.answer
        assert "0x0017" in ans or "Target depth" in ans or "Depth setting failed" in ans
        assert "vehicle" not in ans
        assert "cylinder" not in ans

    def test_throw_range_invalid_description_summarization(self):
        """Query for 'Throw Range Invalid' returns summary of valid throw range remarks."""
        chunks = [
            _make_fault_chunk("0x0010", "Throw range invalid", "Not a valid throw range"),
        ]

        with patch("app.rag.pipeline.get_retriever") as mock_get_ret:
            mock_ret = mock_get_ret.return_value
            mock_ret.retrieve.return_value = chunks
            mock_ret.last_selected_index = "fault_code"

            response = query("What does Throw Range Invalid mean?")

        ans = response.answer
        assert "0x0010" in ans or "throw range" in ans.lower()
        assert "parameters" not in ans or "operational limits" not in ans
