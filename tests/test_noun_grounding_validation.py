"""
Tests for noun-grounding validation and fallback generation in app/rag/generator.py.

Validates that:
  1. Hallucinated/ungrounded terms (e.g. 'internal combustion engines', 'parameters')
     are detected by _extract_ungrounded_terms.
  2. Grounded terms present in the context pass validation without triggering warnings.
  3. _format_no_definition_fallback produces the exact required format:
     "The document does not provide a definition. It only states: <text> [<doc>, <code>]"
"""
from app.rag.generator import (
    _extract_ungrounded_terms,
    _format_no_definition_fallback,
)
from app.rag.retriever import RetrievedChunk


def _make_retrieved(text: str, code: str = "R9", doc: str = "IRL Fault Codes.pdf") -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk={
                "chunk_id": "c1",
                "chunk_text": text,
                "document_name": doc,
                "error_code": code,
                "error_description": text,
                "page_number": 1,
            },
            score=1.0,
        )
    ]


class TestNounGroundingValidation:

    def test_misfire_error_hallucination_detected(self):
        """
        Query: 'What is a misfire error?'
        Retrieved: 'R9 Misfired | Fired but still rocket present'
        Hallucinated LLM answer contains combustion, engines, ignition, cylinders, fuel.
        """
        retrieved = _make_retrieved("R9 Misfired | Fired but still rocket present", code="R9")
        hallucinated_answer = (
            "A misfire error occurs in internal combustion engines when the ignition "
            "system fails to ignite fuel in a cylinder [IRL Fault Codes.pdf, R9]."
        )

        ungrounded = _extract_ungrounded_terms(hallucinated_answer, retrieved)
        assert len(ungrounded) > 0
        assert "combustion" in ungrounded
        assert "engines" in ungrounded
        assert "ignition" in ungrounded

    def test_throw_range_invalid_hallucination_detected(self):
        """
        Query: 'What does throw range invalid mean?'
        Retrieved: 'Throw range invalid for R10 | not a valid throw range'
        Hallucinated LLM answer contains operational, limits, parameters, input.
        """
        retrieved = _make_retrieved("Throw range invalid for R10", code="R10")
        hallucinated_answer = (
            "Throw range invalid means input parameters exceeded operational limits "
            "for R10 [IRL Fault Codes.pdf, R10]."
        )

        ungrounded = _extract_ungrounded_terms(hallucinated_answer, retrieved)
        assert len(ungrounded) > 0
        assert "parameters" in ungrounded
        assert "operational" in ungrounded

    def test_grounded_answer_passes_validation(self):
        """
        Grounded answer containing only terms from context and standard stop words passes.
        """
        retrieved = _make_retrieved("Throw range invalid for R10", code="R10")
        grounded_answer = (
            "The document does not provide a definition. It only states: Throw range invalid for R10 "
            "[IRL Fault Codes.pdf, R10]"
        )

        ungrounded = _extract_ungrounded_terms(grounded_answer, retrieved)
        assert ungrounded == []

    def test_format_no_definition_fallback(self):
        """
        Verify the fallback string format when context lacks an explicit definition.
        """
        retrieved = _make_retrieved("Throw range invalid for R10", code="R10")
        fallback = _format_no_definition_fallback(retrieved)

        assert "The document does not provide a definition. It only states:" in fallback
        assert "Throw range invalid for R10" in fallback
        assert "[IRL Fault Codes.pdf, R10]" in fallback
