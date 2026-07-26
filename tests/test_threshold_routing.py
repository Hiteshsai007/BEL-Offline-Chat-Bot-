"""
Verify that DIRECT_ANSWER_THRESHOLD routing sends the right queries to the
fast path vs. the LLM path.

These tests use the flip-list from the threshold analysis (Step 2) to
confirm that:
  - Exact code lookups (score=1.0) always take the fast path.
  - Semantic queries scoring >= 0.80 take the fast path.
  - Semantic queries scoring < 0.80 take the LLM path (generate() called).
  - Queries that retrieve nothing return NOT_FOUND without calling generate().

The retriever is mocked to return controlled scores — these tests verify
pipeline *routing*, not retrieval accuracy.  Retrieval accuracy against the
real embedding model is validated by scripts/verify_threshold_scores.py.
"""
from unittest.mock import MagicMock, patch

from app.rag.pipeline import query
from app.rag.retriever import RetrievedChunk


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_chunk(code: str, desc: str, remarks: str = "") -> dict:
    """Build a minimal chunk dict matching the corpus schema."""
    return {
        "error_code": code,
        "error_description": desc,
        "error_remarks": remarks,
        "document_name": "IRL Fault Codes.pdf",
        "chunk_text": f"Error Code: {code} | Error Description: {desc} | Error Remarks: {remarks}",
    }


def _mock_retriever(results: list[RetrievedChunk]):
    """Return a mock Retriever that yields the given results."""
    mock = MagicMock()
    mock.retrieve.return_value = results
    return mock


# ── Category A: Exact code lookups always take the fast path ────────────────

@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_exact_code_lookup_takes_fast_path(mock_get_ret, mock_gen):
    """A1: 'What does error 0x0003 mean?' → exact lookup, score=1.0 → fast path."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(chunk=_make_chunk("0x0003", "Fire aborted", "Dynamic accuracy failed"), score=1.0),
    ])
    result = query("What does error 0x0003 mean?")
    assert result.found is True
    assert result.top_score == 1.0
    assert "0x0003" in result.answer
    mock_gen.assert_not_called()


@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_exact_code_lookup_multi_code_takes_fast_path(mock_get_ret, mock_gen):
    """A5: '0x0002 0x0003' → exact lookup, score=1.0 → fast path."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(chunk=_make_chunk("0x0002", "DEBAR Zone - fire restricted"), score=1.0),
        RetrievedChunk(chunk=_make_chunk("0x0003", "Fire aborted"), score=1.0),
    ])
    result = query("0x0002 0x0003")
    assert result.found is True
    assert result.top_score == 1.0
    mock_gen.assert_not_called()


# ── Category B: Semantic queries at the old threshold (0.60-0.80) ───────────
# These would have been fast-path at 0.60 but should now be LLM-path at 0.80.

@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_misfire_query_routes_to_llm(mock_get_ret, mock_gen):
    """B1: 'What is a misfire error?' → score 0.72 → LLM path at 0.80."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(chunk=_make_chunk("0x0005", "R1 Misfired", "Fired but still rocket present"), score=0.72),
    ])
    mock_gen.return_value = {
        "answer": "A misfire error indicates...",
        "citations": ["IRL Fault Codes.pdf, 0x0005"],
        "latency_ms": 3500,
        "guardrail_triggered": False,
    }
    result = query("What is a misfire error?")
    assert result.found is True
    assert result.top_score == 0.72
    mock_gen.assert_called_once()


@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_throw_range_query_routes_to_llm(mock_get_ret, mock_gen):
    """B2: 'What does throw range invalid mean?' → score 0.76 → LLM path at 0.80."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(
            chunk=_make_chunk("0x0018", "Throw range invalid for R1 (Note 6)", "not a valid throw range"),
            score=0.76,
        ),
    ])
    mock_gen.return_value = {
        "answer": "Throw range invalid means...",
        "citations": ["IRL Fault Codes.pdf, 0x0018"],
        "latency_ms": 4000,
        "guardrail_triggered": False,
    }
    result = query("What does throw range invalid mean?")
    assert result.found is True
    mock_gen.assert_called_once()


@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_fire_abort_query_routes_to_llm(mock_get_ret, mock_gen):
    """B3: 'What causes fire abort?' → score 0.70 → LLM path at 0.80."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(
            chunk=_make_chunk("0x0003", "Fire aborted", "Dynamic accuracy failed (+/- 0.5 Deg)"),
            score=0.70,
        ),
    ])
    mock_gen.return_value = {
        "answer": "A fire abort occurs when...",
        "citations": ["IRL Fault Codes.pdf, 0x0003"],
        "latency_ms": 3000,
        "guardrail_triggered": False,
    }
    result = query("What causes fire abort?")
    assert result.found is True
    mock_gen.assert_called_once()


@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_interlock_query_routes_to_llm(mock_get_ret, mock_gen):
    """B4: 'What are interlock failures?' → score 0.68 → LLM path at 0.80."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(chunk=_make_chunk("0x0004", "Fire interlocks", "Internal interlocks in IRL failed"), score=0.68),
    ])
    mock_gen.return_value = {
        "answer": "Interlock failure means...",
        "citations": ["IRL Fault Codes.pdf, 0x0004"],
        "latency_ms": 3500,
        "guardrail_triggered": False,
    }
    result = query("What are interlock failures?")
    assert result.found is True
    mock_gen.assert_called_once()


@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_comparative_query_routes_to_llm(mock_get_ret, mock_gen):
    """B6: 'What's the difference between R1 and R2 misfires?' → score 0.67 → LLM."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(chunk=_make_chunk("0x0005", "R1 Misfired", "Fired but still rocket present"), score=0.67),
        RetrievedChunk(chunk=_make_chunk("0x0006", "R2 Misfired", "Fired but still rocket present"), score=0.65),
    ])
    mock_gen.return_value = {
        "answer": "Both R1 and R2 misfires indicate...",
        "citations": ["IRL Fault Codes.pdf, 0x0005"],
        "latency_ms": 4000,
        "guardrail_triggered": False,
    }
    result = query("What's the difference between R1 and R2 misfires?")
    assert result.found is True
    mock_gen.assert_called_once()


@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_fix_query_routes_to_llm(mock_get_ret, mock_gen):
    """B7: 'How do I fix a misfire on R3?' → score 0.60 → LLM path at 0.80."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(chunk=_make_chunk("0x0007", "R3 Misfired", "Fired but still rocket present"), score=0.60),
    ])
    mock_gen.return_value = {
        "answer": "The documentation does not specify a repair procedure...",
        "citations": ["IRL Fault Codes.pdf, 0x0007"],
        "latency_ms": 3500,
        "guardrail_triggered": False,
    }
    result = query("How do I fix a misfire on R3?")
    assert result.found is True
    mock_gen.assert_called_once()


# ── Boundary: exactly at threshold → fast path ──────────────────────────────

@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_score_exactly_at_threshold_takes_fast_path(mock_get_ret, mock_gen):
    """A query scoring exactly 0.80 should take the fast path (>=)."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(chunk=_make_chunk("0x0003", "Fire aborted"), score=0.80),
    ])
    result = query("What causes fire aborted?")
    assert result.found is True
    assert result.top_score == 0.80
    mock_gen.assert_not_called()


@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_score_just_below_threshold_routes_to_llm(mock_get_ret, mock_gen):
    """A query scoring 0.799 should take the LLM path."""
    mock_get_ret.return_value = _mock_retriever([
        RetrievedChunk(chunk=_make_chunk("0x0003", "Fire aborted"), score=0.799),
    ])
    mock_gen.return_value = {
        "answer": "A fire abort occurs when...",
        "citations": ["IRL Fault Codes.pdf, 0x0003"],
        "latency_ms": 3000,
        "guardrail_triggered": False,
    }
    result = query("What causes fire aborted?")
    assert result.found is True
    mock_gen.assert_called_once()


# ── NOT_FOUND: no retrieval results → no LLM call ───────────────────────────

@patch("app.rag.pipeline.generate")
@patch("app.rag.pipeline.get_retriever")
def test_empty_retrieval_returns_not_found_without_llm(mock_get_ret, mock_gen):
    """C3: 'What are the most common problems?' → [] → NOT_FOUND, no LLM."""
    mock_get_ret.return_value = _mock_retriever([])
    result = query("What are the most common problems?")
    assert result.found is False
    assert "not available" in result.answer.lower()
    mock_gen.assert_not_called()


# ── Threshold is read from config ───────────────────────────────────────────

def test_threshold_value_from_config():
    """DIRECT_ANSWER_THRESHOLD should be loaded from config.yaml, not hardcoded."""
    from app.settings import DIRECT_ANSWER_THRESHOLD
    assert DIRECT_ANSWER_THRESHOLD == 0.80
