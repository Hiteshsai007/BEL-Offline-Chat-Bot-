"""
Tests for app/rag/generator.py — LLM call + citation guardrail interaction.

These cover the untested parts of generator.py that test_security_citation_guardrail.py
does not touch: _call_ollama and generate, including the guardrail retry loop.
"""
from unittest.mock import MagicMock, patch

import httpx

from app.rag.generator import (
    DEGRADED_MSG,
    INSUFFICIENT_MSG,
    _call_ollama,
    generate,
)
from app.rag.retriever import RetrievedChunk

DOC = "IRL Fault Codes.pdf"


def _chunk(code: str = "0x0003", score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk={
            "error_code": code,
            "document_name": DOC,
            "error_description": "Fire aborted",
            "error_remarks": "Fire command aborted by operator",
            "chunk_text": f"Error Code: {code} | Error Description: Fire aborted",
            "chunk_type": "table",
        },
        score=score,
    )


def test_generate_success_with_citation() -> None:
    """Happy path: Ollama returns an answer with a valid citation."""
    context = [_chunk("0x0003")]
    answer = "Fire aborted [IRL Fault Codes.pdf, 0x0003]."

    with patch("app.rag.generator._call_ollama", return_value=(answer, 1200.0)):
        result = generate("What is 0x0003?", context)

    assert result["answer"] == answer
    assert result["citations"] == [f"{DOC}, 0x0003"]
    assert result["guardrail_triggered"] is False
    assert result["latency_ms"] == 1200


def test_generate_guardrail_triggers_retry() -> None:
    """First answer has no citation; retry succeeds."""
    context = [_chunk("0x0003")]
    bad_answer = "Fire aborted by operator."
    good_answer = "Fire aborted [IRL Fault Codes.pdf, 0x0003]."

    with patch(
        "app.rag.generator._call_ollama",
        side_effect=[(bad_answer, 800.0), (good_answer, 900.0)],
    ):
        result = generate("What is 0x0003?", context)

    assert result["answer"] == good_answer
    assert result["guardrail_triggered"] is True
    assert result["citations"] == [f"{DOC}, 0x0003"]


def test_generate_guardrail_both_fail() -> None:
    """Both attempts lack citations -> INSUFFICIENT_MSG."""
    context = [_chunk("0x0003")]
    bad_answer = "Fire aborted by operator."

    with patch("app.rag.generator._call_ollama", return_value=(bad_answer, 800.0)):
        result = generate("What is 0x0003?", context)

    assert result["answer"] == INSUFFICIENT_MSG
    assert result["guardrail_triggered"] is True
    assert result["citations"] == []


def test_generate_ollama_connect_error() -> None:
    """Connection error returns DEGRADED_MSG with error marker."""
    context = [_chunk("0x0003")]

    with patch(
        "app.rag.generator._call_ollama",
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        result = generate("What is 0x0003?", context)

    assert result["answer"] == DEGRADED_MSG
    assert result.get("error") == "ollama_unavailable"


def test_generate_ollama_generic_error() -> None:
    """Generic HTTP error returns DEGRADED_MSG."""
    context = [_chunk("0x0003")]

    with patch(
        "app.rag.generator._call_ollama",
        side_effect=RuntimeError("unexpected boom"),
    ):
        result = generate("What is 0x0003?", context)

    assert result["answer"] == DEGRADED_MSG
    assert "error" in result


def test_call_ollama_parses_response() -> None:
    """
    _call_ollama must return the parsed answer and latency.

    H-14 regression guard: resp.json() must be called inside the
    httpx.Client context manager.  If someone moves it outside the
    ``with`` block this test will still pass *today* because httpx
    buffers the body, but the assertion on json() being called verifies
    the current structure.  A future streaming change would break an
    out-of-block resp.json() and this comment documents that risk.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "  Test answer  "}
    mock_response.raise_for_status.return_value = None

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch("app.rag.generator.httpx.Client") as MockClient:
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
        MockClient.return_value.__exit__ = MagicMock(return_value=False)

        answer, elapsed = _call_ollama("prompt", "system")

    mock_client.post.assert_called_once()
    mock_response.json.assert_called_once()
    assert answer == "Test answer"
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0
