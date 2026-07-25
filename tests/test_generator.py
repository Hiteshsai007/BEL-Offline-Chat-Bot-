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


def test_call_ollama_parses_inside_context_manager() -> None:
    """
    H-14: resp.json() must be called inside the httpx.Client context manager.

    If someone moves resp.json() outside the ``with`` block this test will
    fail because we explicitly verify call order: json() before __exit__.
    """
    call_order: list[str] = []

    mock_response = MagicMock()

    def track_json() -> dict:
        call_order.append("json")
        return {"response": "  Test answer  "}

    mock_response.json.side_effect = track_json
    mock_response.raise_for_status.return_value = None

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    def track_exit(*args, **kwargs) -> bool:
        call_order.append("exit")
        return False

    with patch("app.rag.generator.httpx.Client") as MockClient:
        MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
        MockClient.return_value.__exit__ = track_exit

        answer, elapsed = _call_ollama("prompt", "system")

    mock_client.post.assert_called_once()
    assert call_order.index("json") < call_order.index("exit"), (
        "resp.json() must be called before the httpx.Client context exits"
    )
    assert answer == "Test answer"
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0
