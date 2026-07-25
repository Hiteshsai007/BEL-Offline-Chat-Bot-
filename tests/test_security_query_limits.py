"""
tests/test_security_query_limits.py

Covers security finding S-3: QueryRequest.question had no length limit, so an
arbitrarily large request body was passed straight into
SentenceTransformer.encode() and then concatenated into the Ollama prompt --
a cheap local denial of service.

Cross-platform notes
--------------------
* Pure HTTP-level assertions with the RAG pipeline mocked; no filesystem,
  network or model access, so behaviour is identical on Windows and Linux.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import MAX_QUESTION_CHARS, app
from app.rag.pipeline import QueryResponse as PipelineResponse
from app.settings import SERVER_PORT

client = TestClient(app)

SAME_ORIGIN = f"http://127.0.0.1:{SERVER_PORT}"


def _stub_response() -> PipelineResponse:
    return PipelineResponse(
        answer="**0x0003** — Fire aborted",
        citations=["IRL Fault Codes.pdf, 0x0003"],
        retrieved_chunks=[],
        top_score=1.0,
        latency_ms=8,
        found=True,
    )


def test_overlong_question_rejected_with_422() -> None:
    """A question beyond the limit must be rejected by validation."""
    oversized = "A" * (MAX_QUESTION_CHARS + 1)

    with patch("app.main.rag_query") as mock_query:
        response = client.post(
            "/query",
            json={"question": oversized},
            headers={"Origin": SAME_ORIGIN},
        )

    assert response.status_code == 422, (
        f"Expected 422 for a {len(oversized)}-char question, "
        f"got {response.status_code}."
    )
    mock_query.assert_not_called(), (
        "Validation must reject before the embedder/LLM is invoked -- "
        "otherwise the DoS this fix addresses still lands."
    )


def test_massively_overlong_question_rejected() -> None:
    """A megabyte-scale body must also be refused, not merely truncated."""
    with patch("app.main.rag_query") as mock_query:
        response = client.post(
            "/query",
            json={"question": "A" * 1_000_000},
            headers={"Origin": SAME_ORIGIN},
        )

    assert response.status_code == 422
    mock_query.assert_not_called()


def test_normal_length_question_still_succeeds() -> None:
    """A realistic fault-code question must be unaffected by the limit."""
    with patch("app.main.rag_query", return_value=_stub_response()) as mock_query:
        response = client.post(
            "/query",
            json={"question": "What does error code 0x0003 mean?"},
            headers={"Origin": SAME_ORIGIN},
        )

    assert response.status_code == 200, (
        f"A normal question must succeed, got {response.status_code}: {response.text}"
    )
    mock_query.assert_called_once()


def test_question_exactly_at_limit_is_accepted() -> None:
    """The boundary value itself must be allowed (limit is inclusive)."""
    with patch("app.main.rag_query", return_value=_stub_response()):
        response = client.post(
            "/query",
            json={"question": "A" * MAX_QUESTION_CHARS},
            headers={"Origin": SAME_ORIGIN},
        )

    assert response.status_code == 200


def test_empty_question_rejected() -> None:
    """An empty string is refused by min_length before reaching the pipeline."""
    with patch("app.main.rag_query") as mock_query:
        response = client.post(
            "/query", json={"question": ""}, headers={"Origin": SAME_ORIGIN}
        )

    assert response.status_code == 422
    mock_query.assert_not_called()


def test_whitespace_only_question_rejected() -> None:
    """
    Whitespace passes min_length but is still meaningless; the handler's
    explicit strip() guard must continue to catch it.
    """
    with patch("app.main.rag_query") as mock_query:
        response = client.post(
            "/query", json={"question": "    "}, headers={"Origin": SAME_ORIGIN}
        )

    assert response.status_code == 422
    mock_query.assert_not_called()


def test_limit_is_documented_and_reasonable() -> None:
    """
    Guard against someone tightening the limit to a value that would start
    rejecting genuine questions.
    """
    assert MAX_QUESTION_CHARS >= 500, (
        "Limit is too tight for a plain-language fault-code question."
    )
    assert MAX_QUESTION_CHARS <= 10_000, (
        "Limit is too loose to meaningfully bound embedding/prompt cost."
    )
