"""
tests/test_security_origin.py

Covers security finding S-1: POST /query and POST /reload had no CSRF/origin
protection, so a malicious page in the operator's browser could drive them via
a cross-origin form post.

These tests assert the same-origin guard in app/security.py actually rejects
cross-origin browser requests while leaving legitimate use unaffected.

Cross-platform notes
--------------------
* No filesystem or shell assumptions -- these are pure HTTP-level assertions
  against FastAPI's TestClient, identical on Windows and Linux.
* SERVER_PORT is read from config rather than hardcoded, so the expected
  same-origin value stays correct if the port is retuned in app/config.yaml.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.settings import SERVER_PORT

client = TestClient(app)

SAME_ORIGIN = f"http://127.0.0.1:{SERVER_PORT}"
EVIL_ORIGIN = "http://evil.example.com"


# ── POST /reload ────────────────────────────────────────────────────────────

def test_reload_rejected_from_cross_origin() -> None:
    """A cross-origin POST /reload must be rejected with 403, not executed."""
    with patch("app.rag.retriever.get_retriever") as mock_get:
        response = client.post("/reload", headers={"Origin": EVIL_ORIGIN})

    assert response.status_code == 403, (
        f"Expected 403 for cross-origin /reload, got {response.status_code}. "
        "A malicious page could otherwise force an index reload."
    )
    mock_get.assert_not_called(), "Guard must reject before touching the retriever."


def test_reload_rejected_via_cross_origin_referer() -> None:
    """Origin absent but Referer cross-origin must still be rejected."""
    response = client.post(
        "/reload", headers={"Referer": f"{EVIL_ORIGIN}/attack.html"}
    )
    assert response.status_code == 403


def test_reload_allowed_from_same_origin() -> None:
    """The real UI (same origin) must still be able to reload the index."""
    with patch("app.rag.retriever.get_retriever") as mock_get:
        response = client.post("/reload", headers={"Origin": SAME_ORIGIN})

    assert response.status_code == 200, (
        f"Same-origin /reload must succeed, got {response.status_code}: {response.text}"
    )
    assert response.json()["status"] == "ok"
    mock_get.assert_called_once()


def test_reload_allowed_from_localhost_origin() -> None:
    """'localhost' is an equally valid loopback alias and must be accepted."""
    with patch("app.rag.retriever.get_retriever"):
        response = client.post(
            "/reload", headers={"Origin": f"http://localhost:{SERVER_PORT}"}
        )
    assert response.status_code == 200


def test_reload_allowed_for_non_browser_client() -> None:
    """
    A client sending no Origin/Referer (CLI, curl, probe) is not CSRF-reachable
    and must not be blocked -- documented residual risk in app/security.py.
    """
    with patch("app.rag.retriever.get_retriever"):
        response = client.post("/reload")
    assert response.status_code == 200


# ── POST /query ─────────────────────────────────────────────────────────────

def test_query_rejected_from_cross_origin() -> None:
    """Cross-origin POST /query must be rejected before the RAG pipeline runs."""
    with patch("app.main.rag_query") as mock_query:
        response = client.post(
            "/query",
            json={"question": "What does 0x0003 mean?"},
            headers={"Origin": EVIL_ORIGIN},
        )

    assert response.status_code == 403, (
        f"Expected 403 for cross-origin /query, got {response.status_code}."
    )
    mock_query.assert_not_called(), (
        "Guard must reject before spending embedding/LLM time."
    )


def test_query_allowed_from_same_origin() -> None:
    """The real UI must still be able to submit questions."""
    from app.rag.pipeline import QueryResponse as PipelineResponse

    stub = PipelineResponse(
        answer="**0x0003** — Fire aborted",
        citations=["IRL Fault Codes.pdf, 0x0003"],
        retrieved_chunks=[],
        top_score=1.0,
        latency_ms=12,
        found=True,
    )
    with patch("app.main.rag_query", return_value=stub) as mock_query:
        response = client.post(
            "/query",
            json={"question": "What does 0x0003 mean?"},
            headers={"Origin": SAME_ORIGIN},
        )

    assert response.status_code == 200, (
        f"Same-origin /query must succeed, got {response.status_code}: {response.text}"
    )
    mock_query.assert_called_once()


# ── Read-only routes must stay unguarded ────────────────────────────────────

def test_health_not_blocked_by_origin_guard() -> None:
    """
    /health changes no state; gating it would break monitoring and the UI's
    status poll. It must remain reachable regardless of Origin.
    """
    response = client.get("/health", headers={"Origin": EVIL_ORIGIN})
    assert response.status_code in (200, 503)
