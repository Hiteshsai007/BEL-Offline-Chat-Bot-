"""
tests/test_security_error_leak.py

Covers security finding S-2: app/main.py returned raw ``str(e)`` exception text
to the HTTP client on two paths (the /health Ollama probe and POST /reload),
leaking absolute filesystem paths and the operator's username into the JSON
response body -- which app/static/app.js renders directly into the chat UI.

Each test forces the underlying call to raise an exception whose message
contains obviously sensitive markers, then asserts none of those markers reach
the response body while a generic message does.

Cross-platform notes
--------------------
* The sentinel strings below deliberately include both a Windows-style path
  (C:\\Users\\...) and a POSIX one (/home/...) so the assertion is meaningful
  regardless of which OS the suite runs on.
* No real network or filesystem access -- both failure paths are mocked.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.settings import SERVER_PORT

client = TestClient(app)

SAME_ORIGIN = f"http://127.0.0.1:{SERVER_PORT}"

# Markers that must never appear in an HTTP response body.
SECRET_PATH_WIN = r"C:\Users\bel-operator\secret-index"
SECRET_PATH_NIX = "/home/bel-operator/secret-index"
SECRET_MESSAGE = (
    f"Permission denied opening {SECRET_PATH_NIX} and {SECRET_PATH_WIN} "
    "(uid=1000 user=bel-operator)"
)


def _assert_no_leak(body: str) -> None:
    """Fail if any sensitive fragment of the exception reached the client."""
    for marker in (SECRET_PATH_WIN, SECRET_PATH_NIX, "bel-operator", "uid=1000"):
        assert marker not in body, (
            f"Response body leaked sensitive exception detail {marker!r}. "
            f"Body was: {body}"
        )


# ── /health — line 160-161 in the reviewed code ─────────────────────────────

def test_health_does_not_leak_exception_text() -> None:
    """A failing Ollama probe must not echo the exception into the response."""
    with patch("app.main.httpx.AsyncClient") as mock_client:
        mock_client.side_effect = OSError(SECRET_MESSAGE)
        response = client.get("/health")

    assert response.status_code in (200, 503)
    _assert_no_leak(response.text)


def test_health_reports_generic_unreachable_marker() -> None:
    """The generic replacement value must still convey that Ollama is down."""
    with patch("app.main.httpx.AsyncClient") as mock_client:
        mock_client.side_effect = OSError(SECRET_MESSAGE)
        response = client.get("/health")

    data = response.json()
    assert data["ollama"] == "unreachable", (
        f"Expected generic 'unreachable' marker, got {data['ollama']!r}."
    )
    # app.js branches on `data.ollama !== 'ok'`, so any non-'ok' value keeps
    # the UI's "Ollama not running" state working.
    assert data["ollama"] != "ok"
    assert data["server"] == "ok"


# ── POST /reload — line 179 in the reviewed code ────────────────────────────

def test_reload_does_not_leak_exception_text() -> None:
    """A failing reload must return a generic 500, not the raw exception."""
    with patch("app.rag.retriever.get_retriever") as mock_get:
        mock_get.side_effect = RuntimeError(SECRET_MESSAGE)
        response = client.post("/reload", headers={"Origin": SAME_ORIGIN})

    assert response.status_code == 500
    _assert_no_leak(response.text)


def test_reload_returns_generic_error_message() -> None:
    """The client should get an actionable but non-revealing message."""
    with patch("app.rag.retriever.get_retriever") as mock_get:
        mock_get.side_effect = RuntimeError(SECRET_MESSAGE)
        response = client.post("/reload", headers={"Origin": SAME_ORIGIN})

    detail = response.json()["detail"]
    assert detail == "Index reload failed. See the server log for details."


def test_reload_logs_real_exception_server_side(caplog) -> None:
    """
    The real cause must still reach the server log -- suppressing the leak must
    not also suppress the operator's ability to diagnose the failure.
    """
    with caplog.at_level("ERROR"):
        with patch("app.rag.retriever.get_retriever") as mock_get:
            mock_get.side_effect = RuntimeError(SECRET_MESSAGE)
            client.post("/reload", headers={"Origin": SAME_ORIGIN})

    assert SECRET_MESSAGE in caplog.text, (
        "The genuine exception text must be logged server-side for diagnosis."
    )
