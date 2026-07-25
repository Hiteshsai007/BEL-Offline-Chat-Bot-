from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_health() -> None:
    """
    Ensures that the FastAPI app boots up properly and the /health endpoint is alive.
    In environments like CI (GitHub Actions), Ollama is not running, so the endpoint
    returns 503 Service Unavailable, which we accept as long as the server itself is running.
    """
    response = client.get("/health")
    assert response.status_code in (200, 503)

    data = response.json()
    assert data["server"] == "ok"


def test_health_reflects_degraded_startup() -> None:
    """
    H-7: If startup partially fails (e.g. FAISS index missing), /health must
    surface the degraded state rather than always reporting ok.
    """
    with patch(
        "app.rag.retriever.get_retriever",
        side_effect=FileNotFoundError("FAISS index missing"),
    ):
        # Fresh TestClient so lifespan runs with the patched retriever
        degraded_client = TestClient(app)
        response = degraded_client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["ready"] is False
