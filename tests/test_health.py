from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_health():
    """
    Ensures that the FastAPI app boots up properly and the /health endpoint is alive.
    In environments like CI (GitHub Actions), Ollama is not running, so the endpoint
    returns 503 Service Unavailable, which we accept as long as the server itself is running.
    """
    response = client.get("/health")
    assert response.status_code in (200, 503)

    data = response.json()
    assert data["server"] == "ok"
