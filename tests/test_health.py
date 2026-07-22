from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_health():
    """
    Ensures that the FastAPI app boots up properly and the /health endpoint is alive.
    This is a vital first test for the CI/CD pipeline.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
