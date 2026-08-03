from fastapi.testclient import TestClient

import app.main as main


def test_select_model_rejects_unknown_tag(monkeypatch):
    async def fake_list_models():
        return {"current": "qwen2.5:3b", "available": ["qwen2.5:3b"]}

    monkeypatch.setattr(main, "list_models", fake_list_models)

    with TestClient(main.app) as client:
        response = client.post("/model/select", json={"model": "missing-model"})

    assert response.status_code == 422
    assert "not installed in Ollama" in response.json()["detail"]
