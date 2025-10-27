"""Integration tests for system-level FastAPI endpoints."""


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200

    payload = response.json()
    assert payload["name"] == "AI Assessment API"
    assert payload["version"] == "1.0.0"
    assert payload["endpoints"]["rag"] == "/api/rag"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert isinstance(payload["onnx_enabled"], bool)
    assert isinstance(payload["int8_enabled"], bool)
