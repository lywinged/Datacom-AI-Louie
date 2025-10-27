"""Integration tests for Chat API routes with stubbed chat service."""


def test_chat_message_returns_echo_response(client, dummy_chat_service):
    """Test /message endpoint returns ChatResponse with token counts."""
    response = client.post(
        "/api/chat/message",
        json={"message": "Hello there!", "max_history": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "Echo: Hello there!"
    assert payload["total_tokens"] == 8
    assert "prompt_tokens" in payload
    assert "completion_tokens" in payload
    assert "cost_usd" in payload
    assert "latency_ms" in payload


def test_chat_history_and_clear_workflow(client, dummy_chat_service):
    """Test /history GET and DELETE endpoints work correctly."""
    client.post("/api/chat/message", json={"message": "First message"})
    history_response = client.get("/api/chat/history")

    assert history_response.status_code == 200
    history_payload = history_response.json()
    assert history_payload["total_messages"] == 2  # user + assistant

    clear_response = client.delete("/api/chat/history")
    assert clear_response.status_code == 200

    empty_history = client.get("/api/chat/history").json()
    assert empty_history["total_messages"] == 0


def test_chat_stream_endpoint(client, dummy_chat_service):
    """Test /stream endpoint returns streaming response."""
    response = client.post(
        "/api/chat/stream",
        json={"message": "Stream test", "stream": True},
    )

    # SSE endpoint should return 200 with text/event-stream content type
    assert response.status_code == 200
    # Note: Full SSE testing requires async client, here we just check endpoint exists


def test_chat_metrics_endpoint(client, dummy_chat_service):
    """Test /metrics endpoint returns metrics information."""
    response = client.get("/api/chat/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert "message" in payload or "endpoint" in payload
    # Metrics endpoint returns info about Prometheus metrics


def test_chat_message_with_stream_false(client, dummy_chat_service):
    """Test that stream=false returns complete response immediately."""
    response = client.post(
        "/api/chat/message",
        json={"message": "Quick question", "stream": False, "max_history": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "message" in payload
    assert isinstance(payload["message"], str)
    assert payload["total_tokens"] > 0
