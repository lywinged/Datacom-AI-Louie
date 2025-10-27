"""Integration tests for Code Assistant API routes."""


def test_code_generate_returns_successful_payload(client, dummy_code_assistant):
    """Test /generate endpoint returns CodeResponse with successful code generation."""
    response = client.post(
        "/api/code/generate",
        json={"task": "Write a Python function to add two numbers", "language": "python"},
    )

    assert response.status_code == 200
    payload = response.json()

    # Check core fields
    assert payload["test_passed"] is True
    assert "def add" in payload["code"]
    assert payload["final_test_result"]["exit_code"] == 0
    assert payload["language"] == "python"

    # Check additional response fields
    assert "generation_time_ms" in payload
    assert "tokens_used" in payload
    assert "cost_usd" in payload
    assert "total_retries" in payload
    assert "retry_attempts" in payload


def test_code_generate_with_test_framework(client, dummy_code_assistant):
    """Test /generate endpoint works with specified test framework."""
    response = client.post(
        "/api/code/generate",
        json={
            "task": "Write a function to check if a number is prime",
            "language": "python",
            "test_framework": "pytest",
            "max_retries": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "code" in payload
    assert payload["language"] == "python"


def test_code_generate_includes_test_result(client, dummy_code_assistant):
    """Test /generate endpoint includes detailed test result."""
    response = client.post(
        "/api/code/generate",
        json={"task": "Implement a calculator", "language": "python"},
    )

    assert response.status_code == 200
    payload = response.json()

    # Check final_test_result structure
    test_result = payload["final_test_result"]
    assert "passed" in test_result
    assert "stdout" in test_result
    assert "stderr" in test_result
    assert "exit_code" in test_result
    assert "execution_time_ms" in test_result


def test_code_generate_includes_token_usage(client, dummy_code_assistant):
    """Test /generate endpoint includes token usage and cost."""
    response = client.post(
        "/api/code/generate",
        json={"task": "Simple hello world", "language": "python"},
    )

    assert response.status_code == 200
    payload = response.json()

    # Check token usage
    assert "tokens_used" in payload
    assert "token_usage" in payload
    assert "token_cost_usd" in payload

    if payload["token_usage"]:
        assert "prompt" in payload["token_usage"]
        assert "completion" in payload["token_usage"]
        assert "total" in payload["token_usage"]


def test_code_generate_with_different_language(client, dummy_code_assistant):
    """Test /generate endpoint supports multiple languages."""
    response = client.post(
        "/api/code/generate",
        json={"task": "Write a hello world program", "language": "rust"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["language"] == "rust"


def test_code_health_reports_stub_model(client, dummy_code_assistant):
    """Test /health endpoint returns service status and model info."""
    response = client.get("/api/code/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model"] == dummy_code_assistant.model_name
    assert payload["service"] == "code_assistant"


def test_code_metrics_returns_defaults(client):
    """Test /metrics endpoint returns CodeMetrics structure."""
    response = client.get("/api/code/metrics")
    assert response.status_code == 200

    payload = response.json()
    assert "total_requests" in payload
    assert "success_rate" in payload
    assert "avg_retries" in payload
    assert "avg_generation_time_ms" in payload
    assert "languages_used" in payload


def test_code_generate_retry_attempts_field(client, dummy_code_assistant):
    """Test /generate endpoint includes retry attempts in response."""
    response = client.post(
        "/api/code/generate",
        json={
            "task": "Write buggy code",
            "language": "python",
            "max_retries": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    # Check retry fields
    assert "retry_attempts" in payload
    assert isinstance(payload["retry_attempts"], list)
    assert "total_retries" in payload
    assert isinstance(payload["total_retries"], int)
