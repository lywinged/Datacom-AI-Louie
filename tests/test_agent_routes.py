"""Integration tests for Planning Agent API routes."""


def test_agent_plan_returns_itinerary(client, dummy_planning_agent):
    request_payload = {
        "prompt": "Plan a relaxing weekend in Wellington",
        "constraints": {
            "budget": 1500,
            "currency": "NZD",
            "days": 3,
            "origin_city": "Auckland",
            "destination_city": "Wellington",
        },
    }

    response = client.post("/api/agent/plan", json=request_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["constraints_satisfied"] is True
    assert payload["itinerary"]["destination"] == "Wellington"
    assert payload["total_iterations"] == 1
    assert payload["reasoning_trace"][0]["step_number"] == 1


def test_agent_health_reports_stub_model(client, dummy_planning_agent):
    response = client.get("/api/agent/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model"] == dummy_planning_agent.model_name


def test_agent_metrics_returns_defaults(client):
    response = client.get("/api/agent/metrics")
    assert response.status_code == 200

    payload = response.json()
    assert "total_plans" in payload
    assert "history" in payload
