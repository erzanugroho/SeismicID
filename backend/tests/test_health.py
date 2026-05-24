"""Smoke test for /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "uptime_seconds" in body
    assert isinstance(body["uptime_seconds"], int | float)


def test_api_health_alias_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_endpoint_reports_structured_checks() -> None:
    client = TestClient(app)
    response = client.get("/api/health/readiness")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ready", "degraded"}
    assert isinstance(body["ok"], bool)
    assert body["checks"]["db"]["ok"] is True
    assert "event_count" in body["checks"]["db"]
    assert "forecast" in body["checks"]
    assert "model_dir" in body["checks"]
