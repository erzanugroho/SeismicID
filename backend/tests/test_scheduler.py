"""Tests for scheduler — manual trigger + audit log."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def test_scheduler_runs_endpoint_initial_empty() -> None:
    client = TestClient(app)
    r = client.get("/api/scheduler/runs")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_scheduler_unknown_job() -> None:
    client = TestClient(app)
    r = client.post("/api/scheduler/trigger/nonexistent")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "unknown" in body["error"].lower()


def test_scheduler_trigger_forecast_recompute() -> None:
    client = TestClient(app)
    r = client.post("/api/scheduler/trigger/forecast_recompute")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    # Verify audit log shows it
    runs = client.get("/api/scheduler/runs").json()["items"]
    assert any(r["job_name"] == "forecast_recompute" and r["status"] == "success" for r in runs)
