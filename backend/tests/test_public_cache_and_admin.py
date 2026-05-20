"""Production-safe public cache/admin behavior tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


def test_latest_endpoint_does_not_run_forecast_on_empty_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public reads must not trigger heavy ML computation."""
    calls = 0

    def fake_run_forecast(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal calls
        calls += 1
        return {"rows_written": 1}

    monkeypatch.setattr("backend.app.api.routes.forecasts.run_forecast", fake_run_forecast)

    client = TestClient(app)
    r = client.get("/api/forecasts/latest", params={"horizon": 30, "threshold": 5.0})

    assert r.status_code == 200
    assert "count" in r.json()
    assert calls == 0


def test_top_endpoint_does_not_run_forecast_on_empty_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-N public reads must stay cache-only too."""
    calls = 0

    def fake_run_forecast(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal calls
        calls += 1
        return {"rows_written": 1}

    monkeypatch.setattr("backend.app.api.routes.forecasts.run_forecast", fake_run_forecast)

    client = TestClient(app)
    r = client.get("/api/forecasts/top", params={"n": 10, "horizon": 30, "threshold": 5.0})

    assert r.status_code == 200
    assert r.json()["items"] == []
    assert calls == 0


def test_forecast_run_requires_admin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "dev-secret")
    from backend.app.config import get_settings

    get_settings.cache_clear()
    client = TestClient(app)

    missing = client.post("/api/forecasts/run")
    assert missing.status_code == 401

    wrong = client.post("/api/forecasts/run", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 403

    ok = client.post("/api/forecasts/run", headers={"Authorization": "Bearer dev-secret"}, params={"force_demo": True})
    assert ok.status_code == 200
    assert ok.json()["rows_written"] > 0


def test_forecast_status_reports_cached_metadata() -> None:
    from backend.app.db.metadata import set_metadata_value

    client = TestClient(app)
    client.post("/api/forecasts/run", headers={"Authorization": "Bearer test-admin-token"}, params={"force_demo": True})
    set_metadata_value("new_events_since_last_forecast", "not-a-number")

    r = client.get("/api/forecast/status")

    assert r.status_code == 200
    body = r.json()
    assert body["forecast_last_computed_at"] is not None
    assert body["forecast_model_version"] is not None
    assert body["trigger_mode"] == "any_new_event"
    assert body["fallback_hours"] == 3
    assert body["new_events_since_last_forecast"] == 0


def test_scheduler_tick_runs_forecast_for_any_new_small_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any new catalog event, even small magnitude, must trigger forecast."""
    from backend.app.db.sqlite import get_connection, migrate
    from backend.app.scheduler.jobs import scheduler_tick

    migrate()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO realtime_events
               (event_id, time, lat, lon, depth, magnitude, mag_type, source, place)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                "small-1",
                datetime.now(UTC).isoformat(),
                -7.1,
                110.1,
                10.0,
                2.6,
                "ml",
                "usgs",
                "small test quake",
            ),
        )

    calls = 0

    def fake_ingest_realtime():
        return {"raw": 1, "deduped": 1, "stored": 1}

    def fake_run_forecast():
        nonlocal calls
        calls += 1
        return {"rows_written": 16, "mode": "test"}

    monkeypatch.setattr("backend.app.scheduler.jobs.ingest_realtime", fake_ingest_realtime, raising=False)
    monkeypatch.setattr("backend.app.scheduler.jobs.run_forecast", fake_run_forecast, raising=False)

    out = scheduler_tick(now=datetime.now(UTC))

    assert out["forecast_ran"] is True
    assert out["reason"] == "new_events"
    assert out["new_events"] == 1
    assert calls == 1


def test_scheduler_tick_batches_inside_debounce(monkeypatch: pytest.MonkeyPatch) -> None:
    """A burst inside debounce window should be recorded but not run duplicate forecast."""
    from backend.app.db.sqlite import get_connection, migrate
    from backend.app.scheduler.jobs import scheduler_tick, set_status_value

    migrate()
    now = datetime.now(UTC)
    set_status_value("last_forecast_at", (now - timedelta(minutes=2)).isoformat())
    set_status_value("last_seen_event_id", "prev")

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO realtime_events
               (event_id, time, lat, lon, depth, magnitude, mag_type, source, place)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("prev", (now - timedelta(minutes=3)).isoformat(), -7.1, 110.1, 10.0, 2.6, "ml", "usgs", "prev"),
        )
        conn.execute(
            """INSERT INTO realtime_events
               (event_id, time, lat, lon, depth, magnitude, mag_type, source, place)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("small-2", now.isoformat(), -7.2, 110.2, 10.0, 2.7, "ml", "usgs", "new"),
        )

    calls = 0

    def fake_ingest_realtime():
        return {"raw": 1, "deduped": 1, "stored": 1}

    def fake_run_forecast():
        nonlocal calls
        calls += 1
        return {"rows_written": 16, "mode": "test"}

    monkeypatch.setattr("backend.app.scheduler.jobs.ingest_realtime", fake_ingest_realtime, raising=False)
    monkeypatch.setattr("backend.app.scheduler.jobs.run_forecast", fake_run_forecast, raising=False)

    out = scheduler_tick(now=now)

    assert out["forecast_ran"] is False
    assert out["reason"] == "debounced"
    assert out["new_events"] == 1
    assert calls == 0


def test_scheduler_trigger_requires_admin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "dev-secret")
    from backend.app.config import get_settings

    get_settings.cache_clear()
    client = TestClient(app)

    r = client.post("/api/scheduler/trigger/nonexistent")
    assert r.status_code == 401

    ok = client.post("/api/scheduler/trigger/nonexistent", headers={"Authorization": "Bearer dev-secret"})
    assert ok.status_code == 200
    assert ok.json()["ok"] is False
