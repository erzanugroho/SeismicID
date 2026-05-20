"""Tests for forecast service + /api/forecasts."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.forecast_service import format_sentence, run_forecast


def test_run_forecast_demo_mode_seeds_data() -> None:
    out = run_forecast(force_demo=True)
    assert out["mode"] == "demo_seed"
    assert out["rows_written"] > 0


def test_get_latest_endpoint_returns_data_after_run() -> None:
    client = TestClient(app)
    client.post("/api/forecasts/run", params={"force_demo": True})
    r = client.get("/api/forecasts/latest", params={"horizon": 30, "threshold": 5.0})
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_days"] == 30
    assert body["mag_threshold"] == 5.0
    assert body["count"] > 100  # Indonesia grid has thousands of cells


def test_top_endpoint_returns_sentences() -> None:
    client = TestClient(app)
    client.post("/api/forecasts/run", params={"force_demo": True})
    r = client.get("/api/forecasts/top", params={"n": 5, "horizon": 30, "threshold": 5.0})
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 5
    assert len(body["sentences"]) == 5
    # Validate the canonical Indonesian sentence pattern
    s = body["sentences"][0]
    assert "% probabilitas M≥" in s
    assert "dalam 30 hari" in s


def test_area_endpoint_returns_16_forecasts() -> None:
    client = TestClient(app)
    client.post("/api/forecasts/run", params={"force_demo": True})
    r = client.get("/api/forecasts/latest", params={"horizon": 30, "threshold": 5.0})
    cid = r.json()["items"][0]["cell_id"]
    r2 = client.get(f"/api/forecasts/area/{cid}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["area"]["cell_id"] == cid
    assert len(body["forecasts"]) == 16  # 4 horizons × 4 thresholds


def test_area_404_for_unknown_cell() -> None:
    client = TestClient(app)
    r = client.get("/api/forecasts/area/UNKNOWN")
    assert r.status_code == 404


def test_format_sentence_indonesian() -> None:
    s = format_sentence(
        {"full_label": "Sulawesi Tengah - Palu"},
        probability=0.124,
        horizon_days=30,
        mag_threshold=5.0,
    )
    assert s == "Sulawesi Tengah - Palu, 12.4% probabilitas M≥5.0 dalam 30 hari"


def test_invalid_horizon_returns_400() -> None:
    client = TestClient(app)
    r = client.get("/api/forecasts/latest", params={"horizon": 99, "threshold": 5.0})
    assert r.status_code == 400
