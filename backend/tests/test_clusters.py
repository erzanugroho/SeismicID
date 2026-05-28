"""Tests for cluster aggregation service + /api/forecasts cluster endpoints."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.forecast_service import (
    _aggregate_cluster,
    _cluster_id_from_label,
    format_cluster_sentence,
    get_cluster_forecasts,
    get_top_clusters,
    run_forecast,
)


# --------------------------------------------------------------------------- #
# Pure-function unit tests (no DB needed)
# --------------------------------------------------------------------------- #


def test_cluster_id_slug_is_url_safe_and_stable() -> None:
    assert _cluster_id_from_label("Sulawesi Tengah - Palu") == "sulawesi-tengah-palu"
    assert (
        _cluster_id_from_label("Lepas Pantai Sumatera Barat - dekat Padang")
        == "lepas-pantai-sumatera-barat-dekat-padang"
    )
    # Non-ASCII fold-down
    assert _cluster_id_from_label("Café São Paulo!!!") == "cafe-sao-paulo"
    # Empty / null safety
    assert _cluster_id_from_label("") == "unknown"
    assert _cluster_id_from_label("---") == "unknown"


def test_aggregate_cluster_metrics_obey_ordering_chain() -> None:
    """Synthetic members → verify prob_any_cell ≥ prob_max ≥ prob_top3_mean ≥ prob_mean."""
    members = [
        {
            "cell_id": f"C{i:03d}",
            "probability": p,
            "lat": -1.0 + i * 0.1,
            "lon": 119.0 + i * 0.1,
            "lat_min": -1.0 + i * 0.1 - 0.25,
            "lat_max": -1.0 + i * 0.1 + 0.25,
            "lon_min": 119.0 + i * 0.1 - 0.25,
            "lon_max": 119.0 + i * 0.1 + 0.25,
            "province": "Sulawesi Tengah",
            "subregion": "Palu",
            "region_macro": "Sulawesi",
            "is_offshore": 0,
            "full_label": "Sulawesi Tengah - Palu",
            "computed_at": "2026-05-25T00:00:00+00:00",
        }
        for i, p in enumerate([0.05, 0.03, 0.02, 0.01, 0.005])
    ]
    c = _aggregate_cluster("Sulawesi Tengah - Palu", members)

    assert c["n_cells"] == 5
    assert c["cluster_id"] == "sulawesi-tengah-palu"
    assert c["prob_max"] == 0.05
    # top-3 mean = mean of [0.05, 0.03, 0.02] = 0.0333…
    assert math.isclose(c["prob_top3_mean"], (0.05 + 0.03 + 0.02) / 3, abs_tol=1e-9)
    # mean = (0.05+0.03+0.02+0.01+0.005)/5 = 0.023
    assert math.isclose(c["prob_mean"], 0.023, abs_tol=1e-9)
    # any-cell = 1 - prod(1-p)
    expected_any = 1.0 - math.prod(1.0 - p for p in [0.05, 0.03, 0.02, 0.01, 0.005])
    assert math.isclose(c["prob_any_cell"], expected_any, abs_tol=1e-9)

    # The fundamental ordering chain
    assert c["prob_any_cell"] >= c["prob_max"]
    assert c["prob_max"] >= c["prob_top3_mean"]
    assert c["prob_top3_mean"] >= c["prob_mean"]

    # bbox is union of member bboxes
    assert c["lat_min"] == min(m["lat_min"] for m in members)
    assert c["lat_max"] == max(m["lat_max"] for m in members)
    assert len(c["top_cells"]) == 3
    assert [tc["probability"] for tc in c["top_cells"]] == [0.05, 0.03, 0.02]


def test_aggregate_cluster_handles_single_cell() -> None:
    members = [
        {
            "cell_id": "C001",
            "probability": 0.07,
            "lat": -1.0,
            "lon": 119.0,
            "lat_min": -1.25,
            "lat_max": -0.75,
            "lon_min": 118.75,
            "lon_max": 119.25,
            "province": "Sulawesi Tengah",
            "subregion": "Palu",
            "region_macro": "Sulawesi",
            "is_offshore": 1,
            "full_label": "Lepas Pantai Sulawesi Tengah - dekat Palu",
            "computed_at": "2026-05-25T00:00:00+00:00",
        }
    ]
    c = _aggregate_cluster("Lepas Pantai Sulawesi Tengah - dekat Palu", members)
    assert c["n_cells"] == 1
    assert c["prob_max"] == 0.07
    assert c["prob_top3_mean"] == 0.07
    assert c["prob_mean"] == 0.07
    assert math.isclose(c["prob_any_cell"], 0.07, abs_tol=1e-9)
    assert c["is_offshore"] is True
    assert c["n_offshore_cells"] == 1


def test_aggregate_cluster_clips_saturated_probabilities() -> None:
    """p=1.0 must not blow up the cumulative log calc."""
    members = [
        {
            "cell_id": "Cx",
            "probability": 1.0,  # would otherwise yield log(0) = -inf
            "lat": 0.0,
            "lon": 100.0,
            "lat_min": -0.25,
            "lat_max": 0.25,
            "lon_min": 99.75,
            "lon_max": 100.25,
            "full_label": "Test",
            "province": "Test",
            "subregion": "Test",
            "region_macro": "Test",
            "is_offshore": 0,
            "computed_at": None,
        }
    ]
    c = _aggregate_cluster("Test", members)
    assert math.isfinite(c["prob_any_cell"])
    assert 0.0 <= c["prob_any_cell"] <= 1.0


# --------------------------------------------------------------------------- #
# Service-level integration (uses demo seed)
# --------------------------------------------------------------------------- #


def test_get_cluster_forecasts_returns_clusters_after_demo_run() -> None:
    run_forecast(force_demo=True)
    clusters = get_cluster_forecasts(horizon_days=30, mag_threshold=5.0)
    assert clusters, "demo seed should yield non-empty clusters"

    # Every cluster must satisfy invariants and basic shape
    for c in clusters:
        n = c["n_cells"]
        assert n >= 1
        assert c["cluster_id"]
        assert c["cluster_label"]
        assert len(c["cell_ids"]) == n
        assert len(c["top_cells"]) <= min(3, n)
        for k in ("prob_max", "prob_top3_mean", "prob_any_cell", "prob_mean"):
            v = c[k]
            assert 0.0 <= v <= 1.0, f"{k}={v} outside [0,1]"
        # Ordering chain
        assert c["prob_any_cell"] + 1e-9 >= c["prob_max"]
        assert c["prob_max"] + 1e-9 >= c["prob_top3_mean"]
        assert c["prob_top3_mean"] + 1e-9 >= c["prob_mean"]


def test_get_top_clusters_default_sort_by_top3_mean() -> None:
    run_forecast(force_demo=True)
    top = get_top_clusters(horizon_days=30, mag_threshold=5.0, n=10)
    assert 0 < len(top) <= 10
    probs = [c["prob_top3_mean"] for c in top]
    assert probs == sorted(probs, reverse=True), "must be sorted desc by top3_mean"


def test_get_top_clusters_sort_by_max() -> None:
    run_forecast(force_demo=True)
    top = get_top_clusters(horizon_days=30, mag_threshold=5.0, n=10, sort_by="max")
    probs = [c["prob_max"] for c in top]
    assert probs == sorted(probs, reverse=True)


def test_get_top_clusters_sort_by_any_cell() -> None:
    run_forecast(force_demo=True)
    top = get_top_clusters(horizon_days=30, mag_threshold=5.0, n=10, sort_by="any_cell")
    probs = [c["prob_any_cell"] for c in top]
    assert probs == sorted(probs, reverse=True)


def test_get_cluster_forecasts_filters_by_region_macro() -> None:
    run_forecast(force_demo=True)
    clusters = get_cluster_forecasts(
        horizon_days=30, mag_threshold=5.0, region_macro="Sulawesi"
    )
    assert clusters
    assert all(c["region_macro"] == "Sulawesi" for c in clusters)


def test_get_cluster_forecasts_rejects_invalid_sort_by() -> None:
    run_forecast(force_demo=True)
    try:
        get_cluster_forecasts(horizon_days=30, mag_threshold=5.0, sort_by="garbage")
    except ValueError as e:
        assert "sort_by" in str(e)
    else:
        raise AssertionError("expected ValueError for invalid sort_by")


def test_format_cluster_sentence_shape_indonesian() -> None:
    cluster = {
        "cluster_label": "Sulawesi Tengah - Palu",
        "n_cells": 6,
        "prob_top3_mean": 0.0421,
        "prob_max": 0.058,
        "prob_any_cell": 0.084,
        "prob_mean": 0.020,
    }
    s = format_cluster_sentence(cluster, horizon_days=30, mag_threshold=5.0)
    assert "Sulawesi Tengah - Palu" in s
    assert "(6 cell)" in s
    assert "4.2%" in s
    assert "rata-rata 3 cell tertinggi" in s
    assert "M≥5.0" in s
    assert "30 hari" in s

    s_max = format_cluster_sentence(cluster, horizon_days=30, mag_threshold=5.0, sort_by="max")
    assert "5.8%" in s_max
    assert "cell tertinggi" in s_max


# --------------------------------------------------------------------------- #
# HTTP endpoint smoke tests
# --------------------------------------------------------------------------- #


def test_top_clusters_endpoint_default_returns_sorted_top3_mean() -> None:
    client = TestClient(app)
    client.post(
        "/api/forecasts/run",
        params={"force_demo": True},
        headers={"Authorization": "Bearer test-admin-token"},
    )
    r = client.get(
        "/api/forecasts/top-clusters",
        params={"n": 5, "horizon": 30, "threshold": 5.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sort_by"] == "top3_mean"
    assert body["horizon_days"] == 30
    assert body["mag_threshold"] == 5.0
    assert len(body["items"]) <= 5
    assert len(body["sentences"]) == len(body["items"])
    if body["items"]:
        first = body["items"][0]
        for k in ("cluster_id", "cluster_label", "n_cells",
                  "prob_max", "prob_top3_mean", "prob_any_cell", "prob_mean",
                  "top_cells", "cell_ids", "lat_min", "lat_max",
                  "lon_min", "lon_max"):
            assert k in first, f"missing key {k}"
        assert first["n_cells"] == len(first["cell_ids"])


def test_top_clusters_endpoint_supports_sort_by_max() -> None:
    client = TestClient(app)
    client.post(
        "/api/forecasts/run",
        params={"force_demo": True},
        headers={"Authorization": "Bearer test-admin-token"},
    )
    r = client.get(
        "/api/forecasts/top-clusters",
        params={"n": 5, "horizon": 30, "threshold": 5.0, "sort_by": "max"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sort_by"] == "max"
    probs = [c["prob_max"] for c in body["items"]]
    assert probs == sorted(probs, reverse=True)


def test_top_clusters_invalid_sort_by_returns_400() -> None:
    client = TestClient(app)
    r = client.get(
        "/api/forecasts/top-clusters",
        params={"horizon": 30, "threshold": 5.0, "sort_by": "garbage"},
    )
    assert r.status_code == 400


def test_clusters_latest_endpoint_filters_by_region() -> None:
    client = TestClient(app)
    client.post(
        "/api/forecasts/run",
        params={"force_demo": True},
        headers={"Authorization": "Bearer test-admin-token"},
    )
    r = client.get(
        "/api/forecasts/clusters-latest",
        params={"horizon": 30, "threshold": 5.0, "region_macro": "Sulawesi"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert all(c["region_macro"] == "Sulawesi" for c in body["items"])


def test_clusters_latest_min_probability_filters_low_clusters() -> None:
    client = TestClient(app)
    client.post(
        "/api/forecasts/run",
        params={"force_demo": True},
        headers={"Authorization": "Bearer test-admin-token"},
    )
    r = client.get(
        "/api/forecasts/clusters-latest",
        params={
            "horizon": 30,
            "threshold": 5.0,
            "min_probability": 0.001,
            "sort_by": "top3_mean",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert all(c["prob_top3_mean"] >= 0.001 for c in body["items"])


def test_clusters_consistent_across_endpoints_same_horizon_threshold() -> None:
    """top-clusters[:n] should equal clusters-latest[:n] under the same sort."""
    client = TestClient(app)
    client.post(
        "/api/forecasts/run",
        params={"force_demo": True},
        headers={"Authorization": "Bearer test-admin-token"},
    )
    top_r = client.get(
        "/api/forecasts/top-clusters",
        params={"n": 8, "horizon": 30, "threshold": 5.0, "sort_by": "top3_mean"},
    )
    latest_r = client.get(
        "/api/forecasts/clusters-latest",
        params={"horizon": 30, "threshold": 5.0, "sort_by": "top3_mean"},
    )
    top_ids = [c["cluster_id"] for c in top_r.json()["items"]]
    latest_ids = [c["cluster_id"] for c in latest_r.json()["items"][:8]]
    assert top_ids == latest_ids
