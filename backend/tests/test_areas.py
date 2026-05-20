"""Tests for /api/areas endpoint and bootstrap."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def test_get_areas_bootstraps_lazily() -> None:
    client = TestClient(app)
    response = client.get("/api/areas")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 0
    assert "items" in body
    sample = body["items"][0]
    for key in ("cell_id", "lat", "lon", "lat_min", "lat_max", "full_label", "region_macro"):
        assert key in sample


def test_areas_filter_by_region_macro() -> None:
    client = TestClient(app)
    # Trigger bootstrap
    client.get("/api/areas")
    # Filter Sulawesi
    response = client.get("/api/areas", params={"region_macro": "Sulawesi"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 0
    assert all(item["region_macro"] == "Sulawesi" for item in body["items"])


def test_areas_filter_by_province() -> None:
    client = TestClient(app)
    client.get("/api/areas")  # ensure bootstrapped
    response = client.get("/api/areas", params={"province": "Sulawesi Tengah"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 0
    palu_cells = [it for it in body["items"] if it["subregion"] == "Palu"]
    assert len(palu_cells) >= 1


def test_bootstrap_endpoint_is_idempotent() -> None:
    client = TestClient(app)
    r1 = client.post("/api/areas/bootstrap", headers={"Authorization": "Bearer test-admin-token"})
    assert r1.status_code == 200
    total = r1.json()["total"]

    r2 = client.post("/api/areas/bootstrap", headers={"Authorization": "Bearer test-admin-token"})
    assert r2.status_code == 200
    assert r2.json()["total"] == total  # second call should not duplicate
