"""Tests for data ingestion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import respx
from fastapi.testclient import TestClient
from httpx import Response

from backend.app.data.ingest import dedup_events, ingest_realtime
from backend.app.data.sources.base import Event
from backend.app.data.sources.bmkg import BMKGSource
from backend.app.data.sources.usgs import USGSSource
from backend.app.main import app

USGS_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "id": "us123",
            "geometry": {"type": "Point", "coordinates": [119.87, -0.9, 10]},
            "properties": {
                "mag": 5.5, "magType": "mw",
                "place": "Sulawesi Tengah",
                "time": int(datetime(2024, 1, 1, 12, 0, tzinfo=UTC).timestamp() * 1000),
            },
        },
        {
            "id": "us456",
            "geometry": {"type": "Point", "coordinates": [110.0, -7.0, 30]},
            "properties": {
                "mag": 4.2, "magType": "mb",
                "place": "Jawa Tengah",
                "time": int(datetime(2024, 1, 2, 8, 30, tzinfo=UTC).timestamp() * 1000),
            },
        },
    ],
}

BMKG_AUTO = {
    "Infogempa": {
        "gempa": {
            "DateTime": "2024-01-01T12:00:30+00:00",
            "Lintang": "0.92 LS",
            "Bujur": "119.85 BT",
            "Magnitude": "5.4",
            "Kedalaman": "10 km",
            "Wilayah": "Sulawesi Tengah",
        }
    }
}

BMKG_TERKINI = {
    "Infogempa": {
        "gempa": [
            {  # Same as USGS us123 within dedup window
                "DateTime": "2024-01-01T12:00:30+00:00",
                "Lintang": "0.92 LS", "Bujur": "119.85 BT",
                "Magnitude": "5.4", "Kedalaman": "10 km", "Wilayah": "x",
            },
            {  # New unique BMKG event
                "DateTime": "2024-03-10T05:00:00+00:00",
                "Lintang": "1.50 LS", "Bujur": "100.00 BT",
                "Magnitude": "3.8", "Kedalaman": "5 km", "Wilayah": "Sumatera Barat",
            },
        ]
    }
}

BMKG_DIRASAKAN = {"Infogempa": {"gempa": []}}


def _t(year: int = 2024, month: int = 1, day: int = 1, hour: int = 12) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


@respx.mock
def test_usgs_parses_geojson() -> None:
    respx.get("https://earthquake.usgs.gov/fdsnws/event/1/query").mock(return_value=Response(200, json=USGS_GEOJSON))
    src = USGSSource()
    events = src.fetch(_t(), _t(month=2))
    assert len(events) == 2
    assert {e.source for e in events} == {"usgs"}
    assert events[0].event_id == "usgs_us123"
    assert events[0].magnitude == 5.5


@respx.mock
def test_bmkg_parses_lat_lon_with_hemisphere() -> None:
    respx.get("https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json").mock(return_value=Response(200, json=BMKG_AUTO))
    respx.get("https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json").mock(return_value=Response(200, json=BMKG_TERKINI))
    respx.get("https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json").mock(return_value=Response(200, json=BMKG_DIRASAKAN))
    src = BMKGSource()
    events = src.fetch()
    # autogempa(1) + terkini(2) but the dup with autogempa shares same id → 2 unique within BMKG
    assert len(events) == 2
    e0 = next(e for e in events if abs(e.lat + 0.92) < 0.01)
    assert e0.lat < 0  # parsed LS as negative
    assert e0.lon > 0  # BT positive
    assert e0.source == "bmkg"


def test_dedup_keeps_usgs_drops_bmkg_match() -> None:
    t1 = _t()
    usgs = Event("usgs_x", t1, -0.9, 119.87, 10.0, 5.5, "mw", "usgs", "x", None)
    bmkg = Event("bmkg_y", t1 + timedelta(seconds=30), -0.92, 119.85, 10.0, 5.4, None, "bmkg", "y", None)
    bmkg2 = Event("bmkg_unique", _t(month=3), -1.5, 100.0, 5.0, 3.8, None, "bmkg", "z", None)
    out = dedup_events([usgs, bmkg, bmkg2])
    sources = {e.event_id for e in out}
    assert "usgs_x" in sources
    assert "bmkg_y" not in sources
    assert "bmkg_unique" in sources
    assert len(out) == 2


def test_dedup_independent_events_kept() -> None:
    e1 = Event("a", _t(month=1), -0.9, 119.0, 10.0, 5.0, None, "usgs", "x", None)
    e2 = Event("b", _t(month=2), 1.0, 100.0, 5.0, 4.0, None, "usgs", "y", None)
    out = dedup_events([e1, e2])
    assert len(out) == 2


@respx.mock
def test_ingest_realtime_end_to_end() -> None:
    respx.get("https://earthquake.usgs.gov/fdsnws/event/1/query").mock(return_value=Response(200, json=USGS_GEOJSON))
    respx.get("https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json").mock(return_value=Response(200, json=BMKG_AUTO))
    respx.get("https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json").mock(return_value=Response(200, json=BMKG_TERKINI))
    respx.get("https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json").mock(return_value=Response(200, json=BMKG_DIRASAKAN))
    out = ingest_realtime(lookback_hours=24 * 365 * 5)  # wide so the 2024 events are included by USGS mock
    assert out["raw"] >= 3
    assert out["stored"] >= 1


@respx.mock
def test_events_endpoint_returns_data() -> None:
    # Insert a single event directly via mocking ingest endpoint
    respx.get("https://earthquake.usgs.gov/fdsnws/event/1/query").mock(return_value=Response(200, json=USGS_GEOJSON))
    respx.get("https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json").mock(return_value=Response(200, json=BMKG_AUTO))
    respx.get("https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json").mock(return_value=Response(200, json=BMKG_TERKINI))
    respx.get("https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json").mock(return_value=Response(200, json=BMKG_DIRASAKAN))

    client = TestClient(app)
    client.post(
        "/api/events/ingest",
        params={"lookback_hours": 24 * 365 * 5},
        headers={"Authorization": "Bearer test-admin-token"},
    )
    r = client.get("/api/events", params={"days": 365 * 5})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert all("event_id" in it for it in body["items"])
