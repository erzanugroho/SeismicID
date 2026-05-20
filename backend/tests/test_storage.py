"""Tests for storage layer (SQLite + Parquet)."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from backend.app.data.catalog import (
    append_historical_events,
    archive_forecast,
    list_forecast_archive_days,
    read_forecast_archive,
    read_historical_events,
    read_training_features,
    storage_summary,
    write_declustered_events,
    write_training_features,
)
from backend.app.db.sqlite import get_connection, migrate


# ---------- SQLite layer ----------


def test_migrate_creates_all_tables() -> None:
    migrate()
    expected = {
        "area_labels",
        "current_forecasts",
        "realtime_events",
        "scheduler_runs",
        "model_metadata",
        "evaluation_results",
    }
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = {row["name"] for row in cur.fetchall()}
    assert expected.issubset(tables)


def test_migrate_is_idempotent() -> None:
    migrate()
    migrate()  # should not raise
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS n FROM area_labels")
        # Just confirms the table is queryable; row count irrelevant here.
        assert cur.fetchone()["n"] >= 0


def test_realtime_events_roundtrip() -> None:
    migrate()
    sample = (
        "ev1",
        "2024-01-01T00:00:00Z",
        -0.9,
        119.87,
        10.0,
        5.5,
        "mw",
        "usgs",
        "Sulawesi Tengah",
        "{}",
    )
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO realtime_events
               (event_id, time, lat, lon, depth, magnitude, mag_type, source, place, raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            sample,
        )
        cur = conn.execute("SELECT * FROM realtime_events WHERE event_id = ?", ("ev1",))
        row = cur.fetchone()
    assert row is not None
    assert row["magnitude"] == 5.5
    assert row["source"] == "usgs"


# ---------- Parquet layer ----------


@pytest.fixture
def sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "us_a1",
                "time": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                "lat": -0.9,
                "lon": 119.87,
                "depth": 10.0,
                "magnitude": 5.5,
                "mag_type": "mw",
                "source": "usgs",
                "place": "Sulawesi Tengah",
            },
            {
                "event_id": "us_a2",
                "time": datetime(2024, 1, 2, 8, 30, tzinfo=timezone.utc),
                "lat": -7.0,
                "lon": 110.0,
                "depth": 30.0,
                "magnitude": 4.2,
                "mag_type": "mb",
                "source": "usgs",
                "place": "Jawa Tengah",
            },
        ]
    )


def test_historical_append_and_read(sample_events: pd.DataFrame) -> None:
    n = append_historical_events(sample_events)
    assert n == 2

    df = read_historical_events()
    assert len(df) == 2
    assert set(df["event_id"]) == {"us_a1", "us_a2"}


def test_historical_append_dedup(sample_events: pd.DataFrame) -> None:
    append_historical_events(sample_events)
    n = append_historical_events(sample_events)  # same data again
    assert n == 0  # no new rows
    assert len(read_historical_events()) == 2


def test_historical_filter_min_mag(sample_events: pd.DataFrame) -> None:
    append_historical_events(sample_events)
    df = read_historical_events(min_mag=5.0)
    assert len(df) == 1
    assert df.iloc[0]["event_id"] == "us_a1"


def test_historical_filter_bbox(sample_events: pd.DataFrame) -> None:
    append_historical_events(sample_events)
    # Sulawesi-only bbox
    df = read_historical_events(bbox=(-3.0, 119.0, 1.0, 122.0))
    assert len(df) == 1
    assert df.iloc[0]["event_id"] == "us_a1"


def test_declustered_roundtrip(sample_events: pd.DataFrame) -> None:
    df = sample_events.copy()
    df["is_mainshock"] = [True, False]
    df["cluster_id"] = [0, 0]
    n = write_declustered_events(df)
    assert n == 2


def test_training_features_roundtrip() -> None:
    feats = pd.DataFrame(
        {"cell_id": ["c1", "c2"], "feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]}
    )
    n = write_training_features(feats)
    assert n == 2
    out = read_training_features()
    assert len(out) == 2
    assert "feat_a" in out.columns


def test_forecast_archive_roundtrip() -> None:
    df = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "horizon_days": [30, 30],
            "mag_threshold": [5.0, 5.0],
            "probability": [0.12, 0.05],
        }
    )
    day = date(2025, 1, 15)
    archive_forecast(df, day=day)
    days = list_forecast_archive_days()
    assert day in days
    out = read_forecast_archive(day)
    assert len(out) == 2


def test_storage_summary() -> None:
    summary = storage_summary()
    for key in ("historical", "declustered", "training", "archive_days"):
        assert key in summary
