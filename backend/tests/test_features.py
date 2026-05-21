"""Tests for core feature engineering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.app.core.grid import generate_grid
from backend.app.features.builder import (
    assign_cell_id,
    build_features_for_snapshots,
    compute_window_features,
    default_snapshots,
    feature_columns,
)
from backend.app.features.spatial import neighbor_map


@pytest.fixture
def cells_subset():
    """Small grid subset around Sulawesi for fast tests."""
    all_cells = generate_grid()
    return [c for c in all_cells if -2 <= c.lat <= 2 and 119 <= c.lon <= 121]


def _synth_events(t0: datetime, n: int, lat: float, lon: float, mag_floor: float = 4.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        [
            {
                "event_id": f"ev_{i}",
                "time": t0 + timedelta(days=int(rng.integers(0, 60))),
                "lat": lat + rng.normal(0, 0.05),
                "lon": lon + rng.normal(0, 0.05),
                "depth": float(rng.uniform(5, 50)),
                "magnitude": mag_floor + rng.exponential(0.4),
                "source": "usgs",
            }
            for i in range(n)
        ]
    )


def test_compute_window_features_keys() -> None:
    df = pd.DataFrame(
        [
            {"event_id": "a", "time": datetime(2024, 1, 1, tzinfo=UTC), "lat": -0.9, "lon": 119.87, "depth": 10.0, "magnitude": 5.0},
            {"event_id": "b", "time": datetime(2024, 1, 5, tzinfo=UTC), "lat": -0.9, "lon": 119.87, "depth": 15.0, "magnitude": 4.5},
        ]
    )
    snap = datetime(2024, 1, 31, tzinfo=UTC)
    feats = compute_window_features(df, snap, mc=4.0)
    expected = set(feature_columns()) - {"neighbor_event_count_30d_mean", "neighbor_max_mag_30d_max"}
    assert expected.issubset(feats.keys())
    assert feats["event_count_30d"] == 2
    assert feats["max_mag_30d"] == 5.0


def test_compute_features_zero_when_outside_window() -> None:
    df = pd.DataFrame(
        [{"event_id": "old", "time": datetime(2020, 1, 1, tzinfo=UTC), "lat": -1, "lon": 120, "depth": 10, "magnitude": 5.0}]
    )
    feats = compute_window_features(df, datetime(2024, 1, 1, tzinfo=UTC), mc=4.0)
    assert feats["event_count_30d"] == 0
    assert feats["max_mag_30d"] == 0.0
    assert feats["time_since_last_M5_days"] > 365  # many days passed


def test_assign_cell_id_basic(cells_subset) -> None:
    """An event near (-0.9, 119.87) lands in some Sulawesi cell."""
    events = pd.DataFrame([{"lat": -0.9, "lon": 119.87, "magnitude": 5.0, "depth": 10.0, "time": datetime(2024, 1, 1, tzinfo=UTC), "event_id": "x"}])
    out = assign_cell_id(events, cells_subset)
    assert out["cell_id"].notna().all()


def test_neighbor_map_8_per_cell(cells_subset) -> None:
    nbrs = neighbor_map(cells_subset, k=4)
    for cid, ns in nbrs.items():
        assert len(ns) <= 4
        assert cid not in ns


def test_default_snapshots_count() -> None:
    snaps = default_snapshots(
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 3, 1, tzinfo=UTC),
        freq_days=7,
    )
    assert 8 <= len(snaps) <= 10


def test_build_features_smoke(cells_subset) -> None:
    """End-to-end: 100 synthetic events → features dataframe shape."""
    events = _synth_events(datetime(2024, 1, 1, tzinfo=UTC), n=100, lat=-0.9, lon=119.87)
    snaps = [datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC)]
    df = build_features_for_snapshots(events, snaps, cells=cells_subset)
    expected_rows = len(cells_subset) * len(snaps)
    assert len(df) == expected_rows
    for col in feature_columns():
        assert col in df.columns
