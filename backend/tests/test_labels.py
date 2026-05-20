"""Tests for multi-output labels."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.app.core.grid import generate_grid
from backend.app.features.labels import (
    HORIZONS,
    THRESHOLDS,
    all_label_columns,
    build_labels,
    label_column_name,
    positive_rates,
    time_split,
)


@pytest.fixture
def cells_subset():
    return [c for c in generate_grid() if -2 <= c.lat <= 2 and 119 <= c.lon <= 121]


def test_label_column_naming() -> None:
    assert label_column_name(30, 5.0) == "label_h30_m50"
    assert label_column_name(7, 4.5) == "label_h7_m45"


def test_all_label_columns_count() -> None:
    cols = all_label_columns()
    assert len(cols) == len(HORIZONS) * len(THRESHOLDS)


def test_build_labels_positive_when_future_event(cells_subset) -> None:
    """Snapshot at T0; event at T0+10d in same cell with M=5.5 → h14_m55=1."""
    snap = datetime(2024, 1, 1, tzinfo=timezone.utc)
    future_event_time = snap + timedelta(days=10)
    events = pd.DataFrame(
        [
            {
                "event_id": "x",
                "time": future_event_time,
                "lat": -0.9,
                "lon": 119.87,
                "magnitude": 5.5,
                "depth": 10.0,
                "source": "usgs",
            }
        ]
    )
    labels = build_labels(events, [snap], cells=cells_subset)
    # Find the cell that contains -0.9, 119.87
    target = labels[labels["cell_id"].str.contains("m00")].copy() if labels["cell_id"].str.contains("m00").any() else None
    assert "label_h14_m55" in labels.columns
    # In some cell within the buffer the label must be 1
    assert (labels["label_h14_m55"] == 1).any()


def test_build_labels_negative_when_no_future_event(cells_subset) -> None:
    snap = datetime(2024, 1, 1, tzinfo=timezone.utc)
    events = pd.DataFrame(columns=["event_id", "time", "lat", "lon", "magnitude", "depth", "source"])
    labels = build_labels(events, [snap], cells=cells_subset)
    for col in all_label_columns():
        assert (labels[col] == 0).all()


def test_label_horizon_isolation(cells_subset) -> None:
    """Event at T+50 → label_h7_*=0 but label_h60_*=1."""
    snap = datetime(2024, 1, 1, tzinfo=timezone.utc)
    events = pd.DataFrame(
        [
            {
                "event_id": "y",
                "time": snap + timedelta(days=50),
                "lat": -0.9,
                "lon": 119.87,
                "magnitude": 5.5,
                "depth": 10.0,
                "source": "usgs",
            }
        ]
    )
    labels = build_labels(events, [snap], cells=cells_subset)
    assert (labels["label_h7_m55"] == 0).all()
    assert (labels["label_h60_m55"] == 1).any()


def test_positive_rates_returns_floats() -> None:
    df = pd.DataFrame({col: [0, 1, 1, 0] for col in all_label_columns()})
    rates = positive_rates(df)
    assert all(isinstance(v, float) and 0 <= v <= 1 for v in rates.values())


def test_time_split() -> None:
    df = pd.DataFrame(
        {
            "cell_id": ["a"] * 5,
            "snapshot": [
                "2019-06-01T00:00:00+00:00",
                "2020-06-01T00:00:00+00:00",
                "2021-06-01T00:00:00+00:00",
                "2022-06-01T00:00:00+00:00",
                "2024-06-01T00:00:00+00:00",
            ],
        }
    )
    train, val, test = time_split(df)
    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 2
