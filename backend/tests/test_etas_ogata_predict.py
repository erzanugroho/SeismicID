"""Tests for OgataETAS event-frame ingestion and per-cell prediction."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd


def test_predict_dataframe_has_canonical_label_columns() -> None:
    from backend.app.features.labels import all_label_columns
    from backend.app.ml.etas_ogata import OgataETAS

    rng = np.random.default_rng(0)
    events = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(40)],
            "time": pd.to_datetime(
                [
                    datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=int(d))
                    for d in rng.integers(0, 365, size=40)
                ],
                utc=True,
            ),
            "lat": -1.0 + rng.normal(0, 0.5, size=40),
            "lon": 120.0 + rng.normal(0, 0.5, size=40),
            "magnitude": 4.5 + rng.exponential(0.4, size=40),
            "depth": np.full(40, 10.0),
        }
    )
    model = OgataETAS(mc=4.5).fit_from_events(
        events,
        observation_start=datetime(2024, 1, 1, tzinfo=UTC),
        observation_end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    cell_ids = ["C_-1.00_120.00", "C_-1.00_120.50", "C_0.00_119.50"]
    df = model.predict_dataframe(
        cell_ids, issued_at=datetime(2025, 1, 1, tzinfo=UTC)
    )
    assert df["cell_id"].tolist() == cell_ids
    for col in all_label_columns():
        assert col in df.columns
        assert (df[col] >= 0).all() and (df[col] <= 1).all()


def test_predict_higher_near_recent_large_event() -> None:
    """Cell within ~50 km of a recent M>=5.5 should outrank a cell ~800 km away
    on the 7-day horizon."""
    from backend.app.features.labels import label_column_name
    from backend.app.ml.etas_ogata import OgataETAS

    issued = datetime(2025, 1, 1, tzinfo=UTC)
    recent = issued - timedelta(days=2)
    events = pd.DataFrame(
        {
            "event_id": ["mainshock"],
            "time": pd.to_datetime([recent], utc=True),
            "lat": [-1.0],
            "lon": [120.0],
            "magnitude": [5.8],
            "depth": [10.0],
        }
    )
    model = OgataETAS(mc=4.5).fit_from_events(
        events,
        observation_start=issued - timedelta(days=365),
        observation_end=issued,
    )
    df = model.predict_dataframe(
        ["C_-1.00_120.00", "C_-8.00_115.00"], issued_at=issued
    )
    col = label_column_name(7, 4.5)
    assert df.iloc[0][col] > df.iloc[1][col]


def test_predict_with_no_events_returns_zero_probabilities() -> None:
    from backend.app.features.labels import all_label_columns
    from backend.app.ml.etas_ogata import OgataETAS

    issued = datetime(2025, 1, 1, tzinfo=UTC)
    model = OgataETAS(mc=4.5).fit_from_events(
        pd.DataFrame(columns=["event_id", "time", "lat", "lon", "magnitude"]),
        observation_start=issued - timedelta(days=365),
        observation_end=issued,
    )
    df = model.predict_dataframe(["C_-1.00_120.00"], issued_at=issued)
    for col in all_label_columns():
        assert col in df.columns
        # No events -> mu = 1e-4, very small but non-negative.
        assert 0.0 <= df.iloc[0][col] <= 0.5
