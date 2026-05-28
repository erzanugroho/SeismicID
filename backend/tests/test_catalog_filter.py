"""Tests for Mc-based catalog filter."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_filter_below_mc_drops_subthreshold_events() -> None:
    from backend.app.ml.catalog_filter import filter_below_mc

    df = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "magnitude": [3.8, 4.2, 4.6, 5.1],
            "time": pd.to_datetime(["2024-01-01"] * 4, utc=True),
        }
    )
    out = filter_below_mc(df, mc=4.5)
    assert sorted(out["event_id"].tolist()) == ["c", "d"]


def test_filter_below_mc_preserves_audit_attrs() -> None:
    from backend.app.ml.catalog_filter import filter_below_mc

    df = pd.DataFrame({"event_id": ["a", "b"], "magnitude": [3.0, 5.0]})
    out = filter_below_mc(df, mc=4.5)
    assert out.attrs.get("mc_filter_dropped") == 1
    assert out.attrs.get("mc_value") == 4.5


def test_filter_below_mc_handles_empty_input() -> None:
    from backend.app.ml.catalog_filter import filter_below_mc

    out = filter_below_mc(pd.DataFrame(), mc=4.5)
    assert out.empty


def test_filter_below_mc_no_magnitude_column_passes_through() -> None:
    from backend.app.ml.catalog_filter import filter_below_mc

    df = pd.DataFrame({"event_id": ["a"]})
    out = filter_below_mc(df, mc=4.5)
    assert len(out) == 1


def test_filter_below_mc_nan_values_are_dropped() -> None:
    from backend.app.ml.catalog_filter import filter_below_mc

    df = pd.DataFrame({"event_id": ["a", "b", "c"], "magnitude": [4.7, np.nan, 5.0]})
    out = filter_below_mc(df, mc=4.5)
    assert sorted(out["event_id"].tolist()) == ["a", "c"]
