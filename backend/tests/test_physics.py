"""Tests for physics-informed features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from backend.app.features.physics import (
    fault_type_to_int,
    static_physics_features,
    z_value_quiescence,
)


def test_palu_near_palu_koro_fault() -> None:
    """Palu (-0.9, 119.87) should be near the Palu-Koro transform fault."""
    f = static_physics_features(-0.9, 119.87)
    assert f["fault_type"] in ("transform", "subduction")
    assert f["nearest_fault_km"] < 80  # within 80 km


def test_mentawai_in_subduction_with_slab_depth() -> None:
    """Mentawai (-2.5, 99.5) is in front of Sunda megathrust with slab depth."""
    f = static_physics_features(-2.5, 99.5)
    assert f["slab_depth_km"] is not None
    assert f["slab_depth_km"] >= 0


def test_kalimantan_no_slab_or_far_fault() -> None:
    """Central Kalimantan (~ -2, 113) is intraplate; no shallow slab."""
    f = static_physics_features(-2.0, 113.0)
    assert f["fault_type"] in ("subduction", "transform", "reverse", "normal")
    # Slab may be deep or None
    if f["slab_depth_km"] is not None:
        assert f["slab_depth_km"] > 100


def test_z_value_zero_for_empty_events() -> None:
    df = pd.DataFrame(columns=["time", "lat", "lon", "magnitude"])
    z = z_value_quiescence(df, pd.Timestamp("2024-01-01", tz="UTC"))
    assert z == 0.0


def test_z_value_positive_after_quiescence() -> None:
    """High activity in reference + zero in test window → positive Z."""
    rng = np.random.default_rng(0)
    base = datetime(2018, 1, 1, tzinfo=UTC)
    # Reference window has many events; test window has none
    rows = []
    for _i in range(500):
        rows.append({"time": base + timedelta(days=int(rng.integers(0, 365 * 4))), "lat": -1, "lon": 120, "magnitude": 4.0 + rng.exponential(0.3)})
    df = pd.DataFrame(rows)
    z = z_value_quiescence(df, pd.Timestamp("2024-01-01", tz="UTC"))
    assert z > 0


def test_fault_type_encoding() -> None:
    assert fault_type_to_int("subduction") == 1
    assert fault_type_to_int("transform") == 2
    assert fault_type_to_int(None) == 0
    assert fault_type_to_int("unknown") == 0
