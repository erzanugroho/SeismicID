"""Physics-informed features: fault distance, slab depth, Z-value quiescence."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backend.app.geo.fault_db import nearest_fault
from backend.app.geo.slab_model import slab_depth_km


def static_physics_features(lat: float, lon: float) -> dict[str, float | str | None]:
    """Static features tied to the cell location: nearest fault + slab depth."""
    f, dist = nearest_fault(lat, lon)
    slab = slab_depth_km(lat, lon)
    return {
        "nearest_fault_km": float(dist),
        "fault_type": f.fault_type,
        "fault_slip_rate": float(f.slip_rate_mm_yr),
        "slab_depth_km": slab,
    }


def fault_type_to_int(name: str | None) -> int:
    if name is None:
        return 0
    mapping = {"subduction": 1, "transform": 2, "reverse": 3, "normal": 4}
    return mapping.get(name, 0)


def z_value_quiescence(
    events: pd.DataFrame,
    snapshot: pd.Timestamp,
    *,
    test_window_days: int = 365,
    reference_window_days: int = 1825,  # 5 years
    min_events: int = 30,
) -> float:
    """Z-value quiescence (Wiemer & Wyss 1994 simplified).

    Z = (mean_reference - mean_test) / sqrt(var_reference/n_test + var_test/n_test)
    Positive Z = activity dropped (quiescence). Returns 0.0 if not enough events.
    """
    if events.empty:
        return 0.0

    df = events.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    snap = pd.Timestamp(snapshot)
    if snap.tzinfo is None:
        snap = snap.tz_localize("UTC")

    ref_start = snap - pd.Timedelta(days=reference_window_days)
    test_start = snap - pd.Timedelta(days=test_window_days)
    df_ref = df[(df["time"] >= ref_start) & (df["time"] < test_start)]
    df_test = df[df["time"] >= test_start]

    if len(df_ref) < min_events:
        return 0.0

    # Discretize into 30-day bins, count events per bin, then compute Z
    def _binned_counts(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        if len(d) == 0:
            n_bins = max(1, int((end - start).days // 30))
            return np.zeros(n_bins)
        edges = pd.date_range(start=start, end=end, freq="30D")
        if len(edges) < 2:
            return np.array([float(len(d))])
        counts, _ = np.histogram(
            d["time"].astype("int64") // 1_000_000_000,
            bins=edges.astype("int64") // 1_000_000_000,
        )
        return counts.astype(float)

    ref_counts = _binned_counts(df_ref, ref_start, test_start)
    test_counts = _binned_counts(df_test, test_start, snap)
    if len(ref_counts) < 3 or len(test_counts) < 1:
        return 0.0

    mu_ref, var_ref = ref_counts.mean(), ref_counts.var(ddof=1) if len(ref_counts) > 1 else 0.0
    mu_test, var_test = test_counts.mean(), test_counts.var(ddof=1) if len(test_counts) > 1 else var_ref
    n_ref, n_test = len(ref_counts), len(test_counts)
    denom = math.sqrt(var_ref / max(n_ref, 1) + var_test / max(n_test, 1))
    if denom == 0:
        return 0.0
    return float((mu_ref - mu_test) / denom)
