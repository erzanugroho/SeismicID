"""Core feature engineering: rolling per-cell statistics over historical events.

Output: feature DataFrame indexed by (cell_id, snapshot_date) with columns:
  - event_count_{30,90,365}d
  - max_mag_{30,90}d
  - mean_depth_30d, std_depth_30d
  - log_energy_30d
  - moment_release_ratio_30d_vs_365d
  - b_value_{90,365,1095}d
  - b_value_slope_1y
  - iet_mean_30d, iet_cv_30d
  - time_since_last_M4_days, time_since_last_M5_days
  - activity_trend_90d
  - neighbor_event_count_30d_mean, neighbor_max_mag_30d_max
  - nearest_fault_km, fault_type_int, fault_slip_rate, slab_depth_km

Feature is built per cell + buffer (cell + 8-neighbors). Static in-cell features
use only events with centroid inside the cell bounds; spatial neighbor features
aggregate across the 8 nearest cells. The four physics-informed columns at the
end are constant per cell (cached) so the cost of adding them is negligible.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.grid import GridCell, generate_grid
from backend.app.core.logging import get_logger
from backend.app.data.completeness import build_mc_lookup, lookup_mc
from backend.app.features.labels import assign_cell_id_vec
from backend.app.features.physics import fault_type_to_int, static_physics_features
from backend.app.features.seismology import (
    b_value_slope,
    compute_b_value,
    iet_stats,
    seismic_energy,
)
from backend.app.features.spatial import neighbor_map

logger = get_logger(__name__)

# Sentinel for missing slab depth (cells far from any subduction trench).
# Chosen to be well outside the realistic range of slab depths (0–700 km) so
# tree models can split it as "very far / no slab influence" without confusing
# it for an actual depth measurement.
_MISSING_SLAB_DEPTH = 9999.0

# Names of the static physics-informed features added per cell.
PHYSICS_STATIC_FEATURES = (
    "nearest_fault_km",
    "fault_type_int",
    "fault_slip_rate",
    "slab_depth_km",
)


def _physics_features_for_cell(lat: float, lon: float) -> dict[str, float]:
    """Compute the four static physics features for a single cell."""
    physics = static_physics_features(lat, lon)
    slab = physics["slab_depth_km"]
    nearest = physics["nearest_fault_km"]
    slip = physics["fault_slip_rate"]
    fault_type = physics["fault_type"]
    return {
        "nearest_fault_km": float(nearest) if nearest is not None else 0.0,
        "fault_type_int": float(
            fault_type_to_int(fault_type if isinstance(fault_type, str) else None)
        ),
        "fault_slip_rate": float(slip) if slip is not None else 0.0,
        "slab_depth_km": float(slab) if slab is not None else _MISSING_SLAB_DEPTH,
    }


def assign_cell_id(events: pd.DataFrame, cells: list[GridCell]) -> pd.DataFrame:
    """Add a cell_id column using the shared vectorized grid assignment."""
    return assign_cell_id_vec(events, cells)


def compute_window_features(
    events: pd.DataFrame,
    snapshot: datetime,
    *,
    mc: float = 4.0,
) -> dict[str, float]:
    """Compute features over a window of events ending at `snapshot`."""
    if events.empty:
        return _empty_features()

    snap_ts = pd.Timestamp(snapshot)
    if snap_ts.tzinfo is not None:
        snap_ts = snap_ts.tz_convert("UTC").tz_localize(None)
    snap_np = snap_ts.to_datetime64()

    times_series = pd.to_datetime(events["time"], errors="coerce")
    if times_series.dt.tz is not None:
        times_series = times_series.dt.tz_convert("UTC").dt.tz_localize(None)
    times = times_series.to_numpy()

    # Filter up to snapshot
    valid_idx = times <= snap_np
    if not np.any(valid_idx):
        return _empty_features()

    times = times[valid_idx]
    mags = events["magnitude"].to_numpy()[valid_idx]
    depths = events["depth"].to_numpy()[valid_idx]

    feats: dict[str, float] = {}

    t_30 = snap_np - np.timedelta64(30, 'D')
    t_90 = snap_np - np.timedelta64(90, 'D')
    t_365 = snap_np - np.timedelta64(365, 'D')
    t_1095 = snap_np - np.timedelta64(1095, 'D')

    # Subsets
    idx_30 = times >= t_30
    idx_90 = times >= t_90
    idx_365 = times >= t_365
    idx_1095 = times >= t_1095

    # Event counts
    feats["event_count_30d"] = float(np.sum(idx_30))
    feats["event_count_90d"] = float(np.sum(idx_90))
    feats["event_count_365d"] = float(np.sum(idx_365))

    # 30d stats
    mags_30 = mags[idx_30]
    depths_30 = depths[idx_30]
    times_30 = times[idx_30]

    feats["max_mag_30d"] = float(np.max(mags_30)) if len(mags_30) > 0 else 0.0
    feats["mean_depth_30d"] = float(np.mean(depths_30)) if len(depths_30) > 0 else 0.0
    feats["std_depth_30d"] = float(np.std(depths_30, ddof=0)) if len(depths_30) > 1 else 0.0

    if len(mags_30) > 0:
        energies = seismic_energy(mags_30)
        feats["log_energy_30d"] = float(np.log10(np.sum(10**energies)))
        mu, cv = iet_stats(times_30)
        feats["iet_mean_30d"] = mu if not np.isnan(mu) else 0.0
        feats["iet_cv_30d"] = cv if not np.isnan(cv) else 0.0
    else:
        feats["log_energy_30d"] = 0.0
        feats["iet_mean_30d"] = 0.0
        feats["iet_cv_30d"] = 0.0

    # 90d stats
    mags_90 = mags[idx_90]
    feats["max_mag_90d"] = float(np.max(mags_90)) if len(mags_90) > 0 else 0.0

    # Moment release ratio
    e30 = 10 ** feats["log_energy_30d"] if feats["log_energy_30d"] > 0 else 0.0
    mags_365 = mags[idx_365]
    e365 = float(np.sum(10 ** seismic_energy(mags_365))) if len(mags_365) > 0 else 1.0
    feats["moment_release_ratio_30d_vs_365d"] = e30 / e365 if e365 > 0 else 0.0

    # b-values
    for d, idx in ((90, idx_90), (365, idx_365), (1095, idx_1095)):
        sub_mags = mags[idx]
        b, _ = compute_b_value(sub_mags, mc=mc) if len(sub_mags) > 0 else (float("nan"), float("nan"))
        feats[f"b_value_{d}d"] = b if not np.isnan(b) else 1.0

    # b-value slope
    feats["b_value_slope_1y"] = (
        b_value_slope(times[idx_365], mags_365, mc=mc, window_days=90)
        if len(mags_365) > 50
        else 0.0
    )
    if np.isnan(feats["b_value_slope_1y"]):
        feats["b_value_slope_1y"] = 0.0

    # Time since last M>=X
    for thresh, key in ((4.0, "M4"), (5.0, "M5")):
        idx_thresh = mags >= thresh
        if np.any(idx_thresh):
            last_t = np.max(times[idx_thresh])
            diff_days = float((snap_np - last_t) / np.timedelta64(1, 'D'))
            feats[f"time_since_last_{key}_days"] = diff_days
        else:
            feats[f"time_since_last_{key}_days"] = 9999.0

    # Activity trend
    if len(mags_90) >= 5:
        # construct 7D bins
        bins = np.array([snap_np - np.timedelta64(i * 7, 'D') for i in range(13)][::-1])
        counts, _ = np.histogram(times[idx_90], bins=bins)
        if counts.sum() > 0:
            x = np.arange(len(counts), dtype=float)
            slope, _ = np.polyfit(x, counts, 1)
            feats["activity_trend_90d"] = float(slope)
        else:
            feats["activity_trend_90d"] = 0.0
    else:
        feats["activity_trend_90d"] = 0.0

    return feats


def _empty_features() -> dict[str, float]:
    keys = [
        "event_count_30d", "event_count_90d", "event_count_365d",
        "max_mag_30d", "max_mag_90d", "mean_depth_30d", "std_depth_30d",
        "log_energy_30d", "moment_release_ratio_30d_vs_365d",
        "b_value_90d", "b_value_365d", "b_value_1095d", "b_value_slope_1y",
        "iet_mean_30d", "iet_cv_30d",
        "time_since_last_M4_days", "time_since_last_M5_days",
        "activity_trend_90d",
    ]
    return {k: 9999.0 if "time_since" in k else (1.0 if "b_value_" in k and "slope" not in k else 0.0) for k in keys}


def build_features_for_snapshots(
    events: pd.DataFrame,
    snapshots: list[datetime],
    cells: list[GridCell] | None = None,
) -> pd.DataFrame:
    """Build feature dataset for every (cell, snapshot)."""
    cells = cells or generate_grid()

    # Pre-convert events time to datetime & align timezone
    events_prep = events.copy()
    events_prep["time"] = pd.to_datetime(events_prep["time"], utc=True).dt.tz_localize(None)

    # Precompute naive ISO strings for snapshots
    snap_strs = {}
    for snap in snapshots:
        snap_ts = pd.Timestamp(snap)
        if snap_ts.tzinfo is not None:
            snap_ts = snap_ts.tz_convert("UTC").tz_localize(None)
        else:
            snap_ts = snap_ts.tz_localize(None)
        snap_strs[snap] = snap_ts.isoformat()

    cells_with_id = assign_cell_id(events_prep, cells)
    by_cell = dict(tuple(cells_with_id.dropna(subset=["cell_id"]).groupby("cell_id")))
    nbrs = neighbor_map(cells)
    mc_table = build_mc_lookup(events_prep) if not events_prep.empty else {}

    # Pre-compute static physics features per cell. ``static_physics_features``
    # iterates every fault polyline so caching here saves a lot of work when
    # we evaluate many snapshots.
    physics_per_cell: dict[str, dict[str, float]] = {
        c.cell_id: _physics_features_for_cell(c.lat, c.lon) for c in cells
    }

    # Precompute in-cell features for all cells and snapshots
    cell_snap_feats = {}
    empty_f = _empty_features()

    for c in cells:
        cell_events = by_cell.get(c.cell_id, pd.DataFrame(columns=events_prep.columns))
        if cell_events.empty:
            for snap in snapshots:
                cell_snap_feats[(c.cell_id, snap_strs[snap])] = empty_f.copy()
            continue

        for snap in snapshots:
            mc = lookup_mc(mc_table, c.lat, c.lon, snap, default=4.0)
            cell_snap_feats[(c.cell_id, snap_strs[snap])] = compute_window_features(cell_events, snap, mc=mc)

    rows: list[dict[str, Any]] = []
    for c in cells:
        nbr_ids = nbrs.get(c.cell_id, [])
        cell_physics = physics_per_cell[c.cell_id]
        for snap in snapshots:
            snap_str = snap_strs[snap]
            f: dict[str, Any] = cell_snap_feats[(c.cell_id, snap_str)].copy()

            nbr_counts = []
            nbr_max_mags = []
            for nid in nbr_ids:
                nf = cell_snap_feats.get((nid, snap_str), empty_f)
                nbr_counts.append(nf.get("event_count_30d", 0.0))
                nbr_max_mags.append(nf.get("max_mag_30d", 0.0))

            f["neighbor_event_count_30d_mean"] = float(np.mean(nbr_counts)) if nbr_counts else 0.0
            f["neighbor_max_mag_30d_max"] = float(np.max(nbr_max_mags)) if nbr_max_mags else 0.0
            # Append static physics-informed features (constant per cell).
            f.update(cell_physics)
            f["cell_id"] = c.cell_id
            f["snapshot"] = snap_str
            rows.append(f)

    df = pd.DataFrame(rows)
    logger.info("features_built", rows=len(df), n_features=len(df.columns) - 2)
    return df


def default_snapshots(start: datetime, end: datetime, freq_days: int = 7) -> list[datetime]:
    snaps = []
    cursor = start
    step = timedelta(days=freq_days)
    while cursor <= end:
        snaps.append(cursor)
        cursor += step
    return snaps


def feature_columns() -> list[str]:
    return (
        list(_empty_features().keys())
        + ["neighbor_event_count_30d_mean", "neighbor_max_mag_30d_max"]
        + list(PHYSICS_STATIC_FEATURES)
    )
