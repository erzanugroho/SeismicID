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

SHORT_TERM_WINDOWS = {
    "1h": pd.Timedelta(hours=1),
    "6h": pd.Timedelta(hours=6),
    "24h": pd.Timedelta(hours=24),
    "7d": pd.Timedelta(days=7),
}
SHORT_TERM_THRESHOLDS = {"M45": 4.5, "M50": 5.0}
SHORT_TERM_RADII_KM = (100, 300)


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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def _short_term_empty_features() -> dict[str, float]:
    feats: dict[str, float] = {}
    for radius in SHORT_TERM_RADII_KM:
        for label in SHORT_TERM_THRESHOLDS:
            for window_label in SHORT_TERM_WINDOWS:
                feats[f"count_{label}_{window_label}_r{radius}km"] = 0.0
        feats[f"log_energy_7d_r{radius}km"] = 0.0
        feats[f"rate_ratio_24h_vs_7d_r{radius}km"] = 0.0
    for label in ("M5", "M6"):
        feats[f"nearest_{label}_dist_km"] = 9999.0
        feats[f"nearest_{label}_time_days"] = 9999.0
    return feats


def _radius_neighbor_map(cells: list[GridCell], radius_km: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    lat_step = radius_km / 111.0 + 0.6
    by_lat_bucket: dict[int, list[GridCell]] = {}
    for cell in cells:
        by_lat_bucket.setdefault(int(cell.lat), []).append(cell)
    for cell in cells:
        candidates: list[GridCell] = []
        for bucket in range(int(cell.lat - lat_step) - 1, int(cell.lat + lat_step) + 2):
            candidates.extend(by_lat_bucket.get(bucket, []))
        ids = []
        lon_step = radius_km / max(30.0, 111.0 * abs(np.cos(np.radians(cell.lat)))) + 0.6
        for other in candidates:
            if abs(other.lon - cell.lon) > lon_step or abs(other.lat - cell.lat) > lat_step:
                continue
            if _haversine_km(cell.lat, cell.lon, other.lat, other.lon) <= radius_km:
                ids.append(other.cell_id)
        out[cell.cell_id] = ids
    return out


def _compute_short_term_spatial_features(
    by_cell: dict[str, pd.DataFrame],
    cells: list[GridCell],
    radius_maps: dict[int, dict[str, list[str]]],
    snapshot: datetime,
) -> dict[str, dict[str, float]]:
    snap_ts = pd.Timestamp(snapshot)
    if snap_ts.tzinfo is not None:
        snap_ts = snap_ts.tz_convert("UTC").tz_localize(None)
    empty = _short_term_empty_features()
    out: dict[str, dict[str, float]] = {}
    since_7d = snap_ts - SHORT_TERM_WINDOWS["7d"]

    # Pre-filter each populated cell to the 7-day window used by all Sprint 5 features.
    recent_by_cell: dict[str, pd.DataFrame] = {}
    for cid, df in by_cell.items():
        if df.empty:
            continue
        times = pd.to_datetime(df["time"], errors="coerce")
        if times.dt.tz is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        recent = df.assign(_time=times)
        recent = recent[(recent["_time"].notna()) & (recent["_time"] <= snap_ts) & (recent["_time"] >= since_7d)]
        if not recent.empty:
            recent_by_cell[cid] = recent

    for cell in cells:
        feats = empty.copy()
        for radius, radius_map in radius_maps.items():
            parts = [recent_by_cell[cid] for cid in radius_map.get(cell.cell_id, []) if cid in recent_by_cell]
            if not parts:
                continue
            ev = pd.concat(parts, ignore_index=True)
            if ev.empty:
                continue
            mags = ev["magnitude"].astype(float)
            for label, threshold in SHORT_TERM_THRESHOLDS.items():
                mag_mask = mags >= threshold
                for window_label, delta in SHORT_TERM_WINDOWS.items():
                    t0 = snap_ts - delta
                    feats[f"count_{label}_{window_label}_r{radius}km"] = float(((ev["_time"] >= t0) & mag_mask).sum())
            if not ev.empty:
                log_e = seismic_energy(ev["magnitude"].astype(float).to_numpy())
                m_max = float(np.max(log_e))
                feats[f"log_energy_7d_r{radius}km"] = float(m_max + np.log10(np.sum(10 ** (log_e - m_max))))
            count_24h = feats[f"count_M45_24h_r{radius}km"]
            count_7d = feats[f"count_M45_7d_r{radius}km"]
            feats[f"rate_ratio_24h_vs_7d_r{radius}km"] = float((count_24h + 0.1) / ((count_7d / 7.0) + 0.1))

        # Nearest recent M5/M6 event within 7 days, exact distance from cell centroid.
        all_neighbor_ids = set(radius_maps[300].get(cell.cell_id, []))
        parts = [recent_by_cell[cid] for cid in all_neighbor_ids if cid in recent_by_cell]
        if parts:
            ev = pd.concat(parts, ignore_index=True)
            for label, threshold in (("M5", 5.0), ("M6", 6.0)):
                sub = ev[ev["magnitude"].astype(float) >= threshold]
                if sub.empty:
                    continue
                best_dist = 9999.0
                best_days = 9999.0
                lat_col = "lat" if "lat" in sub.columns else "latitude"
                lon_col = "lon" if "lon" in sub.columns else "longitude"
                for _, erow in sub.iterrows():
                    dist = _haversine_km(cell.lat, cell.lon, float(erow[lat_col]), float(erow[lon_col]))
                    if dist < best_dist:
                        best_dist = dist
                        best_days = float((snap_ts - erow["_time"]) / pd.Timedelta(days=1))
                feats[f"nearest_{label}_dist_km"] = best_dist
                feats[f"nearest_{label}_time_days"] = best_days
        out[cell.cell_id] = feats
    return out


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
        # log_energy_30d = log10(sum_i E_i) where seismic_energy already
        # returns log10 E_i. We use log-sum-exp on log10 to stay numerically
        # stable for catalogs containing M>=8 events (where 10**(1.5*M+4.8)
        # exceeds 10**16 and naive summation loses precision).
        log_e_i = seismic_energy(mags_30)            # log10 of each E_i
        m_max = float(np.max(log_e_i))
        # log10(sum 10**x_i) = m + log10(sum 10**(x_i - m))
        feats["log_energy_30d"] = float(
            m_max + np.log10(np.sum(10 ** (log_e_i - m_max)))
        )
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

    # Moment release ratio (numerator and denominator both via log-sum-exp).
    if len(mags_30) > 0:
        e30_log = feats["log_energy_30d"]
        mags_365 = mags[idx_365]
        if len(mags_365) > 0:
            log_e_365 = seismic_energy(mags_365)
            m_max_365 = float(np.max(log_e_365))
            log_e365_total = m_max_365 + float(np.log10(np.sum(10 ** (log_e_365 - m_max_365))))
            # ratio = 10**(e30_log - log_e365_total)
            feats["moment_release_ratio_30d_vs_365d"] = float(10 ** (e30_log - log_e365_total))
        else:
            feats["moment_release_ratio_30d_vs_365d"] = 0.0
    else:
        feats["moment_release_ratio_30d_vs_365d"] = 0.0
        mags_365 = mags[idx_365]  # still needed below for b-value

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
    empty_short = _short_term_empty_features()
    radius_maps = {radius: _radius_neighbor_map(cells, radius) for radius in SHORT_TERM_RADII_KM}
    short_term_by_snap = {
        snap_strs[snap]: _compute_short_term_spatial_features(by_cell, cells, radius_maps, snap)
        for snap in snapshots
    }

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
            # Append additive Sprint 5 short-term regional seismicity features.
            f.update(short_term_by_snap.get(snap_str, {}).get(c.cell_id, empty_short))
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
        + list(_short_term_empty_features().keys())
        + ["neighbor_event_count_30d_mean", "neighbor_max_mag_30d_max"]
        + list(PHYSICS_STATIC_FEATURES)
    )
