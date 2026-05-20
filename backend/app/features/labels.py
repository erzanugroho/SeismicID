"""Multi-output label generation.

For each (cell, snapshot), label is 1 if any future event with magnitude >=
threshold occurs in (cell ∪ 8 neighbors) within `horizon_days` after snapshot,
else 0.

Label column naming: label_h{horizon}_m{threshold_x10}
e.g., label_h30_m50 = horizon 30 days, threshold M>=5.0
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.grid import GridCell, generate_grid
from backend.app.core.logging import get_logger
from backend.app.features.spatial import neighbor_map

logger = get_logger(__name__)

HORIZONS = (7, 14, 30, 60)
THRESHOLDS = (4.5, 5.0, 5.5, 6.0)


def label_column_name(horizon_days: int, threshold: float) -> str:
    return f"label_h{horizon_days}_m{int(threshold * 10):02d}"


def all_label_columns() -> list[str]:
    return [label_column_name(h, t) for h in HORIZONS for t in THRESHOLDS]


def assign_cell_id_vec(events: pd.DataFrame, cells: list[GridCell]) -> pd.DataFrame:
    """Vectorized cell assignment: floor lat/lon to grid step."""
    if events.empty:
        return events.assign(cell_id=pd.NA)
    df = events.copy()
    step = cells[0].lat_max - cells[0].lat_min if cells else 0.5
    lat_origin = min(c.lat_min for c in cells) if cells else -11.0
    lon_origin = min(c.lon_min for c in cells) if cells else 95.0
    cell_lookup = {(round(c.lat_min, 4), round(c.lon_min, 4)): c.cell_id for c in cells}

    lat_min = lat_origin + step * np.floor((df["lat"].to_numpy() - lat_origin) / step)
    lon_min = lon_origin + step * np.floor((df["lon"].to_numpy() - lon_origin) / step)
    df["cell_id"] = [
        cell_lookup.get((round(la, 4), round(lo, 4)))
        for la, lo in zip(lat_min, lon_min)
    ]
    return df


def build_labels(
    events: pd.DataFrame,
    snapshots: list[datetime],
    cells: list[GridCell] | None = None,
) -> pd.DataFrame:
    """Generate label dataframe with rows = (cell_id, snapshot) and 16 label cols."""
    cells = cells or generate_grid()
    
    # 1. Assign cell_id to each event
    df = assign_cell_id_vec(events, cells)
    df = df.dropna(subset=["cell_id"]).copy()
    
    # Pre-convert events time to naive datetime64 UTC
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    
    nbrs = neighbor_map(cells)
    
    # 2. Map cell_id to its buffer cells (self + neighbors)
    cell_to_buffer = {c.cell_id: [c.cell_id] + nbrs.get(c.cell_id, []) for c in cells}
    
    # 3. We want to map each event to all the cells it affects.
    # An event in cell `cid` affects any cell `c` if `cid` is in `c`'s buffer.
    # Build a reverse mapping: cell_id -> list of cells whose buffers contain it.
    affected_cells_lookup = {c.cell_id: [] for c in cells}
    for c_id, buffer_list in cell_to_buffer.items():
        for b_id in buffer_list:
            if b_id in affected_cells_lookup:
                affected_cells_lookup[b_id].append(c_id)
                
    # 4. Expand events: for each event, duplicate it for all cells it affects
    expanded_rows = []
    for _, row in df.iterrows():
        event_cell = row["cell_id"]
        event_time = row["time"]
        event_mag = row["magnitude"]
        
        for c_id in affected_cells_lookup.get(event_cell, []):
            expanded_rows.append({
                "cell_id": c_id,
                "time": event_time,
                "magnitude": event_mag
            })
            
    if expanded_rows:
        expanded_df = pd.DataFrame(expanded_rows)
        by_cell = {cid: group for cid, group in expanded_df.groupby("cell_id")}
    else:
        by_cell = {}
        
    # Precompute naive timestamps for snapshots
    snap_timestamps = []
    for snap in snapshots:
        snap_ts = pd.Timestamp(snap)
        if snap_ts.tzinfo is not None:
            snap_ts = snap_ts.tz_convert("UTC").tz_localize(None)
        else:
            snap_ts = snap_ts.tz_localize(None)
        snap_timestamps.append(snap_ts)

    label_cols = all_label_columns()
    rows: list[dict[str, Any]] = []
    empty_label_dict = {col: 0 for col in label_cols}

    for c in cells:
        cell_events = by_cell.get(c.cell_id)
        if cell_events is None or cell_events.empty:
            for snap_ts in snap_timestamps:
                row = {"cell_id": c.cell_id, "snapshot": snap_ts.isoformat()}
                row.update(empty_label_dict)
                rows.append(row)
            continue
            
        times = cell_events["time"].to_numpy()
        mags = cell_events["magnitude"].to_numpy()
        
        for snap_ts in snap_timestamps:
            snap_np = snap_ts.to_datetime64()
            row = {"cell_id": c.cell_id, "snapshot": snap_ts.isoformat()}
            
            future_mask = times > snap_np
            if not np.any(future_mask):
                row.update(empty_label_dict)
                rows.append(row)
                continue
                
            future_times = times[future_mask]
            future_mags = mags[future_mask]
            
            for h in HORIZONS:
                horizon_end = snap_np + np.timedelta64(h, 'D')
                in_horizon_mask = future_times <= horizon_end
                
                if np.any(in_horizon_mask):
                    h_mags = future_mags[in_horizon_mask]
                    for t in THRESHOLDS:
                        has_event = np.any(h_mags >= t)
                        row[label_column_name(h, t)] = 1 if has_event else 0
                else:
                    for t in THRESHOLDS:
                        row[label_column_name(h, t)] = 0
            rows.append(row)
            
    out = pd.DataFrame(rows)
    logger.info("labels_built", rows=len(out), n_label_columns=len(all_label_columns()))
    return out


def join_features_and_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (cell_id, snapshot)."""
    return features.merge(labels, on=["cell_id", "snapshot"], how="inner")


def positive_rates(dataset: pd.DataFrame) -> dict[str, float]:
    """Per-head positive class fraction (for class imbalance awareness)."""
    return {col: float(dataset[col].mean()) for col in all_label_columns() if col in dataset}


def time_split(
    dataset: pd.DataFrame,
    *,
    train_end_year: int = 2020,
    val_end_year: int = 2021,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based split: train <= train_end_year, val = val_end_year, test > val_end_year.
    If default values are used and result in empty splits (due to date drift),
    dynamically falls back to a 60/20/20 time-based split.
    """
    df = dataset.copy()
    if df.empty:
        return df, df, df
        
    df["snapshot_dt"] = pd.to_datetime(df["snapshot"], utc=True)
    
    train = df[df["snapshot_dt"].dt.year <= train_end_year]
    val = df[df["snapshot_dt"].dt.year == val_end_year]
    test = df[df["snapshot_dt"].dt.year > val_end_year]
    
    if train.empty or val.empty:
        years = df["snapshot_dt"].dt.year
        max_year = years.max()
        min_year = years.min()
        
        # Try relative split based on dataset years
        if max_year - min_year >= 2:
            r_train_end = max_year - 2
            r_val_end = max_year - 1
            train = df[df["snapshot_dt"].dt.year <= r_train_end]
            val = df[df["snapshot_dt"].dt.year == r_val_end]
            test = df[df["snapshot_dt"].dt.year > r_val_end]
            
        # If still empty, fall back to percentile split
        if train.empty or val.empty:
            sorted_times = df["snapshot_dt"].sort_values()
            n = len(df)
            idx1 = int(n * 0.6)
            idx2 = int(n * 0.8)
            if idx1 > 0 and idx2 > idx1 and idx2 < n:
                t1 = sorted_times.iloc[idx1]
                t2 = sorted_times.iloc[idx2]
                train = df[df["snapshot_dt"] <= t1]
                val = df[(df["snapshot_dt"] > t1) & (df["snapshot_dt"] <= t2)]
                test = df[df["snapshot_dt"] > t2]
            else:
                train = df.iloc[:int(n * 0.6)]
                val = df.iloc[int(n * 0.6):int(n * 0.8)]
                test = df.iloc[int(n * 0.8):]
                
    return train.drop(columns=["snapshot_dt"]), val.drop(columns=["snapshot_dt"]), test.drop(columns=["snapshot_dt"])
