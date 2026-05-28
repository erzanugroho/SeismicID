"""Catalog declustering — Reasenberg (1985) algorithm.

For each event, define an interaction zone in time and space based on the
mainshock magnitude. Events that fall inside the zone of a previous mainshock
are flagged as aftershocks (cluster members). Otherwise they become the new
cluster mainshock.

Parameters follow Reasenberg defaults. Output: DataFrame with `is_mainshock`
and `cluster_id` columns added.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backend.app.core.logging import get_logger
from backend.app.data.catalog import (
    read_historical_events,
    write_declustered_events,
)

logger = get_logger(__name__)

TAU_MIN_DAYS = 1.0
TAU_HARD_CAP_DAYS = 1825.0  # 5 years failsafe
P_CONFIDENCE = 0.95
RFACT = 10.0


def _tau_max_days(mag: float) -> float:
    """Magnitude-dependent temporal window length (Reasenberg 1985, app. A).

    tau follows ~10**(0.43*M - 0.06) days. M5≈123d, M6≈331d, M7≈891d.
    Fixes the previous flat 10-day window which under-declustered M>=7
    aftershock sequences.
    """
    days = 10 ** (0.43 * float(mag) - 0.06)
    return float(max(TAU_MIN_DAYS, min(days, TAU_HARD_CAP_DAYS)))


def _interaction_radius_km(mag: float) -> float:
    """Crustal rupture radius (Wells & Coppersmith 1994, simplified)."""
    log_l = -2.44 + 0.59 * mag
    length_km = 10 ** log_l
    return RFACT * length_km / 2.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def decluster(events: pd.DataFrame) -> pd.DataFrame:
    """Apply Reasenberg-style declustering.

    Returns the input DataFrame with two extra columns:
        is_mainshock (bool), cluster_id (int; -1 means standalone mainshock)
    """
    if events.empty:
        return events.assign(is_mainshock=True, cluster_id=-1)

    df = events.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    n = len(df)
    is_main = np.ones(n, dtype=bool)
    cluster = np.full(n, -1, dtype=np.int64)
    next_cluster_id = 0

    times = df["time"].astype("int64").to_numpy() // 1_000_000_000  # seconds
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    mags = df["magnitude"].to_numpy()

    for i in range(n):
        for j in range(i - 1, -1, -1):
            dt_days = (times[i] - times[j]) / 86400.0
            tau_j = _tau_max_days(mags[j])
            if dt_days > tau_j:
                if dt_days > TAU_HARD_CAP_DAYS:
                    break
                continue
            if mags[j] <= mags[i]:
                continue
            r_km = _interaction_radius_km(mags[j])
            if _haversine_km(lats[i], lons[i], lats[j], lons[j]) <= r_km:
                is_main[i] = False
                cluster[i] = cluster[j] if cluster[j] != -1 else next_cluster_id
                if cluster[j] == -1:
                    cluster[j] = next_cluster_id
                    next_cluster_id += 1
                break

    df["is_mainshock"] = is_main
    df["cluster_id"] = cluster
    n_main = int(is_main.sum())
    logger.info("decluster_done", total=n, mainshocks=n_main, aftershocks=n - n_main)
    return df


def run_decluster_pipeline() -> int:
    """Read historical events -> decluster -> write declustered Parquet."""
    df = read_historical_events()
    if df.empty:
        logger.warning("decluster_skip_no_events")
        return 0
    out = decluster(df)
    return write_declustered_events(out)
