"""Magnitude of completeness (Mc) estimation per region per epoch.

Method: Maximum Curvature (MAXC) of frequency-magnitude distribution.
Mc = magnitude bin with the highest event count (mode of histogram).
Add a +0.2 correction to account for known MAXC underestimation bias.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.logging import get_logger

logger = get_logger(__name__)

REGIONS = {
    "Sumatera": (-6.0, 95.0, 6.0, 106.5),
    "Jawa": (-9.0, 105.0, -5.5, 115.0),
    "BaliNusa": (-11.0, 114.0, -8.0, 126.0),
    "Kalimantan": (-4.5, 108.5, 4.5, 119.0),
    "Sulawesi": (-7.5, 118.5, 5.0, 127.0),
    "MalukuPapua": (-11.0, 124.0, 3.0, 141.0),
}

# 5-year epochs from 2000 to today
EPOCH_YEARS = [2000, 2005, 2010, 2015, 2020, 2025, 2030]


def estimate_mc_maxc(magnitudes: np.ndarray, *, bin_width: float = 0.1, correction: float = 0.2) -> float:
    """Maximum curvature method. Returns Mc with bias correction."""
    if len(magnitudes) < 30:
        return float("nan")
    bins = np.arange(magnitudes.min(), magnitudes.max() + bin_width, bin_width)
    counts, edges = np.histogram(magnitudes, bins=bins)
    if counts.sum() == 0:
        return float("nan")
    mc_idx = int(np.argmax(counts))
    mc = (edges[mc_idx] + edges[mc_idx + 1]) / 2
    return float(mc + correction)


def _region_for(lat: float, lon: float) -> str | None:
    for name, (lat_min, lon_min, lat_max, lon_max) in REGIONS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return None


def build_mc_lookup(events: pd.DataFrame) -> dict[tuple[str, int, int], float]:
    """Build Mc lookup keyed on (region, epoch_start_year, epoch_end_year)."""
    if events.empty:
        return {}
    df = events.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["region"] = df.apply(lambda r: _region_for(r["lat"], r["lon"]), axis=1)
    df = df.dropna(subset=["region"])

    out: dict[tuple[str, int, int], float] = {}
    for region in REGIONS:
        for i in range(len(EPOCH_YEARS) - 1):
            ep_start, ep_end = EPOCH_YEARS[i], EPOCH_YEARS[i + 1]
            mask = (
                (df["region"] == region)
                & (df["time"].dt.year >= ep_start)
                & (df["time"].dt.year < ep_end)
            )
            mags = df.loc[mask, "magnitude"].to_numpy()
            mc = estimate_mc_maxc(mags)
            if not np.isnan(mc):
                out[(region, ep_start, ep_end)] = mc
    logger.info("mc_lookup_built", entries=len(out))
    return out


def lookup_mc(
    table: dict[tuple[str, int, int], float],
    lat: float,
    lon: float,
    when: datetime,
    *,
    default: float = 4.0,
) -> float:
    region = _region_for(lat, lon)
    if region is None:
        return default
    year = when.year
    for (r, ep_start, ep_end), mc in table.items():
        if r == region and ep_start <= year < ep_end:
            return mc
    return default


def mc_lookup_to_dataframe(table: dict[tuple[str, int, int], float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (region, ep_start, ep_end), mc in sorted(table.items()):
        rows.append({"region": region, "epoch_start": ep_start, "epoch_end": ep_end, "mc": mc})
    return pd.DataFrame(rows)
