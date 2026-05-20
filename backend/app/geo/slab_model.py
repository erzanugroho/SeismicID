"""Slab depth model.

When real USGS Slab2.0 grid is downloaded, use it. Otherwise use an
analytical approximation: at any point, slab depth ≈ k * distance_to_trench
for distances within ~500 km of the trench. Returns NaN for points not in
known subduction zones.
"""

from __future__ import annotations

import math
from pathlib import Path

from backend.app.geo.fault_db import FAULTS, _distance_to_polyline_km


def has_slab_data(geo_dir: Path) -> bool:
    return any(p.suffix == ".grd" for p in geo_dir.glob("*slab*.grd"))


SUBDUCTION_DIP_DEG = 25.0
MAX_DEPTH_KM = 700.0


def slab_depth_km(lat: float, lon: float, max_distance_km: float = 500.0) -> float | None:
    """Approximate slab depth at point.

    Strategy: nearest subduction-zone polyline → use distance × tan(dip).
    Cap at MAX_DEPTH_KM. Returns None if outside any subduction influence.
    """
    best_d = float("inf")
    for f in FAULTS:
        if f.fault_type != "subduction":
            continue
        d = _distance_to_polyline_km(lat, lon, f.polyline)
        if d < best_d:
            best_d = d
    if best_d > max_distance_km:
        return None
    depth = best_d * math.tan(math.radians(SUBDUCTION_DIP_DEG))
    return float(min(depth, MAX_DEPTH_KM))
