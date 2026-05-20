"""Indonesian active faults database (simplified).

Hardcoded major fault zones used as a substitute for PUSGEN 2017 when
the real shapefile is not yet downloaded. Each fault is approximated as
a polyline (list of lat/lon vertices) plus type and slip rate. For higher
precision, replace with real shapefile via load_pusgen_faults() (Task 7+).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fault:
    name: str
    fault_type: str  # subduction|transform|reverse|normal
    slip_rate_mm_yr: float
    polyline: tuple[tuple[float, float], ...]  # ((lat, lon), ...)


# Polylines are coarse approximations sourced from major regional studies
# (Sunda megathrust, Sumatra fault, Palu-Koro, Sorong, etc.).
FAULTS: tuple[Fault, ...] = (
    Fault("Sunda Megathrust (Sumatra segment)", "subduction", 50.0, (
        (5.5, 94.0), (3.0, 95.0), (0.0, 97.5), (-3.0, 100.5), (-5.5, 102.5), (-7.0, 105.5),
    )),
    Fault("Sunda Megathrust (Java segment)", "subduction", 50.0, (
        (-7.0, 105.5), (-9.0, 109.0), (-10.0, 113.0), (-10.5, 116.0),
    )),
    Fault("Sunda Megathrust (Bali-Sumba segment)", "subduction", 45.0, (
        (-10.5, 116.0), (-11.0, 119.0), (-11.0, 123.0),
    )),
    Fault("Sumatra Fault", "transform", 12.0, (
        (5.5, 95.5), (3.0, 97.0), (0.0, 99.5), (-3.0, 102.0), (-5.0, 104.0),
    )),
    Fault("Palu-Koro Fault", "transform", 35.0, (
        (-0.5, 119.5), (-1.5, 120.0), (-2.5, 120.5),
    )),
    Fault("Sorong Fault", "transform", 30.0, (
        (-1.0, 131.0), (-1.0, 134.0), (-1.0, 136.0),
    )),
    Fault("Tarera-Aiduna Fault", "transform", 25.0, (
        (-3.0, 134.0), (-4.0, 136.0), (-4.5, 138.0),
    )),
    Fault("Banda Sea Subduction", "subduction", 60.0, (
        (-7.0, 124.0), (-7.5, 127.0), (-8.0, 130.0), (-8.5, 132.0),
    )),
    Fault("North Sulawesi Trench", "subduction", 40.0, (
        (1.5, 123.0), (1.5, 125.0), (2.0, 127.0),
    )),
    Fault("Philippine Trench (north Maluku)", "subduction", 50.0, (
        (1.0, 126.5), (3.0, 127.0), (5.0, 127.5),
    )),
    Fault("New Guinea Trench (Papua)", "subduction", 40.0, (
        (-1.5, 137.0), (-1.5, 139.0), (-2.0, 141.0),
    )),
    Fault("Matano Fault (Sulawesi)", "transform", 30.0, (
        (-2.5, 121.0), (-2.7, 122.0),
    )),
    Fault("Lawanopo Fault (Sulawesi)", "transform", 20.0, (
        (-3.5, 121.5), (-4.0, 122.5),
    )),
    Fault("Cimandiri Fault (W Java)", "reverse", 4.0, (
        (-7.0, 106.5), (-7.2, 107.5),
    )),
    Fault("Lembang Fault (Bandung)", "reverse", 3.0, (
        (-6.85, 107.5), (-6.85, 107.7),
    )),
    Fault("Opak Fault (Yogyakarta)", "normal", 2.0, (
        (-7.7, 110.3), (-8.1, 110.4),
    )),
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _distance_to_polyline_km(lat: float, lon: float, polyline: tuple[tuple[float, float], ...]) -> float:
    """Min haversine distance from point to any segment of the polyline.

    Approximation: sample each segment at 0.1° intervals and take min point distance.
    Sufficient for grid-cell-level features.
    """
    best = float("inf")
    for (la1, lo1), (la2, lo2) in zip(polyline, polyline[1:]):
        seg_len = math.hypot(la2 - la1, lo2 - lo1)
        steps = max(1, int(seg_len * 10))  # ~0.1° per step
        for s in range(steps + 1):
            t = s / steps
            la = la1 + (la2 - la1) * t
            lo = lo1 + (lo2 - lo1) * t
            d = _haversine_km(lat, lon, la, lo)
            if d < best:
                best = d
    return best


def nearest_fault(lat: float, lon: float) -> tuple[Fault, float]:
    """Return (nearest_fault, distance_km) for a given point."""
    best_d = float("inf")
    best_f: Fault = FAULTS[0]
    for f in FAULTS:
        d = _distance_to_polyline_km(lat, lon, f.polyline)
        if d < best_d:
            best_d, best_f = d, f
    return best_f, best_d


def has_real_pusgen(geo_dir: Path) -> bool:
    """Check if a real PUSGEN/GEM shapefile is present."""
    return any((geo_dir / name).exists() for name in ("pusgen_faults.shp", "gem_active_faults.geojson"))
