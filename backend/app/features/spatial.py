"""Spatial neighbor utilities for grid cells."""

from __future__ import annotations

import math

from backend.app.core.grid import GridCell, generate_grid


def find_neighbors(cells: list[GridCell], cell_id: str, *, k: int = 8) -> list[str]:
    """Return cell_ids of k nearest neighbors by centroid haversine distance."""
    target = next((c for c in cells if c.cell_id == cell_id), None)
    if target is None:
        return []
    others = [c for c in cells if c.cell_id != cell_id]
    others.sort(key=lambda c: _haversine_km(target.lat, target.lon, c.lat, c.lon))
    return [c.cell_id for c in others[:k]]


def neighbor_map(cells: list[GridCell] | None = None, *, k: int = 8) -> dict[str, list[str]]:
    """Precompute neighbors for all cells. Cached friendly."""
    cells = cells or generate_grid()
    return {c.cell_id: find_neighbors(cells, c.cell_id, k=k) for c in cells}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))
