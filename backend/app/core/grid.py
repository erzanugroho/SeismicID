"""Indonesia grid generation for forecast cells.

Deterministic 0.5°×0.5° grid covering the Indonesian bounding box. Each cell
gets a stable cell_id based on rounded lat/lon coordinates so the ID survives
config tweaks as long as `grid_step` is unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from backend.app.config import Settings, get_settings


@dataclass(frozen=True)
class GridCell:
    """One forecast cell."""

    cell_id: str
    lat: float
    lon: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(lat_min, lon_min, lat_max, lon_max)."""
        return (self.lat_min, self.lon_min, self.lat_max, self.lon_max)


def make_cell_id(lat: float, lon: float) -> str:
    """Stable cell ID from cell center coordinates.

    Format: "C{lat10}_{lon10}" where coords are multiplied by 10 (positive ints)
    e.g., lat=-0.75, lon=119.25 → "C-08_1193".
    """
    lat_int = round(lat * 10)
    lon_int = round(lon * 10)
    return f"C{lat_int:+03d}_{lon_int:+04d}".replace("+", "p").replace("-", "m")


def iter_grid_cells(settings: Settings | None = None) -> Iterator[GridCell]:
    """Yield grid cells over the configured bounding box."""
    s = settings or get_settings()
    step = s.grid_step
    half = step / 2.0

    # Use integer multiplication to avoid floating drift.
    n_lat = round((s.grid_lat_max - s.grid_lat_min) / step)
    n_lon = round((s.grid_lon_max - s.grid_lon_min) / step)

    for i in range(n_lat):
        lat_min = s.grid_lat_min + i * step
        lat_max = lat_min + step
        center_lat = lat_min + half
        for j in range(n_lon):
            lon_min = s.grid_lon_min + j * step
            lon_max = lon_min + step
            center_lon = lon_min + half
            yield GridCell(
                cell_id=make_cell_id(center_lat, center_lon),
                lat=round(center_lat, 4),
                lon=round(center_lon, 4),
                lat_min=round(lat_min, 4),
                lat_max=round(lat_max, 4),
                lon_min=round(lon_min, 4),
                lon_max=round(lon_max, 4),
            )


def generate_grid(settings: Settings | None = None) -> list[GridCell]:
    """Return all grid cells as a list."""
    return list(iter_grid_cells(settings))
