"""Tests for grid generator."""

from __future__ import annotations

from backend.app.config import get_settings
from backend.app.core.grid import generate_grid, iter_grid_cells, make_cell_id


def test_grid_count_matches_bbox() -> None:
    """Grid covers full Indonesia bbox at 0.5° step."""
    cells = generate_grid()
    s = get_settings()
    n_lat = round((s.grid_lat_max - s.grid_lat_min) / s.grid_step)
    n_lon = round((s.grid_lon_max - s.grid_lon_min) / s.grid_step)
    assert len(cells) == n_lat * n_lon
    # Sanity: about 34 lat * 92 lon = 3128 raw cells (no land filter at this layer).
    assert 3000 < len(cells) < 4000


def test_cells_have_unique_ids() -> None:
    cells = generate_grid()
    ids = {c.cell_id for c in cells}
    assert len(ids) == len(cells)


def test_cell_bounds_consistent() -> None:
    """Center is midpoint of bounds; bounds are step apart."""
    s = get_settings()
    for c in iter_grid_cells():
        assert abs(c.lat - (c.lat_min + c.lat_max) / 2) < 1e-9
        assert abs(c.lon - (c.lon_min + c.lon_max) / 2) < 1e-9
        assert abs((c.lat_max - c.lat_min) - s.grid_step) < 1e-9
        assert abs((c.lon_max - c.lon_min) - s.grid_step) < 1e-9


def test_make_cell_id_stable() -> None:
    """Cell id is deterministic from coordinates."""
    a = make_cell_id(-0.75, 119.75)
    b = make_cell_id(-0.75, 119.75)
    assert a == b
    assert a != make_cell_id(-0.25, 119.75)


def test_cell_for_palu_exists() -> None:
    """The cell containing Palu (~-0.9, 119.87) is in the grid."""
    cells = generate_grid()
    matches = [c for c in cells if c.lat_min <= -0.9 <= c.lat_max and c.lon_min <= 119.87 <= c.lon_max]
    assert len(matches) == 1
