"""Tests for fallback geocoder."""

from __future__ import annotations

from pathlib import Path

from backend.app.core.geocode import label_cell_fallback, label_cells


def test_palu_falls_in_sulawesi_tengah() -> None:
    label = label_cell_fallback(-0.90, 119.87)
    assert label.province == "Sulawesi Tengah"
    assert label.subregion == "Palu"
    assert label.full_label == "Sulawesi Tengah - Palu"
    assert label.is_offshore is False
    assert label.region_macro == "Sulawesi"


def test_jakarta_falls_in_dki() -> None:
    label = label_cell_fallback(-6.2, 106.85)
    assert "Jakarta" in label.full_label
    assert label.region_macro == "Jawa"
    assert label.is_offshore is False


def test_padang_falls_in_sumatera_barat() -> None:
    label = label_cell_fallback(-0.95, 100.35)
    assert label.province == "Sumatera Barat"
    assert label.subregion == "Padang"
    assert label.region_macro == "Sumatera"


def test_offshore_indian_ocean_marked_offshore() -> None:
    """Far southwest of Sumatera = open ocean."""
    label = label_cell_fallback(-9.0, 95.5)
    assert label.is_offshore is True
    assert "Lepas Pantai" in label.full_label


def test_label_cells_batch_with_fallback(tmp_path: Path) -> None:
    """Batch labelling without shapefile uses fallback for every cell."""
    cells = [
        ("c1", -0.90, 119.87),
        ("c2", -6.20, 106.85),
        ("c3", -9.0, 95.5),
    ]
    out = label_cells(cells, geo_dir=tmp_path)
    assert set(out.keys()) == {"c1", "c2", "c3"}
    assert out["c1"].province == "Sulawesi Tengah"
    assert out["c3"].is_offshore is True
