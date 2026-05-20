"""Cell → (province, sub-region) labelling.

Two modes:
1. Shapefile mode: spatial join via geopandas to GADM shapefile (accurate).
2. Fallback mode: hardcoded province bboxes + anchor cities (no internet).

Always returns a `Label` with `province`, `subregion`, `full_label`,
`is_offshore`, `region_macro`. Offshore = cell centroid not inside any
province bbox/polygon → mapped to nearest province by centroid distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from backend.app.core._provinces import PROVINCES, ProvinceBox
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Label:
    province: str
    subregion: str
    full_label: str
    is_offshore: bool
    region_macro: str


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _nearest_anchor(lat: float, lon: float, province: ProvinceBox) -> str:
    """Return the name of the closest anchor city in the given province."""
    best_dist = float("inf")
    best_name = province.name
    for anchor in province.anchors:
        d = _haversine_km(lat, lon, anchor.lat, anchor.lon)
        if d < best_dist:
            best_dist = d
            best_name = anchor.name
    return best_name


def _province_containing(lat: float, lon: float) -> ProvinceBox | None:
    """Smallest-area bbox that contains (lat, lon), or None."""
    candidates = [p for p in PROVINCES if p.contains(lat, lon)]
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.area)


def _nearest_province(lat: float, lon: float) -> ProvinceBox:
    """Province whose bbox center is closest to (lat, lon)."""
    return min(
        PROVINCES,
        key=lambda p: _haversine_km(lat, lon, p.center[0], p.center[1]),
    )


def label_cell_fallback(lat: float, lon: float) -> Label:
    """Geocode a single cell using the bbox+anchor fallback."""
    province = _province_containing(lat, lon)
    if province is not None:
        sub = _nearest_anchor(lat, lon, province)
        return Label(
            province=province.name,
            subregion=sub,
            full_label=f"{province.name} - {sub}",
            is_offshore=False,
            region_macro=province.macro,
        )

    nearest = _nearest_province(lat, lon)
    sub = _nearest_anchor(lat, lon, nearest)
    return Label(
        province=nearest.name,
        subregion=sub,
        full_label=f"Lepas Pantai {nearest.name} - dekat {sub}",
        is_offshore=True,
        region_macro=nearest.macro,
    )


def shapefile_available(geo_dir: Path) -> bool:
    """Check whether a usable GADM shapefile is present."""
    candidates = ["gadm41_IDN_1.shp", "gadm36_IDN_1.shp", "gadm_IDN_1.shp"]
    return any((geo_dir / c).exists() for c in candidates)


def label_cells_shapefile(
    cells: list[tuple[str, float, float]],
    geo_dir: Path,
) -> dict[str, Label]:
    """Bulk geocode using GADM shapefile via geopandas.

    cells: list of (cell_id, lat, lon).
    Falls back to bbox approach for any cell that fails to match a polygon.
    """
    try:
        import geopandas as gpd  # type: ignore[import-not-found]
        from shapely.geometry import Point  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("geopandas_not_installed_fallback_to_bbox")
        return {cid: label_cell_fallback(lat, lon) for cid, lat, lon in cells}

    shp_path: Path | None = None
    for candidate in ("gadm41_IDN_1.shp", "gadm36_IDN_1.shp", "gadm_IDN_1.shp"):
        p = geo_dir / candidate
        if p.exists():
            shp_path = p
            break

    if shp_path is None:
        logger.warning("gadm_shapefile_missing_fallback_to_bbox", geo_dir=str(geo_dir))
        return {cid: label_cell_fallback(lat, lon) for cid, lat, lon in cells}

    logger.info("geocode_shapefile_loading", path=str(shp_path))
    gdf = gpd.read_file(shp_path)
    name_col = next((c for c in ("NAME_1", "name_1", "PROVINCE", "province") if c in gdf.columns), None)
    if name_col is None:
        logger.warning("shapefile_missing_name_column_fallback")
        return {cid: label_cell_fallback(lat, lon) for cid, lat, lon in cells}

    points = gpd.GeoDataFrame(
        {"cell_id": [c[0] for c in cells]},
        geometry=[Point(c[2], c[1]) for c in cells],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, gdf[[name_col, "geometry"]], how="left", predicate="within")

    out: dict[str, Label] = {}
    for cell_id, lat, lon in cells:
        match = joined[joined["cell_id"] == cell_id]
        if not match.empty and not match.iloc[0][name_col] is None:
            province_name = str(match.iloc[0][name_col])
            macro = _macro_for(province_name)
            sub = _nearest_anchor_for_name(lat, lon, province_name)
            out[cell_id] = Label(
                province=province_name,
                subregion=sub,
                full_label=f"{province_name} - {sub}",
                is_offshore=False,
                region_macro=macro,
            )
        else:
            out[cell_id] = label_cell_fallback(lat, lon)
    return out


def _macro_for(province_name: str) -> str:
    for p in PROVINCES:
        if p.name.lower() == province_name.lower():
            return p.macro
    # Heuristic for shapefile names not in our table
    name = province_name.lower()
    if "sumat" in name or "aceh" in name or "riau" in name or "lampung" in name or "jambi" in name or "bangka" in name or "bengkulu" in name:
        return "Sumatera"
    if "jawa" in name or "jakarta" in name or "banten" in name or "yogyakarta" in name:
        return "Jawa"
    if "bali" in name or "nusa tenggara" in name:
        return "BaliNusa"
    if "kalimantan" in name:
        return "Kalimantan"
    if "sulawesi" in name or "gorontalo" in name:
        return "Sulawesi"
    if "maluku" in name or "papua" in name:
        return "MalukuPapua"
    return "Other"


def _nearest_anchor_for_name(lat: float, lon: float, province_name: str) -> str:
    for p in PROVINCES:
        if p.name.lower() == province_name.lower():
            return _nearest_anchor(lat, lon, p)
    # Unknown province: fall back to nearest by overall distance
    return _nearest_anchor(lat, lon, _nearest_province(lat, lon))


def label_cells(
    cells: list[tuple[str, float, float]],
    geo_dir: Path,
) -> dict[str, Label]:
    """Top-level entry point. Picks shapefile mode if available, else fallback."""
    if shapefile_available(geo_dir):
        return label_cells_shapefile(cells, geo_dir)
    logger.info("geocode_fallback_mode", reason="gadm_shapefile_not_found")
    return {cid: label_cell_fallback(lat, lon) for cid, lat, lon in cells}
