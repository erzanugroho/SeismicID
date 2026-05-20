"""Area-labels service: bootstrap (populate from grid + geocoder) and queries."""

from __future__ import annotations

from typing import Any

from backend.app.config import get_settings
from backend.app.core.geocode import label_cells
from backend.app.core.grid import generate_grid
from backend.app.core.logging import get_logger
from backend.app.db.sqlite import get_connection, migrate
from backend.app.features.physics import static_physics_features

logger = get_logger(__name__)


def count_area_labels() -> int:
    migrate()  # ensure table exists
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS n FROM area_labels")
        row = cur.fetchone()
        return int(row["n"]) if row else 0


def bootstrap_area_labels(force: bool = False) -> int:
    """Populate `area_labels` from grid generator + geocoder.

    Returns number of inserted rows. No-op if table already populated and not forced.
    """
    migrate()
    if not force and count_area_labels() > 0:
        logger.info("area_labels_already_populated", count=count_area_labels())
        return 0

    settings = get_settings()
    cells = generate_grid(settings)
    logger.info("area_labels_bootstrap_start", n_cells=len(cells))

    pairs = [(c.cell_id, c.lat, c.lon) for c in cells]
    labels = label_cells(pairs, settings.geo_path)

    rows: list[tuple[Any, ...]] = []
    for c in cells:
        lbl = labels[c.cell_id]
        physics = static_physics_features(c.lat, c.lon)
        rows.append(
            (
                c.cell_id,
                c.lat,
                c.lon,
                c.lat_min,
                c.lat_max,
                c.lon_min,
                c.lon_max,
                lbl.province,
                lbl.subregion,
                lbl.full_label,
                1 if lbl.is_offshore else 0,
                lbl.region_macro,
                physics["nearest_fault_km"],
                physics["fault_type"],
                physics["fault_slip_rate"],
                physics["slab_depth_km"],
            )
        )

    with get_connection() as conn:
        if force:
            conn.execute("DELETE FROM area_labels")
        conn.execute("BEGIN")
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO area_labels
                   (cell_id, lat, lon, lat_min, lat_max, lon_min, lon_max,
                    province, subregion, full_label, is_offshore, region_macro,
                    nearest_fault_km, fault_type, fault_slip_rate, slab_depth_km)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    logger.info("area_labels_bootstrap_done", inserted=len(rows))
    return len(rows)


def list_areas(province: str | None = None, region_macro: str | None = None) -> list[dict]:
    where = []
    args: list[Any] = []
    if province:
        where.append("province = ?")
        args.append(province)
    if region_macro:
        where.append("region_macro = ?")
        args.append(region_macro)
    sql = "SELECT * FROM area_labels"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY full_label"

    with get_connection() as conn:
        cur = conn.execute(sql, args)
        return [dict(row) for row in cur.fetchall()]
