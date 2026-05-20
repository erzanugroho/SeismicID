"""GET /api/areas — list grid cells with labels."""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.services.area_service import bootstrap_area_labels, count_area_labels, list_areas

router = APIRouter(prefix="/areas", tags=["areas"])


@router.get("")
def get_areas(
    province: str | None = Query(default=None),
    region_macro: str | None = Query(default=None, description="Sumatera|Jawa|BaliNusa|Kalimantan|Sulawesi|MalukuPapua"),
) -> dict:
    """Return all grid cells with labels.

    Bootstraps `area_labels` lazily on first call.
    """
    if count_area_labels() == 0:
        bootstrap_area_labels()
    items = list_areas(province=province, region_macro=region_macro)
    return {"count": len(items), "items": items}


@router.post("/bootstrap")
def trigger_bootstrap(force: bool = False) -> dict:
    """Re-populate area_labels (admin)."""
    inserted = bootstrap_area_labels(force=force)
    return {"inserted": inserted, "total": count_area_labels()}
