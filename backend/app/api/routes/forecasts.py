"""GET/POST /api/forecasts endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.deps import guarded_admin_job, require_admin_token
from backend.app.config import get_settings
from backend.app.features.labels import HORIZONS, THRESHOLDS
from backend.app.services.forecast_service import (
    ALLOWED_CLUSTER_SORTS,
    format_cluster_sentence,
    format_sentence,
    get_area_forecasts,
    get_cluster_forecasts,
    get_forecast_status,
    get_latest_forecasts,
    get_tier_distribution,
    get_top_clusters,
    get_top_forecasts,
    run_forecast,
)

router = APIRouter(prefix="/forecasts", tags=["forecasts"])
status_router = APIRouter(prefix="/forecast", tags=["forecasts"])


def _validate(horizon: int, threshold: float) -> None:
    if horizon not in HORIZONS:
        raise HTTPException(400, f"horizon must be one of {list(HORIZONS)}")
    if threshold not in THRESHOLDS:
        raise HTTPException(400, f"threshold must be one of {list(THRESHOLDS)}")


@router.get("/latest")
def latest(
    horizon: int = Query(default=None),
    threshold: float = Query(default=None),
    min_probability: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int | None = Query(default=None, ge=1, le=5000),
) -> dict:
    s = get_settings()
    h = horizon or s.default_horizon_days
    t = threshold or s.default_mag_threshold
    _validate(h, t)
    items = get_latest_forecasts(horizon_days=h, mag_threshold=t, min_probability=min_probability, limit=limit)
    status = get_forecast_status()
    return {
        "horizon_days": h,
        "mag_threshold": t,
        "count": len(items),
        "items": items,
        "baseline_type": status.get("forecast_baseline_type"),
        "forecast_mode": status.get("forecast_mode"),
        "computed_at": status.get("forecast_last_computed_at"),
    }


@router.get("/top")
def top(
    n: int = Query(default=10, ge=1, le=100),
    horizon: int = Query(default=None),
    threshold: float = Query(default=None),
) -> dict:
    s = get_settings()
    h = horizon or s.default_horizon_days
    t = threshold or s.default_mag_threshold
    _validate(h, t)
    items = get_top_forecasts(horizon_days=h, mag_threshold=t, n=n)
    sentences = [
        format_sentence(it, it["probability"], horizon_days=h, mag_threshold=t) for it in items
    ]
    return {"horizon_days": h, "mag_threshold": t, "n": n, "items": items, "sentences": sentences}


@router.get("/top-risk")
def top_risk(
    horizon: int = Query(default=None),
    threshold: float = Query(default=None),
    limit: int = Query(default=10, ge=1, le=100),
) -> dict:
    return top(n=limit, horizon=horizon, threshold=threshold)


@router.get("/area/{cell_id}")
def area(cell_id: str) -> dict:
    out = get_area_forecasts(cell_id)
    if not out:
        raise HTTPException(404, f"cell {cell_id} not found")
    return out


def _validate_cluster_sort(sort_by: str) -> None:
    if sort_by not in ALLOWED_CLUSTER_SORTS:
        raise HTTPException(
            400, f"sort_by must be one of {sorted(ALLOWED_CLUSTER_SORTS)}, got {sort_by!r}"
        )


@router.get("/top-clusters")
def top_clusters(
    n: int = Query(default=10, ge=1, le=100),
    horizon: int = Query(default=None),
    threshold: float = Query(default=None),
    sort_by: str = Query(default="top3_mean"),
) -> dict:
    """Top-N subregion clusters ranked by the chosen aggregation metric.

    sort_by:
      - top3_mean (default) — mean of the 3 highest cells in the cluster
      - max                 — single worst cell in the cluster
      - any_cell            — 1 - Π(1 - pᵢ); probability that ≥1 cell exceeds threshold
      - mean                — mean of all cells (rarely the right choice)
    """
    s = get_settings()
    h = horizon or s.default_horizon_days
    t = threshold or s.default_mag_threshold
    _validate(h, t)
    _validate_cluster_sort(sort_by)
    items = get_top_clusters(horizon_days=h, mag_threshold=t, n=n, sort_by=sort_by)
    sentences = [
        format_cluster_sentence(c, horizon_days=h, mag_threshold=t, sort_by=sort_by) for c in items
    ]
    return {
        "horizon_days": h,
        "mag_threshold": t,
        "n": n,
        "sort_by": sort_by,
        "count": len(items),
        "items": items,
        "sentences": sentences,
    }


@router.get("/clusters-latest")
def clusters_latest(
    horizon: int = Query(default=None),
    threshold: float = Query(default=None),
    sort_by: str = Query(default="top3_mean"),
    region_macro: str | None = Query(default=None),
    province: str | None = Query(default=None),
    min_probability: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int | None = Query(default=None, ge=1, le=1000),
) -> dict:
    """All clusters for (horizon, threshold), sorted by the chosen metric."""
    s = get_settings()
    h = horizon or s.default_horizon_days
    t = threshold or s.default_mag_threshold
    _validate(h, t)
    _validate_cluster_sort(sort_by)
    items = get_cluster_forecasts(
        horizon_days=h,
        mag_threshold=t,
        sort_by=sort_by,
        region_macro=region_macro,
        province=province,
        min_probability=min_probability,
    )
    if limit:
        items = items[:limit]
    return {
        "horizon_days": h,
        "mag_threshold": t,
        "sort_by": sort_by,
        "count": len(items),
        "items": items,
    }


@router.get("/status")
def status() -> dict:
    return get_forecast_status()


@status_router.get("/status")
def singular_status() -> dict:
    return get_forecast_status()


@router.get("/tier-distribution")
def tier_distribution(
    hours: int = Query(default=24, ge=1, le=24 * 30),
) -> dict:
    """How many forecast runs landed in each tier over the past ``hours``.

    Surface for operators to confirm the ETAS-Ogata fallback tier is firing
    when expected. Without this endpoint the ENABLE_ETAS_BASELINE_TIER flag
    is invisible at runtime.
    """
    return get_tier_distribution(hours=hours)


@router.post("/run", dependencies=[Depends(require_admin_token)])
@guarded_admin_job("forecast_recompute")
def trigger_run(force_demo: bool = False) -> dict:
    return run_forecast(force_demo=force_demo)
