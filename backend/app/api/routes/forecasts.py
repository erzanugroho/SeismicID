"""GET/POST /api/forecasts endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.deps import require_admin_token
from backend.app.config import get_settings
from backend.app.features.labels import HORIZONS, THRESHOLDS
from backend.app.services.forecast_service import (
    format_sentence,
    get_area_forecasts,
    get_forecast_status,
    get_latest_forecasts,
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
) -> dict:
    s = get_settings()
    h = horizon or s.default_horizon_days
    t = threshold or s.default_mag_threshold
    _validate(h, t)
    items = get_latest_forecasts(horizon_days=h, mag_threshold=t)
    return {"horizon_days": h, "mag_threshold": t, "count": len(items), "items": items}


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


@router.get("/area/{cell_id}")
def area(cell_id: str) -> dict:
    out = get_area_forecasts(cell_id)
    if not out:
        raise HTTPException(404, f"cell {cell_id} not found")
    return out


@router.get("/status")
def status() -> dict:
    return get_forecast_status()


@status_router.get("/status")
def singular_status() -> dict:
    return get_forecast_status()


@router.post("/run", dependencies=[Depends(require_admin_token)])
def trigger_run(force_demo: bool = False) -> dict:
    return run_forecast(force_demo=force_demo)
