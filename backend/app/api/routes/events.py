"""GET /api/events — recent earthquakes feed."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import require_admin_token
from backend.app.data.ingest import ingest_realtime, list_events

router = APIRouter(prefix="/events", tags=["events"])


@router.get("")
def get_events(
    days: int = Query(default=7, ge=1, le=3650),
    min_mag: float | None = Query(default=None, ge=0, le=10),
    source: str | None = Query(default=None, pattern="^(usgs|bmkg)$"),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    items = list_events(days=days, min_mag=min_mag, source=source, limit=limit)
    return {"count": len(items), "items": items}


@router.post("/ingest", dependencies=[Depends(require_admin_token)])
def trigger_ingest(
    fetch_usgs: bool = True,
    fetch_bmkg: bool = True,
    lookback_hours: int = 24,
) -> dict:
    return ingest_realtime(
        fetch_usgs=fetch_usgs, fetch_bmkg=fetch_bmkg, lookback_hours=lookback_hours
    )
