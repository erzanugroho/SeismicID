"""GET /api/scheduler/runs and POST /api/scheduler/trigger."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import require_admin_token
from backend.app.scheduler.runner import list_runs, trigger_job

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


@router.get("/runs")
def get_runs(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    items = list_runs(limit=limit)
    return {"count": len(items), "items": items}


@router.post("/trigger/{job_name}", dependencies=[Depends(require_admin_token)])
def trigger(job_name: str) -> dict:
    return trigger_job(job_name)
