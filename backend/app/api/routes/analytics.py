"""Admin analytics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import require_admin_token
from backend.app.services.analytics import daily_active_users

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dau", dependencies=[Depends(require_admin_token)])
def get_dau(days: int = Query(default=14, ge=1, le=90)) -> dict:
    return daily_active_users(days)
