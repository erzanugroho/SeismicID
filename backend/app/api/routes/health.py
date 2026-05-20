"""Health check endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter

from backend.app.config import get_settings

router = APIRouter(tags=["health"])

_START_TIME = time.monotonic()


@router.get("/health")
def health() -> dict:
    """Liveness/readiness probe."""
    settings = get_settings()
    return {
        "status": "ok",
        "name": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
        "uptime_seconds": round(time.monotonic() - _START_TIME, 2),
    }
