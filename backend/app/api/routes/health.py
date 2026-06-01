"""Health check endpoints."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.db.sqlite import get_connection, migrate
from backend.app.services.forecast_service import get_forecast_status

router = APIRouter(tags=["health"])

_START_TIME = time.monotonic()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@router.get("/health")
def health() -> dict:
    """Lightweight liveness probe."""
    settings = get_settings()
    return {
        "status": "ok",
        "name": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
        "role": settings.app_role,
        "uptime_seconds": round(time.monotonic() - _START_TIME, 2),
    }


@router.get("/health/readiness")
def readiness() -> dict[str, Any]:
    """Quality/readiness probe: DB, model, forecast freshness, scheduler health."""
    settings = get_settings()
    checks: dict[str, Any] = {}

    try:
        migrate()
        with get_connection() as conn:
            event_count = int(conn.execute("SELECT COUNT(*) AS n FROM realtime_events").fetchone()["n"])
            model_row = conn.execute(
                "SELECT version FROM model_metadata WHERE is_active = 1 ORDER BY training_date DESC LIMIT 1"
            ).fetchone()
            last_success = conn.execute(
                """SELECT job_name, started_at, finished_at, status
                   FROM scheduler_runs
                   WHERE status = 'success'
                   ORDER BY started_at DESC
                   LIMIT 1"""
            ).fetchone()
            recent_event = conn.execute(
                """SELECT time, magnitude, place, source
                   FROM realtime_events
                   ORDER BY time DESC
                   LIMIT 1"""
            ).fetchone()
        checks["db"] = {"ok": True, "event_count": event_count}
        checks["active_model"] = {"ok": model_row is not None, "version": model_row["version"] if model_row else None}
        checks["scheduler_last_success"] = dict(last_success) if last_success else None
        checks["recent_event"] = dict(recent_event) if recent_event else None
    except Exception as exc:  # noqa: BLE001
        checks["db"] = {"ok": False, "error": str(exc)}

    age_hours = None
    forecast_fresh = False
    try:
        status = get_forecast_status()
        last_forecast = _parse_dt(status.get("forecast_last_computed_at"))
        if last_forecast:
            age_hours = (datetime.now(UTC) - last_forecast).total_seconds() / 3600
            forecast_fresh = age_hours <= max(settings.forecast_fallback_hours * 2, 6)
        checks["forecast"] = {
            "ok": forecast_fresh,
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
            "mode": status.get("forecast_mode"),
            "model_version": status.get("forecast_model_version"),
        }
    except Exception as exc:  # noqa: BLE001
        checks["forecast"] = {"ok": False, "error": str(exc)}

    model_path = Path(settings.models_path)
    checks["model_dir"] = {"ok": model_path.exists(), "path": str(model_path)}
    checks["telegram"] = {
        "bot_configured": bool(settings.telegram_bot_token),
        "admin_chat_configured": bool(settings.telegram_chat_id),
        "webhook_secret_configured": bool(settings.telegram_webhook_secret),
    }
    try:
        settings.backup_path.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=settings.backup_path, prefix="health-", delete=True):
            pass
        checks["backup_dir"] = {"ok": True, "path": str(settings.backup_path)}
    except Exception as exc:  # noqa: BLE001
        checks["backup_dir"] = {"ok": False, "path": str(settings.backup_path), "error": str(exc)}

    ok = bool(checks.get("db", {}).get("ok")) and forecast_fresh and bool(checks.get("backup_dir", {}).get("ok"))
    return {"status": "ready" if ok else "degraded", "ok": ok, "checks": checks}
