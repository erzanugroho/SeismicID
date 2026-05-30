"""Background jobs.

Each job logs its start/end + outcome to scheduler_runs.
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.data.ingest import ingest_realtime
from backend.app.db.metadata import get_metadata_value, set_metadata_value
from backend.app.db.sqlite import get_connection, migrate
from backend.app.services.forecast_service import run_forecast
from backend.app.services.telegram_alerts import send_daily_forecast_report, send_forecast_alert

logger = get_logger(__name__)


def set_status_value(key: str, value: str | None) -> None:
    """Set scheduler/forecast status metadata."""
    set_metadata_value(key, value)


def get_status_value(key: str) -> str | None:
    """Get scheduler/forecast status metadata."""
    return get_metadata_value(key)


def _record_start(job_name: str) -> int:
    migrate()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO scheduler_runs (job_name, started_at, status) VALUES (?,?,?)",
            (job_name, datetime.now(UTC).isoformat(), "running"),
        )
        run_id = cur.lastrowid
        if run_id is None:
            raise RuntimeError("failed to record scheduler run start")
        return int(run_id)


def _record_end(
    run_id: int,
    *,
    status: str,
    error: str | None = None,
    items: int | None = None,
    meta: dict | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """UPDATE scheduler_runs
               SET finished_at = ?, status = ?, error = ?, items_processed = ?, metadata_json = ?
               WHERE id = ?""",
            (
                datetime.now(UTC).isoformat(),
                status,
                error,
                items,
                json.dumps(meta) if meta else None,
                run_id,
            ),
        )


def _run_job_with_logging(job_name: str, fn: Callable[[], dict]) -> dict:
    run_id = _record_start(job_name)
    try:
        result = fn()
        items = None
        meta = None
        if isinstance(result, dict):
            items = result.get("rows_written") or result.get("stored") or result.get("items")
            meta = result
        _record_end(run_id, status="success", items=items, meta=meta)
        logger.info("scheduler_job_done", job=job_name, items=items)
        return result
    except Exception as e:  # noqa: BLE001
        _record_end(run_id, status="error", error=str(e))
        logger.error("scheduler_job_error", job=job_name, error=str(e), tb=traceback.format_exc())
        raise


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _latest_event() -> dict[str, Any] | None:
    migrate()
    with get_connection() as conn:
        row = conn.execute(
            """SELECT event_id, time, magnitude, source, place
               FROM realtime_events
               ORDER BY time DESC, event_id DESC
               LIMIT 1"""
        ).fetchone()
    return dict(row) if row else None


def _event_time(event_id: str | None) -> str | None:
    if not event_id:
        return None
    migrate()
    with get_connection() as conn:
        row = conn.execute("SELECT time FROM realtime_events WHERE event_id = ?", (event_id,)).fetchone()
    return None if row is None else row["time"]


def _count_events_after(marker_time: str | None) -> int:
    migrate()
    with get_connection() as conn:
        if marker_time is None:
            row = conn.execute("SELECT COUNT(*) AS n FROM realtime_events").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM realtime_events WHERE time > ?",
                (marker_time,),
            ).fetchone()
    return int(row["n"])


def _should_run_fallback(now: datetime) -> bool:
    settings = get_settings()
    last_forecast = _parse_dt(get_status_value("last_forecast_at"))
    if last_forecast is None:
        return True
    return now - last_forecast >= timedelta(hours=settings.forecast_fallback_hours)


def _inside_debounce(now: datetime) -> bool:
    settings = get_settings()
    last_forecast = _parse_dt(get_status_value("last_forecast_at"))
    if last_forecast is None:
        return False
    return now - last_forecast < timedelta(minutes=settings.forecast_debounce_minutes)


# === Job functions ===

def realtime_fetch() -> dict:
    return _run_job_with_logging("realtime_fetch", ingest_realtime)


def forecast_recompute() -> dict:
    result = _run_job_with_logging("forecast_recompute", run_forecast)
    send_forecast_alert(result)
    return result


def scheduler_tick(*, now: datetime | None = None) -> dict:
    """One-shot worker tick: fetch data, detect any new event, maybe run forecast.

    Forecast trigger policy is `any_new_event`: every new catalog event counts,
    including small earthquakes. Debounce prevents duplicate runs during bursts.
    """
    settings = get_settings()
    now = now or datetime.now(UTC)
    checked_at = now.isoformat()
    ingest_result = ingest_realtime()

    latest = _latest_event()
    last_seen_id = get_status_value("last_seen_event_id")
    marker_time = get_status_value("last_seen_event_time") or _event_time(last_seen_id)
    new_events = _count_events_after(marker_time) if latest else 0

    set_status_value("last_checked_at", checked_at)
    set_status_value("last_ingest_result", json.dumps(ingest_result, default=str))
    set_status_value("new_events_since_last_forecast", str(new_events))

    forecast_ran = False
    forecast_result: dict | None = None
    reason = "no_new_events"

    if new_events > 0:
        if _inside_debounce(now):
            reason = "debounced"
        else:
            reason = "new_events"
            forecast_result = run_forecast()
            forecast_ran = True
    elif _should_run_fallback(now):
        reason = "fallback"
        forecast_result = run_forecast()
        forecast_ran = True

    if forecast_ran and latest:
        set_status_value("last_seen_event_id", str(latest["event_id"]))
        set_status_value("last_seen_event_time", str(latest["time"]))
        set_status_value("new_events_since_last_forecast", "0")
    set_status_value("last_forecast_reason", reason)

    alert_sent = send_forecast_alert(forecast_result) if forecast_ran else False

    out = {
        "ok": True,
        "trigger_mode": settings.forecast_trigger_mode,
        "checked_at": checked_at,
        "new_events": new_events,
        "latest_event": latest,
        "forecast_ran": forecast_ran,
        "reason": reason,
        "ingest": ingest_result,
        "forecast": forecast_result,
        "telegram_alert_sent": alert_sent,
    }
    logger.info("scheduler_tick_done", **out)
    return out


def scheduler_tick_job() -> dict:
    return _run_job_with_logging("scheduler_tick", scheduler_tick)


def telegram_daily_report() -> dict:
    ok = send_daily_forecast_report()
    return {"ok": ok}


def retrain() -> dict:
    """Retrain heavy job. Skipped if not enough data; logs warning otherwise."""

    def _do_retrain() -> dict:
        from backend.app.data.catalog import read_historical_events

        df = read_historical_events()
        if len(df) < 10000:
            logger.warning("retrain_skipped_insufficient_data", n=len(df))
            return {"skipped": True, "n_events": len(df)}
        # Real retrain pipeline lives in scripts/train_initial.py; left as a hook.
        return {"skipped": True, "reason": "real_retrain_runs_via_script"}

    return _run_job_with_logging("retrain", _do_retrain)
