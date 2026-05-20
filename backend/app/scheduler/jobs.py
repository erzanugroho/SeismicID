"""Background jobs.

Each job logs its start/end + outcome to scheduler_runs.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from backend.app.core.logging import get_logger
from backend.app.db.sqlite import get_connection, migrate

logger = get_logger(__name__)


def _record_start(job_name: str) -> int:
    migrate()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO scheduler_runs (job_name, started_at, status) VALUES (?,?,?)",
            (job_name, datetime.now(timezone.utc).isoformat(), "running"),
        )
        return int(cur.lastrowid)


def _record_end(run_id: int, *, status: str, error: str | None = None, items: int | None = None, meta: dict | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            """UPDATE scheduler_runs
               SET finished_at = ?, status = ?, error = ?, items_processed = ?, metadata_json = ?
               WHERE id = ?""",
            (
                datetime.now(timezone.utc).isoformat(),
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


# === Job functions ===

def realtime_fetch() -> dict:
    from backend.app.data.ingest import ingest_realtime
    return _run_job_with_logging("realtime_fetch", ingest_realtime)


def forecast_recompute() -> dict:
    from backend.app.services.forecast_service import run_forecast
    return _run_job_with_logging("forecast_recompute", run_forecast)


def retrain() -> dict:
    """Retrain heavy job. Skipped if not enough data; logs warning otherwise."""
    def _do_retrain():
        from backend.app.data.catalog import read_historical_events
        df = read_historical_events()
        if len(df) < 10000:
            logger.warning("retrain_skipped_insufficient_data", n=len(df))
            return {"skipped": True, "n_events": len(df)}
        # Real retrain pipeline lives in scripts/train_initial.py; left as a hook.
        return {"skipped": True, "reason": "real_retrain_runs_via_script"}
    return _run_job_with_logging("retrain", _do_retrain)
