"""APScheduler background scheduler."""

from __future__ import annotations

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.app.api.deps import guarded_admin_job
from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.scheduler import jobs

logger = get_logger(__name__)

_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    settings = get_settings()
    settings.ensure_dirs()
    # Use APScheduler's in-memory jobstore. Jobs are declared from code on every
    # startup, so persisting job definitions in the main SQLite DB is unnecessary
    # and can block scheduler startup when only the APScheduler table is corrupt.
    _scheduler = BackgroundScheduler(
        executors={"default": ThreadPoolExecutor(max_workers=2)},
        timezone="UTC",
    )
    return _scheduler


def start_scheduler() -> BackgroundScheduler:
    s = get_settings()
    sched = get_scheduler()
    if sched.running:
        return sched

    sched.add_job(
        jobs.scheduler_tick_job,
        trigger=IntervalTrigger(minutes=s.forecast_fetch_interval_minutes),
        id="scheduler_tick",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    sched.add_job(
        jobs.telegram_daily_report,
        trigger=CronTrigger(hour=s.telegram_daily_report_hour_utc, minute=0),
        id="telegram_daily_report",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    sched.add_job(
        jobs.sqlite_backup,
        trigger=CronTrigger(hour=18, minute=0),  # 01:00 WIB
        id="sqlite_backup",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    sched.add_job(
        jobs.retrain,
        trigger=CronTrigger(day_of_week=s.sched_retrain_cron_day, hour=s.sched_retrain_cron_hour),
        id="retrain",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )

    sched.start()
    logger.info("scheduler_started", jobs=[j.id for j in sched.get_jobs()])
    return sched


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("scheduler_stopped")


def list_runs(limit: int = 50) -> list[dict]:
    from backend.app.db.sqlite import get_connection, migrate

    migrate()
    with get_connection() as conn:
        cur = conn.execute(
            """SELECT id, job_name, started_at, finished_at, status, error, items_processed, metadata_json
               FROM scheduler_runs
               ORDER BY started_at DESC
               LIMIT ?""",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def trigger_job(job_name: str) -> dict:
    """Manual trigger of a registered job (runs synchronously in caller thread)."""
    fn = {
        "scheduler_tick": jobs.scheduler_tick_job,
        "realtime_fetch": jobs.realtime_fetch,
        "forecast_recompute": jobs.forecast_recompute,
        "telegram_daily_report": jobs.telegram_daily_report,
        "sqlite_backup": jobs.sqlite_backup,
        "retrain": jobs.retrain,
    }.get(job_name)
    if fn is None:
        return {"ok": False, "error": f"unknown job '{job_name}'"}
    guarded = guarded_admin_job(job_name)(fn)
    out = guarded()
    return {"ok": True, "result": out if isinstance(out, dict) else str(out)}
