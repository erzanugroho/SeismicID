"""APScheduler background scheduler with SQLite-backed jobstore."""

from __future__ import annotations

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

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
    jobstore_url = f"sqlite:///{settings.sqlite_full_path}"
    _scheduler = BackgroundScheduler(
        jobstores={"default": SQLAlchemyJobStore(url=jobstore_url, tablename="apscheduler_jobs")},
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
        jobs.realtime_fetch,
        trigger=IntervalTrigger(minutes=s.sched_realtime_fetch_min),
        id="realtime_fetch",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    sched.add_job(
        jobs.forecast_recompute,
        trigger=IntervalTrigger(minutes=s.sched_forecast_recompute_min),
        id="forecast_recompute",
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
        "realtime_fetch": jobs.realtime_fetch,
        "forecast_recompute": jobs.forecast_recompute,
        "retrain": jobs.retrain,
    }.get(job_name)
    if fn is None:
        return {"ok": False, "error": f"unknown job '{job_name}'"}
    out = fn()
    return {"ok": True, "result": out if isinstance(out, dict) else str(out)}
