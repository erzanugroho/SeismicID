#!/usr/bin/env python
"""Run the SeismicID APScheduler worker in the foreground."""

from __future__ import annotations

import signal
import time

from backend.app.config import get_settings
from backend.app.core.logging import configure_logging, get_logger
from backend.app.db.sqlite import migrate
from backend.app.scheduler.runner import start_scheduler, stop_scheduler

logger = get_logger(__name__)
_STOP = False


def _handle_stop(signum: int, frame: object) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    migrate()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    scheduler = start_scheduler()
    logger.info("worker_started", jobs=[job.id for job in scheduler.get_jobs()])
    try:
        while not _STOP:
            time.sleep(1)
    finally:
        stop_scheduler()
        logger.info("worker_stopped")


if __name__ == "__main__":
    main()
