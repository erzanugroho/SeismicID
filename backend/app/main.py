"""FastAPI application entry point.

Usage:
    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import areas, events, forecasts, health
from backend.app.api.routes import model as model_route
from backend.app.api.routes import scheduler as scheduler_route
from backend.app.config import get_settings
from backend.app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Startup / shutdown lifecycle. Scheduler hook lands in Task 12."""
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()
    # Apply DB migrations so all tables exist before any request lands.
    from backend.app.db.sqlite import migrate

    migrate()

    # Start scheduler only for worker/combined roles unless explicitly disabled.
    import os

    scheduler_enabled = (
        os.environ.get("DISABLE_SCHEDULER") != "1"
        and settings.app_role.lower() in {"worker", "combined"}
    )
    if scheduler_enabled:
        try:
            from backend.app.scheduler.runner import start_scheduler

            start_scheduler()
        except Exception as e:  # noqa: BLE001
            logger.warning("scheduler_start_failed", error=str(e))

    logger.info(
        "app_startup",
        name=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
    )
    yield

    # Graceful shutdown
    try:
        from backend.app.scheduler.runner import stop_scheduler

        stop_scheduler()
    except Exception as e:  # noqa: BLE001
        logger.warning("scheduler_stop_failed", error=str(e))
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    app = FastAPI(
        title="Gempa Forecast API",
        version=settings.app_version,
        description=(
            "Sistem forecast probabilitas gempa bumi Indonesia. "
            "Output: 'Area X, Y% probabilitas M>=Z dalam N hari.'"
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(health.router, prefix="/api")
    app.include_router(areas.router, prefix="/api")
    app.include_router(events.router, prefix="/api")
    app.include_router(forecasts.router, prefix="/api")
    app.include_router(forecasts.status_router, prefix="/api")
    app.include_router(model_route.router, prefix="/api")
    app.include_router(scheduler_route.router, prefix="/api")
    # Also expose /health (without /api) for typical k8s/docker probes
    app.include_router(health.router)

    # Static frontend (mounted last so /api/* takes precedence)
    frontend_dir = settings.project_root / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    else:
        logger.warning("frontend_dir_missing", path=str(frontend_dir))
        # Fallback root endpoint
        Path(frontend_dir).mkdir(parents=True, exist_ok=True)

    return app


app = create_app()
