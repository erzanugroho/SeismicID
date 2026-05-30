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
    else:
        # Loud about WHY the scheduler is off, so a misconfigured deploy
        # (e.g. APP_ROLE unset/web on Railway) doesn't silently freeze forecasts.
        logger.warning(
            "scheduler_disabled",
            app_role=settings.app_role,
            disable_scheduler_env=os.environ.get("DISABLE_SCHEDULER"),
            reason="app_role not in {worker,combined} or DISABLE_SCHEDULER=1",
        )

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
    # Hide interactive API docs / OpenAPI schema in production so the full
    # endpoint map (including admin routes) is not exposed publicly.
    is_prod = settings.app_env.lower() in {"production", "prod"}
    app = FastAPI(
        title="Gempa Forecast API",
        version=settings.app_version,
        description=(
            "Sistem forecast probabilitas gempa bumi Indonesia. "
            "Output: 'Area X, Y% probabilitas M>=Z dalam N hari.'"
        ),
        lifespan=lifespan,
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/openapi.json",
    )

    # Security headers applied to every response (defends clickjacking, MIME
    # sniffing, SSL strip, and limits referrer/feature leakage).
    @app.middleware("http")
    async def security_headers(request, call_next):  # noqa: ANN001, ANN202
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(self), microphone=(), camera=()"
        )
        if is_prod:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
            )
        return response

    # API is public read-only and does not use cookies/sessions, so credentials
    # are not needed; methods are restricted to what the app actually uses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
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
