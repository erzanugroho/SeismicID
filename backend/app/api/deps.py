"""Shared FastAPI dependencies and lightweight admin job guards."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from fastapi import Header, HTTPException, status

from backend.app.config import get_settings

_RUNNING_JOBS: set[str] = set()
_LAST_STARTED: dict[str, float] = {}
_GUARD_LOCK = threading.Lock()


def require_admin_token(authorization: str | None = Header(default=None)) -> None:
    """Require ``Authorization: Bearer <ADMIN_TOKEN>`` for heavy/admin actions."""
    settings = get_settings()
    expected = settings.admin_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin actions are disabled because ADMIN_TOKEN is not configured",
        )

    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided = authorization[len(prefix) :]
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid admin token")


def guarded_admin_job(job_name: str, cooldown_seconds: int | None = None) -> Callable:
    """Prevent accidental parallel/repeated heavy admin job execution in-process."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = get_settings()
            cooldown = (
                cooldown_seconds
                if cooldown_seconds is not None
                else (
                    settings.admin_retrain_cooldown_seconds
                    if job_name == "retrain"
                    else settings.admin_job_cooldown_seconds
                )
            )
            now = time.monotonic()
            # Include the configured SQLite path so isolated test databases and
            # blue/green local runs do not throttle one another in the same process.
            guard_key = f"{settings.sqlite_full_path}:{job_name}"
            with _GUARD_LOCK:
                if guard_key in _RUNNING_JOBS:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"job '{job_name}' is already running; try again later",
                    )
                last = _LAST_STARTED.get(guard_key)
                if last is not None and now - last < cooldown:
                    wait = int(cooldown - (now - last)) + 1
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"job '{job_name}' is cooling down; retry in {wait}s",
                    )
                _RUNNING_JOBS.add(guard_key)
                _LAST_STARTED[guard_key] = now
            try:
                return fn(*args, **kwargs)
            finally:
                with _GUARD_LOCK:
                    _RUNNING_JOBS.discard(guard_key)

        return wrapper

    return decorator
