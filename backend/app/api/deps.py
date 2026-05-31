"""Shared FastAPI dependencies and lightweight admin job guards."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from fastapi import Header, HTTPException, Request, status

from backend.app.config import get_settings

_RUNNING_JOBS: set[str] = set()
_LAST_STARTED: dict[str, float] = {}
_GUARD_LOCK = threading.Lock()

# --- Brute-force protection for admin auth -------------------------------
# Sliding-window per-IP counter. In-memory is sufficient for a single-process
# deployment; a multi-replica setup would move this to Redis.
_AUTH_ATTEMPTS: dict[str, list[float]] = {}
_AUTH_LOCK = threading.Lock()
_AUTH_MAX_ATTEMPTS = 5
_AUTH_WINDOW_SECONDS = 60.0

_AI_ATTEMPTS: dict[str, list[float]] = {}
_AI_LOCK = threading.Lock()
_AI_MAX_ATTEMPTS = 20
_AI_WINDOW_SECONDS = 60.0


def rate_limit_admin_auth(request: Request) -> None:
    """Throttle admin login attempts to slow down brute-force guessing.

    Allows at most ``_AUTH_MAX_ATTEMPTS`` attempts per client IP within a
    rolling ``_AUTH_WINDOW_SECONDS`` window; further attempts get 429.
    """
    client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    now = time.monotonic()
    with _AUTH_LOCK:
        attempts = [t for t in _AUTH_ATTEMPTS.get(client_ip, []) if now - t < _AUTH_WINDOW_SECONDS]
        if len(attempts) >= _AUTH_MAX_ATTEMPTS:
            retry_in = int(_AUTH_WINDOW_SECONDS - (now - attempts[0])) + 1
            attempts.append(now)
            _AUTH_ATTEMPTS[client_ip] = attempts
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"too many attempts; retry in {retry_in}s",
                headers={"Retry-After": str(retry_in)},
            )
        attempts.append(now)
        _AUTH_ATTEMPTS[client_ip] = attempts


def rate_limit_ai(request: Request) -> None:
    """Throttle public AI endpoints to reduce abuse/cost spikes."""
    client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    now = time.monotonic()
    with _AI_LOCK:
        attempts = [t for t in _AI_ATTEMPTS.get(client_ip, []) if now - t < _AI_WINDOW_SECONDS]
        if len(attempts) >= _AI_MAX_ATTEMPTS:
            retry_in = int(_AI_WINDOW_SECONDS - (now - attempts[0])) + 1
            attempts.append(now)
            _AI_ATTEMPTS[client_ip] = attempts
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"too many AI requests; retry in {retry_in}s",
                headers={"Retry-After": str(retry_in)},
            )
        attempts.append(now)
        _AI_ATTEMPTS[client_ip] = attempts


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
