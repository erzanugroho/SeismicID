"""Shared FastAPI dependencies."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from backend.app.config import get_settings


def require_admin_token(authorization: str | None = Header(default=None)) -> None:
    """Require `Authorization: Bearer <ADMIN_TOKEN>` for heavy/admin actions."""
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

    provided = authorization[len(prefix):]
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid admin token")
