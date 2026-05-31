"""Telegram webhook endpoint."""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from backend.app.config import get_settings
from backend.app.services.telegram_bot import handle_update

router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Receive Telegram webhook updates.

    If TELEGRAM_WEBHOOK_SECRET is configured, Telegram must send matching
    X-Telegram-Bot-Api-Secret-Token header.
    """
    settings = get_settings()
    if settings.telegram_webhook_secret:
        if not x_telegram_bot_api_secret_token or not hmac.compare_digest(
            x_telegram_bot_api_secret_token,
            settings.telegram_webhook_secret,
        ):
            raise HTTPException(status_code=401, detail="invalid telegram webhook secret")
    update = await request.json()
    ok = handle_update(update)
    return {"ok": ok}
