"""Telegram alert helpers for scheduler runs."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.services.forecast_service import get_top_forecasts

logger = get_logger(__name__)


def _post_telegram(text: str) -> bool:
    settings = get_settings()
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.info("telegram_alert_skipped", reason="not_configured")
        return False

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": settings.telegram_chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok = 200 <= resp.status < 300
            logger.info("telegram_alert_sent", ok=ok, status=resp.status)
            return ok
    except urllib.error.HTTPError as e:
        logger.warning("telegram_alert_failed", status=e.code)
    except Exception as e:  # noqa: BLE001
        logger.warning("telegram_alert_failed", error=str(e))
    return False


def send_forecast_alert(result: dict[str, Any] | None) -> bool:
    """Send compact top-risk alert after forecast run if Telegram env is configured."""
    settings = get_settings()
    top = get_top_forecasts(horizon_days=30, mag_threshold=5.0, n=5)
    if not top:
        return False

    max_prob = float(top[0].get("probability") or 0.0)
    if max_prob < settings.telegram_alert_min_probability:
        logger.info("telegram_alert_skipped", reason="below_threshold", max_probability=max_prob)
        return False

    lines = [
        "<b>SeismicID forecast alert</b>",
        "M ≥ 5.0 · horizon 30 hari",
        "",
    ]
    for i, item in enumerate(top, 1):
        prob = float(item.get("probability") or 0.0) * 100
        label = item.get("full_label") or item.get("cell_id") or "—"
        lines.append(f"{i}. {label}: <b>{prob:.2f}%</b>")
    if result:
        mode = result.get("mode") or result.get("baseline_type") or result.get("forecast_mode")
        if mode:
            lines.append("")
            lines.append(f"mode: {mode}")
    lines.append("")
    lines.append("Eksperimental — bukan peringatan dini. Gunakan BMKG untuk info resmi.")
    return _post_telegram("\n".join(lines))
