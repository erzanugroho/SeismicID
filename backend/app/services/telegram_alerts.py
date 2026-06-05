"""Telegram alert helpers for scheduler runs."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.db.metadata import get_metadata_value, set_metadata_value
from backend.app.db.sqlite import get_connection, migrate
from backend.app.services.forecast_service import get_top_forecasts
from backend.app.services.telegram_bot import _e, _report, _send

logger = get_logger(__name__)
WIB = timezone(timedelta(hours=7))


def post_admin_alert(text: str) -> bool:
    return _post_telegram(text)


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


def _top_snapshot(top: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for item in top:
        items.append(
            {
                "cell_id": item.get("cell_id"),
                "label": item.get("full_label") or item.get("cell_id") or "—",
                "probability": float(item.get("probability") or 0.0),
            }
        )
    return {"created_at": datetime.now(UTC).isoformat(), "items": items}


def _load_snapshot(key: str) -> dict[str, Any] | None:
    raw = get_metadata_value(key)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _significant_change(current: dict[str, Any], previous: dict[str, Any] | None) -> tuple[bool, str]:
    settings = get_settings()
    cur_items = current.get("items") or []
    prev_items = (previous or {}).get("items") or []
    if not cur_items:
        return False, "empty_current"
    cur_top = cur_items[0]
    cur_prob = float(cur_top.get("probability") or 0.0)
    if cur_prob < settings.telegram_alert_min_probability:
        return False, "below_threshold"
    if not prev_items:
        return False, "baseline_snapshot_created"
    prev_top = prev_items[0]
    prev_prob = float(prev_top.get("probability") or 0.0)
    if cur_top.get("cell_id") != prev_top.get("cell_id"):
        return True, "top_cell_changed"
    abs_delta = abs(cur_prob - prev_prob)
    rel_delta = abs_delta / max(prev_prob, 1e-9)
    crossed_threshold = prev_prob < settings.telegram_alert_min_probability <= cur_prob
    if crossed_threshold:
        return True, "crossed_threshold"
    if abs_delta >= settings.telegram_significant_abs_delta:
        return True, "absolute_delta"
    if rel_delta >= settings.telegram_significant_rel_delta:
        return True, "relative_delta"
    return False, "no_significant_change"


def _wib_now_text() -> str:
    return datetime.now(WIB).strftime("%d %b %Y · %H:%M WIB")


def _reason_text(reason: str | None) -> str:
    return {
        "top_cell_changed": "Wilayah risiko tertinggi berubah dibanding snapshot sebelumnya.",
        "crossed_threshold": "Probabilitas melewati ambang pantau.",
        "absolute_delta": "Perubahan probabilitas absolut cukup besar dibanding snapshot sebelumnya.",
        "relative_delta": "Perubahan probabilitas relatif cukup besar dibanding snapshot sebelumnya.",
    }.get(reason or "", "Perubahan risiko signifikan terdeteksi dibanding snapshot sebelumnya.")


def _model_text(result: dict[str, Any] | None) -> str:
    mode = (result or {}).get("mode") or (result or {}).get("baseline_type") or (result or {}).get("forecast_mode")
    if not mode:
        return "ML ensemble terkalibrasi publik"
    return str(mode).replace("ml_ensemble_public_calibrated", "ML ensemble terkalibrasi publik").replace("_", " ")


def _previous_top_text(previous: dict[str, Any] | None) -> str | None:
    items = (previous or {}).get("items") or []
    if not items:
        return None
    top = items[0]
    label = top.get("label") or top.get("cell_id") or "—"
    prob = float(top.get("probability") or 0.0) * 100
    return f"{label} ({prob:.2f}%)"


def _count_risk_cells() -> tuple[int, int]:
    migrate()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
              SUM(CASE WHEN probability >= 0.05 THEN 1 ELSE 0 END) AS above_5,
              SUM(CASE WHEN probability >= 0.08 THEN 1 ELSE 0 END) AS above_8
            FROM current_forecasts
            WHERE horizon_days = 30 AND mag_threshold = 5.0
            """
        ).fetchone()
    return int(row["above_5"] or 0), int(row["above_8"] or 0)


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _format_event_time(value: Any) -> str:
    if not value:
        return "waktu tidak tersedia"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(WIB).strftime("%d %b %Y · %H:%M WIB")
    except ValueError:
        return str(value)


def _top_cell_recent_event(top_item: dict[str, Any] | None) -> str:
    if not top_item:
        return "Tidak ada area teratas."
    cell_id = top_item.get("cell_id")
    if not cell_id:
        return "Tidak ada gempa signifikan terbaru dalam radius pantau."
    migrate()
    with get_connection() as conn:
        cell = conn.execute("SELECT lat, lon FROM area_labels WHERE cell_id = ?", (cell_id,)).fetchone()
        if not cell:
            return "Tidak ada gempa signifikan terbaru dalam radius pantau."
        events = conn.execute(
            """
            SELECT time, magnitude, place, lat, lon
            FROM realtime_events
            WHERE magnitude >= 4.5
            ORDER BY time DESC
            LIMIT 80
            """
        ).fetchall()
    best: tuple[float, dict[str, Any]] | None = None
    cell_lat = float(cell["lat"])
    cell_lon = float(cell["lon"])
    for row in events:
        lat = row["lat"]
        lon = row["lon"]
        if lat is None or lon is None:
            continue
        dist = _km(cell_lat, cell_lon, float(lat), float(lon))
        if dist <= 300 and (best is None or dist < best[0]):
            best = (dist, dict(row))
    if not best:
        return "Tidak ada gempa signifikan terbaru dalam radius pantau."
    dist, event = best
    time_text = _format_event_time(event.get("time"))
    return f"{float(event.get('magnitude') or 0.0):.1f} — {dist:.0f} km dari area teratas — {_e(event.get('place') or 'lokasi tidak tersedia')} — {time_text}"


def _format_top_message(title: str, top: list[dict[str, Any]], *, reason: str | None = None, result: dict[str, Any] | None = None, previous: dict[str, Any] | None = None) -> str:
    if reason:
        current_top = top[0] if top else None
        previous_top = _previous_top_text(previous)
        above_5, above_8 = _count_risk_cells()
        lines = [
            "⚠️ <b>Perubahan Risiko SeismicID</b>",
            "",
            "Terjadi perubahan signifikan pada ranking risiko nasional.",
            "",
            f"🕒 Update: {_wib_now_text()}",
            "⏳ Horizon: 30 hari",
            "🌋 Ambang gempa: ≥ 5.0",
            "",
            "🔥 <b>Top 5 Risiko Saat Ini</b>",
        ]
        for i, item in enumerate(top[:5], 1):
            prob = float(item.get("probability") or 0.0) * 100
            label = item.get("full_label") or item.get("cell_id") or "—"
            lines.append(f"{i}. {_e(label)}")
            lines.append(f"   {prob:.2f}%")
        if previous_top or current_top:
            current_label = (current_top or {}).get("full_label") or (current_top or {}).get("cell_id") or "—"
            current_prob = float((current_top or {}).get("probability") or 0.0) * 100
            lines.extend(["", "📈 <b>Perubahan Utama</b>"])
            if previous_top:
                lines.append(f"Area teratas sebelumnya: {_e(previous_top)}")
            lines.append(f"Area teratas sekarang: {_e(current_label)} ({current_prob:.2f}%)")
        lines.extend([
            "",
            "📌 <b>Pemicu Alert</b>",
            _reason_text(reason),
            "",
            "🌋 <b>Aktivitas Sekitar Area Teratas</b>",
            _top_cell_recent_event(current_top),
            "",
            "🧭 <b>Kondisi Nasional</b>",
            f"Cell di atas 5%: {above_5}",
            f"Cell di atas 8%: {above_8}",
            "",
            "🧠 <b>Model</b>",
            _model_text(result),
            "",
            "Catatan:",
            "Ini sinyal probabilistik, bukan prediksi pasti dan bukan peringatan dini.",
            "Untuk info resmi, gunakan BMKG/BNPB.",
        ])
        return "\n".join(lines)

    lines = [
        f"<b>{title}</b>",
        "M ≥ 5.0 · horizon 30 hari",
        "",
    ]
    for i, item in enumerate(top, 1):
        prob = float(item.get("probability") or 0.0) * 100
        label = item.get("full_label") or item.get("cell_id") or "—"
        lines.append(f"{i}. {label}: <b>{prob:.2f}%</b>")
    lines.append("")
    lines.append("Eksperimental — bukan peringatan dini. Gunakan BMKG untuk info resmi.")
    return "\n".join(lines)

def send_forecast_alert(result: dict[str, Any] | None) -> bool:
    """Send Telegram alert only when risk changes significantly."""
    top = get_top_forecasts(horizon_days=30, mag_threshold=5.0, n=5)
    if not top:
        return False
    current = _top_snapshot(top)
    previous = _load_snapshot("telegram_last_forecast_snapshot")
    should_send, reason = _significant_change(current, previous)
    set_metadata_value("telegram_last_forecast_snapshot", json.dumps(current))
    if not should_send:
        logger.info("telegram_alert_skipped", reason=reason)
        return False
    ok = _post_telegram(_format_top_message("SeismicID perubahan risiko", top, reason=reason, result=result, previous=previous))
    if ok:
        set_metadata_value("telegram_last_alert_snapshot", json.dumps(current))
        set_metadata_value("telegram_last_alert_at", current["created_at"])
    return ok


def _active_user_chat_ids() -> list[str]:
    migrate()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT l.chat_id
            FROM telegram_user_locations l
            LEFT JOIN telegram_bot_opt_outs o ON o.chat_id = l.chat_id
            WHERE l.stopped_at IS NULL AND o.chat_id IS NULL
            ORDER BY l.updated_at DESC
            """
        ).fetchall()
    return [str(r["chat_id"]) for r in rows]


def _top_section(top: list[dict[str, Any]]) -> str:
    lines = ["🔥 <b>Top 10 Risiko Nasional</b>", "M ≥ 5.0 · horizon 30 hari"]
    for i, item in enumerate(top[:10], 1):
        prob = float(item.get("probability") or 0.0) * 100
        label = item.get("full_label") or item.get("cell_id") or "—"
        cell_id = item.get("cell_id") or ""
        lines.append(f"{i}. {_e(label)}: <b>{prob:.2f}%</b> · <code>{_e(cell_id)}</code>")
    return "\n".join(lines)


def send_daily_forecast_report() -> bool:
    """Send once-per-day Telegram forecast summary. Scheduled for 07:00 WIB (00:00 UTC)."""
    top = get_top_forecasts(horizon_days=30, mag_threshold=5.0, n=10)
    if not top:
        return False
    today = datetime.now(UTC).date().isoformat()
    if get_metadata_value("telegram_last_daily_report_date") == today:
        logger.info("telegram_daily_report_skipped", reason="already_sent", date=today)
        return False

    chat_ids = _active_user_chat_ids()
    sent = 0
    top_section = _top_section(top)
    for chat_id in chat_ids:
        text = "\n\n".join([_report(chat_id), top_section])
        if _send(chat_id, text):
            sent += 1

    # Keep admin/home channel useful too, even if no user has set location yet.
    admin_ok = _post_telegram(_format_top_message("SeismicID laporan pagi", top))
    ok = sent > 0 or admin_ok
    if ok:
        set_metadata_value("telegram_last_daily_report_date", today)
        set_metadata_value("telegram_last_daily_report_at", datetime.now(UTC).isoformat())
        set_metadata_value("telegram_last_daily_report_recipients", str(sent))
    return ok
