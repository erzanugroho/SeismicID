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


ALERT_HORIZON_DAYS = 30
ALERT_THRESHOLDS = (5.0, 5.5, 6.0)
ALERT_PROFILES: dict[float, dict[str, float | int]] = {
    5.0: {"n": 20, "min_prob": 0.03, "abs_delta": 0.005, "rel_delta": 0.25},
    5.5: {"n": 20, "min_prob": 0.01, "abs_delta": 0.0035, "rel_delta": 0.20},
    6.0: {"n": 50, "min_prob": 0.003, "abs_delta": 0.0015, "rel_delta": 0.20},
}


def _rank_map(snapshot: dict[str, Any] | None) -> dict[str, int]:
    return {
        str(item.get("cell_id")): i
        for i, item in enumerate((snapshot or {}).get("items") or [], 1)
        if item.get("cell_id")
    }


def _significant_change(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    threshold: float = 5.0,
) -> tuple[bool, str]:
    profile = ALERT_PROFILES.get(threshold, ALERT_PROFILES[5.0])
    cur_items = current.get("items") or []
    prev_items = (previous or {}).get("items") or []
    if not cur_items:
        return False, "empty_current"
    cur_top = cur_items[0]
    cur_prob = float(cur_top.get("probability") or 0.0)
    min_prob = float(profile["min_prob"])
    if cur_prob < min_prob:
        return False, "below_threshold"
    if not prev_items:
        return False, "baseline_snapshot_created"

    prev_top = prev_items[0]
    prev_prob = float(prev_top.get("probability") or 0.0)
    if cur_top.get("cell_id") != prev_top.get("cell_id"):
        return True, "top_cell_changed"

    abs_delta = abs(cur_prob - prev_prob)
    rel_delta = abs_delta / max(prev_prob, 1e-9)
    if prev_prob < min_prob <= cur_prob:
        return True, "crossed_threshold"
    if abs_delta >= float(profile["abs_delta"]):
        return True, "absolute_delta"
    if rel_delta >= float(profile["rel_delta"]):
        return True, "relative_delta"

    if threshold >= 6.0:
        prev_ranks = _rank_map(previous)
        cur_ranks = _rank_map(current)
        for item in cur_items[:5]:
            cell_id = str(item.get("cell_id") or "")
            if cell_id and prev_ranks.get(cell_id, 999) > 5:
                return True, "entered_top5"
        for cell_id, cur_rank in cur_ranks.items():
            if cur_rank <= 10 and prev_ranks.get(cell_id, cur_rank) - cur_rank >= 10:
                return True, "rank_jump"

    return False, "no_significant_change"


def _wib_now_text() -> str:
    return datetime.now(WIB).strftime("%d %b %Y · %H:%M WIB")


def _reason_text(reason: str | None) -> str:
    return {
        "top_cell_changed": "Wilayah risiko tertinggi berubah dibanding snapshot sebelumnya.",
        "crossed_threshold": "Probabilitas melewati ambang pantau.",
        "absolute_delta": "Perubahan probabilitas absolut cukup besar dibanding snapshot sebelumnya.",
        "relative_delta": "Perubahan probabilitas relatif cukup besar dibanding snapshot sebelumnya.",
        "entered_top5": "Area baru masuk Top 5 nasional untuk ambang magnitude terkait.",
        "rank_jump": "Area risiko naik tajam dalam ranking nasional.",
    }.get(reason or "", "Perubahan risiko signifikan terdeteksi dibanding snapshot sebelumnya.")


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

def _format_multi_threshold_alert(
    tops: dict[float, list[dict[str, Any]]],
    triggers: list[dict[str, Any]],
    previous_by_threshold: dict[float, dict[str, Any] | None],
) -> str:
    primary_threshold = 5.0 if tops.get(5.0) else triggers[0]["threshold"]
    primary_top = tops.get(float(primary_threshold), [])
    top_cells = primary_top[:3]
    prob_lookup: dict[float, dict[str, float]] = {}
    for threshold, items in tops.items():
        prob_lookup[threshold] = {str(item.get("cell_id")): float(item.get("probability") or 0.0) for item in items}

    current_top = primary_top[0] if primary_top else None
    previous_top = _previous_top_text(previous_by_threshold.get(float(primary_threshold)))
    above_5, above_8 = _count_risk_cells()
    lines = [
        "⚠️ <b>Perubahan Risiko SeismicID</b>",
        "",
        "Terjadi perubahan signifikan pada risiko nasional.",
        "",
        f"🕒 Update: {_wib_now_text()}",
        "⏳ Horizon: 30 hari",
        "",
        "📊 <b>Threshold Terdeteksi</b>",
    ]
    for trigger in triggers:
        threshold = float(trigger["threshold"])
        lines.append(f"• M≥{threshold:.1f} — {_reason_text(str(trigger['reason']))}")

    lines.extend(["", "🔥 <b>Top Risiko Saat Ini</b>"])
    for i, item in enumerate(top_cells, 1):
        cell_id = str(item.get("cell_id") or "")
        label = item.get("full_label") or cell_id or "—"
        parts = []
        for threshold in ALERT_THRESHOLDS:
            prob = prob_lookup.get(threshold, {}).get(cell_id)
            if prob is not None:
                parts.append(f"{threshold:.1f}: {prob * 100:.2f}%")
        lines.append(f"{i}. {_e(label)}")
        lines.append(f"   {' · '.join(parts) if parts else '—'}")

    if previous_top or current_top:
        current_label = (current_top or {}).get("full_label") or (current_top or {}).get("cell_id") or "—"
        current_prob = float((current_top or {}).get("probability") or 0.0) * 100
        lines.extend(["", "📈 <b>Perubahan Utama</b>"])
        if previous_top:
            lines.append(f"Area teratas sebelumnya: {_e(previous_top)}")
        lines.append(f"Area teratas sekarang: {_e(current_label)} ({current_prob:.2f}%)")

    lines.extend([
        "",
        "🌋 <b>Aktivitas Sekitar Area Teratas</b>",
        _top_cell_recent_event(current_top),
        "",
        "🧭 <b>Kondisi Nasional</b>",
        f"Cell M≥5.0 di atas 5%: {above_5}",
        f"Cell M≥5.0 di atas 8%: {above_8}",
        "",
        "Catatan:",
        "Ini sinyal probabilistik, bukan prediksi pasti dan bukan peringatan dini.",
        "Untuk info resmi, gunakan BMKG/BNPB.",
    ])
    return "\n".join(lines)


def send_forecast_alert(result: dict[str, Any] | None) -> bool:
    """Send one combined Telegram alert for H30 M≥5.0/5.5/6.0 significant changes."""
    tops: dict[float, list[dict[str, Any]]] = {}
    current_by_threshold: dict[float, dict[str, Any]] = {}
    previous_by_threshold: dict[float, dict[str, Any] | None] = {}
    triggers: list[dict[str, Any]] = []

    for threshold in ALERT_THRESHOLDS:
        profile = ALERT_PROFILES[threshold]
        top = get_top_forecasts(horizon_days=ALERT_HORIZON_DAYS, mag_threshold=threshold, n=int(profile["n"]))
        if not top:
            continue
        tops[threshold] = top
        current = _top_snapshot(top)
        previous = _load_snapshot(f"telegram_last_forecast_snapshot_m{str(threshold).replace('.', '')}")
        current_by_threshold[threshold] = current
        previous_by_threshold[threshold] = previous
        should_send, reason = _significant_change(current, previous, threshold=threshold)
        if should_send:
            triggers.append({"threshold": threshold, "reason": reason})

    for threshold, current in current_by_threshold.items():
        set_metadata_value(f"telegram_last_forecast_snapshot_m{str(threshold).replace('.', '')}", json.dumps(current))

    if not triggers:
        logger.info("telegram_alert_skipped", reason="no_multi_threshold_change")
        return False

    last_alert_at = get_metadata_value("telegram_last_alert_at")
    if last_alert_at:
        try:
            last_dt = datetime.fromisoformat(last_alert_at.replace("Z", "+00:00"))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=UTC)
            age_minutes = (datetime.now(UTC) - last_dt.astimezone(UTC)).total_seconds() / 60
            if age_minutes < 60:
                logger.info("telegram_alert_skipped", reason="global_cooldown", age_minutes=round(age_minutes, 1), triggers=triggers)
                return False
        except ValueError:
            pass

    ok = _post_telegram(_format_multi_threshold_alert(tops, triggers, previous_by_threshold))
    if ok:
        set_metadata_value("telegram_last_alert_at", datetime.now(UTC).isoformat())
        set_metadata_value("telegram_last_alert_triggers", json.dumps(triggers))
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
