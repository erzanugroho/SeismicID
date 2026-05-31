"""AI MVP v1: daily briefing, cell explanation, changelog."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone

from backend.app.config import get_settings
from backend.app.db.sqlite import get_connection
from backend.app.services.ai_guardrails import guard_public_text, validate_public_text
from backend.app.services.ai_provider import generate_text, ai_enabled


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _cache_get(key: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT payload_json FROM ai_cache WHERE cache_key=? AND expires_at > datetime('now')",
            (key,),
        ).fetchone()
    return json.loads(row[0]) if row else None


def _cache_set(key: str, payload: dict, ttl_minutes: int | None = None) -> None:
    ttl = ttl_minutes or get_settings().ai_cache_ttl_minutes
    expires = (_now() + timedelta(minutes=ttl)).replace(microsecond=0).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO ai_cache(cache_key, payload_json, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
              payload_json=excluded.payload_json,
              created_at=datetime('now'),
              expires_at=excluded.expires_at
            """,
            (key, json.dumps(payload, ensure_ascii=False), expires),
        )


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _fallback_briefing(top: list[dict], events: list[dict], horizon: int, threshold: float) -> str:
    if top:
        best = top[0]
        area = best.get("full_label") or best.get("cell_id")
        prob = float(best.get("probability") or 0) * 100
        text = f"Ringkasan risiko {horizon} hari M≥{threshold}: area tertinggi saat ini {area} dengan probabilitas {prob:.3f}%."
    else:
        text = f"Ringkasan risiko {horizon} hari M≥{threshold}: data forecast belum tersedia."
    if events:
        mags = ", ".join(f"{float(e['magnitude']):.1f}" for e in events[:3] if e.get("magnitude") is not None)
        text += f" Gempa besar terbaru dalam buffer: {mags}."
    return guard_public_text(text)


def daily_briefing(horizon: int = 30, threshold: float = 5.0, force: bool = False) -> dict:
    key = f"daily_briefing:v1:{horizon}:{threshold}"
    if not force and (cached := _cache_get(key)):
        cached["cached"] = True
        return cached
    top = _rows(
        """
        SELECT f.cell_id, f.probability, f.computed_at, f.model_version, a.full_label, a.province, a.region_macro
        FROM current_forecasts f LEFT JOIN area_labels a ON a.cell_id=f.cell_id
        WHERE f.horizon_days=? AND f.mag_threshold=?
        ORDER BY f.probability DESC LIMIT 10
        """,
        (horizon, threshold),
    )
    events = _rows(
        """
        SELECT event_id, time, magnitude, place, source
        FROM realtime_events WHERE magnitude >= 5.0
        ORDER BY time DESC LIMIT 8
        """
    )
    prompt = json.dumps({"top_forecasts": top, "recent_m5_events": events}, ensure_ascii=False, default=str)
    system = (
        "You write short Indonesian earthquake-risk briefings for SeismicID. "
        "Use only supplied data. Never claim an earthquake will happen. "
        "Always mention probabilistic, experimental, not early warning, and BMKG/official authorities."
    )
    ai_text = generate_text(system, prompt, max_tokens=360)
    text = guard_public_text(ai_text) if ai_text else _fallback_briefing(top, events, horizon, threshold)
    payload = {"text": text, "ai_enabled": ai_enabled(), "guardrail": validate_public_text(text), "top": top[:5], "events": events[:5], "cached": False}
    _cache_set(key, payload)
    return payload


def cell_explanation(cell_id: str, horizon: int = 30, threshold: float = 5.0, force: bool = False) -> dict:
    key = f"cell_explain:v1:{cell_id}:{horizon}:{threshold}"
    if not force and (cached := _cache_get(key)):
        cached["cached"] = True
        return cached
    rows = _rows(
        """
        SELECT f.cell_id, f.probability, f.raw_probability, f.computed_at, f.model_version,
               a.full_label, a.province, a.region_macro, a.lat_min, a.lat_max, a.lon_min, a.lon_max
        FROM current_forecasts f LEFT JOIN area_labels a ON a.cell_id=f.cell_id
        WHERE f.cell_id=? AND f.horizon_days=? AND f.mag_threshold=? LIMIT 1
        """,
        (cell_id, horizon, threshold),
    )
    cell = rows[0] if rows else {"cell_id": cell_id}
    events = _rows(
        """
        SELECT event_id, time, magnitude, place, source, lat, lon
        FROM realtime_events
        WHERE magnitude >= 5.0
        ORDER BY time DESC LIMIT 20
        """
    )
    prompt = json.dumps({"cell": cell, "recent_m5_events": events[:10]}, ensure_ascii=False, default=str)
    system = (
        "Explain one SeismicID map cell in Indonesian. Use only supplied data. "
        "Be concise. Say probability is small absolute risk but relative ranking can be useful. "
        "Never predict certainty. Mention not early warning and use BMKG for official information."
    )
    ai_text = generate_text(system, prompt, max_tokens=320)
    if not ai_text:
        prob = float(cell.get("probability") or 0) * 100
        label = cell.get("full_label") or cell_id
        ai_text = f"Cell {label} punya probabilitas {prob:.3f}% untuk M≥{threshold} dalam {horizon} hari. Nilai ini adalah risiko relatif model, bukan prediksi pasti."
    text = guard_public_text(ai_text)
    payload = {"text": text, "ai_enabled": ai_enabled(), "guardrail": validate_public_text(text), "cell": cell, "cached": False}
    _cache_set(key, payload)
    return payload


def auto_changelog(limit: int = 12, force: bool = False) -> dict:
    key = f"auto_changelog:v1:{limit}"
    if not force and (cached := _cache_get(key)):
        cached["cached"] = True
        return cached
    try:
        raw = subprocess.check_output(
            ["git", "log", f"-{limit}", "--pretty=format:%h %s"],
            cwd=str(get_settings().project_root),
            text=True,
            timeout=8,
        )
    except Exception:  # noqa: BLE001
        raw = ""
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    system = "Write concise Indonesian public changelog bullets for SeismicID. No hype. Mention safety if relevant."
    ai_text = generate_text(system, raw, max_tokens=360)
    if not ai_text:
        bullets = [f"- {line.split(' ', 1)[1]}" for line in raw.splitlines() if " " in line]
        if not bullets:
            bullets = [
                "- AI MVP v1 ditambahkan: daily briefing, penjelasan cell, guardrail, dan changelog otomatis.",
                "- Gempa besar terbaru ditampilkan sebagai outline pulsing pada cell terkait.",
                "- Dataset SeismicID tersedia di Hugging Face dalam format parquet.",
                "- Peta memiliki loading state saat cell forecast sedang dimuat.",
            ]
        ai_text = "Perubahan terbaru:\n" + "\n".join(bullets[:limit])
    text = guard_public_text(ai_text)
    payload = {"text": text, "source": raw, "source_hash": digest, "ai_enabled": ai_enabled(), "cached": False}
    _cache_set(key, payload, ttl_minutes=60)
    return payload
