"""GET /api/model/metadata and /api/model/evaluation."""

from __future__ import annotations

import json

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.db.sqlite import get_connection, migrate

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metadata")
def metadata() -> dict:
    settings = get_settings()
    active = settings.models_path / "active.json"
    if not active.exists():
        return {"version": None, "status": "no_active_model"}
    version = json.loads(active.read_text())["version"]
    meta_path = settings.models_path / f"metadata_{version}.json"
    if not meta_path.exists():
        return {"version": version, "status": "metadata_missing"}
    return json.loads(meta_path.read_text())


@router.get("/evaluation")
def evaluation() -> dict:
    """Return latest stored evaluation payload from DB. Empty if none yet."""
    migrate()
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT model_version, eval_type, payload_json, computed_at FROM evaluation_results ORDER BY computed_at DESC"
        )
        items = [dict(r) for r in cur.fetchall()]
    return {"count": len(items), "items": [{**it, "payload_json": json.loads(it["payload_json"])} for it in items]}


def _human_rate(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


@router.get("/performance-v2")
def performance_v2(horizon: int = 30, threshold: float = 5.0, high_risk_top_pct: float = 0.10) -> dict:
    """Human-readable performance proxy for dashboard v2.

    This endpoint compares the current high-risk map against recent observed events.
    It is an operational/readability metric, not a replacement for prospective CSEP evaluation.
    """
    migrate()
    high_risk_top_pct = max(0.01, min(float(high_risk_top_pct), 0.50))
    with get_connection() as conn:
        total_row = conn.execute(
            "SELECT COUNT(*) AS n FROM current_forecasts WHERE horizon_days = ? AND mag_threshold = ?",
            (horizon, threshold),
        ).fetchone()
        total_cells = int(total_row["n"] or 0) if total_row else 0
        high_risk_n = max(1, int(total_cells * high_risk_top_pct)) if total_cells else 0
        high_rows = conn.execute(
            """SELECT cf.cell_id, cf.probability, al.lat_min, al.lat_max, al.lon_min, al.lon_max, al.full_label
               FROM current_forecasts cf
               JOIN area_labels al ON al.cell_id = cf.cell_id
               WHERE cf.horizon_days = ? AND cf.mag_threshold = ?
               ORDER BY cf.probability DESC
               LIMIT ?""",
            (horizon, threshold, high_risk_n),
        ).fetchall()
        high_cells = [dict(r) for r in high_rows]
        event_rows = conn.execute(
            """SELECT event_id, time, lat, lon, magnitude, place
               FROM realtime_events
               WHERE magnitude >= ? AND time >= datetime('now', '-365 days')
               ORDER BY time DESC""",
            (threshold,),
        ).fetchall()

    high_ids = {c["cell_id"] for c in high_cells}
    high_with_event: set[str] = set()
    events_in_high = 0
    total_events_30d = 0
    events_in_high_30d = 0
    monthly: dict[str, dict] = {}

    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    cutoff_30 = now - timedelta(days=30)

    def parse_time(value: str) -> datetime | None:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            return None

    def event_cell_id(ev: dict) -> str | None:
        lat = float(ev["lat"]); lon = float(ev["lon"])
        for c in high_cells:
            if c["lat_min"] <= lat < c["lat_max"] and c["lon_min"] <= lon < c["lon_max"]:
                return str(c["cell_id"])
        return None

    for row in event_rows:
        ev = dict(row)
        dt = parse_time(ev.get("time"))
        if not dt:
            continue
        month = dt.strftime("%Y-%m")
        monthly.setdefault(month, {"month": month, "events": 0, "events_in_high_risk": 0, "hit_rate": None})
        monthly[month]["events"] += 1
        cell_id = event_cell_id(ev)
        in_high = cell_id in high_ids if cell_id else False
        if in_high:
            events_in_high += 1
            high_with_event.add(cell_id)
            monthly[month]["events_in_high_risk"] += 1
        if dt >= cutoff_30:
            total_events_30d += 1
            if in_high:
                events_in_high_30d += 1

    for m in monthly.values():
        m["hit_rate"] = _human_rate(m["events_in_high_risk"] / m["events"]) if m["events"] else None
    monthly_rows = [monthly[k] for k in sorted(monthly.keys())][-12:]

    high_count = len(high_cells)
    high_hit_cells = len(high_with_event)
    false_alarm_cells = max(high_count - high_hit_cells, 0)

    return {
        "mode": "operational_proxy",
        "note": "Membandingkan high-risk map saat ini dengan event historis realtime buffer. Untuk validasi ilmiah, gunakan evaluation_results/CSEP.",
        "horizon_days": horizon,
        "mag_threshold": threshold,
        "high_risk_top_pct": high_risk_top_pct,
        "total_cells": total_cells,
        "high_risk_cells": high_count,
        "cards": {
            "hit_rate_30d": _human_rate(events_in_high_30d / total_events_30d) if total_events_30d else None,
            "hit_rate_365d": _human_rate(events_in_high / len(event_rows)) if event_rows else None,
            "false_alarm_rate_365d": _human_rate(false_alarm_cells / high_count) if high_count else None,
            "high_risk_precision_365d": _human_rate(high_hit_cells / high_count) if high_count else None,
            "observed_events_30d": total_events_30d,
            "observed_events_365d": len(event_rows),
            "high_risk_cells_with_event_365d": high_hit_cells,
        },
        "monthly": monthly_rows,
        "top_high_risk": [
            {"cell_id": c["cell_id"], "label": c["full_label"], "probability": c["probability"]}
            for c in high_cells[:10]
        ],
    }
