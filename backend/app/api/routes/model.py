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


@router.get("/backtest")
def backtest(
    start: str,
    end: str,
    horizon: int = 30,
    threshold: float = 5.0,
    high_risk_top_pct: float = 0.10,
) -> dict:
    """Backtesting UI endpoint.

    Uses available forecast snapshot and observed events in selected period.
    If historical forecast archives are not available, this is a current-map replay proxy.
    """
    from datetime import datetime

    def parse_date(value: str) -> str:
        try:
            return datetime.fromisoformat(value[:10]).date().isoformat()
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc

    start_date = parse_date(start)
    end_date = parse_date(end)
    high_risk_top_pct = max(0.01, min(float(high_risk_top_pct), 0.50))
    migrate()
    with get_connection() as conn:
        total_row = conn.execute(
            "SELECT COUNT(*) AS n FROM current_forecasts WHERE horizon_days = ? AND mag_threshold = ?",
            (horizon, threshold),
        ).fetchone()
        total_cells = int(total_row["n"] or 0) if total_row else 0
        high_risk_n = max(1, int(total_cells * high_risk_top_pct)) if total_cells else 0
        high_cells = [dict(r) for r in conn.execute(
            """SELECT cf.cell_id, cf.probability, al.lat_min, al.lat_max, al.lon_min, al.lon_max, al.full_label
               FROM current_forecasts cf
               JOIN area_labels al ON al.cell_id = cf.cell_id
               WHERE cf.horizon_days = ? AND cf.mag_threshold = ?
               ORDER BY cf.probability DESC
               LIMIT ?""",
            (horizon, threshold, high_risk_n),
        ).fetchall()]
        events = [dict(r) for r in conn.execute(
            """SELECT event_id, time, lat, lon, depth, magnitude, source, place
               FROM realtime_events
               WHERE magnitude >= ? AND substr(time, 1, 10) >= ? AND substr(time, 1, 10) <= ?
               ORDER BY time ASC""",
            (threshold, start_date, end_date),
        ).fetchall()]

    # Add historical catalog when available so older periods (e.g. Jan-Mar 2026)
    # work even if realtime_events buffer is short.
    try:
        import pandas as pd

        hist_path = get_settings().parquet_path / "historical_events.parquet"
        if hist_path.exists():
            df = pd.read_parquet(hist_path, columns=["event_id", "time", "lat", "lon", "depth", "magnitude", "source", "place"])
            df["date"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
            df = df[(df["magnitude"] >= threshold) & (df["date"] >= start_date) & (df["date"] <= end_date)]
            events.extend(df.drop(columns=["date"]).to_dict("records"))
    except Exception:  # noqa: BLE001 - backtest should still work with realtime DB only
        pass

    deduped: dict[str, dict] = {}
    for ev in events:
        event_id = str(ev.get("event_id") or f"{ev.get('time')}-{ev.get('lat')}-{ev.get('lon')}")
        if event_id not in deduped:
            ev["time"] = str(ev.get("time"))
            deduped[event_id] = ev
    events = sorted(deduped.values(), key=lambda x: str(x.get("time")))

    high_ids = {c["cell_id"] for c in high_cells}

    def containing_high_cell(ev: dict) -> dict | None:
        lat = float(ev["lat"]); lon = float(ev["lon"])
        for c in high_cells:
            if c["lat_min"] <= lat < c["lat_max"] and c["lon_min"] <= lon < c["lon_max"]:
                return c
        return None

    event_results = []
    hit_cells = set()
    for ev in events:
        cell = containing_high_cell(ev)
        hit = bool(cell and cell["cell_id"] in high_ids)
        if hit and cell:
            hit_cells.add(cell["cell_id"])
        event_results.append({
            "event_id": ev["event_id"],
            "time": ev["time"],
            "magnitude": ev["magnitude"],
            "place": ev.get("place"),
            "source": ev.get("source"),
            "lat": ev["lat"],
            "lon": ev["lon"],
            "hit": hit,
            "matched_cell_id": cell["cell_id"] if cell else None,
            "matched_label": cell["full_label"] if cell else None,
            "matched_probability": cell["probability"] if cell else None,
        })

    total_events = len(event_results)
    hits = sum(1 for e in event_results if e["hit"])
    misses = total_events - hits
    false_alarm_cells = max(len(high_cells) - len(hit_cells), 0)

    monthly: dict[str, dict] = {}
    for ev in event_results:
        month = str(ev["time"])[:7]
        monthly.setdefault(month, {"month": month, "events": 0, "hits": 0, "misses": 0})
        monthly[month]["events"] += 1
        monthly[month]["hits"] += 1 if ev["hit"] else 0
        monthly[month]["misses"] += 0 if ev["hit"] else 1
    for row in monthly.values():
        row["hit_rate"] = _human_rate(row["hits"] / row["events"]) if row["events"] else None

    return {
        "mode": "current_map_replay",
        "note": "Backtest memakai forecast snapshot yang tersedia saat ini terhadap event pada periode pilihan. Historical forecast archive dapat ditambahkan kemudian untuk prospective replay penuh.",
        "start": start_date,
        "end": end_date,
        "horizon_days": horizon,
        "mag_threshold": threshold,
        "high_risk_top_pct": high_risk_top_pct,
        "total_cells": total_cells,
        "high_risk_cells": len(high_cells),
        "summary": {
            "events": total_events,
            "hits": hits,
            "misses": misses,
            "hit_rate": _human_rate(hits / total_events) if total_events else None,
            "high_risk_cells_hit": len(hit_cells),
            "false_alarm_cells": false_alarm_cells,
            "false_alarm_rate": _human_rate(false_alarm_cells / len(high_cells)) if high_cells else None,
        },
        "monthly": [monthly[k] for k in sorted(monthly)],
        "events": event_results,
        "top_high_risk": [
            {"cell_id": c["cell_id"], "label": c["full_label"], "probability": c["probability"]}
            for c in high_cells[:10]
        ],
    }
