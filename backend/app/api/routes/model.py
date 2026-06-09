"""GET /api/model/metadata and /api/model/evaluation."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends

from backend.app.api.deps import require_admin_token
from backend.app.config import get_settings
from backend.app.db.sqlite import get_connection, migrate
from backend.app.services.canonical_events import canonical_event_stats, rebuild_canonical_events

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

    if not events:
        try:
            import urllib.parse
            import urllib.request

            settings = get_settings()
            params = urllib.parse.urlencode({
                "format": "geojson",
                "starttime": start_date,
                "endtime": end_date,
                "minmagnitude": threshold,
                "minlatitude": settings.grid_lat_min,
                "maxlatitude": settings.grid_lat_max,
                "minlongitude": settings.grid_lon_min,
                "maxlongitude": settings.grid_lon_max,
                "orderby": "time-asc",
                "limit": 20000,
            })
            with urllib.request.urlopen(f"{settings.usgs_base_url}?{params}", timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            for feature in payload.get("features", []):
                props = feature.get("properties") or {}
                coords = (feature.get("geometry") or {}).get("coordinates") or [None, None, None]
                events.append({
                    "event_id": feature.get("id"),
                    "time": datetime.utcfromtimestamp((props.get("time") or 0) / 1000).isoformat() + "Z",
                    "lat": coords[1],
                    "lon": coords[0],
                    "depth": coords[2],
                    "magnitude": props.get("mag"),
                    "source": "usgs_live",
                    "place": props.get("place"),
                })
        except Exception:  # noqa: BLE001 - external fallback optional
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


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@router.get("/canonical-events/status")
def canonical_events_status() -> dict:
    return {"mode": "canonical_events", **canonical_event_stats()}


@router.post("/canonical-events/rebuild", dependencies=[Depends(require_admin_token)])
def canonical_events_rebuild(days: int | None = 60) -> dict:
    return {"mode": "canonical_events_rebuild", **rebuild_canonical_events(days=days)}


@router.get("/pre-event-backtest")
def pre_event_backtest(
    threshold: float = 6.0,
    radius_km: float = 300.0,
    limit: int = 50,
    event_source: str = "canonical",
) -> dict:
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        return {"mode": "pre_event_rank", "error": f"pandas_unavailable: {exc}"}

    settings = get_settings()
    migrate()
    with get_connection() as conn:
        area_rows = [dict(r) for r in conn.execute(
            "SELECT cell_id, full_label, lat, lon, lat_min, lat_max, lon_min, lon_max FROM area_labels"
        ).fetchall()]
        source_mode = "canonical" if event_source.lower() not in {"raw", "realtime"} else "raw"
        canonical_count = conn.execute("SELECT COUNT(*) AS n FROM canonical_events").fetchone()["n"]
        if source_mode == "canonical" and int(canonical_count or 0) > 0:
            events = [dict(r) for r in conn.execute(
                """SELECT canonical_id AS event_id, time, lat, lon, magnitude, place, primary_source AS source, source_count AS members
                   FROM canonical_events
                   WHERE magnitude >= ?
                   ORDER BY time ASC""",
                (threshold,),
            ).fetchall()]
        else:
            source_mode = "raw"
            events = [dict(r) for r in conn.execute(
                """SELECT event_id, time, lat, lon, magnitude, place, source, 1 AS members
                   FROM realtime_events
                   WHERE magnitude >= ?
                   ORDER BY time ASC""",
                (threshold,),
            ).fetchall()]

    def event_cell(lat: float, lon: float) -> dict | None:
        for c in area_rows:
            if c["lat_min"] <= lat < c["lat_max"] and c["lon_min"] <= lon < c["lon_max"]:
                return c
        return None

    deduped: list[dict] = []
    if source_mode == "canonical":
        for ev in events:
            dt = _parse_dt(str(ev.get("time")))
            if not dt or ev.get("lat") is None or ev.get("lon") is None:
                continue
            ev["dt"] = dt
            ev["member_count"] = int(ev.get("members") or 1)
            deduped.append(ev)
    else:
        for ev in events:
            dt = _parse_dt(str(ev.get("time")))
            if not dt or ev.get("lat") is None or ev.get("lon") is None:
                continue
            ev["dt"] = dt
            assigned = False
            for cl in deduped:
                if abs((dt - cl["dt"]).total_seconds()) <= 900 and _km(float(ev["lat"]), float(ev["lon"]), float(cl["lat"]), float(cl["lon"])) <= 120:
                    cl.setdefault("members", []).append(ev)
                    if float(ev.get("magnitude") or 0) > float(cl.get("magnitude") or 0):
                        members = cl["members"]
                        cl.update(ev)
                        cl["members"] = members
                    assigned = True
                    break
            if not assigned:
                ev["members"] = [ev]
                deduped.append(ev)

    archive_dir = settings.parquet_path / "forecast_archive"
    files: list[tuple[datetime, str]] = []
    for f in archive_dir.glob("*/*.parquet"):
        try:
            issued = datetime.strptime(f"{f.parent.name} {f.name[:6]}", "%Y-%m-%d %H%M%S").replace(tzinfo=UTC)
            files.append((issued, str(f)))
        except ValueError:
            continue
    files.sort(key=lambda x: x[0])

    leads = [("last_before", timedelta(seconds=0)), ("1h_before", timedelta(hours=1)), ("6h_before", timedelta(hours=6)), ("24h_before", timedelta(hours=24)), ("7d_before", timedelta(days=7))]
    suffix = str(threshold).replace(".", "")
    horizon_cols = {h: f"label_h{h}_m{suffix}" for h in [7, 14, 30, 60]}

    def choose_file(event_dt: datetime, min_lead: timedelta) -> tuple[datetime, str] | None:
        chosen = None
        cutoff = event_dt - min_lead
        for issued, file_path in files:
            if issued <= cutoff:
                chosen = (issued, file_path)
            else:
                break
        return chosen

    def _rank_from_probability(df, col: str, probability: float) -> int:  # noqa: ANN001
        return int((df[col] > probability).sum() + 1)

    def _best_rank_for_ids(df, ids: set[str], col: str) -> dict:  # noqa: ANN001
        nearby = df[df["cell_id"].isin(ids)].sort_values(col, ascending=False)
        if nearby.empty:
            return {"rank": None, "probability": None, "cell_id": None}
        row = nearby.iloc[0]
        prob = float(row[col])
        return {"rank": _rank_from_probability(df, col, prob), "probability": prob, "cell_id": row["cell_id"]}

    def rank_payload(df, cell_id: str, col: str, cell: dict) -> dict | None:  # noqa: ANN001
        if col not in df.columns:
            return None
        sub = df[df["cell_id"] == cell_id]
        if sub.empty:
            return None
        prob = float(sub.iloc[0][col])
        exact_rank = _rank_from_probability(df, col, prob)
        percentile = float((df[col] <= prob).mean() * 100)
        reciprocal_rank = 1.0 / exact_rank if exact_rank > 0 else 0.0
        ndcg10 = 1.0 / math.log2(exact_rank + 1) if exact_rank <= 10 else 0.0
        lat_step = float(cell["lat_max"] - cell["lat_min"])
        lon_step = float(cell["lon_max"] - cell["lon_min"])
        ring1_ids = {
            c["cell_id"]
            for c in area_rows
            if max(abs(float(c["lat"]) - float(cell["lat"])) / max(lat_step, 1e-9), abs(float(c["lon"]) - float(cell["lon"])) / max(lon_step, 1e-9)) <= 1.01
        }
        ring2_ids = {
            c["cell_id"]
            for c in area_rows
            if max(abs(float(c["lat"]) - float(cell["lat"])) / max(lat_step, 1e-9), abs(float(c["lon"]) - float(cell["lon"])) / max(lon_step, 1e-9)) <= 2.01
        }
        cluster100_ids = {c["cell_id"] for c in area_rows if _km(float(cell["lat"]), float(cell["lon"]), float(c["lat"]), float(c["lon"])) <= 100.0}
        cluster_radius_ids = {c["cell_id"] for c in area_rows if _km(float(cell["lat"]), float(cell["lon"]), float(c["lat"]), float(c["lon"])) <= radius_km}
        neighbor1 = _best_rank_for_ids(df, ring1_ids, col)
        neighbor2 = _best_rank_for_ids(df, ring2_ids, col)
        cluster100 = _best_rank_for_ids(df, cluster100_ids, col)
        cluster_radius = _best_rank_for_ids(df, cluster_radius_ids, col)
        return {
            "probability": prob,
            "rank": exact_rank,
            "percentile": round(percentile, 2),
            "reciprocal_rank": round(reciprocal_rank, 6),
            "ndcg10": round(ndcg10, 6),
            "top10": exact_rank <= 10,
            "top25": exact_rank <= 25,
            "top50": exact_rank <= 50,
            "top100": exact_rank <= 100,
            "neighbor_ring1_best_rank": neighbor1["rank"],
            "neighbor_ring1_top10": bool(neighbor1["rank"] and neighbor1["rank"] <= 10),
            "neighbor_ring1_best_cell_id": neighbor1["cell_id"],
            "neighbor_ring1_best_probability": neighbor1["probability"],
            "neighbor_ring2_best_rank": neighbor2["rank"],
            "neighbor_ring2_top10": bool(neighbor2["rank"] and neighbor2["rank"] <= 10),
            "neighbor_ring2_best_cell_id": neighbor2["cell_id"],
            "neighbor_ring2_best_probability": neighbor2["probability"],
            "cluster100_best_rank": cluster100["rank"],
            "cluster100_top10": bool(cluster100["rank"] and cluster100["rank"] <= 10),
            "cluster100_best_cell_id": cluster100["cell_id"],
            "cluster100_best_probability": cluster100["probability"],
            "cluster_best_rank": cluster_radius["rank"],
            "cluster_top10": bool(cluster_radius["rank"] and cluster_radius["rank"] <= 10),
            "cluster_best_cell_id": cluster_radius["cell_id"],
            "cluster_best_probability": cluster_radius["probability"],
        }

    event_results: list[dict] = []
    columns = ["cell_id", *horizon_cols.values()]
    for ev in deduped[: max(1, min(limit, 500))]:
        cell = event_cell(float(ev["lat"]), float(ev["lon"]))
        item = {"event_id": ev.get("event_id"), "time": ev["dt"].isoformat(), "magnitude": ev.get("magnitude"), "place": ev.get("place"), "source": ev.get("source"), "members": int(ev.get("member_count") or len(ev.get("members") or [])), "cell_id": cell["cell_id"] if cell else None, "label": cell["full_label"] if cell else None, "leads": {}}
        if not cell:
            event_results.append(item)
            continue
        for lead_name, lead_delta in leads:
            chosen = choose_file(ev["dt"], lead_delta)
            if not chosen:
                item["leads"][lead_name] = {"snapshot": None, "available": False}
                continue
            issued, file_path = chosen
            try:
                df = pd.read_parquet(file_path, columns=columns)
                item["leads"][lead_name] = {"snapshot": issued.isoformat(), "available": True, "horizons": {str(h): rank_payload(df, cell["cell_id"], col, cell) for h, col in horizon_cols.items()}}
            except Exception as exc:  # noqa: BLE001
                item["leads"][lead_name] = {"snapshot": issued.isoformat(), "available": False, "error": str(exc)}
        event_results.append(item)

    summary: dict[str, dict] = {}
    for lead_name, _ in leads:
        for h in [7, 14, 30, 60]:
            rows = []
            for ev in event_results:
                hp = ((ev.get("leads") or {}).get(lead_name) or {}).get("horizons") or {}
                val = hp.get(str(h))
                if val:
                    rows.append(val)
            ranks = sorted([r["rank"] for r in rows])
            key = f"{lead_name}_h{h}"
            if rows:
                mrr = sum(float(r.get("reciprocal_rank") or 0.0) for r in rows) / len(rows)
                ndcg10 = sum(float(r.get("ndcg10") or 0.0) for r in rows) / len(rows)
            else:
                mrr = 0.0
                ndcg10 = 0.0
            summary[key] = {
                "events": len(rows),
                "top10_hit_rate": _human_rate(sum(1 for r in rows if r["top10"]) / len(rows)) if rows else None,
                "top25_hit_rate": _human_rate(sum(1 for r in rows if r["top25"]) / len(rows)) if rows else None,
                "top50_hit_rate": _human_rate(sum(1 for r in rows if r["top50"]) / len(rows)) if rows else None,
                "top100_hit_rate": _human_rate(sum(1 for r in rows if r["top100"]) / len(rows)) if rows else None,
                "neighbor_ring1_top10_hit_rate": _human_rate(sum(1 for r in rows if r["neighbor_ring1_top10"]) / len(rows)) if rows else None,
                "neighbor_ring2_top10_hit_rate": _human_rate(sum(1 for r in rows if r["neighbor_ring2_top10"]) / len(rows)) if rows else None,
                "cluster100_top10_hit_rate": _human_rate(sum(1 for r in rows if r["cluster100_top10"]) / len(rows)) if rows else None,
                "cluster_top10_hit_rate": _human_rate(sum(1 for r in rows if r["cluster_top10"]) / len(rows)) if rows else None,
                "mrr": round(mrr, 6) if rows else None,
                "ndcg10": round(ndcg10, 6) if rows else None,
                "median_rank": ranks[len(ranks) // 2] if ranks else None,
            }

    return {"mode": "pre_event_rank", "threshold": threshold, "event_source": source_mode, "radius_km": radius_km, "archive_files": len(files), "archive_start": files[0][0].isoformat() if files else None, "archive_end": files[-1][0].isoformat() if files else None, "events": event_results, "summary": summary, "note": "Prospective-style check: forecast snapshot must be earlier than event time. Cluster metric uses best-ranked cell within radius_km of event cell."}
