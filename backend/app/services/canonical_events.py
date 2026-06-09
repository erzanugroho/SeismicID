"""Canonical earthquake event dedupe service.

This layer is additive: raw/source-level events stay in `realtime_events`; deduped
clusters are stored in `canonical_events` for evaluation and future features.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any

from backend.app.db.sqlite import get_connection, migrate

TIME_WINDOW_SECONDS = 10 * 60
DISTANCE_WINDOW_KM = 100.0
MAG_WINDOW = 0.5

SOURCE_PRIORITY = {
    "bmkg": 3,
    "usgs": 2,
    "emsc": 1,
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


def _source_score(source: str | None) -> int:
    return SOURCE_PRIORITY.get((source or "").lower(), 0)


def _same_cluster(event: dict[str, Any], cluster: dict[str, Any]) -> bool:
    dt1 = event["dt"]
    dt2 = cluster["dt"]
    return (
        abs((dt1 - dt2).total_seconds()) <= TIME_WINDOW_SECONDS
        and _km(float(event["lat"]), float(event["lon"]), float(cluster["lat"]), float(cluster["lon"])) <= DISTANCE_WINDOW_KM
        and abs(float(event["magnitude"]) - float(cluster["magnitude"])) <= MAG_WINDOW
    )


def _canonical_id(cluster: dict[str, Any]) -> str:
    dt = cluster["dt"].strftime("%Y%m%dT%H%M%S")
    seed = f"{dt}:{float(cluster['lat']):.3f}:{float(cluster['lon']):.3f}:{float(cluster['magnitude']):.1f}"
    return "ce_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def rebuild_canonical_events(*, days: int | None = None) -> dict[str, int | str]:
    """Rebuild canonical_events from realtime_events.

    days=None rebuilds all realtime_events. This is safe/idempotent: it replaces
    canonical_events only, never deletes raw realtime_events.
    """
    migrate()
    where = ""
    params: tuple[Any, ...] = ()
    if days is not None:
        where = "WHERE time >= datetime('now', ?)"
        params = (f"-{int(days)} days",)

    with get_connection() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                f"""
                SELECT event_id, time, lat, lon, depth, magnitude, place, source, raw_json
                FROM realtime_events
                {where}
                ORDER BY time ASC, magnitude DESC
                """,
                params,
            ).fetchall()
        ]

    clusters: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        dt = _parse_dt(str(row.get("time")))
        if not dt or row.get("lat") is None or row.get("lon") is None or row.get("magnitude") is None:
            skipped += 1
            continue
        row["dt"] = dt
        target = None
        for cluster in clusters:
            if _same_cluster(row, cluster):
                target = cluster
                break
        if target is None:
            row["members"] = [row]
            clusters.append(row)
            continue
        target["members"].append(row)
        # Canonical representative: prefer largest magnitude; tie by source priority.
        if (
            float(row["magnitude"]) > float(target["magnitude"])
            or (
                float(row["magnitude"]) == float(target["magnitude"])
                and _source_score(row.get("source")) > _source_score(target.get("source"))
            )
        ):
            members = target["members"]
            target.update(row)
            target["members"] = members

    now = datetime.now(UTC).isoformat()
    with get_connection() as conn:
        conn.execute("BEGIN")
        try:
            conn.execute("DELETE FROM canonical_events")
            for cluster in clusters:
                members = cluster.get("members") or []
                sources = sorted({str(m.get("source") or "unknown") for m in members})
                member_payload = [
                    {
                        "event_id": m.get("event_id"),
                        "time": m.get("time"),
                        "magnitude": m.get("magnitude"),
                        "source": m.get("source"),
                    }
                    for m in members
                ]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO canonical_events (
                        canonical_id, time, lat, lon, depth_km, magnitude, place,
                        primary_source, sources_json, source_count, members_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _canonical_id(cluster),
                        cluster["dt"].isoformat(),
                        float(cluster["lat"]),
                        float(cluster["lon"]),
                        None if cluster.get("depth") is None else float(cluster["depth"]),
                        float(cluster["magnitude"]),
                        cluster.get("place"),
                        cluster.get("source"),
                        json.dumps(sources),
                        len(sources),
                        json.dumps(member_payload),
                        now,
                        now,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    return {
        "raw_events": len(rows),
        "canonical_events": len(clusters),
        "deduped_events": max(0, len(rows) - len(clusters)),
        "skipped_events": skipped,
        "updated_at": now,
    }


def canonical_event_stats() -> dict[str, Any]:
    migrate()
    with get_connection() as conn:
        raw = conn.execute("SELECT COUNT(*) AS n FROM realtime_events").fetchone()["n"]
        canonical = conn.execute("SELECT COUNT(*) AS n FROM canonical_events").fetchone()["n"]
        multi = conn.execute("SELECT COUNT(*) AS n FROM canonical_events WHERE source_count > 1").fetchone()["n"]
        latest = conn.execute("SELECT time, magnitude, place, source_count FROM canonical_events ORDER BY time DESC LIMIT 1").fetchone()
    return {
        "raw_events": int(raw or 0),
        "canonical_events": int(canonical or 0),
        "deduped_events": max(0, int(raw or 0) - int(canonical or 0)),
        "multi_source_events": int(multi or 0),
        "latest": dict(latest) if latest else None,
    }
