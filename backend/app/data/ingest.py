"""Ingestion pipeline: USGS + BMKG + EMSC → dedup → SQLite + Parquet.

Dedup rule: time delta <=60s, spatial delta <=0.5°, magnitude delta <=0.5.
USGS is canonical when match.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.data.catalog import EVENT_COLUMNS, append_historical_events
from backend.app.data.sources.base import Event
from backend.app.data.sources.bmkg import BMKGSource
from backend.app.data.sources.emsc import EMSCSource
from backend.app.data.sources.usgs import USGSSource
from backend.app.db.sqlite import get_connection, migrate

logger = get_logger(__name__)

DEDUP_TIME_S = 60
DEDUP_SPACE_DEG = 0.5
DEDUP_MAG = 0.5


def _is_duplicate(a: Event, b: Event) -> bool:
    if abs((a.time - b.time).total_seconds()) > DEDUP_TIME_S:
        return False
    if abs(a.lat - b.lat) > DEDUP_SPACE_DEG or abs(a.lon - b.lon) > DEDUP_SPACE_DEG:
        return False
    return abs(a.magnitude - b.magnitude) <= DEDUP_MAG


def dedup_events(events: list[Event]) -> list[Event]:
    """Dedup with source priority: USGS > EMSC > BMKG.

    USGS is canonical for global event metadata. EMSC ranks above BMKG because
    its event_id is stable across queries; BMKG TEWS reuses no canonical ID.
    """
    if not events:
        return []
    priority = {"usgs": 0, "emsc": 1, "bmkg": 2}
    events = sorted(events, key=lambda e: (priority.get(e.source, 9), e.time))
    out: list[Event] = []
    for ev in events:
        if any(_is_duplicate(ev, kept) for kept in out):
            continue
        out.append(ev)
    return out


def events_to_dataframe(events: list[Event]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows = []
    for e in events:
        rows.append(
            {
                "event_id": e.event_id,
                "time": e.time,
                "lat": e.lat,
                "lon": e.lon,
                "depth": e.depth,
                "magnitude": e.magnitude,
                "mag_type": e.mag_type,
                "source": e.source,
                "place": e.place,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def ingest_historical(start: datetime, end: datetime, *, min_mag: float = 2.5) -> int:
    """Bulk historical from USGS (chunked per-year) → Parquet."""
    settings = get_settings()
    bbox = (settings.grid_lat_min, settings.grid_lon_min, settings.grid_lat_max, settings.grid_lon_max)
    src = USGSSource()
    total = 0
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor.replace(year=cursor.year + 1))
        events = src.fetch(cursor, chunk_end, bbox=bbox, min_mag=min_mag)
        if events:
            df = events_to_dataframe(events)
            n = append_historical_events(df)
            total += n
            logger.info("ingest_historical_chunk", year=cursor.year, fetched=len(events), inserted=n)
        cursor = chunk_end
    return total


def _store_realtime(events: list[Event]) -> int:
    if not events:
        return 0
    migrate()
    rows = [
        (
            e.event_id,
            e.time.isoformat(),
            e.lat,
            e.lon,
            e.depth,
            e.magnitude,
            e.mag_type,
            e.source,
            e.place,
            json.dumps(e.raw, default=str) if e.raw else None,
        )
        for e in events
    ]
    with get_connection() as conn:
        conn.execute("BEGIN")
        try:
            cur = conn.executemany(
                """INSERT OR IGNORE INTO realtime_events
                   (event_id, time, lat, lon, depth, magnitude, mag_type, source, place, raw_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
            inserted = int(cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else conn.total_changes)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    # Also append to historical Parquet for future training
    append_historical_events(events_to_dataframe(events))
    return inserted


def ingest_realtime(
    *,
    fetch_usgs: bool = True,
    fetch_bmkg: bool = True,
    fetch_emsc: bool = True,
    lookback_hours: int | None = None,
) -> dict:
    """Fetch USGS feed (recent) + BMKG (live) + EMSC (FDSN) → dedup → store."""
    settings = get_settings()
    bbox = (settings.grid_lat_min, settings.grid_lon_min, settings.grid_lat_max, settings.grid_lon_max)
    end = datetime.now(UTC)
    if lookback_hours is None:
        lookback_hours = settings.forecast_lookback_hours
    start = end - timedelta(hours=lookback_hours)

    raw: list[Event] = []
    counts: dict[str, int] = {}
    if fetch_usgs:
        try:
            ev_usgs = USGSSource().fetch(start, end, bbox=bbox, min_mag=2.5)
            raw.extend(ev_usgs)
            counts["usgs"] = len(ev_usgs)
        except Exception as e:  # noqa: BLE001
            logger.warning("usgs_realtime_failed", error=str(e))
            counts["usgs"] = 0
    if fetch_bmkg:
        try:
            ev_bmkg = BMKGSource().fetch(bbox=bbox)
            raw.extend(ev_bmkg)
            counts["bmkg"] = len(ev_bmkg)
        except Exception as e:  # noqa: BLE001
            logger.warning("bmkg_realtime_failed", error=str(e))
            counts["bmkg"] = 0
    if fetch_emsc:
        try:
            ev_emsc = EMSCSource().fetch(start, end, bbox=bbox, min_mag=2.5)
            raw.extend(ev_emsc)
            counts["emsc"] = len(ev_emsc)
        except Exception as e:  # noqa: BLE001
            logger.warning("emsc_realtime_failed", error=str(e))
            counts["emsc"] = 0

    deduped = dedup_events(raw)
    n = _store_realtime(deduped)
    return {
        "raw": len(raw),
        "deduped": len(deduped),
        "stored": n,
        "lookback_hours": lookback_hours,
        "by_source": counts,
    }


def list_events(
    *,
    days: int = 7,
    min_mag: float | None = None,
    source: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Read recent events from SQLite (realtime) — for /api/events."""
    migrate()
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    where = ["time >= ?"]
    args: list = [cutoff]
    if min_mag is not None:
        where.append("magnitude >= ?")
        args.append(min_mag)
    if source:
        where.append("source = ?")
        args.append(source)
    sql = (
        "SELECT event_id, time, lat, lon, depth, magnitude, mag_type, source, place "
        "FROM realtime_events WHERE " + " AND ".join(where) + " ORDER BY time DESC LIMIT ?"
    )
    args.append(limit)
    with get_connection() as conn:
        cur = conn.execute(sql, args)
        return [dict(r) for r in cur.fetchall()]
