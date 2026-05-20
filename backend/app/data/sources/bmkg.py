"""BMKG TEWS feed client (3 endpoints).

BMKG returns a single object or a list of dicts with Indonesian field names.
Coordinates can be like "0.90 LS 119.87 BT" → must parse hemisphere markers.
We also accept already-numeric Lat/Lon if upstream changes format.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.data.sources.base import EarthquakeSource, Event

logger = get_logger(__name__)

_COORD_RE = re.compile(r"([0-9.]+)\s*(LS|LU|BT|BB)", re.IGNORECASE)


def _parse_coord(s: str | float | None) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = _COORD_RE.search(str(s))
    if not m:
        try:
            return float(s)
        except (TypeError, ValueError):
            return None
    val = float(m.group(1))
    hemi = m.group(2).upper()
    if hemi in ("LS", "BB"):  # South / West → negative
        val = -val
    return val


def _parse_datetime(s: str | None) -> datetime | None:
    """BMKG sends 'DateTime': '2024-01-15T12:34:56+00:00' or 'Tanggal'+'Jam' separately."""
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_bmkg_record(rec: dict[str, Any], source_tag: str) -> Event | None:
    try:
        # BMKG keys vary — try a few common spellings
        dt = _parse_datetime(rec.get("DateTime") or rec.get("dateTime") or rec.get("datetime"))
        if dt is None:
            return None
        lat = _parse_coord(rec.get("Lintang") or rec.get("Lat") or rec.get("latitude"))
        lon = _parse_coord(rec.get("Bujur") or rec.get("Lon") or rec.get("longitude"))
        if lat is None or lon is None:
            return None
        mag = rec.get("Magnitude") or rec.get("Mag") or rec.get("magnitude")
        if mag is None:
            return None
        depth_str = rec.get("Kedalaman") or rec.get("Depth") or rec.get("depth")
        depth: float | None
        if isinstance(depth_str, (int, float)):
            depth = float(depth_str)
        elif isinstance(depth_str, str):
            try:
                depth = float(depth_str.split()[0])
            except (ValueError, IndexError):
                depth = None
        else:
            depth = None

        # Stable BMKG ID from time+coords (source has no canonical event_id)
        eid = f"bmkg_{int(dt.timestamp())}_{round(lat * 100)}_{round(lon * 100)}"
        return Event(
            event_id=eid,
            time=dt,
            lat=float(lat),
            lon=float(lon),
            depth=depth,
            magnitude=float(mag),
            mag_type=rec.get("MagType"),
            source="bmkg",
            place=rec.get("Wilayah") or rec.get("place"),
            raw=rec,
        )
    except (KeyError, TypeError, ValueError):
        return None


class BMKGSource(EarthquakeSource):
    """Fetches BMKG live feeds. Historical bulk not supported by BMKG API."""

    name = "bmkg"

    def __init__(self, timeout: float = 30.0):
        s = get_settings()
        self.urls = {
            "autogempa": s.bmkg_autogempa_url,
            "terkini": s.bmkg_terkini_url,
            "dirasakan": s.bmkg_dirasakan_url,
        }
        self.timeout = timeout

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    def _request(self, url: str) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()

    def _records_from(self, payload: Any, key_chain: list[str]) -> list[dict]:
        node: Any = payload
        for k in key_chain:
            if isinstance(node, dict):
                node = node.get(k, {})
            else:
                return []
        if isinstance(node, list):
            return node
        if isinstance(node, dict):
            return [node]
        return []

    def fetch(
        self,
        start: datetime | None = None,  # ignored — BMKG only has live feed
        end: datetime | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        min_mag: float | None = None,
    ) -> list[Event]:
        events: list[Event] = []
        for tag, url in self.urls.items():
            try:
                payload = self._request(url)
            except Exception as e:  # noqa: BLE001
                logger.warning("bmkg_endpoint_unavailable", url=url, error=str(e))
                continue
            # Common BMKG nesting: Infogempa.gempa or Infogempa.Gempa
            records = (
                self._records_from(payload, ["Infogempa", "gempa"])
                or self._records_from(payload, ["Infogempa", "Gempa"])
                or self._records_from(payload, ["gempa"])
            )
            for r in records:
                ev = _parse_bmkg_record(r, tag)
                if ev is None:
                    continue
                if min_mag is not None and ev.magnitude < min_mag:
                    continue
                if bbox is not None:
                    lat_min, lon_min, lat_max, lon_max = bbox
                    if not (lat_min <= ev.lat <= lat_max and lon_min <= ev.lon <= lon_max):
                        continue
                events.append(ev)
        # Dedup within BMKG payloads themselves
        seen: set[str] = set()
        unique: list[Event] = []
        for e in events:
            if e.event_id in seen:
                continue
            seen.add(e.event_id)
            unique.append(e)
        logger.info("bmkg_fetch_done", n=len(unique))
        return unique
