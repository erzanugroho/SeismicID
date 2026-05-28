"""BMKG TEWS feed client + realtime HTML scraper.

Two data paths:
1. TEWS JSON endpoints (autogempa, terkini, dirasakan) — limited to ~31 latest
   records each, often only M ≥ 5 plus latest "felt" events.
2. Public realtime HTML page (gempabumi-realtime.bmkg) — Nuxt 3 SSR payload
   contains 100-200 raw earthquake records including micro-quakes (M ≥ ~2.0).

Coordinates can be like "0.90 LS 119.87 BT" → must parse hemisphere markers.
We also accept already-numeric Lat/Lon if upstream changes format.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.data.sources.base import EarthquakeSource, Event

logger = get_logger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_COORD_RE = re.compile(r"([0-9.]+)\s*(LS|LU|BT|BB)", re.IGNORECASE)
_NUXT_PAYLOAD_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.DOTALL)


def _parse_coord(s: str | float | None) -> float | None:
    if s is None:
        return None
    if isinstance(s, int | float):
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
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
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
        if isinstance(depth_str, int | float):
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
        self.realtime_html_url = s.bmkg_realtime_url
        self.timeout = timeout

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    def _request(self, url: str) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    def _request_html(self, url: str) -> str:
        with httpx.Client(timeout=self.timeout, headers={"User-Agent": _USER_AGENT}) as client:
            r = client.get(url, follow_redirects=True)
            r.raise_for_status()
            return r.text

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

    @staticmethod
    def _resolve_nuxt_ref(idx: Any, data: list, depth: int = 0) -> Any:
        """Resolve a Nuxt-3 hydration index reference recursively.

        Nuxt 3 SSR payloads are de-duplicated arrays where every value is replaced
        with an integer index that points back into the same array. We walk the
        graph until we hit a primitive value.
        """
        if depth > 6 or not isinstance(idx, int) or idx < 0 or idx >= len(data):
            return idx
        item = data[idx]
        if isinstance(item, dict):
            return {k: BMKGSource._resolve_nuxt_ref(v, data, depth + 1) for k, v in item.items()}
        if isinstance(item, list):
            return [BMKGSource._resolve_nuxt_ref(v, data, depth + 1) for v in item]
        return item

    @classmethod
    def _parse_realtime_html(cls, html: str) -> list[dict]:
        """Extract earthquake records from the public realtime HTML page.

        BMKG renders this page via Nuxt 3 SSR; the data is embedded in a
        ``window.__NUXT__`` script as a flattened, de-duplicated array. We find
        the script that contains both 'Infogempa' and 'gempa' keys, parse it as
        JSON, then walk the index references to produce a list of plain dicts.
        Each record has keys: eventid, status, waktu, lintang, bujur, dalam,
        mag, fokal, area.
        """
        payload = None
        for script_body in _NUXT_PAYLOAD_RE.findall(html):
            if "Infogempa" in script_body and "gempa" in script_body and len(script_body) > 5000:
                payload = script_body
                break
        if payload is None:
            return []
        try:
            data = json.loads(payload)
        except (ValueError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []

        # Find a dict that has a 'gempa' key and resolve its value (an index list)
        gempa_array: list | None = None
        for item in data:
            if isinstance(item, dict) and "gempa" in item:
                ref = item["gempa"]
                if isinstance(ref, int) and 0 <= ref < len(data):
                    candidate = data[ref]
                    if isinstance(candidate, list):
                        gempa_array = candidate
                        break
                elif isinstance(ref, list):
                    gempa_array = ref
                    break
        if not gempa_array:
            return []

        records = []
        for ref in gempa_array:
            rec = cls._resolve_nuxt_ref(ref, data) if isinstance(ref, int) else ref
            if isinstance(rec, dict):
                records.append(rec)
        return records

    @staticmethod
    def _parse_realtime_record(rec: dict) -> Event | None:
        """Convert a parsed BMKG realtime HTML record into an Event."""
        try:
            waktu = rec.get("waktu") or rec.get("DateTime")
            if not waktu:
                return None
            dt: datetime | None = None
            if isinstance(waktu, str):
                # Format 1: "2026/05/25  04:08:37.728" (UTC, double space)
                # Format 2: "2026-05-25T04:08:37+00:00"
                w = waktu.strip().replace("  ", " ")
                try:
                    dt = datetime.strptime(w, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
                except ValueError:
                    try:
                        dt = datetime.strptime(w, "%Y/%m/%d %H:%M:%S").replace(tzinfo=UTC)
                    except ValueError:
                        try:
                            dt = datetime.fromisoformat(w.replace("Z", "+00:00"))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=UTC)
                            else:
                                dt = dt.astimezone(UTC)
                        except ValueError:
                            return None
            if dt is None:
                return None

            lat = rec.get("lintang")
            lon = rec.get("bujur")
            if lat is None or lon is None:
                return None
            mag = rec.get("mag")
            if mag is None:
                return None
            depth = rec.get("dalam")
            event_id_raw = rec.get("eventid") or rec.get("event_id")
            eid = (
                f"bmkg_{event_id_raw}"
                if event_id_raw
                else f"bmkg_{int(dt.timestamp())}_{round(float(lat) * 100)}_{round(float(lon) * 100)}"
            )
            return Event(
                event_id=eid,
                time=dt,
                lat=float(lat),
                lon=float(lon),
                depth=float(depth) if depth is not None else None,
                magnitude=float(mag),
                mag_type=rec.get("fokal"),
                source="bmkg",
                place=rec.get("area") or rec.get("Wilayah"),
                raw=rec,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _fetch_realtime_html(
        self,
        bbox: tuple[float, float, float, float] | None,
        min_mag: float | None,
    ) -> list[Event]:
        """Fetch and parse the public realtime HTML page (extra coverage of micro-quakes)."""
        if not self.realtime_html_url:
            return []
        try:
            html = self._request_html(self.realtime_html_url)
        except Exception as e:  # noqa: BLE001
            logger.warning("bmkg_realtime_html_unavailable", url=self.realtime_html_url, error=str(e))
            return []
        records = self._parse_realtime_html(html)
        out: list[Event] = []
        for rec in records:
            ev = self._parse_realtime_record(rec)
            if ev is None:
                continue
            if min_mag is not None and ev.magnitude < min_mag:
                continue
            if bbox is not None:
                lat_min, lon_min, lat_max, lon_max = bbox
                if not (lat_min <= ev.lat <= lat_max and lon_min <= ev.lon <= lon_max):
                    continue
            out.append(ev)
        logger.info("bmkg_realtime_html_parsed", n_records=len(records), n_kept=len(out))
        return out

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

        # Realtime HTML scraper — gives 100-200 records including micro-quakes.
        events.extend(self._fetch_realtime_html(bbox=bbox, min_mag=min_mag))

        # Dedup within BMKG payloads themselves
        seen: set[str] = set()
        unique: list[Event] = []
        for ev_ in events:
            if ev_.event_id in seen:
                continue
            seen.add(ev_.event_id)
            unique.append(ev_)
        logger.info("bmkg_fetch_done", n=len(unique))
        return unique
