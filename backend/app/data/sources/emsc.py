"""EMSC (European-Mediterranean Seismological Centre) FDSN event service client.

EMSC's seismicportal.eu speaks the same FDSN-WS protocol as USGS so we can
reuse the same query/parse pattern. EMSC has good global coverage and is a
useful redundant source when BMKG/USGS are slow or incomplete.

Endpoint: GET /fdsnws/event/1/query?format=json&starttime=...&endtime=...
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.data.sources.base import EarthquakeSource, Event

logger = get_logger(__name__)

EMSC_MAX_PER_QUERY = 5000


class EMSCSource(EarthquakeSource):
    name = "emsc"

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = base_url or get_settings().emsc_base_url
        self.timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _request(self, params: dict[str, Any]) -> Any:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self.base_url, params=params)
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _parse(feature: dict) -> Event | None:
        """Parse a GeoJSON feature from EMSC FDSN-WS response."""
        try:
            props = feature.get("properties", {}) or {}
            geom = feature.get("geometry", {}) or {}
            time_s = props.get("time") or props.get("origintime")
            if not time_s:
                return None
            try:
                # EMSC returns ISO 8601 strings, e.g. "2026-05-25T04:08:37.728Z"
                dt = datetime.fromisoformat(str(time_s).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                else:
                    dt = dt.astimezone(UTC)
            except ValueError:
                return None

            mag = props.get("mag") or props.get("magnitude")
            if mag is None:
                return None
            coords = geom.get("coordinates", [None, None, None]) or [None, None, None]
            lon = coords[0] if len(coords) > 0 else None
            lat = coords[1] if len(coords) > 1 else None
            # EMSC returns the third GeoJSON coordinate following the spec's
            # elevation convention: positive = above sea level, negative = below.
            # USGS and BMKG instead emit depth as a positive km value (downward).
            # Prefer the explicit ``depth`` property when present (already
            # positive km); otherwise normalise the elevation-style coord by
            # taking abs() so all sources agree on "positive km below surface".
            depth_raw = props.get("depth")
            if depth_raw is None and len(coords) > 2:
                depth_raw = coords[2]
            depth = abs(float(depth_raw)) if depth_raw is not None else None
            if lat is None or lon is None:
                return None

            event_id_raw = (
                feature.get("id")
                or props.get("source_id")
                or props.get("unid")
                or f"{int(dt.timestamp())}_{round(float(lat) * 100)}_{round(float(lon) * 100)}"
            )
            return Event(
                event_id=f"emsc_{event_id_raw}",
                time=dt,
                lat=float(lat),
                lon=float(lon),
                depth=float(depth) if depth is not None else None,
                magnitude=float(mag),
                mag_type=props.get("magtype") or props.get("mag_type"),
                source="emsc",
                place=props.get("flynn_region") or props.get("region") or props.get("place"),
                raw=props,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def fetch(
        self,
        start: datetime,
        end: datetime,
        bbox: tuple[float, float, float, float] | None = None,
        min_mag: float | None = None,
    ) -> list[Event]:
        params: dict[str, Any] = {
            "format": "json",
            "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "limit": EMSC_MAX_PER_QUERY,
            "orderby": "time-asc",
        }
        if bbox is not None:
            lat_min, lon_min, lat_max, lon_max = bbox
            params.update(
                minlat=lat_min, maxlat=lat_max,
                minlon=lon_min, maxlon=lon_max,
            )
        if min_mag is not None:
            params["minmag"] = min_mag

        try:
            data = self._request(params)
        except Exception as e:  # noqa: BLE001
            logger.warning("emsc_fetch_failed", error=str(e), params=params)
            return []

        # EMSC returns either a FeatureCollection-style dict or a list directly
        if isinstance(data, dict):
            features = data.get("features", []) or []
        elif isinstance(data, list):
            features = data
        else:
            features = []

        events = [e for e in (self._parse(f) for f in features) if e is not None]
        logger.info(
            "emsc_fetch_done",
            n=len(events),
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return events
