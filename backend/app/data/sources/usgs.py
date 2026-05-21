"""USGS ComCat FDSN event service client.

Endpoint: GET /fdsnws/event/1/query?format=geojson&starttime=...&endtime=...
Paging: max 20000 per query → if hit, halve the time window and recurse.
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

USGS_MAX_PER_QUERY = 20000


class USGSSource(EarthquakeSource):
    name = "usgs"

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = base_url or get_settings().usgs_base_url
        self.timeout = timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _request(self, params: dict[str, Any]) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self.base_url, params=params)
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _parse(feature: dict) -> Event | None:
        try:
            props = feature["properties"]
            geom = feature["geometry"]
            time_ms = props.get("time")
            if time_ms is None:
                return None
            dt = datetime.fromtimestamp(time_ms / 1000.0, tz=UTC)
            mag = props.get("mag")
            if mag is None:
                return None
            coords = geom.get("coordinates", [None, None, None])
            return Event(
                event_id=f"usgs_{feature.get('id')}",
                time=dt,
                lat=float(coords[1]),
                lon=float(coords[0]),
                depth=float(coords[2]) if coords[2] is not None else None,
                magnitude=float(mag),
                mag_type=props.get("magType"),
                source="usgs",
                place=props.get("place"),
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
            "format": "geojson",
            "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "orderby": "time-asc",
            "limit": USGS_MAX_PER_QUERY,
        }
        if bbox is not None:
            lat_min, lon_min, lat_max, lon_max = bbox
            params.update(
                minlatitude=lat_min, maxlatitude=lat_max,
                minlongitude=lon_min, maxlongitude=lon_max,
            )
        if min_mag is not None:
            params["minmagnitude"] = min_mag

        try:
            data = self._request(params)
        except Exception as e:  # noqa: BLE001
            logger.error("usgs_fetch_failed", error=str(e), params=params)
            return []

        features = data.get("features", [])
        # If we hit the cap, split window in half and recurse.
        if len(features) >= USGS_MAX_PER_QUERY and (end - start).total_seconds() > 86400:
            mid = start + (end - start) / 2
            logger.info("usgs_paging_split", start=start.isoformat(), mid=mid.isoformat(), end=end.isoformat())
            left = self.fetch(start, mid, bbox, min_mag)
            right = self.fetch(mid, end, bbox, min_mag)
            return left + right

        events = [e for e in (self._parse(f) for f in features) if e is not None]
        logger.info("usgs_fetch_done", n=len(events), start=start.isoformat(), end=end.isoformat())
        return events
