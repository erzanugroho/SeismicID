"""Abstract earthquake source + normalized Event dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class Event:
    event_id: str
    time: datetime  # UTC
    lat: float
    lon: float
    depth: float | None
    magnitude: float
    mag_type: str | None
    source: str
    place: str | None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["time"] = self.time.isoformat()
        d.pop("raw", None)
        return d


class EarthquakeSource(ABC):
    """Pluggable source adapter."""

    name: str = "base"

    @abstractmethod
    def fetch(
        self,
        start: datetime,
        end: datetime,
        bbox: tuple[float, float, float, float] | None = None,
        min_mag: float | None = None,
    ) -> list[Event]: ...
