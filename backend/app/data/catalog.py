"""Parquet I/O for historical events, declustered events, features, and forecast archive.

Conventions:
- Files use snake_case names matching Parquet variants of SQLite tables.
- Append uses dedup-by-key (event_id) so re-runs are idempotent.
- Forecast archive is per-day (YYYY-MM-DD.parquet).

All paths come from `Settings.parquet_path`.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.config import Settings, get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

EVENT_COLUMNS: list[str] = [
    "event_id",
    "time",
    "lat",
    "lon",
    "depth",
    "magnitude",
    "mag_type",
    "source",
    "place",
]


# ---------- helpers ----------


def _settings(s: Settings | None = None) -> Settings:
    return s or get_settings()


def _path_historical(s: Settings) -> Path:
    return s.parquet_path / "historical_events.parquet"


def _path_declustered(s: Settings) -> Path:
    return s.parquet_path / "declustered_events.parquet"


def _path_training(s: Settings) -> Path:
    return s.parquet_path / "training_features.parquet"


def _path_archive(s: Settings, day: date) -> Path:
    return s.parquet_path / "forecast_archive" / f"{day.isoformat()}.parquet"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------- historical events ----------


def read_historical_events(
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    min_mag: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Read historical events with optional filters.

    bbox: (lat_min, lon_min, lat_max, lon_max).
    """
    s = _settings(settings)
    p = _path_historical(s)
    if not p.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.read_parquet(p)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    if start is not None:
        df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["time"] <= pd.Timestamp(end, tz="UTC")]
    if min_mag is not None:
        df = df[df["magnitude"] >= min_mag]
    if bbox is not None:
        lat_min, lon_min, lat_max, lon_max = bbox
        df = df[
            (df["lat"] >= lat_min)
            & (df["lat"] <= lat_max)
            & (df["lon"] >= lon_min)
            & (df["lon"] <= lon_max)
        ]
    return df.reset_index(drop=True)


def append_historical_events(df: pd.DataFrame, settings: Settings | None = None) -> int:
    """Append events to the historical Parquet, deduping by event_id.

    Returns number of *new* rows actually written (excluding already-present IDs).
    """
    s = _settings(settings)
    p = _path_historical(s)

    if df.empty:
        return 0

    if "event_id" not in df.columns:
        raise ValueError("DataFrame must include 'event_id'")

    df = df.drop_duplicates(subset=["event_id"]).copy()

    if p.exists():
        existing = pd.read_parquet(p, columns=["event_id"])
        new_df = df[~df["event_id"].isin(existing["event_id"])]
        if new_df.empty:
            logger.info("append_historical_no_new_rows", incoming=len(df))
            return 0
        full = pd.concat([pd.read_parquet(p), new_df], ignore_index=True)
    else:
        new_df = df
        full = df

    _ensure_parent(p)
    full.to_parquet(p, index=False)
    logger.info("append_historical_done", appended=len(new_df), total=len(full))
    return int(len(new_df))


# ---------- declustered events ----------


def read_declustered_events(settings: Settings | None = None) -> pd.DataFrame:
    s = _settings(settings)
    p = _path_declustered(s)
    if not p.exists():
        return pd.DataFrame(columns=[*EVENT_COLUMNS, "is_mainshock", "cluster_id"])
    return pd.read_parquet(p)


def write_declustered_events(df: pd.DataFrame, settings: Settings | None = None) -> int:
    s = _settings(settings)
    p = _path_declustered(s)
    _ensure_parent(p)
    df.to_parquet(p, index=False)
    logger.info("write_declustered_done", n=len(df), path=str(p))
    return int(len(df))


# ---------- training features ----------


def read_training_features(settings: Settings | None = None) -> pd.DataFrame:
    s = _settings(settings)
    p = _path_training(s)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def write_training_features(df: pd.DataFrame, settings: Settings | None = None) -> int:
    s = _settings(settings)
    p = _path_training(s)
    _ensure_parent(p)
    df.to_parquet(p, index=False)
    logger.info("write_training_features_done", n=len(df), path=str(p))
    return int(len(df))


# ---------- forecast archive ----------


def archive_forecast(
    df: pd.DataFrame,
    day: date | None = None,
    settings: Settings | None = None,
) -> Path:
    """Snapshot forecast (current_forecasts table content) to per-day Parquet."""
    s = _settings(settings)
    day = day or date.today()
    p = _path_archive(s, day)
    _ensure_parent(p)
    df.to_parquet(p, index=False)
    logger.info("forecast_archive_done", day=day.isoformat(), n=len(df), path=str(p))
    return p


def read_forecast_archive(
    day: date,
    settings: Settings | None = None,
) -> pd.DataFrame:
    s = _settings(settings)
    p = _path_archive(s, day)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def list_forecast_archive_days(settings: Settings | None = None) -> list[date]:
    s = _settings(settings)
    archive_dir = s.parquet_path / "forecast_archive"
    if not archive_dir.exists():
        return []
    days: list[date] = []
    for f in archive_dir.glob("*.parquet"):
        try:
            days.append(date.fromisoformat(f.stem))
        except ValueError:
            continue
    return sorted(days)


# ---------- helpers for tests / introspection ----------


def storage_summary(settings: Settings | None = None) -> dict[str, Any]:
    s = _settings(settings)
    return {
        "historical": _path_historical(s).exists(),
        "declustered": _path_declustered(s).exists(),
        "training": _path_training(s).exists(),
        "archive_days": len(list_forecast_archive_days(s)),
    }
