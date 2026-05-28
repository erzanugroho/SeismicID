"""Parquet I/O for historical events, declustered events, features, and forecast archive.

Conventions:
- Files use snake_case names matching Parquet variants of SQLite tables.
- Append uses dedup-by-key (event_id) so re-runs are idempotent.
- Forecast archive is per-run (immutable) under
  ``forecast_archive/<YYYY-MM-DD>/<HHMMSSZ>_<model_version>.parquet``.
  Backwards-compatible reads still recognise the legacy single-file layout
  (``forecast_archive/<YYYY-MM-DD>.parquet``) so older deployments keep
  working until they are rewritten.

All paths come from `Settings.parquet_path`.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
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


def _path_archive_legacy(s: Settings, day: date) -> Path:
    """Legacy single-file-per-day archive path, kept for backward compat."""
    return s.parquet_path / "forecast_archive" / f"{day.isoformat()}.parquet"


def _archive_dir_for_day(s: Settings, day: date) -> Path:
    """New per-run archive directory for a given UTC day."""
    return s.parquet_path / "forecast_archive" / day.isoformat()


_SAFE_VERSION_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_model_version(model_version: str | None) -> str:
    if not model_version:
        return "unknown"
    return _SAFE_VERSION_RE.sub("_", str(model_version)).strip("_") or "unknown"


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
    *,
    model_version: str | None = None,
    issued_at: datetime | None = None,
    raw_df: pd.DataFrame | None = None,
    baseline_type: str = "ml",
) -> Path:
    """Write a snapshot of the forecast frame to an immutable per-run file.

    Each call produces a distinct Parquet file under
    ``data/parquet/forecast_archive/<UTC-date>/<HHMMSS>Z_<model_version>.parquet``
    so prospective evaluation always sees an unmodified history (the original
    overwriting behaviour was flagged in the audit). Three metadata columns
    are appended to the frame to make the archive self-describing:

    * ``forecast_run_id``  — ``<date>T<HHMMSSZ>_<model_version>``
    * ``issued_at_utc``    — ISO-8601 UTC timestamp of the run
    * ``model_version``    — the active model version (``"unknown"`` if absent)

    When ``raw_df`` is supplied (the calibrated, pre-public-cap predictions),
    every probability column is duplicated with a ``raw_`` prefix so prospective
    skill scoring can audit the calibrated value while the UI keeps using the
    capped ``label_*`` columns.

    The return value is the path to the written Parquet file.
    """
    s = _settings(settings)
    issued = issued_at or datetime.now(UTC)
    issued = issued.replace(tzinfo=UTC) if issued.tzinfo is None else issued.astimezone(UTC)
    day = day or issued.date()

    archive_dir = _archive_dir_for_day(s, day)
    archive_dir.mkdir(parents=True, exist_ok=True)

    safe_version = _safe_model_version(model_version)
    ts_str = issued.strftime("%H%M%SZ")
    run_id = f"{day.isoformat()}T{ts_str}_{safe_version}"
    fname = f"{ts_str}_{safe_version}.parquet"
    path = archive_dir / fname

    # If two runs land in the exact same second (extremely unlikely but
    # plausible on cron + manual trigger collisions), de-dup with a counter
    # so we never overwrite an existing immutable run.
    if path.exists():
        i = 1
        while True:
            candidate = archive_dir / f"{ts_str}_{safe_version}_{i:02d}.parquet"
            if not candidate.exists():
                path = candidate
                break
            i += 1

    annotated = df.copy()

    # Merge raw (pre-cap, pre-shrinkage) probabilities under raw_<col> names so
    # downstream tooling can compare display vs scoring values without joining
    # against a separate file. Falls back to no-op if columns mismatch.
    if raw_df is not None and not raw_df.empty and "cell_id" in raw_df.columns:
        prob_cols = [c for c in raw_df.columns if c.startswith("label_h")]
        if prob_cols:
            renamed = raw_df[["cell_id", *prob_cols]].rename(
                columns={c: f"raw_{c}" for c in prob_cols}
            )
            annotated = annotated.merge(renamed, on="cell_id", how="left")

    annotated["forecast_run_id"] = run_id
    annotated["issued_at_utc"] = issued.isoformat()
    annotated["model_version"] = model_version or "unknown"
    annotated["baseline_type"] = baseline_type
    annotated.to_parquet(path, index=False)
    logger.info(
        "forecast_archive_done",
        day=day.isoformat(),
        n=len(annotated),
        path=str(path),
        run_id=run_id,
        with_raw=bool(raw_df is not None and not raw_df.empty),
    )
    return path


def read_forecast_archive(
    day: date,
    settings: Settings | None = None,
    *,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Read the most recent forecast archive for a given UTC day.

    Honors both the new per-run directory layout and the legacy single-file
    layout. Pass ``run_id`` to fetch a specific run; otherwise the most
    recent file (lex-largest filename → latest UTC timestamp) is returned.
    """
    s = _settings(settings)
    archive_dir = _archive_dir_for_day(s, day)
    if archive_dir.exists() and archive_dir.is_dir():
        files = sorted(archive_dir.glob("*.parquet"))
        if run_id is not None:
            for f in files:
                # ``forecast_run_id`` lives in the data; we can also match by
                # filename pattern derived from the run id.
                df_one = pd.read_parquet(f)
                if (
                    "forecast_run_id" in df_one.columns
                    and (df_one["forecast_run_id"] == run_id).any()
                ):
                    return df_one
            return pd.DataFrame()
        if files:
            return pd.read_parquet(files[-1])
    legacy = _path_archive_legacy(s, day)
    if legacy.exists():
        return pd.read_parquet(legacy)
    return pd.DataFrame()


def list_forecast_archive_days(settings: Settings | None = None) -> list[date]:
    s = _settings(settings)
    archive_dir = s.parquet_path / "forecast_archive"
    if not archive_dir.exists():
        return []
    days: set[date] = set()
    for f in archive_dir.glob("*.parquet"):
        try:
            days.add(date.fromisoformat(f.stem))
        except ValueError:
            continue
    for d in archive_dir.iterdir():
        if d.is_dir():
            try:
                days.add(date.fromisoformat(d.name))
            except ValueError:
                continue
    return sorted(days)


def list_forecast_archive_runs(
    day: date,
    settings: Settings | None = None,
) -> list[Path]:
    """Return every per-run archive file for a given UTC day, sorted ascending."""
    s = _settings(settings)
    archive_dir = _archive_dir_for_day(s, day)
    runs: list[Path] = []
    if archive_dir.exists() and archive_dir.is_dir():
        runs.extend(sorted(archive_dir.glob("*.parquet")))
    legacy = _path_archive_legacy(s, day)
    if legacy.exists():
        runs.append(legacy)
    return runs


# ---------- helpers for tests / introspection ----------


def storage_summary(settings: Settings | None = None) -> dict[str, Any]:
    s = _settings(settings)
    return {
        "historical": _path_historical(s).exists(),
        "declustered": _path_declustered(s).exists(),
        "training": _path_training(s).exists(),
        "archive_days": len(list_forecast_archive_days(s)),
    }
