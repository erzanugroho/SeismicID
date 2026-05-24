"""Bulk historical USGS download into Parquet with resume and coverage reports.

Recommended layers:
    # Training main layer: modern complete-ish M>=4.5
    python scripts/bootstrap_data.py --start 1970 --end 2026 --min-mag 4.5

    # Modern activity layer: smaller events for recent temporal features
    python scripts/bootstrap_data.py --start 2010 --end 2026 --min-mag 2.5
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.core.logging import configure_logging, get_logger  # noqa: E402
from backend.app.data.catalog import append_historical_events, read_historical_events  # noqa: E402
from backend.app.data.ingest import events_to_dataframe  # noqa: E402
from backend.app.data.sources.usgs import USGSSource  # noqa: E402

logger = get_logger(__name__)


def _window_bounds(start_year: int, end_year: int, chunk: str) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    if chunk == "year":
        for year in range(start_year, end_year + 1):
            windows.append((datetime(year, 1, 1, tzinfo=UTC), datetime(year + 1, 1, 1, tzinfo=UTC)))
    elif chunk == "month":
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                nxt_year = year + 1 if month == 12 else year
                nxt_month = 1 if month == 12 else month + 1
                windows.append((datetime(year, month, 1, tzinfo=UTC), datetime(nxt_year, nxt_month, 1, tzinfo=UTC)))
    else:
        raise ValueError(f"Unsupported chunk={chunk}")
    return windows


def _has_year(df: pd.DataFrame, year: int, min_mag: float) -> bool:
    if df.empty:
        return False
    sub = df[(df["time"].dt.year == year) & (df["magnitude"] >= min_mag)]
    return not sub.empty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1970)
    parser.add_argument("--end", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--min-mag", type=float, default=4.5)
    parser.add_argument("--chunk", choices=["year", "month"], default="year")
    parser.add_argument("--resume", action="store_true", help="Skip years already present for this min magnitude")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")
    settings = get_settings()
    bbox = (settings.grid_lat_min, settings.grid_lon_min, settings.grid_lat_max, settings.grid_lon_max)
    existing = read_historical_events()
    if not existing.empty:
        existing["time"] = pd.to_datetime(existing["time"], utc=True)

    src = USGSSource()
    total_fetched = 0
    total_inserted = 0
    skipped = 0

    for start, end in _window_bounds(args.start, args.end, args.chunk):
        if args.resume and args.chunk == "year" and _has_year(existing, start.year, args.min_mag):
            skipped += 1
            logger.info("bootstrap_skip_existing_year", year=start.year, min_mag=args.min_mag)
            continue
        logger.info("bootstrap_fetch_window", start=start.isoformat(), end=end.isoformat(), min_mag=args.min_mag)
        if args.dry_run:
            continue
        events = src.fetch(start, end, bbox=bbox, min_mag=args.min_mag)
        total_fetched += len(events)
        if not events:
            continue
        n = append_historical_events(events_to_dataframe(events))
        total_inserted += n
        logger.info("bootstrap_window_done", fetched=len(events), inserted=n)

    logger.info(
        "bootstrap_complete",
        fetched=total_fetched,
        inserted=total_inserted,
        skipped_windows=skipped,
        min_mag=args.min_mag,
    )
    print({"fetched": total_fetched, "inserted": total_inserted, "skipped_windows": skipped})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
