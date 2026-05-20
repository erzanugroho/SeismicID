"""Bulk historical USGS download into Parquet.

Usage:
    python scripts/bootstrap_data.py --start 2000 --end 2024
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.logging import configure_logging, get_logger  # noqa: E402
from backend.app.data.ingest import ingest_historical  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=datetime.now().year)
    parser.add_argument("--min-mag", type=float, default=2.5)
    args = parser.parse_args()

    configure_logging("INFO")
    start = datetime(args.start, 1, 1, tzinfo=timezone.utc)
    end = datetime(args.end, 12, 31, tzinfo=timezone.utc)
    n = ingest_historical(start, end, min_mag=args.min_mag)
    logger.info("bootstrap_complete", events_added=n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
