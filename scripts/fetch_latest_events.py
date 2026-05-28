"""Fetch latest realtime earthquake events into local storage."""

from __future__ import annotations

import argparse
import json

from backend.app.data.ingest import ingest_realtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch latest USGS/BMKG/EMSC earthquake events")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        help="Hours of catalog to fetch per source (default: forecast_lookback_hours config = 72)",
    )
    parser.add_argument("--no-usgs", action="store_true")
    parser.add_argument("--no-bmkg", action="store_true")
    parser.add_argument("--no-emsc", action="store_true")
    args = parser.parse_args()

    out = ingest_realtime(
        fetch_usgs=not args.no_usgs,
        fetch_bmkg=not args.no_bmkg,
        fetch_emsc=not args.no_emsc,
        lookback_hours=args.lookback_hours,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
