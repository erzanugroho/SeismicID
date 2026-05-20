"""Run forecast manually and persist cached results."""

from __future__ import annotations

import argparse
import json

from backend.app.services.forecast_service import run_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Run earthquake forecast once")
    parser.add_argument("--force-demo", action="store_true", help="Force physics-aware demo seed mode")
    args = parser.parse_args()

    out = run_forecast(force_demo=args.force_demo)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
