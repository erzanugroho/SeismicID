"""Import external historical earthquake CSV catalogs into canonical Parquet.

Supported presets:
- isc_gem: ISC-GEM CSV export.
- noaa_significant: NOAA/NCEI Significant Earthquake Database CSV.
- bmkg: BMKG/manual CSV with Indonesian or English column names.
- generic: auto-detect common time/lat/lon/magnitude columns.

Usage:
    python scripts/import_catalog_csv.py --source isc_gem --file data/import/isc_gem.csv --min-mag 5.5
    python scripts/import_catalog_csv.py --source noaa_significant --file data/import/noaa.csv --min-mag 6.0
    python scripts/import_catalog_csv.py --source bmkg --file data/import/bmkg_historical.csv --min-mag 2.5
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.data.catalog import EVENT_COLUMNS, append_historical_events  # noqa: E402


def _first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _parse_time(df: pd.DataFrame) -> pd.Series:
    direct = _first_col(df, ["time", "datetime", "date_time", "origin_time", "DateTime", "Date"])
    if direct:
        return pd.to_datetime(df[direct], errors="coerce", utc=True)
    year = _first_col(df, ["year", "Year"])
    month = _first_col(df, ["month", "Month"])
    day = _first_col(df, ["day", "Day"])
    if year and month and day:
        hour = _first_col(df, ["hour", "Hour"]) or "__zero_hour"
        minute = _first_col(df, ["minute", "Minute"]) or "__zero_minute"
        second = _first_col(df, ["second", "Second"]) or "__zero_second"
        tmp = pd.DataFrame({
            "year": pd.to_numeric(df[year], errors="coerce"),
            "month": pd.to_numeric(df[month], errors="coerce").fillna(1),
            "day": pd.to_numeric(df[day], errors="coerce").fillna(1),
            "hour": pd.to_numeric(df[hour], errors="coerce").fillna(0) if hour in df else 0,
            "minute": pd.to_numeric(df[minute], errors="coerce").fillna(0) if minute in df else 0,
            "second": pd.to_numeric(df[second], errors="coerce").fillna(0) if second in df else 0,
        })
        return pd.to_datetime(tmp, errors="coerce", utc=True)
    raise ValueError("Could not detect time columns")


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    lat_col = _first_col(df, ["lat", "latitude", "Latitude", "Lintang"])
    lon_col = _first_col(df, ["lon", "longitude", "Longitude", "Bujur"])
    mag_col = _first_col(df, ["mag", "magnitude", "Magnitude", "mw", "MW", "Ms", "mb"])
    depth_col = _first_col(df, ["depth", "depth_km", "Depth", "Kedalaman"])
    place_col = _first_col(df, ["place", "location", "Location", "Wilayah", "country"])
    id_col = _first_col(df, ["event_id", "id", "EventID", "eventid", "iscid"])
    mag_type_col = _first_col(df, ["mag_type", "magnitude_type", "MagType", "magType"])

    if not (lat_col and lon_col and mag_col):
        raise ValueError("CSV must contain latitude, longitude, and magnitude columns")

    out = pd.DataFrame(columns=EVENT_COLUMNS)
    out["time"] = _parse_time(df)
    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    out["depth"] = pd.to_numeric(df[depth_col], errors="coerce") if depth_col else pd.NA
    out["magnitude"] = pd.to_numeric(df[mag_col], errors="coerce")
    out["mag_type"] = df[mag_type_col].astype(str) if mag_type_col else pd.NA
    out["source"] = source
    out["place"] = df[place_col].astype(str) if place_col else source

    if id_col:
        out["event_id"] = source + "_" + df[id_col].astype(str)
    else:
        keys = (out["time"].astype(str) + "|" + out["lat"].astype(str) + "|" + out["lon"].astype(str) + "|" + out["magnitude"].astype(str))
        out["event_id"] = [source + "_" + hashlib.sha1(k.encode()).hexdigest()[:16] for k in keys]

    return out.dropna(subset=["time", "lat", "lon", "magnitude"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["isc_gem", "noaa_significant", "bmkg", "generic"], required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--min-mag", type=float, default=None)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.file, encoding=args.encoding)
    out = normalize(df, args.source)
    settings = get_settings()
    out = out[
        (out["lat"] >= settings.grid_lat_min)
        & (out["lat"] <= settings.grid_lat_max)
        & (out["lon"] >= settings.grid_lon_min)
        & (out["lon"] <= settings.grid_lon_max)
    ]
    if args.min_mag is not None:
        out = out[out["magnitude"] >= args.min_mag]

    print(f"normalized={len(out)} source={args.source} range={out['time'].min()}..{out['time'].max() if not out.empty else None}")
    if args.dry_run:
        print(out.head().to_string())
        return 0
    inserted = append_historical_events(out)
    print(f"inserted={inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
