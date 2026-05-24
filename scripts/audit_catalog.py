"""Catalog coverage audit for Indonesia earthquake datasets.

Usage:
    python scripts/audit_catalog.py
    python scripts/audit_catalog.py --csv reports/catalog_audit.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.data.catalog import read_historical_events  # noqa: E402


def _counts_by_year(df: pd.DataFrame, min_mag: float) -> pd.DataFrame:
    sub = df[df["magnitude"] >= min_mag].copy()
    if sub.empty:
        return pd.DataFrame(columns=["year", f"m_ge_{min_mag:g}"])
    out = sub.groupby(sub["time"].dt.year).size().rename(f"m_ge_{min_mag:g}").reset_index()
    out.columns = ["year", f"m_ge_{min_mag:g}"]
    return out


def build_report() -> tuple[str, pd.DataFrame]:
    df = read_historical_events()
    if df.empty:
        return "No historical events found.", pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], utc=True)
    lines: list[str] = []
    lines.append("Catalog audit")
    lines.append(f"rows={len(df)}")
    lines.append(f"time_range={df['time'].min()} .. {df['time'].max()}")
    lines.append(f"lat_range={df['lat'].min():.4f} .. {df['lat'].max():.4f}")
    lines.append(f"lon_range={df['lon'].min():.4f} .. {df['lon'].max():.4f}")
    lines.append(f"mag_range={df['magnitude'].min():.2f} .. {df['magnitude'].max():.2f}")
    lines.append(f"duplicate_event_ids={int(df['event_id'].duplicated().sum())}")
    lines.append("source_counts=" + str(df["source"].value_counts(dropna=False).to_dict()))
    lines.append("null_counts=" + str(df.isna().sum().to_dict()))

    thresholds = [2.5, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
    for threshold in thresholds:
        lines.append(f"count_m>={threshold:g}={int((df['magnitude'] >= threshold).sum())}")

    report = None
    for threshold in thresholds:
        part = _counts_by_year(df, threshold)
        report = part if report is None else report.merge(part, on="year", how="outer")
    assert report is not None
    report = report.fillna(0).sort_values("year")
    for col in report.columns:
        if col != "year":
            report[col] = report[col].astype(int)

    years = set(range(int(df["time"].dt.year.min()), int(df["time"].dt.year.max()) + 1))
    present = set(df["time"].dt.year.astype(int).unique())
    missing = sorted(years - present)
    lines.append("missing_years=" + (",".join(map(str, missing)) if missing else "none"))
    lines.append("\nby_year_thresholds:\n" + report.to_string(index=False))
    return "\n".join(lines), report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="Write by-year threshold report to CSV")
    args = parser.parse_args()
    text, report = build_report()
    print(text)
    if args.csv and not report.empty:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.csv, index=False)
        print(f"csv_written={args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
