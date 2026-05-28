"""Aftershock decay sanity test for OgataETAS — Phase 3 Task 3.3.

For each reference Indonesian mainshock (Lombok 2018 M6.9, Palu 2018 M7.5,
Mamuju 2021 M6.2):
  1. Slice the catalog 5 years before the event.
  2. Fit OgataETAS on that slice.
  3. Predict the per-day rate at the mainshock cell for 60 days following.
  4. Plot rate-vs-time on log-log; the slope should land in p ∈ [0.8, 1.5]
     (literature range for Omori law).

Usage:
    python -m scripts.analyze_etas_aftershock_decay
    python -m scripts.analyze_etas_aftershock_decay --catalog data/parquet/historical_events.parquet --out docs/notebooks/etas_aftershock_validation.md

The script runs on the WSL Python (`./.venv-wsl/bin/python`). It writes a
markdown report next to a directory of PNG plots.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.ml.etas_ogata import OgataETAS  # noqa: E402


@dataclass(frozen=True)
class ReferenceEvent:
    name: str
    when: datetime
    lat: float
    lon: float
    magnitude: float


REFERENCE_EVENTS: tuple[ReferenceEvent, ...] = (
    ReferenceEvent(
        "Lombok 2018",
        datetime(2018, 8, 5, 11, 46, 38, tzinfo=timezone.utc),
        -8.288, 116.452, 6.9,
    ),
    ReferenceEvent(
        "Palu 2018",
        datetime(2018, 9, 28, 10, 2, 45, tzinfo=timezone.utc),
        -0.178, 119.840, 7.5,
    ),
    ReferenceEvent(
        "Mamuju 2021",
        datetime(2021, 1, 14, 18, 35, 49, tzinfo=timezone.utc),
        -2.969, 118.892, 6.2,
    ),
)


def fit_pre_event_etas(
    events: pd.DataFrame, mainshock: ReferenceEvent, *, mc: float = 4.5,
    lookback_years: int = 5,
) -> OgataETAS:
    obs_end = mainshock.when
    obs_start = obs_end - timedelta(days=365 * lookback_years)
    df = events[
        (events["time"] >= obs_start) & (events["time"] <= obs_end)
    ].copy()
    if df.empty:
        raise RuntimeError(
            f"no events in 5y window before {mainshock.name} — catalog gap?"
        )
    return OgataETAS(mc=mc).fit_from_events(
        df, observation_start=obs_start, observation_end=obs_end,
    )


def post_event_rates(
    model: OgataETAS, mainshock: ReferenceEvent, *, days_after: int = 60,
) -> pd.DataFrame:
    """Per-day rate at the mainshock cell for ``days_after`` days post-event."""
    rows: list[dict] = []
    base = mainshock.when
    for d in range(1, days_after + 1):
        t = base + timedelta(days=d)
        t_query_days = (t - model._t0).total_seconds() / 86400  # type: ignore[attr-defined]
        rate_per_km2_day = model._cell_rate(  # type: ignore[attr-defined]
            mainshock.lat, mainshock.lon, t_query_days=t_query_days,
        )
        # Per ~50x50 km cell — comparable to predict_dataframe default.
        rate_cell_day = rate_per_km2_day * 2500.0
        rows.append({"day": d, "rate": rate_cell_day})
    return pd.DataFrame(rows)


def fit_omori_slope(rates: pd.DataFrame) -> float:
    """Linear regression of log10(rate) vs log10(day) → power-law exponent p."""
    valid = rates[(rates["rate"] > 0) & (rates["day"] > 0)].copy()
    if len(valid) < 5:
        return float("nan")
    x = np.log10(valid["day"].to_numpy())
    y = np.log10(valid["rate"].to_numpy())
    slope, _intercept = np.polyfit(x, y, 1)
    return float(-slope)


def analyze(events: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for ev in REFERENCE_EVENTS:
        try:
            model = fit_pre_event_etas(events, ev)
            rates = post_event_rates(model, ev)
            p_hat = fit_omori_slope(rates)
            ok = 0.8 <= p_hat <= 1.5
            report[ev.name] = {
                "p_estimate": round(p_hat, 3) if np.isfinite(p_hat) else None,
                "in_range": bool(ok),
                "params": {k: round(float(v), 4) for k, v in model.params_.items()},
                "rates_csv": str(out_dir / f"{ev.name.replace(' ', '_')}.csv"),
            }
            rates.to_csv(out_dir / f"{ev.name.replace(' ', '_')}.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            report[ev.name] = {"error": str(exc)}
    return report


def write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        "# ETAS aftershock decay sanity check",
        "",
        "Phase 3 Task 3.3. For each reference Indonesian mainshock we fit",
        "`OgataETAS` on the 5-year pre-event window and project the per-day",
        "rate at the mainshock cell for 60 days post-event. The Omori",
        "exponent is recovered by log-log regression; literature range is",
        "p ∈ [0.8, 1.5].",
        "",
        "| Event | p̂ | In [0.8, 1.5]? | μ | K | c | p (fit) | α |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in report.items():
        if "error" in entry:
            lines.append(f"| {name} | — | — | error: {entry['error']} | | | | |")
            continue
        params = entry["params"]
        lines.append(
            f"| {name} | {entry['p_estimate']} | {'✅' if entry['in_range'] else '❌'} "
            f"| {params['mu']} | {params['K']} | {params['c']} | "
            f"{params['p']} | {params['alpha']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", default="data/parquet/historical_events.parquet",
        help="Parquet catalog path",
    )
    parser.add_argument(
        "--out", default="docs/notebooks/etas_aftershock_validation.md",
        help="Markdown report destination",
    )
    args = parser.parse_args()
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"catalog not found: {catalog_path}", file=sys.stderr)
        return 1
    events = pd.read_parquet(catalog_path)
    if "time" in events.columns:
        events["time"] = pd.to_datetime(events["time"], utc=True)
    out_dir = Path(args.out).parent / "etas_aftershock_data"
    report = analyze(events, out_dir)
    write_markdown_report(report, Path(args.out))
    print(f"wrote {args.out}")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
