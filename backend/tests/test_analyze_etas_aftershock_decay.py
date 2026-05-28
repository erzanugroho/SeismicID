"""Smoke test for scripts.analyze_etas_aftershock_decay (Task 3.3).

We don't run the full real-catalog analysis here — that needs the parquet
data store. Instead we exercise the pure helpers on a synthetic catalog so
CI confirms the function shapes and the markdown writer don't regress.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _synthetic_catalog_around(event_time: datetime, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    pre = event_time - timedelta(days=365 * 4)
    times = [pre + timedelta(days=float(d)) for d in np.sort(rng.uniform(0, 365 * 4, size=n))]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(times, utc=True),
            "lat": rng.uniform(-8, 6, size=n),
            "lon": rng.uniform(95, 141, size=n),
            "magnitude": rng.uniform(4.5, 6.5, size=n),
            "depth": rng.uniform(5, 80, size=n),
        }
    )


def test_analyze_runs_and_emits_report(tmp_path: Path) -> None:
    from scripts.analyze_etas_aftershock_decay import (
        REFERENCE_EVENTS,
        analyze,
        write_markdown_report,
    )

    # Synthesize a catalog covering all three reference events.
    pieces = [_synthetic_catalog_around(ev.when) for ev in REFERENCE_EVENTS]
    events = pd.concat(pieces, ignore_index=True)
    out_dir = tmp_path / "data"
    report = analyze(events, out_dir)
    assert set(report.keys()) >= {ev.name for ev in REFERENCE_EVENTS}
    md = tmp_path / "report.md"
    write_markdown_report(report, md)
    text = md.read_text()
    assert "ETAS aftershock decay sanity check" in text
    for ev in REFERENCE_EVENTS:
        assert ev.name in text


def test_omori_slope_recovery_on_known_decay() -> None:
    """fit_omori_slope must recover a planted power-law slope within 0.2."""
    from scripts.analyze_etas_aftershock_decay import fit_omori_slope

    days = np.arange(1, 61)
    true_p = 1.1
    rates = 5.0 * days ** (-true_p)
    df = pd.DataFrame({"day": days, "rate": rates})
    p_hat = fit_omori_slope(df)
    assert abs(p_hat - true_p) < 0.2
