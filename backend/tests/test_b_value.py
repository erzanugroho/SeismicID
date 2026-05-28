"""Tests for the Aki-Utsu Gutenberg-Richter b-value estimator (Task 3.1)."""
from __future__ import annotations

import numpy as np


def test_aki_utsu_recovers_b_within_tolerance() -> None:
    from backend.app.ml.b_value import estimate_b_aki_utsu

    rng = np.random.default_rng(0)
    mc = 4.5
    true_b = 0.95
    # GR magnitudes follow exponential tail with rate beta = b * ln(10)
    mags = mc + rng.exponential(1.0 / (true_b * np.log(10)), size=5000)
    b_hat = estimate_b_aki_utsu(mags, mc=mc)
    assert abs(b_hat - true_b) < 0.10


def test_b_value_returns_default_for_small_sample() -> None:
    from backend.app.ml.b_value import estimate_b_aki_utsu

    b_hat = estimate_b_aki_utsu(np.array([4.6, 4.7, 5.0]), mc=4.5, default=1.0)
    assert b_hat == 1.0


def test_b_value_drops_subthreshold_events() -> None:
    """Mags below mc must be filtered before computing the mean."""
    from backend.app.ml.b_value import estimate_b_aki_utsu

    rng = np.random.default_rng(7)
    mc = 4.5
    mags_above = mc + rng.exponential(1.0 / (1.0 * np.log(10)), size=2000)
    mags_below = rng.uniform(2.0, 4.0, size=10000)
    mixed = np.concatenate([mags_above, mags_below])
    b_hat = estimate_b_aki_utsu(mixed, mc=mc)
    assert 0.85 < b_hat < 1.15


def test_b_value_returns_default_when_mean_at_mc() -> None:
    """Edge case: degenerate catalog where mean(mags) == mc."""
    from backend.app.ml.b_value import estimate_b_aki_utsu

    mags = np.full(100, 4.5)
    b_hat = estimate_b_aki_utsu(mags, mc=4.5, default=1.0)
    assert b_hat == 1.0


def test_etas_predict_consumes_per_cell_b_value_dict() -> None:
    """OgataETAS.predict_dataframe must accept a dict {cell_id: b}."""
    from datetime import datetime, timedelta, timezone

    import pandas as pd
    from backend.app.core.grid import generate_grid
    from backend.app.ml.etas_ogata import OgataETAS

    rng = np.random.default_rng(13)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    times = [base + timedelta(days=float(d)) for d in np.sort(rng.uniform(0, 365, size=40))]
    events = pd.DataFrame(
        {
            "time": pd.to_datetime(times, utc=True),
            "lat": rng.uniform(-8, 6, size=40),
            "lon": rng.uniform(95, 141, size=40),
            "magnitude": rng.uniform(4.5, 6.0, size=40),
            "depth": rng.uniform(5, 80, size=40),
        }
    )
    cells = [c.cell_id for c in generate_grid()[:3]]
    model = OgataETAS(mc=4.5).fit_from_events(
        events,
        observation_start=events["time"].min().to_pydatetime(),
        observation_end=events["time"].max().to_pydatetime(),
    )
    # Lower b → heavier upper-mag tail → higher P at higher thresholds.
    b_low = {cells[0]: 0.7, cells[1]: 0.7, cells[2]: 0.7}
    b_high = {cells[0]: 1.3, cells[1]: 1.3, cells[2]: 1.3}
    issued = events["time"].max().to_pydatetime()
    pred_low = model.predict_dataframe(cells, issued_at=issued, b_value=b_low)
    pred_high = model.predict_dataframe(cells, issued_at=issued, b_value=b_high)
    # Pick highest threshold available — low b should yield >= prob there.
    high_thr_cols = [c for c in pred_low.columns if c.endswith("_m60")]
    assert high_thr_cols, "expected a label_h*_m60 column"
    col = high_thr_cols[0]
    assert (pred_low[col] >= pred_high[col]).all()
