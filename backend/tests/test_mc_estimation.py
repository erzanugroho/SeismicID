"""Tests for magnitude of completeness (Mc) estimation via MAXC."""
from __future__ import annotations

import numpy as np


def test_maxc_recovers_known_mc_from_synthetic_gr_catalog() -> None:
    """Synthetic catalog with b=1.0 and Mc=4.5 should be recovered to within
    +/- 0.2 magnitude units by the MAXC estimator."""
    from backend.app.ml.mc_estimation import estimate_mc_maxc

    rng = np.random.default_rng(42)
    # Above-Mc population follows Gutenberg-Richter (exponential tail above Mc).
    n_above = 5000
    mags_above = 4.5 + rng.exponential(scale=1.0 / np.log(10.0) / 1.0, size=n_above)
    # Below-Mc events represent sub-completeness rolloff.
    n_below = 800
    mags_below = 3.0 + rng.uniform(0, 1.5, size=n_below)
    mags = np.concatenate([mags_above, mags_below])

    mc = estimate_mc_maxc(mags, bin_width=0.1)
    assert 4.3 <= mc <= 4.7, f"Mc estimate {mc} outside expected range"


def test_maxc_returns_nan_for_too_few_events() -> None:
    from backend.app.ml.mc_estimation import estimate_mc_maxc

    mc = estimate_mc_maxc(np.array([4.5, 4.7, 5.1]), bin_width=0.1)
    assert np.isnan(mc), "Should return NaN when sample too small"


def test_estimate_mc_per_region_groups_by_cell_id() -> None:
    """Per-region helper should return one Mc per group, NaN for sparse cells."""
    import pandas as pd

    from backend.app.ml.mc_estimation import estimate_mc_per_region

    rng = np.random.default_rng(0)
    big = pd.DataFrame(
        {
            "cell_id": ["A"] * 200,
            "magnitude": 4.5 + rng.exponential(0.5, size=200),
        }
    )
    small = pd.DataFrame({"cell_id": ["B"] * 5, "magnitude": [4.6, 4.7, 5.0, 4.8, 4.9]})
    events = pd.concat([big, small], ignore_index=True)

    out = estimate_mc_per_region(events, region_col="cell_id", min_events=50)
    assert "A" in out and "B" in out
    assert np.isfinite(out["A"]), "cell A should have a real Mc"
    assert np.isnan(out["B"]), "cell B too small, should be NaN"
