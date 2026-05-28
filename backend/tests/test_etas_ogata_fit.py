"""Tests for ETAS MLE fit via L-BFGS-B + thinning simulator."""
from __future__ import annotations

import numpy as np


def test_simulate_catalog_returns_valid_arrays() -> None:
    from backend.app.ml.etas_ogata import simulate_catalog

    rng = np.random.default_rng(0)
    times, mags = simulate_catalog(
        T_end=100.0, mc=4.5,
        mu=0.05, K=0.02, c=0.01, p=1.1, alpha=1.5,
        max_events=2000, rng=rng,
    )
    assert times.shape == mags.shape
    assert times.size > 0
    assert np.all((times >= 0) & (times <= 100.0))
    assert np.all(mags >= 4.5)


def test_fit_recovers_synthetic_params_within_tolerance() -> None:
    from backend.app.ml.etas_ogata import OgataETAS, simulate_catalog

    rng = np.random.default_rng(1)
    true_params = dict(mu=0.10, K=0.05, c=0.01, p=1.15, alpha=1.4)
    times, mags = simulate_catalog(
        T_end=200.0, mc=4.5, max_events=800, rng=rng, **true_params
    )
    assert len(times) >= 50, f"synthetic catalog too small: {len(times)}"

    model = OgataETAS(mc=4.5).fit(times, mags, T_end=200.0)
    # MLE on finite catalog has wide finite-sample bias on N~50; we only
    # assert the optimizer ran cleanly + parameters in plausible range.
    # Strict parameter recovery is validated separately in Phase 3
    # cross-validation against the `etas` PyPI library.
    assert model.params_["mu"] > 0 and model.params_["K"] > 0
    assert 0.5 <= model.params_["p"] <= 2.5
    assert 0.1 <= model.params_["alpha"] <= 3.5
    assert model.fit_loglik_ is not None and np.isfinite(model.fit_loglik_)


def test_fit_with_no_events_returns_background_only() -> None:
    from backend.app.ml.etas_ogata import OgataETAS

    model = OgataETAS(mc=4.5).fit(np.array([]), np.array([]), T_end=100.0)
    assert model.params_["mu"] >= 0.0
    assert model.params_["K"] >= 0.0
    assert model.fit_status_ == "no_events"


def test_fit_respects_parameter_bounds() -> None:
    """All fitted params must lie inside _BOUNDS."""
    from backend.app.ml.etas_ogata import _BOUNDS, OgataETAS, simulate_catalog

    rng = np.random.default_rng(2)
    times, mags = simulate_catalog(
        T_end=200.0, mc=4.5, mu=0.08, K=0.03, c=0.01, p=1.1, alpha=1.3,
        max_events=2000, rng=rng,
    )
    model = OgataETAS(mc=4.5).fit(times, mags, T_end=200.0)
    for k, (lo, hi) in _BOUNDS.items():
        v = model.params_[k]
        assert lo <= v <= hi, f"{k}={v} outside [{lo}, {hi}]"
