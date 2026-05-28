"""Tests for temporal Ogata ETAS log-likelihood."""
from __future__ import annotations

import numpy as np


def test_loglik_returns_finite_for_simple_catalog() -> None:
    from backend.app.ml.etas_ogata import ogata_loglik

    times = np.array([0.5, 1.2, 2.4, 3.8, 5.1, 7.3])  # days since T0
    mags = np.array([5.0, 4.6, 4.8, 5.2, 4.7, 5.0])
    params = dict(mu=0.05, K=0.02, c=0.01, p=1.1, alpha=1.5)
    ll = ogata_loglik(times, mags, T_end=10.0, mc=4.5, **params)
    assert np.isfinite(ll), "log-likelihood must be finite"


def test_loglik_higher_at_truth_than_far_from_truth() -> None:
    """Catalog generated with mu=0.05 should score better at mu=0.05 than mu=10."""
    from backend.app.ml.etas_ogata import ogata_loglik

    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0, 100, size=80))
    mags = 4.5 + rng.exponential(0.5, size=80)
    common = dict(K=0.02, c=0.01, p=1.1, alpha=1.5)
    ll_good = ogata_loglik(times, mags, T_end=100.0, mc=4.5, mu=0.05, **common)
    ll_bad = ogata_loglik(times, mags, T_end=100.0, mc=4.5, mu=10.0, **common)
    assert ll_good > ll_bad, f"good {ll_good} should beat bad {ll_bad}"


def test_loglik_empty_catalog_equals_minus_mu_T() -> None:
    """Empty catalog: log L = -mu * T_end (only background integral)."""
    from backend.app.ml.etas_ogata import ogata_loglik

    ll = ogata_loglik(
        np.array([]), np.array([]), T_end=50.0, mc=4.5,
        mu=0.1, K=0.02, c=0.01, p=1.1, alpha=1.5,
    )
    assert abs(ll - (-0.1 * 50.0)) < 1e-9


def test_loglik_p_eq_1_branch_is_finite() -> None:
    """The p == 1 branch uses log integral; ensure no division-by-zero."""
    from backend.app.ml.etas_ogata import ogata_loglik

    times = np.array([1.0, 3.0, 7.0])
    mags = np.array([4.6, 4.8, 5.0])
    ll = ogata_loglik(
        times, mags, T_end=10.0, mc=4.5,
        mu=0.05, K=0.02, c=0.01, p=1.0, alpha=1.5,
    )
    assert np.isfinite(ll)
