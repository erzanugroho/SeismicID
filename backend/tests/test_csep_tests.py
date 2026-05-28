"""Tests for the dedicated CSEP-tests module (Phase 4 Task 4.2).

The L-test and N-test logic already lives in ``evaluate.py``; this task
exposes it as a stable public API at ``backend.app.ml.csep_tests`` and adds
a synthetic-ground-truth check so the two-sided pass band (per Issue 5 of
test_probability_audit) is not silently regressed.
"""
from __future__ import annotations

import numpy as np


def test_csep_tests_module_exposes_l_and_n_tests() -> None:
    from backend.app.ml import csep_tests

    assert hasattr(csep_tests, "run_l_test")
    assert hasattr(csep_tests, "run_n_test")
    assert hasattr(csep_tests, "run_csep_tests")


def test_n_test_passes_on_well_calibrated_synthetic_forecast() -> None:
    """If forecast rate matches the data-generating process, N-test must pass."""
    from backend.app.ml.csep_tests import run_n_test

    rng = np.random.default_rng(0)
    n_cells = 500
    p = np.full(n_cells, 0.05)
    y = (rng.uniform(size=n_cells) < p).astype(int)
    out = run_n_test(y, p)
    # Two-sided pass band: 0.025 <= q <= 0.975. Use both p_value sides.
    assert out["status"] == "pass", out


def test_l_test_passes_on_well_calibrated_synthetic_forecast() -> None:
    from backend.app.ml.csep_tests import run_l_test

    rng = np.random.default_rng(7)
    n = 200
    p = np.full(n, 0.05)
    y = (rng.uniform(size=n) < p).astype(int)
    out = run_l_test(y, p, n_sim=500)
    assert out["status"] == "pass", out


def test_n_test_fails_when_forecast_underestimates_rate() -> None:
    """Hard signal: if true rate is 5x higher than forecast, N-test must reject."""
    from backend.app.ml.csep_tests import run_n_test

    rng = np.random.default_rng(11)
    n_cells = 1000
    p_forecast = np.full(n_cells, 0.01)
    p_true = np.full(n_cells, 0.05)
    y = (rng.uniform(size=n_cells) < p_true).astype(int)
    out = run_n_test(y, p_forecast)
    assert out["status"] == "fail", out
