"""CSEP-style hypothesis tests for earthquake forecasts (Phase 4 Task 4.2).

The numerical implementations live in ``backend.app.ml.evaluate``. This module
re-exports them under a stable public API so prospective evaluators and the
ETAS validation pipeline can import a single canonical surface:

    from backend.app.ml.csep_tests import run_l_test, run_n_test, run_csep_tests

All three tests use a two-sided 95% pass band (0.025 <= q <= 0.975) per
Issue 5 in ``test_probability_audit`` — one-sided thresholds were silently
accepting implausibly good fits.
"""
from __future__ import annotations

from backend.app.ml.evaluate import (  # noqa: F401
    csep_tests as run_csep_tests,
    run_l_test,
    run_n_test,
    run_s_test,
)

__all__ = ["run_l_test", "run_n_test", "run_s_test", "run_csep_tests"]
