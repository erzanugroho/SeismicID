"""Tests for evaluate_dataset with dual baselines (Poisson + ETAS-Ogata)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.features.labels import all_label_columns


def _toy_truth_and_pred() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = all_label_columns()
    n = 8
    rng = np.random.default_rng(0)
    truth = pd.DataFrame(
        {c: rng.integers(0, 2, size=n).astype(int) for c in cols}
    )
    truth["cell_id"] = [f"c{i % 4}" for i in range(n)]
    truth["snapshot"] = pd.to_datetime(["2024-01-01"] * n, utc=True)
    preds = truth.copy()
    for c in cols:
        # Predictions slightly correlated with truth, deliberately imperfect.
        preds[c] = (truth[c] * 0.6 + 0.2).astype(float)
    return truth, preds


def test_evaluate_reports_bss_for_etas_when_provided() -> None:
    from backend.app.ml.evaluate import evaluate_dataset

    truth, preds = _toy_truth_and_pred()
    cols = all_label_columns()
    unique_cells = truth["cell_id"].drop_duplicates().tolist()
    poisson_b = pd.DataFrame({c: np.full(len(unique_cells), 0.4) for c in cols})
    poisson_b["cell_id"] = unique_cells
    etas_b = pd.DataFrame({c: np.full(len(unique_cells), 0.3) for c in cols})
    etas_b["cell_id"] = unique_cells

    out = evaluate_dataset(
        truth, preds, baseline=poisson_b, baseline_etas=etas_b
    )
    sample_head = next(iter(out["per_head"].values()))
    assert "bss_vs_poisson" in sample_head
    assert "bss_vs_etas" in sample_head


def test_evaluate_without_etas_baseline_omits_etas_bss() -> None:
    """Backward compatibility: omitting baseline_etas keeps behavior unchanged."""
    from backend.app.ml.evaluate import evaluate_dataset

    truth, preds = _toy_truth_and_pred()
    cols = all_label_columns()
    unique_cells = truth["cell_id"].drop_duplicates().tolist()
    poisson_b = pd.DataFrame({c: np.full(len(unique_cells), 0.4) for c in cols})
    poisson_b["cell_id"] = unique_cells

    out = evaluate_dataset(truth, preds, baseline=poisson_b)
    sample_head = next(iter(out["per_head"].values()))
    assert "bss_vs_poisson" in sample_head
    assert "bss_vs_etas" not in sample_head
