"""Tests for information-gain (bits/event) metric (Task 3.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.features.labels import all_label_columns


def test_information_gain_positive_when_model_beats_baseline() -> None:
    from backend.app.ml.evaluate import evaluate_dataset

    cols = all_label_columns()
    cells = [f"C+0{i}_p{i:03d}" for i in range(8)]
    rng = np.random.default_rng(0)
    truth = pd.DataFrame(
        {c: rng.integers(0, 2, size=len(cells)).astype(int) for c in cols}
    )
    truth["cell_id"] = cells
    truth["snapshot"] = pd.to_datetime(["2024-01-01"] * len(cells), utc=True)
    # Model is well calibrated to truth, baseline is flat 0.5.
    preds = truth.copy()
    for c in cols:
        preds[c] = (truth[c] * 0.7 + 0.15).astype(float)
    baseline = pd.DataFrame({c: np.full(len(cells), 0.5) for c in cols})
    baseline["cell_id"] = cells

    out = evaluate_dataset(truth, preds, baseline=baseline)
    sample = next(iter(out["per_head"].values()))
    assert "info_gain_vs_poisson" in sample
    assert sample["info_gain_vs_poisson"] > 0


def test_information_gain_includes_etas_when_dual_baseline() -> None:
    from backend.app.ml.evaluate import evaluate_dataset

    cols = all_label_columns()
    cells = [f"C+0{i}_p{i:03d}" for i in range(8)]
    rng = np.random.default_rng(1)
    truth = pd.DataFrame(
        {c: rng.integers(0, 2, size=len(cells)).astype(int) for c in cols}
    )
    truth["cell_id"] = cells
    truth["snapshot"] = pd.to_datetime(["2024-01-01"] * len(cells), utc=True)
    preds = truth.copy()
    for c in cols:
        preds[c] = (truth[c] * 0.6 + 0.2).astype(float)
    poisson_b = pd.DataFrame({c: np.full(len(cells), 0.5) for c in cols})
    poisson_b["cell_id"] = cells
    etas_b = pd.DataFrame({c: np.full(len(cells), 0.4) for c in cols})
    etas_b["cell_id"] = cells

    out = evaluate_dataset(truth, preds, baseline=poisson_b, baseline_etas=etas_b)
    sample = next(iter(out["per_head"].values()))
    assert "info_gain_vs_poisson" in sample
    assert "info_gain_vs_etas" in sample
