"""Smoke: train_initial evaluation helper wires ETAS-Ogata baseline.

We don't run the whole training pipeline here — that's expensive. Instead we
exercise the extracted ``_evaluate_with_dual_baseline`` helper directly with
synthetic events/test/preds to confirm the skill payload now carries
``bss_vs_etas`` per head.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from backend.app.features.labels import all_label_columns


def _toy_events(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    times = [base + timedelta(days=float(d)) for d in np.sort(rng.uniform(0, 365, size=n))]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(times, utc=True),
            "lat": rng.uniform(-8, 6, size=n),
            "lon": rng.uniform(95, 141, size=n),
            "magnitude": rng.uniform(4.5, 6.0, size=n),
            "depth": rng.uniform(5, 80, size=n),
        }
    )


def _toy_truth_and_pred() -> tuple[pd.DataFrame, pd.DataFrame]:
    from backend.app.core.grid import generate_grid

    cols = all_label_columns()
    cells = [c.cell_id for c in generate_grid()[:4]]
    rng = np.random.default_rng(0)
    truth = pd.DataFrame(
        {c: rng.integers(0, 2, size=len(cells)).astype(int) for c in cols}
    )
    truth["cell_id"] = cells
    truth["snapshot"] = pd.to_datetime(["2020-12-31"] * len(cells), utc=True)
    preds = truth.copy()
    for c in cols:
        preds[c] = (truth[c] * 0.55 + 0.2).astype(float)
    return truth, preds


def test_evaluate_helper_emits_bss_vs_etas() -> None:
    from scripts.train_initial import _evaluate_with_dual_baseline

    events = _toy_events()
    truth, preds = _toy_truth_and_pred()
    cols = all_label_columns()
    cells = truth["cell_id"].tolist()
    poisson_b = pd.DataFrame({c: np.full(len(cells), 0.4) for c in cols})
    poisson_b["cell_id"] = cells

    eval_out = _evaluate_with_dual_baseline(
        test=truth,
        preds=preds,
        train_events=events,
        baseline_for_eval=poisson_b,
        obs_start=events["time"].min().to_pydatetime(),
        obs_end=events["time"].max().to_pydatetime(),
    )
    skill = eval_out["skill_payload"]
    assert skill, "skill payload must not be empty"
    sample = next(iter(skill.values()))
    assert "bss_vs_etas" in sample
    assert "bss_vs_poisson" in sample
