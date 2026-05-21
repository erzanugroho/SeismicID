"""Tests for training + ensemble + calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.features.labels import THRESHOLDS, all_label_columns
from backend.app.ml.calibration import (
    BetaCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    fit_best_calibrator,
)
from backend.app.ml.ensemble import format_top, predict_ensemble
from backend.app.ml.etas import ETASBaseline
from backend.app.ml.train import load_active_models, save_models, train_heads


@pytest.fixture
def synthetic_dataset():
    """Make a small synthetic dataset that has signal: feature `x0` correlates with label."""
    rng = np.random.default_rng(42)
    n = 1500
    feat_cols = ["x0", "x1", "x2", "x3"]
    x = rng.normal(size=(n, len(feat_cols)))
    # True signal: label = 1 if x0 + 0.5*x1 > 1 with noise
    base_score = x[:, 0] + 0.5 * x[:, 1]
    df = pd.DataFrame(x, columns=feat_cols)
    df["cell_id"] = [f"c{i % 20}" for i in range(n)]
    df["snapshot"] = pd.date_range("2018-01-01", periods=n, freq="D").strftime("%Y-%m-%dT%H:%M:%S+00:00")
    for col in all_label_columns():
        # Add noise per head; tighter threshold for higher mag
        thresh = 0.7 + (THRESHOLDS.index(float(col.split("_m")[1]) / 10)) * 0.4
        df[col] = (base_score + rng.normal(0, 0.5, size=n) > thresh).astype(int)
    return df


def test_calibrators_produce_in_range(synthetic_dataset) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200)
    y = (p + rng.normal(0, 0.2, 200) > 0.5).astype(int)
    for cls in (PlattCalibrator, IsotonicCalibrator, BetaCalibrator):
        c = cls()
        c.fit(p, y)
        out = c.predict_proba(p)
        assert np.all((out >= 0) & (out <= 1))


def test_fit_best_calibrator_returns_valid(synthetic_dataset) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 300)
    y = (p > 0.5).astype(int)
    calib, scores = fit_best_calibrator(p, y)
    assert "identity" in scores
    assert callable(calib.predict_proba)


def test_train_and_predict_smoke(synthetic_dataset, tmp_path) -> None:
    """Train heads on tiny data, save, reload, predict."""
    df = synthetic_dataset
    train = df.iloc[:1000]
    val = df.iloc[1000:1200]
    heads = train_heads(train, val, feature_cols=["x0", "x1", "x2", "x3"])
    assert len(heads) >= 1
    # Save + reload
    saved = save_models(heads, version="test_v1", models_dir=tmp_path)
    assert saved.exists()
    loaded, version = load_active_models(models_dir=tmp_path)
    assert version == "test_v1"
    assert set(loaded.keys()) == set(heads.keys())

    # Predict on test split
    test = df.iloc[1200:]
    preds = predict_ensemble(
        loaded,
        test,
        cell_ids=test["cell_id"].tolist(),
    )
    assert len(preds) == len(test)
    # Probabilities in [0, 1]
    for col in all_label_columns():
        if col in preds.columns:
            assert preds[col].between(0, 1).all()


def test_ensemble_format_top() -> None:
    pred = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "label_h30_m50": [0.05, 0.20, 0.10],
        }
    )
    top = format_top(pred, horizon=30, threshold=5.0, n=2)
    assert len(top) == 2
    assert top.iloc[0]["cell_id"] == "b"
    assert top.iloc[0]["probability"] == 0.20


def test_etas_baseline_empty_returns_zero() -> None:
    et = ETASBaseline()
    et.fit(pd.DataFrame(), observation_start=pd.Timestamp("2020-01-01", tz="UTC"), observation_end=pd.Timestamp("2024-01-01", tz="UTC"))
    p = et.predict_probability("c0", 30, 5.0)
    assert p == 0.0


def test_etas_baseline_predicts_higher_for_active_cell() -> None:
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2020-01-01", tz="UTC")
    for i in range(50):
        rows.append({"event_id": f"a{i}", "time": base + pd.Timedelta(days=int(rng.integers(0, 1000))), "cell_id": "c_active", "magnitude": 5.0 + rng.exponential(0.3)})
    for i in range(2):
        rows.append({"event_id": f"q{i}", "time": base + pd.Timedelta(days=int(rng.integers(0, 1000))), "cell_id": "c_quiet", "magnitude": 5.2})
    df = pd.DataFrame(rows)
    et = ETASBaseline()
    et.fit(df, observation_start=base, observation_end=base + pd.Timedelta(days=1000))
    p_active = et.predict_probability("c_active", 30, 5.0)
    p_quiet = et.predict_probability("c_quiet", 30, 5.0)
    assert p_active > p_quiet


def test_evaluate_dataset_calculates_all_metrics(synthetic_dataset) -> None:
    from backend.app.ml.evaluate import evaluate_dataset
    df = synthetic_dataset
    test_df = df.iloc[:100].copy()
    preds_df = df.iloc[:100][["cell_id", "snapshot"]].copy()
    for col in all_label_columns():
        preds_df[col] = np.clip(test_df[col] + np.random.uniform(-0.2, 0.2, len(test_df)), 0.0, 1.0)

    out = evaluate_dataset(test_df, preds_df)
    assert "per_head" in out
    for col in all_label_columns():
        if col in out["per_head"]:
            h_metrics = out["per_head"][col]
            assert "molchan" in h_metrics
            assert "csep" in h_metrics
            assert "n_test" in h_metrics["csep"]
            assert "l_test" in h_metrics["csep"]
            assert "s_test" in h_metrics["csep"]
            assert isinstance(h_metrics["molchan"]["space_time_fraction"], list)
            assert isinstance(h_metrics["molchan"]["miss_rate"], list)
            assert h_metrics["csep"]["n_test"]["status"] in ("pass", "fail")
