"""Model evaluation: ROC, Brier, skill score, reliability, CSEP-style."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from backend.app.features.labels import all_label_columns


def evaluate_head(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray | None = None) -> dict:
    """Per-head metrics. baseline = Poisson predictions (constant rate per cell)."""
    metrics: dict = {}
    if len(np.unique(y_true)) < 2:
        return {"brier": float(brier_score_loss(y_true, y_pred)) if len(y_true) else float("nan")}
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
    metrics["brier"] = float(brier_score_loss(y_true, y_pred))
    if baseline is not None and len(baseline) == len(y_true):
        b_base = brier_score_loss(y_true, baseline)
        metrics["bss_vs_poisson"] = float(1 - metrics["brier"] / max(b_base, 1e-9))
    return metrics


def reliability_diagram(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> dict:
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(y_pred, edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)
    out: dict[str, list[float]] = {"bin_centers": [], "predicted_mean": [], "observed_freq": [], "count": []}
    for i in range(n_bins):
        mask = bins == i
        if mask.sum() == 0:
            continue
        out["bin_centers"].append(float((edges[i] + edges[i + 1]) / 2))
        out["predicted_mean"].append(float(y_pred[mask].mean()))
        out["observed_freq"].append(float(y_true[mask].mean()))
        out["count"].append(int(mask.sum()))
    return out


def roc_points(y_true: np.ndarray, y_pred: np.ndarray, n_points: int = 50) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "auc": float("nan")}
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    if len(fpr) > n_points:
        idx = np.linspace(0, len(fpr) - 1, n_points).astype(int)
        fpr = fpr[idx]
        tpr = tpr[idx]
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_score(y_true, y_pred))}


def molchan_points(y_true: np.ndarray, y_pred: np.ndarray, n_points: int = 50) -> dict:
    total_events = int(y_true.sum())
    if total_events == 0:
        return {"space_time_fraction": [0.0, 1.0], "miss_rate": [1.0, 0.0]}

    thresholds = np.unique(y_pred)
    if len(thresholds) > n_points:
        thresholds = np.percentile(y_pred, np.linspace(0, 100, n_points))
        thresholds = np.unique(thresholds)
    thresholds = sorted(thresholds) + [1.01]

    tau_list = []
    nu_list = []

    n_total = len(y_pred)
    for th in thresholds:
        alarms = y_pred >= th
        tau = float(alarms.sum() / n_total)
        hits = (alarms & (y_true == 1)).sum()
        nu = float(1.0 - hits / total_events)
        tau_list.append(tau)
        nu_list.append(nu)

    sorted_idx = np.argsort(tau_list)
    return {
        "space_time_fraction": [tau_list[i] for i in sorted_idx],
        "miss_rate": [nu_list[i] for i in sorted_idx]
    }


def run_n_test(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n_obs = int(y_true.sum())
    n_pred = float(y_pred.sum())
    if n_pred <= 0:
        return {
            "observed": n_obs,
            "predicted": n_pred,
            "p_value_low": 0.0,
            "p_value_high": 0.0,
            "status": "fail"
        }
    p_low = float(poisson.cdf(n_obs, n_pred))
    p_high = float(1.0 - poisson.cdf(n_obs - 1, n_pred)) if n_obs > 0 else 1.0
    status = "pass" if (p_low >= 0.025 and p_high >= 0.025) else "fail"
    return {
        "observed": n_obs,
        "predicted": n_pred,
        "p_value_low": p_low,
        "p_value_high": p_high,
        "status": status
    }


def run_l_test(y_true: np.ndarray, y_pred: np.ndarray, n_sim: int = 1000) -> dict:
    p_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    log_p = np.log(p_clipped)
    log_1_p = np.log(1 - p_clipped)
    obs_ll = float(np.sum(y_true * log_p + (1 - y_true) * log_1_p))

    rng = np.random.default_rng(42)
    n = len(y_pred)
    # Memory-aware adaptive simulation. The naive (n_sim, n) allocation
    # blows past tens of GiB once ``n`` is on the order of a few hundred
    # thousand. Chunk the simulations so peak memory stays bounded
    # regardless of input size, while keeping the same statistical sample.
    if n == 0:
        return {
            "observed_log_likelihood": obs_ll,
            "mean_sim_log_likelihood": 0.0,
            "quantile": 1.0,
            "status": "pass",
        }
    # Aim for ~64 MiB per chunk: rows = 64 MiB / (n * 8 B) but at least 1.
    target_bytes = 64 * 1024 * 1024
    rows_per_chunk = max(1, min(n_sim, target_bytes // max(1, n * 8)))
    sim_lls = np.empty(n_sim, dtype=np.float64)
    written = 0
    while written < n_sim:
        rows = min(rows_per_chunk, n_sim - written)
        sim_block = rng.random((rows, n)) < y_pred
        sim_lls[written : written + rows] = np.sum(
            sim_block * log_p + (1 - sim_block) * log_1_p, axis=1
        )
        written += rows

    quantile = float(np.mean(sim_lls <= obs_ll))
    # Two-sided pass band: the observed log-likelihood must sit inside the
    # central 95% of the simulated distribution. The previous one-sided
    # ``quantile >= 0.025`` accepted any "too good to be true" extreme.
    status = "pass" if 0.025 <= quantile <= 0.975 else "fail"
    return {
        "observed_log_likelihood": obs_ll,
        "mean_sim_log_likelihood": float(np.mean(sim_lls)),
        "quantile": quantile,
        "status": status,
    }


def run_s_test(y_true: np.ndarray, y_pred: np.ndarray, n_sim: int = 1000) -> dict:
    n_obs = int(y_true.sum())
    if n_obs == 0:
        return {
            "observed_spatial_log_likelihood": 0.0,
            "mean_sim_spatial_log_likelihood": 0.0,
            "quantile": 1.0,
            "status": "pass"
        }

    sum_p = y_pred.sum()
    if sum_p <= 0:
        return {
            "observed_spatial_log_likelihood": -9999.0,
            "mean_sim_spatial_log_likelihood": 0.0,
            "quantile": 0.0,
            "status": "fail"
        }

    w = y_pred / sum_p
    w_clipped = np.clip(w, 1e-9, 1.0)
    log_w = np.log(w_clipped)

    obs_s_ll = float(np.sum(y_true * log_w))

    rng = np.random.default_rng(42)
    sim_indices = rng.choice(len(y_pred), size=(n_sim, n_obs), p=w)
    sim_s_lls = np.sum(log_w[sim_indices], axis=1)

    quantile = float(np.mean(sim_s_lls <= obs_s_ll))
    # Two-sided pass band (mirrors the L-test fix). One-sided was treating
    # implausibly good fits as ``pass``.
    status = "pass" if 0.025 <= quantile <= 0.975 else "fail"
    return {
        "observed_spatial_log_likelihood": obs_s_ll,
        "mean_sim_spatial_log_likelihood": float(np.mean(sim_s_lls)),
        "quantile": quantile,
        "status": status,
    }


def csep_tests(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n_test": run_n_test(y_true, y_pred),
        "l_test": run_l_test(y_true, y_pred),
        "s_test": run_s_test(y_true, y_pred)
    }


def evaluate_dataset(
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    baseline: pd.DataFrame | None = None,
) -> dict:
    """Evaluate per-head against test set.

    Merges on ``(cell_id, snapshot)`` when both columns are present in
    ``predictions``; otherwise falls back to ``cell_id`` only. The dual-key
    merge prevents a Cartesian explosion when the test set has the same
    ``cell_id`` repeated across many snapshots and ``predictions`` does too
    — the original ``cell_id``-only merge could blow up to hundreds of
    millions of rows and trigger a multi-hundred-GiB allocation downstream.
    """
    out: dict = {"per_head": {}}
    label_cols = all_label_columns()
    use_snapshot = "snapshot" in predictions.columns and "snapshot" in test.columns
    merge_keys = ["cell_id", "snapshot"] if use_snapshot else ["cell_id"]

    for head in label_cols:
        if head not in test.columns or head not in predictions.columns:
            continue
        left = test[merge_keys + [head]]
        right_cols = merge_keys + [head]
        right = predictions[right_cols].rename(columns={head: f"{head}_pred"})
        merged = left.merge(right, on=merge_keys, how="left").fillna(
            {f"{head}_pred": 0.5}
        )
        y = merged[head].to_numpy()
        p = merged[f"{head}_pred"].to_numpy()
        b = (
            baseline.set_index("cell_id").reindex(merged["cell_id"])[head].fillna(0.05).to_numpy()
            if baseline is not None and head in baseline.columns
            else None
        )
        out["per_head"][head] = {
            **evaluate_head(y, p, baseline=b),
            "reliability": reliability_diagram(y, p),
            "roc": roc_points(y, p),
            "molchan": molchan_points(y, p),
            "csep": csep_tests(y, p),
        }
    return out
