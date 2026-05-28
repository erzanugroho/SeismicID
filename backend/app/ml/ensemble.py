"""Ensemble (XGB + LGBM + Poisson baseline) + Bayesian prior blend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.app.core.logging import get_logger
from backend.app.features.labels import (
    HORIZONS,
    THRESHOLDS,
    label_column_name,
)
from backend.app.ml.calibration import IdentityCalibrator
from backend.app.ml.train import HeadModel

logger = get_logger(__name__)

# Hard probability floor for numerical stability. Anything <= this value is
# treated as "data minim / floor only" for downstream UIs.
PROB_FLOOR: float = 1e-6


@dataclass
class EnsembleConfig:
    weight_xgb: float = 0.4
    weight_lgbm: float = 0.4
    weight_poisson: float = 0.2
    bayesian_alpha: float = 5.0  # pseudo-count toward prior (Poisson rate)


def _safe_proba(clf, x: np.ndarray) -> np.ndarray:
    if clf is None:
        return np.full(len(x), 0.0, dtype=np.float64)
    p = clf.predict_proba(x)
    if p.shape[1] == 1:
        return p[:, 0]
    return p[:, 1]


def predict_ensemble(
    heads: dict[str, HeadModel],
    features: pd.DataFrame,
    *,
    cell_ids: list[str] | None = None,
    snapshots: list | pd.Series | None = None,
    poisson_predictions: pd.DataFrame | None = None,
    cell_event_counts: dict[str, int] | None = None,
    config: EnsembleConfig | None = None,
    base_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run ensemble per head + Bayesian blend.

    Returns DataFrame with cell_id (+ optional snapshot) + 16 calibrated
    probability columns. The optional ``snapshots`` argument lets callers
    propagate the per-row snapshot timestamp so downstream evaluation can
    merge on both ``(cell_id, snapshot)`` instead of ``cell_id`` alone —
    which would otherwise cartesian-explode when the input has duplicate
    cell_ids across snapshots.
    """
    cfg = config or EnsembleConfig()
    cell_ids = cell_ids or features.get("cell_id", pd.Series(dtype=str)).tolist()
    if snapshots is None and "snapshot" in features.columns:
        snapshots = features["snapshot"].tolist()
    feat_cols = next(iter(heads.values())).feature_names if heads else []
    x = (
        features[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
        if feat_cols
        else np.zeros((len(cell_ids), 0), dtype=np.float32)
    )

    # One-time visibility into the calibrator mix for this prediction batch.
    # Helps explain post-hoc compression behaviour: if every head logs
    # ``IsotonicCalibrator`` we should never see ``posthoc_recalibration_applied``,
    # and vice versa. Useful when chasing under-prediction bugs.
    calibrator_counts: dict[str, int] = {}
    for hm in heads.values():
        name = hm.calibrator.__class__.__name__
        calibrator_counts[name] = calibrator_counts.get(name, 0) + 1
    logger.debug("ensemble_calibrator_distribution", **calibrator_counts)

    # Normalised evidence vector. ``cell_event_counts`` is intentionally
    # treated symmetrically: ``None`` and an empty dict both mean "no per-cell
    # information available" and trigger a moderate default. A non-empty dict
    # with missing keys means we know about the catalog but this cell wasn't
    # represented; we still allow at least one effective sample so the
    # posterior never collapses to zero just because the prior is zero.
    n_cells = len(cell_ids)
    if not cell_event_counts:
        n_evidence_default = 10.0
        n_evidence = np.full(n_cells, n_evidence_default, dtype=np.float64)
    else:
        n_evidence = np.array(
            [max(float(cell_event_counts.get(cid, 0)), 1.0) for cid in cell_ids],
            dtype=np.float64,
        )

    # Lazy-loaded fallback base rates for posthoc compression. Only computed
    # the first time a head with IdentityCalibrator is encountered, so we
    # never pay the cost when proper calibrators are in place.
    _base_rates_loaded: dict[str, float] | None = base_rates

    out = pd.DataFrame({"cell_id": cell_ids})
    if snapshots is not None:
        out["snapshot"] = pd.Series(list(snapshots)).values
    posthoc_count = 0
    for head, hm in heads.items():
        p_xgb = _safe_proba(hm.booster_xgb, x)
        p_lgbm = _safe_proba(hm.booster_lgbm, x)
        if poisson_predictions is not None and head in poisson_predictions.columns:
            poisson_aligned = (
                poisson_predictions.set_index("cell_id")
                .reindex(cell_ids)[head]
                .fillna(0.0)
                .to_numpy()
            )
        else:
            poisson_aligned = np.zeros(n_cells)

        w_sum = cfg.weight_xgb + cfg.weight_lgbm + cfg.weight_poisson
        ens = (
            cfg.weight_xgb * p_xgb
            + cfg.weight_lgbm * p_lgbm
            + cfg.weight_poisson * poisson_aligned
        ) / w_sum

        # Calibrate ensemble.
        try:
            calibrated = hm.calibrator.predict_proba(ens)
        except Exception:  # noqa: BLE001
            calibrated = ens

        # Per-head posthoc compression. Apply ONLY when this specific head's
        # calibrator is the IdentityCalibrator (no proper calibration). A
        # single Identity sibling must not drag previously-calibrated heads
        # through the compression: the audit explicitly flagged that.
        if isinstance(hm.calibrator, IdentityCalibrator):
            if _base_rates_loaded is None:
                from backend.app.ml.posthoc_calibration import _fallback_base_rates

                _base_rates_loaded = _fallback_base_rates()
            from backend.app.ml.posthoc_calibration import recalibrate_head as _recal_head

            br = _base_rates_loaded.get(head, 0.005)
            calibrated = _recal_head(calibrated, br)
            posthoc_count += 1

        # Bayesian blend with the Poisson baseline as prior, but only where
        # the prior is actually informative (positive and finite). When the
        # prior is missing or zero, falling back to the calibrated estimate
        # preserves whatever signal the ML stack provides instead of forcing
        # the cell down to the floor.
        prior = poisson_aligned
        prior_valid = (prior > 0.0) & np.isfinite(prior)
        alpha = cfg.bayesian_alpha
        denom = n_evidence + alpha
        blended = np.where(
            prior_valid,
            (n_evidence * calibrated + alpha * prior) / denom,
            calibrated,
        )
        out[head] = np.clip(blended, PROB_FLOOR, 1 - PROB_FLOOR)

    if posthoc_count > 0:
        logger.info(
            "posthoc_recalibration_applied",
            n_heads=posthoc_count,
            total_heads=len(heads),
        )

    return out


def enforce_probability_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce monotonic probability constraints on a multi-output forecast frame.

    For every cell row we require:
        * ``P(longer horizon, fixed threshold) ≥ P(shorter horizon, fixed threshold)``
        * ``P(lower threshold, fixed horizon) ≥ P(higher threshold, fixed horizon)``

    The fix is rank-preserving on each axis: we run a cumulative max along
    horizons (ascending) per threshold, then a reverse cumulative max along
    thresholds per horizon. Because the second pass takes a max over a
    superset that already satisfies horizon-monotonicity, both invariants
    hold simultaneously after the second pass.
    """
    if df.empty:
        return df

    out = df.copy()

    horizons_sorted = sorted(HORIZONS)
    thresholds_sorted = sorted(THRESHOLDS)

    # Pass 1: enforce horizon monotonicity per threshold.
    for t in thresholds_sorted:
        cols = [label_column_name(h, t) for h in horizons_sorted]
        present = [c for c in cols if c in out.columns]
        if len(present) >= 2:
            arr = out[present].to_numpy(dtype=np.float64)
            arr = np.maximum.accumulate(arr, axis=1)
            for i, c in enumerate(present):
                out[c] = arr[:, i]

    # Pass 2: enforce threshold monotonicity per horizon (lower t ≥ higher t).
    for h in horizons_sorted:
        cols = [label_column_name(h, t) for t in thresholds_sorted]
        present = [c for c in cols if c in out.columns]
        if len(present) >= 2:
            arr = out[present].to_numpy(dtype=np.float64)
            arr = arr[:, ::-1]
            arr = np.maximum.accumulate(arr, axis=1)
            arr = arr[:, ::-1]
            for i, c in enumerate(present):
                out[c] = arr[:, i]

    return out


def format_top(df: pd.DataFrame, *, horizon: int, threshold: float, n: int = 10) -> pd.DataFrame:
    col = label_column_name(horizon, threshold)
    if col not in df.columns:
        return pd.DataFrame()
    top = df.nlargest(n, col)[["cell_id", col]].rename(columns={col: "probability"})
    top["horizon_days"] = horizon
    top["mag_threshold"] = threshold
    return top.reset_index(drop=True)
