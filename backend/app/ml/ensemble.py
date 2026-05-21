"""Ensemble (XGB + LGBM + ETAS) + Bayesian prior blend + post-hoc recalibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.app.core.logging import get_logger
from backend.app.features.labels import label_column_name
from backend.app.ml.calibration import IdentityCalibrator
from backend.app.ml.train import HeadModel

logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    weight_xgb: float = 0.4
    weight_lgbm: float = 0.4
    weight_etas: float = 0.2
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
    etas_predictions: pd.DataFrame | None = None,
    cell_event_counts: dict[str, int] | None = None,
    config: EnsembleConfig | None = None,
    base_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run ensemble per head + Bayesian blend.

    Returns DataFrame with cell_id + 16 calibrated probability columns.
    """
    cfg = config or EnsembleConfig()
    cell_ids = cell_ids or features.get("cell_id", pd.Series(dtype=str)).tolist()
    feat_cols = next(iter(heads.values())).feature_names if heads else []
    x = features[feat_cols].fillna(0.0).to_numpy(dtype=np.float32) if feat_cols else np.zeros((len(cell_ids), 0), dtype=np.float32)

    # Pre-compute post-hoc recalibration if needed
    needs_posthoc = any(
        isinstance(hm.calibrator, IdentityCalibrator) for hm in heads.values()
    )
    if needs_posthoc:
        if base_rates is None:
            from backend.app.ml.posthoc_calibration import _fallback_base_rates
            base_rates = _fallback_base_rates()
        from backend.app.ml.posthoc_calibration import recalibrate_head as _recal_head

    out = pd.DataFrame({"cell_id": cell_ids})
    for head, hm in heads.items():
        p_xgb = _safe_proba(hm.booster_xgb, x)
        p_lgbm = _safe_proba(hm.booster_lgbm, x)
        if etas_predictions is not None and head in etas_predictions.columns:
            etas_aligned = etas_predictions.set_index("cell_id").reindex(cell_ids)[head].fillna(0.0).to_numpy()
        else:
            etas_aligned = np.zeros(len(cell_ids))

        w_sum = cfg.weight_xgb + cfg.weight_lgbm + cfg.weight_etas
        ens = (cfg.weight_xgb * p_xgb + cfg.weight_lgbm * p_lgbm + cfg.weight_etas * etas_aligned) / w_sum

        # Calibrate ensemble
        try:
            calibrated = hm.calibrator.predict_proba(ens)
        except Exception:
            calibrated = ens

        # Post-hoc recalibration BEFORE Bayesian blend
        # This compresses inflated ML outputs to realistic base rates
        if needs_posthoc and isinstance(hm.calibrator, IdentityCalibrator):
            br = base_rates.get(head, 0.005) if base_rates else 0.005
            calibrated = _recal_head(calibrated, br)

        # Bayesian blend with ETAS as prior, weighted by evidence
        # Use lower default (10) when event counts unknown → more regularization toward prior
        n_evidence = (
            np.array([cell_event_counts.get(cid, 0) for cid in cell_ids], dtype=np.float64)
            if cell_event_counts else np.full(len(cell_ids), 10.0)
        )
        alpha = cfg.bayesian_alpha
        prior = etas_aligned
        posterior = (n_evidence * calibrated + alpha * prior) / (n_evidence + alpha)
        out[head] = np.clip(posterior, 1e-6, 1 - 1e-6)

    if needs_posthoc:
        logger.info("posthoc_recalibration_applied", n_heads=len(heads))

    return out


def format_top(df: pd.DataFrame, *, horizon: int, threshold: float, n: int = 10) -> pd.DataFrame:
    col = label_column_name(horizon, threshold)
    if col not in df.columns:
        return pd.DataFrame()
    top = df.nlargest(n, col)[["cell_id", col]].rename(columns={col: "probability"})
    top["horizon_days"] = horizon
    top["mag_threshold"] = threshold
    return top.reset_index(drop=True)
