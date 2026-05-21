"""Post-hoc probability recalibration for uncalibrated ensemble outputs.

When the training pipeline produces IdentityCalibrator (no proper calibration),
raw XGBoost/LightGBM outputs are systematically inflated due to scale_pos_weight.

This module applies **rank-preserving rescaling** so that:
1. The median output matches the empirical base rate from historical data.
2. The ranking between cells is perfectly preserved (Spearman rho = 1.0).
3. Output stays in valid probability range [epsilon, 1 - epsilon].

Two strategies:
- `EmpiricalBaseRateCalibrator`: Uses historical event data to compute per-head
  base rates, then applies logit-space affine rescaling.
- `QuantileRescaler`: Non-parametric — maps raw output quantiles to a target
  distribution derived from Poisson base rates.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backend.app.core.logging import get_logger
from backend.app.features.labels import HORIZONS, THRESHOLDS, label_column_name

logger = get_logger(__name__)

# Epsilon for numerical stability
_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_base_rates(
    events: pd.DataFrame,
    n_cells: int,
    *,
    observation_days: float | None = None,
) -> dict[str, float]:
    """Compute empirical per-cell base rate P(≥1 event | horizon, threshold).

    Uses Poisson assumption: P = 1 - exp(-λ·horizon)
    where λ = total_events / (n_cells · observation_days).

    Returns dict mapping label column name to base rate.
    """
    if events.empty or n_cells == 0:
        # Fallback to conservative USGS-style global rates
        return _fallback_base_rates()

    df = events.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    tmin = df["time"].min()
    tmax = df["time"].max()
    obs_days = observation_days or max(1.0, (tmax - tmin).total_seconds() / 86400)

    rates: dict[str, float] = {}
    for h in HORIZONS:
        for t in THRESHOLDS:
            col = label_column_name(h, t)
            n_events = int((df["magnitude"] >= t).sum())
            # Per-cell daily rate
            lam = n_events / (n_cells * obs_days) if obs_days > 0 else 0.0
            # Poisson probability of ≥1 event in horizon
            p = 1.0 - math.exp(-lam * h)
            rates[col] = max(p, _EPS)
    return rates


def _fallback_base_rates() -> dict[str, float]:
    """Conservative global base rates when no data available.

    Based on approximate Indonesian seismicity:
    ~15 M≥5.0 events/month across ~3000 cells → per-cell 30-day rate ≈ 0.5%.
    Scaled by Gutenberg-Richter for other thresholds.
    """
    rates: dict[str, float] = {}
    for h in HORIZONS:
        for t in THRESHOLDS:
            col = label_column_name(h, t)
            # Annual rate per cell, Gutenberg-Richter scaling
            annual_rate = 0.06 * math.exp(-(t - 4.5) / 0.43)  # ~6% for M≥4.5/year/cell
            daily_rate = annual_rate / 365.0
            p = 1.0 - math.exp(-daily_rate * h)
            rates[col] = max(p, _EPS)
    return rates


def recalibrate_head(
    raw_probs: np.ndarray,
    base_rate: float,
) -> np.ndarray:
    """Rank-preserving rescaling of raw probabilities to match target base rate.

    Uses logit-space affine transform:
        calibrated = sigmoid(a * logit(raw) + b)
    where a and b are chosen so that:
        median(calibrated) ≈ base_rate

    The slope `a` controls spread — we use a < 1 to compress the inflated range.
    The intercept `b` shifts the center to match the base rate.

    Parameters
    ----------
    raw_probs : array of raw (inflated) probabilities
    base_rate : target median probability (from empirical data)

    Returns
    -------
    calibrated : array of rescaled probabilities
    """
    if len(raw_probs) == 0:
        return raw_probs

    raw = np.clip(raw_probs, _EPS, 1 - _EPS)
    logits = _logit(raw)

    raw_median = np.median(raw)
    raw_median_logit = _logit(np.array([raw_median]))[0]
    target_logit = _logit(np.array([max(base_rate, _EPS)]))[0]

    # Compute compression factor
    # We want to map [raw_min_logit, raw_max_logit] → a narrower range
    # centered on target_logit
    logit_range = logits.max() - logits.min()
    if logit_range < _EPS:
        return np.full_like(raw, base_rate)

    # Target range: base_rate should be median, max should be ~10× base_rate
    # but capped at reasonable values
    target_max = min(base_rate * 15, 0.50)  # Cap at 50%
    target_max_logit = _logit(np.array([target_max]))[0]
    target_range = (target_max_logit - target_logit) * 2  # symmetric around median

    # Compression slope
    a = min(target_range / logit_range, 1.0)  # Never expand, only compress
    a = max(a, 0.05)  # Minimum slope to maintain some differentiation

    # Shift: center compressed logits on target median
    b = target_logit - a * raw_median_logit

    calibrated_logits = a * logits + b
    calibrated = _sigmoid(calibrated_logits)

    return np.clip(calibrated, _EPS, 1 - _EPS)


def recalibrate_predictions(
    predictions: pd.DataFrame,
    base_rates: dict[str, float],
) -> pd.DataFrame:
    """Apply post-hoc recalibration to all probability columns in predictions.

    Parameters
    ----------
    predictions : DataFrame with cell_id + label columns (label_h*_m*)
    base_rates : dict mapping label column name to empirical base rate

    Returns
    -------
    DataFrame with recalibrated probabilities (same structure, same ranking)
    """
    out = predictions.copy()
    label_cols = [c for c in predictions.columns if c.startswith("label_h")]

    for col in label_cols:
        if col not in out.columns:
            continue
        raw = out[col].to_numpy(dtype=np.float64)
        br = base_rates.get(col, 0.005)  # Default 0.5% if unknown

        calibrated = recalibrate_head(raw, br)
        out[col] = calibrated

        logger.debug(
            "posthoc_recalibrated",
            head=col,
            base_rate=f"{br:.4f}",
            raw_median=f"{np.median(raw):.4f}",
            cal_median=f"{np.median(calibrated):.4f}",
            cal_max=f"{calibrated.max():.4f}",
        )

    return out
