"""Aki-Utsu Gutenberg-Richter b-value estimator (Phase 3 Task 3.1).

The b-value characterizes the slope of the magnitude-frequency distribution:
    log10(N(M >= m)) = a - b * (m - mc)
For Indonesia, b is regionally close to 1.0 but varies between subduction
segments. Per-region b unlocks more accurate threshold scaling in
``OgataETAS.predict_dataframe``.

Reference: Aki (1965), Utsu (1965). Maximum-likelihood estimator for the
exponential tail above the completeness magnitude:
    b_hat = log10(e) / (mean(M) - mc)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# Below this many events the MLE is unstable and we fall back to the prior.
MIN_EVENTS = 50


def estimate_b_aki_utsu(
    magnitudes: Iterable[float] | np.ndarray,
    *,
    mc: float,
    default: float = 1.0,
) -> float:
    """Estimate the Gutenberg-Richter b-value via Aki-Utsu MLE.

    Parameters
    ----------
    magnitudes:
        Earthquake magnitudes. Values strictly below ``mc`` are dropped to
        respect the completeness assumption.
    mc:
        Completeness magnitude (estimated separately, e.g. via MAXC).
    default:
        Returned when the catalog above ``mc`` is too small or degenerate.

    Returns
    -------
    float
        The estimated b-value, or ``default`` if the input is insufficient.
    """
    mags = np.asarray(magnitudes, dtype=np.float64).ravel()
    mags = mags[~np.isnan(mags)]
    mags = mags[mags >= mc]
    if mags.size < MIN_EVENTS:
        return float(default)
    mean_m = float(mags.mean())
    if mean_m <= mc:
        return float(default)
    return float(np.log10(np.e) / (mean_m - mc))
