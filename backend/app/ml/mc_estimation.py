"""Magnitude of completeness (Mc) estimation.

MAXC method (Wiemer & Wyss 2000): Mc = magnitude bin with maximum frequency
in the non-cumulative frequency-magnitude distribution. Simple and robust
enough for catalog filtering before ETAS fitting. For publication-grade Mc,
prefer Lilliefors / GFT — but those require more events than typical regional
cells in the Indonesian catalog.

Used by `OgataETAS` (in etas_ogata.py) to filter sub-completeness events
before MLE; reported in MODEL_CARD per region for transparency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

MIN_EVENTS_FOR_MC = 50


def estimate_mc_maxc(magnitudes: np.ndarray, *, bin_width: float = 0.1) -> float:
    """Return Mc estimate via MAXC. NaN if sample too small.

    Parameters
    ----------
    magnitudes : array of event magnitudes (any unit, typically Mw).
    bin_width  : bin size for the histogram (default 0.1 mag units).

    Notes
    -----
    Mc is taken as the LEFT edge of the most populated bin. This is the
    standard MAXC convention and is conservative (slightly over-estimates Mc
    rather than under-estimates, which protects downstream rate calculations
    from sub-completeness contamination).
    """
    mags = np.asarray(magnitudes, dtype=np.float64)
    mags = mags[np.isfinite(mags)]
    if mags.size < MIN_EVENTS_FOR_MC:
        return float("nan")
    lo = float(np.floor(mags.min() * 10) / 10)
    hi = float(np.ceil(mags.max() * 10) / 10)
    bins = np.arange(lo, hi + bin_width, bin_width)
    if bins.size < 2:
        return float("nan")
    counts, edges = np.histogram(mags, bins=bins)
    if counts.sum() == 0:
        return float("nan")
    peak_idx = int(np.argmax(counts))
    return float(edges[peak_idx])


def estimate_mc_per_region(
    events: "pd.DataFrame",
    *,
    region_col: str = "cell_id",
    bin_width: float = 0.1,
    min_events: int = MIN_EVENTS_FOR_MC,
) -> dict[str, float]:
    """Per-region Mc dict; cells with fewer than ``min_events`` get NaN.

    Useful when feeding ETAS Ogata: each spatial region might have a different
    completeness threshold due to station coverage, and using a single global
    Mc inflates productivity estimates in well-instrumented regions.
    """
    out: dict[str, float] = {}
    if events.empty or region_col not in events.columns:
        return out
    for region, sub in events.groupby(region_col):
        if len(sub) < min_events:
            out[str(region)] = float("nan")
            continue
        out[str(region)] = estimate_mc_maxc(
            sub["magnitude"].to_numpy(), bin_width=bin_width
        )
    return out
