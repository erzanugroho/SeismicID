"""Seismological primitives: b-value, seismic energy, IET stats."""

from __future__ import annotations

import math

import numpy as np


def seismic_energy(magnitude: float | np.ndarray) -> float | np.ndarray:
    """Log10 seismic energy (joules) from magnitude. Gutenberg–Richter (1956)."""
    return 1.5 * np.asarray(magnitude) + 4.8


def compute_b_value(magnitudes: np.ndarray, mc: float, *, dm: float = 0.1) -> tuple[float, float]:
    """Aki (1965) maximum-likelihood b-value with Utsu (1965) correction.

    Returns (b, std_err). NaN if not enough events above Mc.
    """
    arr = np.asarray(magnitudes, dtype=float)
    above = arr[arr >= mc]
    n = len(above)
    if n < 5:
        return (float("nan"), float("nan"))
    mean_m = above.mean()
    # Utsu correction
    b = math.log10(math.e) / (mean_m - (mc - dm / 2))
    std_err = b / math.sqrt(n)
    return float(b), float(std_err)


def b_value_slope(times: np.ndarray, magnitudes: np.ndarray, mc: float, window_days: int = 365) -> float:
    """Linear slope of b-value over time (windows of `window_days`).

    Negative slope = b-value declining (stress accumulation hypothesis).
    """
    if len(times) < 50:
        return float("nan")
    t = np.asarray(times, dtype="datetime64[s]").astype("int64")
    # Build sliding windows
    t0, t1 = t.min(), t.max()
    step = window_days * 86400
    if (t1 - t0) < 2 * step:
        return float("nan")
    centers, bs = [], []
    cursor = t0 + step // 2
    while cursor + step // 2 <= t1:
        mask = (t >= cursor - step // 2) & (t <= cursor + step // 2)
        if mask.sum() >= 10:
            b, _ = compute_b_value(magnitudes[mask], mc)
            if not math.isnan(b):
                centers.append(cursor)
                bs.append(b)
        cursor += step
    if len(bs) < 3:
        return float("nan")
    centers_arr = np.asarray(centers, dtype=float)
    bs_arr = np.asarray(bs)
    # Linear regression slope (b vs days)
    days = (centers_arr - centers_arr[0]) / 86400
    slope, _ = np.polyfit(days, bs_arr, 1)
    return float(slope)


def inter_event_times(times: np.ndarray) -> np.ndarray:
    """Time differences between consecutive events in seconds (sorted)."""
    if len(times) < 2:
        return np.array([], dtype=float)
    sorted_t = np.sort(np.asarray(times, dtype="datetime64[s]").astype("int64"))
    return np.diff(sorted_t).astype(float)


def iet_stats(times: np.ndarray) -> tuple[float, float]:
    """(mean, coefficient_of_variation) of inter-event times. NaN if <2 events."""
    iets = inter_event_times(times)
    if len(iets) < 2:
        return (float("nan"), float("nan"))
    mu = float(iets.mean())
    sigma = float(iets.std(ddof=0))
    cv = sigma / mu if mu > 0 else float("nan")
    return (mu, cv)
