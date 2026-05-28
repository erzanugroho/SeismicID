"""True ETAS (Ogata 1988) — temporal model.

Conditional intensity:
    lambda(t | H_t) = mu + sum_{t_i < t} K * exp(alpha * (M_i - Mc))
                                            * (t - t_i + c) ** (-p)

Log-likelihood (Ogata 1988, eq. 6):
    log L = sum_i log lambda(t_i | H_{t_i})  -  integral_0^T lambda(t) dt

The integral term has a closed form per past event:
    int_{t_i}^{T_end} K * G_i * (t - t_i + c)**(-p) dt
        = K * G_i * [ (T_end - t_i + c)**(1-p) - c**(1-p) ] / (1 - p)   (p != 1)
        = K * G_i * log((T_end - t_i + c) / c)                         (p == 1)
where G_i = exp(alpha * (M_i - Mc)).

The spatial extension lives in the same module (added in Task 1.3) so the
class API can grow without breaking the existing temporal-only callers.
We deliberately keep this module separate from `etas.py` (PoissonBaseline)
to preserve the documented compatibility alias `ETASBaseline = PoissonBaseline`
during the deprecation cycle.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _triggered_intensity(
    t_query: float,
    times: np.ndarray,
    mags: np.ndarray,
    *,
    K: float,
    c: float,
    p: float,
    alpha: float,
    mc: float,
) -> float:
    """Sum of Omori-modified-decay contributions from all events before t_query."""
    mask = times < t_query
    if not np.any(mask):
        return 0.0
    dt = t_query - times[mask] + c
    productivity = np.exp(alpha * (mags[mask] - mc))
    return float(K * np.sum(productivity * dt ** (-p)))


def ogata_loglik(
    times: np.ndarray,
    mags: np.ndarray,
    *,
    T_end: float,
    mc: float,
    mu: float,
    K: float,
    c: float,
    p: float,
    alpha: float,
) -> float:
    """Return log-likelihood under temporal Ogata ETAS.

    Time in days since catalog start, magnitude in Mw (or compatible scale),
    Mc the completeness threshold used to filter the catalog upstream.
    """
    times = np.asarray(times, dtype=np.float64)
    mags = np.asarray(mags, dtype=np.float64)
    if times.size == 0:
        # Empty-catalog likelihood: only the background integral term remains.
        return float(-mu * T_end)

    # First sum: log lambda at each event time (using strict-prior history).
    lam_at = np.empty(times.size, dtype=np.float64)
    for i, ti in enumerate(times):
        triggered = _triggered_intensity(
            ti, times, mags, K=K, c=c, p=p, alpha=alpha, mc=mc
        )
        lam_at[i] = mu + triggered
    log_term = float(np.sum(np.log(np.maximum(lam_at, EPS))))

    # Integral term: closed-form analytic solution per past event + background.
    G = np.exp(alpha * (mags - mc))
    if abs(p - 1.0) < 1e-9:
        integ_each = K * G * np.log((T_end - times + c) / max(c, EPS))
    else:
        integ_each = (
            K
            * G
            * ((T_end - times + c) ** (1 - p) - c ** (1 - p))
            / (1 - p)
        )
    # Past-event integrals: only events that occurred before T_end contribute.
    in_window = times < T_end
    integ_term = float(mu * T_end + np.sum(integ_each[in_window]))

    return log_term - integ_term
