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

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

EPS = 1e-12

_PARAM_ORDER = ("mu", "K", "c", "p", "alpha")
_BOUNDS: dict[str, tuple[float, float]] = {
    "mu":    (1e-6, 50.0),
    "K":     (1e-6, 50.0),
    "c":     (1e-4, 10.0),
    "p":     (0.5,   2.5),
    "alpha": (0.1,   3.5),
}


def _pack(params: dict[str, float]) -> np.ndarray:
    return np.array([params[k] for k in _PARAM_ORDER], dtype=np.float64)


def _unpack(x: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(_PARAM_ORDER, x)}


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

    Vectorized using a strict-lower-triangular dt matrix; O(N^2) memory but
    O(N^2) time with no Python-level loop, which keeps L-BFGS-B fits tractable
    for catalogs up to a few thousand events.
    """
    times = np.asarray(times, dtype=np.float64)
    mags = np.asarray(mags, dtype=np.float64)
    if times.size == 0:
        return float(-mu * T_end)

    # First sum: log lambda at each event time using strict-prior history.
    # past[i, j] = True iff j < i  (event j strictly precedes event i).
    # dt[i, j] = times[i] - times[j] + c when j < i (sorted, so dt >= c > 0).
    n = times.size
    dt = times[:, None] - times[None, :] + c
    past = np.tri(n, n, k=-1, dtype=bool)  # strict lower; past[i, j] iff j < i
    dt = np.where(past, dt, 1.0)  # neutral value where mask is False
    decay = dt ** (-p)
    G_all = np.exp(alpha * (mags - mc))
    contrib = K * G_all[None, :] * decay
    contrib = np.where(past, contrib, 0.0)
    triggered_at_event = contrib.sum(axis=1)
    lam_at = mu + triggered_at_event
    log_term = float(np.sum(np.log(np.maximum(lam_at, EPS))))

    # Integral term: closed-form analytic per past event + background.
    if abs(p - 1.0) < 1e-9:
        integ_each = K * G_all * np.log((T_end - times + c) / max(c, EPS))
    else:
        integ_each = (
            K
            * G_all
            * ((T_end - times + c) ** (1 - p) - c ** (1 - p))
            / (1 - p)
        )
    in_window = times < T_end
    integ_term = float(mu * T_end + np.sum(integ_each[in_window]))

    return log_term - integ_term


@dataclass
class OgataETAS:
    """Temporal Ogata ETAS fitted by maximum likelihood (L-BFGS-B).

    Spatial extension is added in a follow-up task (1.4) so this class can
    grow without breaking temporal-only callers.
    """

    mc: float
    params_: dict[str, float] = field(default_factory=dict)
    fit_loglik_: float | None = None
    fit_status_: str = "unfit"

    def fit(
        self,
        times: np.ndarray,
        mags: np.ndarray,
        *,
        T_end: float,
        x0: dict[str, float] | None = None,
    ) -> "OgataETAS":
        times = np.asarray(times, dtype=np.float64)
        mags = np.asarray(mags, dtype=np.float64)

        if times.size == 0:
            self.params_ = {
                "mu": 1e-4, "K": 1e-4, "c": 0.01, "p": 1.1, "alpha": 1.0
            }
            self.fit_loglik_ = float(-1e-4 * T_end)
            self.fit_status_ = "no_events"
            return self

        x0_dict = x0 or {
            "mu": 0.05, "K": 0.02, "c": 0.01, "p": 1.1, "alpha": 1.5
        }

        def neg_ll(x: np.ndarray) -> float:
            try:
                params = _unpack(x)
                ll = ogata_loglik(
                    times, mags, T_end=T_end, mc=self.mc, **params
                )
                return -ll if np.isfinite(ll) else 1e12
            except Exception:  # noqa: BLE001
                return 1e12

        bounds = [_BOUNDS[k] for k in _PARAM_ORDER]
        res = minimize(
            neg_ll, x0=_pack(x0_dict), method="L-BFGS-B", bounds=bounds
        )
        self.params_ = _unpack(res.x)
        self.fit_loglik_ = float(-res.fun) if np.isfinite(res.fun) else None
        self.fit_status_ = (
            "converged" if res.success else f"warn:{str(res.message)[:40]}"
        )
        return self


def simulate_catalog(
    *,
    T_end: float,
    mc: float,
    mu: float,
    K: float,
    c: float,
    p: float,
    alpha: float,
    max_events: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Thinning simulation of temporal Ogata ETAS for tests / validation.

    Magnitudes are drawn from a Gutenberg-Richter exponential tail with b=1.0
    above ``mc``. Spatial dimension is intentionally absent; this is the
    temporal-only test fixture used to validate likelihood + MLE.
    """
    rng = rng or np.random.default_rng()
    times: list[float] = []
    mags: list[float] = []
    t = 0.0
    while t < T_end and len(times) < max_events:
        t_arr = np.asarray(times)
        m_arr = np.asarray(mags)
        lam_now = mu + (
            _triggered_intensity(
                t + 1e-6, t_arr, m_arr,
                K=K, c=c, p=p, alpha=alpha, mc=mc,
            )
            if t_arr.size else 0.0
        )
        lam_bar = max(lam_now * 1.5, mu * 2.0, 1e-3)
        u = rng.exponential(1.0 / lam_bar)
        t += u
        if t >= T_end:
            break
        lam_t = mu + (
            _triggered_intensity(
                t, t_arr, m_arr, K=K, c=c, p=p, alpha=alpha, mc=mc,
            )
            if t_arr.size else 0.0
        )
        if rng.uniform() <= lam_t / lam_bar:
            times.append(t)
            mags.append(mc + rng.exponential(1.0 / np.log(10.0)))
    return np.array(times), np.array(mags)
