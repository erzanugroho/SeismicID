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

EARTH_RADIUS_KM = 6371.0
# Approximate study-area for Indonesia bounded grid (~3000 cells x 2500 km^2).
# Used to convert temporally-fit mu (events/day across whole catalog) into a
# per-km^2-per-day background density. Refined per-region in Phase 3.
INDONESIA_STUDY_AREA_KM2 = 7.5e6
# Per-event spatial contribution cap: prevents a single near-event trigger
# from saturating P(>=1)=1 just because the kernel integrated over a cell is
# overcounted as (kernel_density * cell_area) for events sitting in the cell.
# Phase 5 replaces this with closed-form circular-integral of the kernel.
PER_EVENT_SPATIAL_CAP = 0.95


def _haversine_km(
    lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance from a single point to many points."""
    p1 = np.radians(lat1)
    l1 = np.radians(lon1)
    p2 = np.radians(lat2)
    l2 = np.radians(lon2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _cell_center_from_id(cell_id: str) -> tuple[float, float]:
    """Parse the canonical grid cell_id produced by ``make_cell_id``.

    Format: ``C{lat10}_{lon10}`` with sign encoded as ``p`` (positive) or
    ``m`` (negative) and the integer being lat*10 / lon*10. Example:
    ``Cm108_p952`` → lat=-10.8, lon=95.2. Falls back to the legacy
    ``C_<lat>_<lon>`` form for older fixtures.
    """
    if cell_id.startswith("C") and "_" in cell_id and not cell_id.startswith("C_"):
        lat_part, lon_part = cell_id[1:].split("_", 1)

        def _decode(token: str) -> float:
            sign = 1.0
            if token.startswith("m"):
                sign, token = -1.0, token[1:]
            elif token.startswith("p"):
                sign, token = 1.0, token[1:]
            return sign * int(token) / 10.0

        return _decode(lat_part), _decode(lon_part)
    parts = cell_id.split("_")
    return float(parts[1]), float(parts[2])

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
    # Spatial state (populated by fit_from_events; empty for temporal-only fit).
    _times_days: np.ndarray = field(default_factory=lambda: np.array([]))
    _mags: np.ndarray = field(default_factory=lambda: np.array([]))
    _lats: np.ndarray = field(default_factory=lambda: np.array([]))
    _lons: np.ndarray = field(default_factory=lambda: np.array([]))
    _t0: "datetime | None" = None
    _spatial: dict[str, float] = field(
        default_factory=lambda: {"d0": 2.0, "gamma": 0.5, "q": 1.5}
    )

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

    def fit_from_events(
        self,
        events: "pd.DataFrame",
        *,
        observation_start: "datetime",
        observation_end: "datetime",
    ) -> "OgataETAS":
        """Ingest a wide event frame, store spatial state, run temporal MLE."""
        import pandas as pd

        df = events.copy()
        if df.empty or "time" not in df.columns:
            T_end = (
                observation_end - observation_start
            ).total_seconds() / 86400
            self.fit(np.array([]), np.array([]), T_end=T_end)
            self._t0 = observation_start
            return self
        df["time"] = pd.to_datetime(df["time"], utc=True)
        mask = (df["time"] >= observation_start) & (df["time"] <= observation_end)
        df = df[mask].sort_values("time")
        T_end = (observation_end - observation_start).total_seconds() / 86400
        if df.empty:
            self.fit(np.array([]), np.array([]), T_end=T_end)
            self._t0 = observation_start
            return self
        times_days = (
            (df["time"] - pd.Timestamp(observation_start)).dt.total_seconds()
            .to_numpy()
            / 86400
        )
        mags = df["magnitude"].to_numpy()
        self.fit(times_days, mags, T_end=T_end)
        self._times_days = times_days
        self._mags = mags
        self._lats = df["lat"].to_numpy()
        self._lons = df["lon"].to_numpy()
        self._t0 = observation_start
        return self

    def _cell_rate(
        self, lat: float, lon: float, *, t_query_days: float
    ) -> float:
        """Per-km^2-per-day intensity at (lat, lon, t_query_days).

        Notes:
          * mu fitted by temporal MLE is catalog-wide events/day; we divide
            by the study-area (Indonesia) to get a per-km^2 density.
          * Each past-event spatial contribution is clipped to a cap so a
            single trigger inside the cell cannot dominate P(>=1).
          * Phase 5 will replace the spatial cap with a closed-form
            integral of the kernel over the cell footprint.
        """
        mu_per_km2 = float(self.params_.get("mu", 0.0)) / INDONESIA_STUDY_AREA_KM2
        if self._times_days.size == 0:
            return mu_per_km2
        past = self._times_days < t_query_days
        if not np.any(past):
            return mu_per_km2
        dt = t_query_days - self._times_days[past] + self.params_["c"]
        omori = dt ** (-self.params_["p"])
        productivity = np.exp(
            self.params_["alpha"] * (self._mags[past] - self.mc)
        )
        r_km = _haversine_km(
            lat, lon, self._lats[past], self._lons[past]
        )
        spatial = np.array(
            [
                spatial_kernel_powerlaw(
                    np.array([r_km[i]]),
                    mag=float(self._mags[past][i]),
                    mc=self.mc,
                    **self._spatial,
                )[0]
                for i in range(r_km.size)
            ]
        )
        per_event = self.params_["K"] * productivity * omori * spatial
        per_event = np.minimum(per_event, PER_EVENT_SPATIAL_CAP)
        triggered = float(np.sum(per_event))
        return mu_per_km2 + triggered

    def predict_dataframe(
        self,
        cell_ids: list[str],
        *,
        issued_at: "datetime",
        cell_area_km2: float = 2500.0,
        b_value: float = 1.0,
    ) -> "pd.DataFrame":
        """Per-cell P(>=1 event in horizon) for the canonical 16-column grid.

        Approximations (refined in Phase 3):
          * Rate at issued_at is treated as constant over each horizon
            (acceptable for h <= 60 d; ignores Omori decay within the horizon).
          * Threshold scaling assumes Gutenberg-Richter b-value (default 1.0)
            so rate(M >= t) = rate_full * 10**(-b * (t - mc)).
          * Cell area defaults to ~50x50 km^2 (a 0.5deg cell near the equator).
        """
        import pandas as pd

        from backend.app.features.labels import (
            HORIZONS, THRESHOLDS, label_column_name,
        )

        if self._t0 is None:
            return pd.DataFrame({"cell_id": cell_ids})
        t_query_days = (issued_at - self._t0).total_seconds() / 86400
        rows: list[dict] = []
        for cid in cell_ids:
            try:
                lat, lon = _cell_center_from_id(cid)
            except (IndexError, ValueError):
                rows.append({"cell_id": cid})
                continue
            rate_per_km2_day = self._cell_rate(
                lat, lon, t_query_days=t_query_days
            )
            rate_cell_day = rate_per_km2_day * cell_area_km2
            row: dict = {"cell_id": cid}
            for h in HORIZONS:
                for t in THRESHOLDS:
                    scaled = rate_cell_day * 10 ** (-b_value * (t - self.mc))
                    p_ge1 = 1.0 - np.exp(-scaled * h)
                    row[label_column_name(h, t)] = float(
                        min(max(p_ge1, 0.0), 1.0)
                    )
            rows.append(row)
        return pd.DataFrame(rows)


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


def spatial_kernel_powerlaw(
    r_km: np.ndarray,
    *,
    mag: float,
    mc: float,
    d0: float,
    gamma: float,
    q: float,
) -> np.ndarray:
    """Magnitude-scaled isotropic power-law spatial kernel (per km^2).

    f(r | M) = (q - 1) / (pi * d^2) * (1 + (r/d)^2) ** (-q)
        d = d0 * exp(gamma * (M - Mc))

    Integral over R^2 equals 1.0 by construction. The (q - 1) coefficient
    requires q > 1 for normalization to converge.
    """
    d = d0 * np.exp(gamma * (mag - mc))
    coeff = (q - 1.0) / (np.pi * d * d)
    return coeff * (1.0 + (r_km / d) ** 2) ** (-q)
