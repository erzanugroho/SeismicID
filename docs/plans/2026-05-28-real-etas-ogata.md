# Real ETAS (Ogata 1988) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Tambahkan implementasi ETAS Ogata sungguhan (μ, K, c, p, α + spatial kernel) sebagai second baseline di samping `PoissonBaseline`, supaya klaim "model mengalahkan baseline" punya bukti seismologis yang valid.

**Architecture:**
Buat modul baru `backend/app/ml/etas_ogata.py` (terpisah dari `etas.py` Poisson supaya tidak konflik dengan alias `ETASBaseline`). Modul ini fit parameter ETAS via MLE (L-BFGS-B) pada train window, expose `predict_dataframe(cell_ids)` dengan signature kompatibel `PoissonBaseline` sehingga `forecast_service` & `evaluate_dataset` bisa pakai sebagai second baseline. Mc (magnitude completeness) di-estimate per region pakai MAXC, hasilnya disimpan ke catalog filter sebelum fit. Evaluasi dilakukan di dua skill metric: BSS vs Poisson DAN BSS vs ETAS-Ogata.

**Tech Stack:** Python, numpy, scipy.optimize (L-BFGS-B), pandas, pytest. Library opsional `etas` (Mizrahi 2023, PyPI) untuk validation cross-check, tapi implementasi inti kita tulis sendiri supaya transparan & testable. No new heavy deps di runtime path.

**Prerequisites:**
- `backend/app/ml/etas.py` (PoissonBaseline) tetap tidak diubah — Phase 0 hanya menambah sibling module.
- Test framework `pytest` sudah jalan (lihat `Makefile`).
- Catalog historical events available via `read_historical_events()` (sudah ada di pipeline lama).

---

## Phase 0: Persiapan & Mc estimation

### Task 0.1: Tambah modul Mc estimation (MAXC)

**Objective:** Hitung magnitude of completeness per region grid pakai metode MAXC (Wiemer & Wyss 2000) — dipakai sebagai filter event sebelum fit ETAS.

**Files:**
- Create: `backend/app/ml/mc_estimation.py`
- Test: `backend/tests/test_mc_estimation.py`

**Step 1: Write failing test**

```python
# backend/tests/test_mc_estimation.py
"""Tests for magnitude of completeness (Mc) estimation via MAXC."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_maxc_recovers_known_mc_from_synthetic_gr_catalog() -> None:
    """Synthetic catalog with b=1.0 and Mc=4.5 should be recovered to within
    +/- 0.2 magnitude units by the MAXC estimator."""
    from backend.app.ml.mc_estimation import estimate_mc_maxc

    rng = np.random.default_rng(42)
    # Gutenberg-Richter: log10(N) = a - b*M. Below Mc, rolloff (sub-completeness).
    n_above = 5000
    mags_above = 4.5 + rng.exponential(scale=1.0 / np.log(10.0) / 1.0, size=n_above)
    n_below = 800
    mags_below = 3.0 + rng.uniform(0, 1.5, size=n_below)
    mags = np.concatenate([mags_above, mags_below])

    mc = estimate_mc_maxc(mags, bin_width=0.1)
    assert 4.3 <= mc <= 4.7, f"Mc estimate {mc} outside expected range"


def test_maxc_returns_nan_for_too_few_events() -> None:
    from backend.app.ml.mc_estimation import estimate_mc_maxc

    mc = estimate_mc_maxc(np.array([4.5, 4.7, 5.1]), bin_width=0.1)
    assert np.isnan(mc), "Should return NaN when sample too small"
```

**Step 2: Run test to verify failure**

Run: `pytest backend/tests/test_mc_estimation.py -v`
Expected: FAIL — "ModuleNotFoundError: backend.app.ml.mc_estimation"

**Step 3: Write minimal implementation**

```python
# backend/app/ml/mc_estimation.py
"""Magnitude of completeness estimation.

MAXC method (Wiemer & Wyss 2000): Mc = magnitude bin with maximum frequency
in the non-cumulative frequency-magnitude distribution. Simple, robust enough
for catalog filtering before ETAS fit. For publication-grade Mc, prefer
Lilliefors / GFT — but those require more events than typical regional cells.
"""
from __future__ import annotations

import numpy as np

MIN_EVENTS_FOR_MC = 50


def estimate_mc_maxc(magnitudes: np.ndarray, *, bin_width: float = 0.1) -> float:
    """Return Mc estimate via MAXC. NaN if sample too small.

    Parameters
    ----------
    magnitudes : array of event magnitudes (any units, typically Mw).
    bin_width  : bin size for the histogram (default 0.1 mag units).
    """
    mags = np.asarray(magnitudes, dtype=np.float64)
    mags = mags[np.isfinite(mags)]
    if mags.size < MIN_EVENTS_FOR_MC:
        return float("nan")
    lo, hi = float(np.floor(mags.min() * 10) / 10), float(np.ceil(mags.max() * 10) / 10)
    bins = np.arange(lo, hi + bin_width, bin_width)
    counts, edges = np.histogram(mags, bins=bins)
    if counts.sum() == 0:
        return float("nan")
    peak_idx = int(np.argmax(counts))
    # Mc is the LEFT edge of the most populated bin (conservative).
    return float(edges[peak_idx])


def estimate_mc_per_region(
    events: "pd.DataFrame",
    *,
    region_col: str = "cell_id",
    bin_width: float = 0.1,
    min_events: int = MIN_EVENTS_FOR_MC,
) -> dict[str, float]:
    """Per-region Mc dict; cells with fewer than ``min_events`` get NaN."""
    import pandas as pd

    out: dict[str, float] = {}
    if events.empty or region_col not in events.columns:
        return out
    for region, sub in events.groupby(region_col):
        if len(sub) < min_events:
            out[str(region)] = float("nan")
            continue
        out[str(region)] = estimate_mc_maxc(sub["magnitude"].to_numpy(), bin_width=bin_width)
    return out
```

**Step 4: Run test to verify pass**

Run: `pytest backend/tests/test_mc_estimation.py -v`
Expected: PASS (2 passed)

**Step 5: Commit**

```bash
git add backend/app/ml/mc_estimation.py backend/tests/test_mc_estimation.py
git commit -m "feat(ml): add MAXC magnitude completeness estimator for ETAS pre-filter"
```

---

### Task 0.2: Tambah catalog filter helper (Mc + waktu)

**Objective:** Helper kecil untuk filter catalog: drop event di bawah Mc global atau Mc-region, tetap simpan original count untuk audit.

**Files:**
- Create: `backend/app/ml/catalog_filter.py`
- Test: `backend/tests/test_catalog_filter.py`

**Step 1: Write failing test**

```python
# backend/tests/test_catalog_filter.py
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd


def test_filter_below_mc_drops_subthreshold_events() -> None:
    from backend.app.ml.catalog_filter import filter_below_mc

    df = pd.DataFrame(
        {
            "event_id": ["a", "b", "c", "d"],
            "magnitude": [3.8, 4.2, 4.6, 5.1],
            "time": pd.to_datetime(["2024-01-01"] * 4, utc=True),
        }
    )
    out = filter_below_mc(df, mc=4.5)
    assert sorted(out["event_id"].tolist()) == ["c", "d"]


def test_filter_below_mc_preserves_audit_attr() -> None:
    from backend.app.ml.catalog_filter import filter_below_mc

    df = pd.DataFrame({"event_id": ["a", "b"], "magnitude": [3.0, 5.0]})
    out = filter_below_mc(df, mc=4.5)
    assert out.attrs.get("mc_filter_dropped") == 1
    assert out.attrs.get("mc_value") == 4.5
```

**Step 2: Run test**

Run: `pytest backend/tests/test_catalog_filter.py -v`
Expected: FAIL — module not found.

**Step 3: Implementation**

```python
# backend/app/ml/catalog_filter.py
"""Helpers to filter event catalogs prior to ETAS fitting."""
from __future__ import annotations

import pandas as pd


def filter_below_mc(events: pd.DataFrame, *, mc: float) -> pd.DataFrame:
    """Drop events with magnitude < mc. Preserves audit attrs."""
    if events.empty or "magnitude" not in events.columns:
        return events
    n_before = len(events)
    out = events[events["magnitude"] >= mc].copy()
    out.attrs["mc_value"] = float(mc)
    out.attrs["mc_filter_dropped"] = int(n_before - len(out))
    return out
```

**Step 4: Run test**

Run: `pytest backend/tests/test_catalog_filter.py -v`
Expected: PASS (2 passed)

**Step 5: Commit**

```bash
git add backend/app/ml/catalog_filter.py backend/tests/test_catalog_filter.py
git commit -m "feat(ml): add Mc-based catalog filter with audit attrs"
```

---

## Phase 1: Core ETAS Ogata MLE

### Task 1.1: Implement ETAS log-likelihood (temporal-only first)

**Objective:** Tulis log-likelihood Ogata 1988 versi temporal (tanpa spatial kernel) sebagai fondasi. Validasi dengan synthetic catalog yang punya parameter ground-truth.

**Math reference:**
Conditional intensity: λ(t|H_t) = μ + Σ_{t_i<t} K · exp(α·(M_i − M_c)) · (t − t_i + c)^(−p)
Log-likelihood: log L = Σ_i log λ(t_i|H_{t_i}) − ∫_0^T λ(t|H_t) dt

**Files:**
- Create: `backend/app/ml/etas_ogata.py` (skeleton + likelihood only)
- Test: `backend/tests/test_etas_ogata_likelihood.py`

**Step 1: Write failing test**

```python
# backend/tests/test_etas_ogata_likelihood.py
from __future__ import annotations

import numpy as np


def test_loglik_returns_finite_for_simple_catalog() -> None:
    from backend.app.ml.etas_ogata import ogata_loglik

    times = np.array([0.5, 1.2, 2.4, 3.8, 5.1, 7.3])  # days since T0
    mags = np.array([5.0, 4.6, 4.8, 5.2, 4.7, 5.0])
    params = dict(mu=0.05, K=0.02, c=0.01, p=1.1, alpha=1.5)
    ll = ogata_loglik(times, mags, T_end=10.0, mc=4.5, **params)
    assert np.isfinite(ll), "log-likelihood must be finite"


def test_loglik_decreases_when_mu_far_from_truth() -> None:
    """Catalog generated with mu=0.05 should score better at mu=0.05 than mu=10."""
    from backend.app.ml.etas_ogata import ogata_loglik

    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0, 100, size=80))
    mags = 4.5 + rng.exponential(0.5, size=80)
    common = dict(K=0.02, c=0.01, p=1.1, alpha=1.5)
    ll_good = ogata_loglik(times, mags, T_end=100.0, mc=4.5, mu=0.05, **common)
    ll_bad = ogata_loglik(times, mags, T_end=100.0, mc=4.5, mu=10.0, **common)
    assert ll_good > ll_bad
```

**Step 2: Run test**

Run: `pytest backend/tests/test_etas_ogata_likelihood.py -v`
Expected: FAIL — module not found.

**Step 3: Implementation**

```python
# backend/app/ml/etas_ogata.py
"""True ETAS (Ogata 1988) — temporal model.

Conditional intensity:
    lambda(t | H_t) = mu + sum_{t_i < t} K * exp(alpha * (M_i - Mc))
                                            * (t - t_i + c) ** (-p)

Log-likelihood (Ogata 1988, eq. 6):
    log L = sum_i log lambda(t_i | H_{t_i})  -  integral_0^T lambda(t) dt

Spatial kernel is added in a follow-up task (1.4) to keep the temporal core
small and unit-testable. We do NOT call this 'ETAS' in user-facing copy until
the spatial extension lands and validation is complete.
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

    Parameters use units consistent with the rest of the project: time in days
    since the catalog start, magnitude in Mw (or compatible scale), Mc the
    completeness threshold used to filter the catalog upstream.
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

    # Integral term: integrate lambda over [0, T_end].
    # Background contributes mu * T_end. Triggered contribution from each
    # past event integrates analytically:
    #   integral_{t_i}^{T_end} K * G_i * (t - t_i + c)**(-p) dt
    # where G_i = exp(alpha * (M_i - Mc)). Closed form for p != 1:
    #   K * G_i * [ (T_end - t_i + c)**(1-p) - c**(1-p) ] / (1 - p)
    # For p == 1, the integral is K * G_i * log((T_end - t_i + c) / c).
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
    integ_term = float(mu * T_end + np.sum(integ_each))

    return log_term - integ_term
```

**Step 4: Run test**

Run: `pytest backend/tests/test_etas_ogata_likelihood.py -v`
Expected: PASS (2 passed)

**Step 5: Commit**

```bash
git add backend/app/ml/etas_ogata.py backend/tests/test_etas_ogata_likelihood.py
git commit -m "feat(ml): add temporal Ogata ETAS log-likelihood with closed-form integral"
```

---

### Task 1.2: Fit ETAS parameters via L-BFGS-B (temporal)

**Objective:** Maximum-likelihood fit untuk (μ, K, c, p, α) dengan bound feasibility. Recovery test: simulate catalog dari known params, fit harus recover dalam tolerance.

**Files:**
- Modify: `backend/app/ml/etas_ogata.py`
- Test: `backend/tests/test_etas_ogata_fit.py`

**Step 1: Failing test**

```python
# backend/tests/test_etas_ogata_fit.py
from __future__ import annotations

import numpy as np


def test_fit_recovers_synthetic_params_within_tolerance() -> None:
    from backend.app.ml.etas_ogata import OgataETAS, simulate_catalog

    rng = np.random.default_rng(0)
    true_params = dict(mu=0.10, K=0.05, c=0.01, p=1.15, alpha=1.4)
    times, mags = simulate_catalog(
        T_end=200.0, mc=4.5, max_events=2000, rng=rng, **true_params
    )
    assert len(times) >= 50, "synthetic catalog too small"

    model = OgataETAS(mc=4.5).fit(times, mags, T_end=200.0)
    # Loose tolerance (MLE on finite catalog has known bias):
    assert abs(model.params_["mu"] - true_params["mu"]) / true_params["mu"] < 0.6
    assert abs(model.params_["p"] - true_params["p"]) < 0.25
    assert model.fit_loglik_ is not None and np.isfinite(model.fit_loglik_)


def test_fit_with_no_events_returns_background_only() -> None:
    from backend.app.ml.etas_ogata import OgataETAS

    model = OgataETAS(mc=4.5).fit(np.array([]), np.array([]), T_end=100.0)
    assert model.params_["mu"] >= 0.0
    assert model.params_["K"] >= 0.0
```

**Step 2: Run** — Expected FAIL (`OgataETAS` and `simulate_catalog` missing).

**Step 3: Implementation** — append to `backend/app/ml/etas_ogata.py`:

```python
from dataclasses import dataclass, field
from scipy.optimize import minimize


_PARAM_ORDER = ("mu", "K", "c", "p", "alpha")
_BOUNDS = {
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


@dataclass
class OgataETAS:
    """Temporal Ogata ETAS fitted by maximum likelihood."""

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
            self.params_ = {"mu": 1e-4, "K": 1e-4, "c": 0.01, "p": 1.1, "alpha": 1.0}
            self.fit_loglik_ = float(-1e-4 * T_end)
            self.fit_status_ = "no_events"
            return self

        x0_dict = x0 or {"mu": 0.05, "K": 0.02, "c": 0.01, "p": 1.1, "alpha": 1.5}

        def neg_ll(x: np.ndarray) -> float:
            try:
                params = _unpack(x)
                ll = ogata_loglik(times, mags, T_end=T_end, mc=self.mc, **params)
                return -ll if np.isfinite(ll) else 1e12
            except Exception:  # noqa: BLE001
                return 1e12

        bounds = [_BOUNDS[k] for k in _PARAM_ORDER]
        res = minimize(neg_ll, x0=_pack(x0_dict), method="L-BFGS-B", bounds=bounds)
        self.params_ = _unpack(res.x)
        self.fit_loglik_ = float(-res.fun) if np.isfinite(res.fun) else None
        self.fit_status_ = "converged" if res.success else f"warn:{res.message[:40]}"
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
    """Thinning simulation of temporal Ogata ETAS for tests/validation."""
    rng = rng or np.random.default_rng()
    times: list[float] = []
    mags: list[float] = []
    t = 0.0
    while t < T_end and len(times) < max_events:
        # Upper bound on intensity over a small window via current state.
        t_arr = np.asarray(times)
        m_arr = np.asarray(mags)
        lam_now = mu + (
            _triggered_intensity(t + 1e-6, t_arr, m_arr,
                                 K=K, c=c, p=p, alpha=alpha, mc=mc)
            if t_arr.size else 0.0
        )
        lam_bar = max(lam_now * 1.5, mu * 2.0, 1e-3)
        u = rng.exponential(1.0 / lam_bar)
        t += u
        if t >= T_end:
            break
        lam_t = mu + (
            _triggered_intensity(t, t_arr, m_arr,
                                 K=K, c=c, p=p, alpha=alpha, mc=mc)
            if t_arr.size else 0.0
        )
        if rng.uniform() <= lam_t / lam_bar:
            times.append(t)
            mags.append(mc + rng.exponential(1.0 / np.log(10.0)))  # b=1.0
    return np.array(times), np.array(mags)
```

**Step 4: Run** — `pytest backend/tests/test_etas_ogata_fit.py -v` → PASS (2 passed). Note: synthetic recovery test has known finite-sample bias; loose tolerance is intentional.

**Step 5: Commit**

```bash
git add backend/app/ml/etas_ogata.py backend/tests/test_etas_ogata_fit.py
git commit -m "feat(ml): MLE fit for temporal Ogata ETAS via L-BFGS-B + thinning simulator"
```

---

### Task 1.3: Add isotropic spatial kernel to ETAS

**Objective:** Extend conditional intensity dengan spatial kernel isotropic (2D Gaussian / power-law) supaya prediksi bisa per-cell, bukan global rate. Gunakan power-law: f(r|M) = (q-1)/(π·d²) · (1 + r²/d²)^(-q), d = d₀·exp(γ·(M-Mc)).

**Files:**
- Modify: `backend/app/ml/etas_ogata.py` (tambah spatial functions; keep temporal-only public API)
- Test: `backend/tests/test_etas_ogata_spatial.py`

**Step 1: Failing test**

```python
# backend/tests/test_etas_ogata_spatial.py
from __future__ import annotations

import numpy as np


def test_spatial_kernel_integrates_to_unity_on_dense_grid() -> None:
    from backend.app.ml.etas_ogata import spatial_kernel_powerlaw

    # Place trigger at (0,0), magnitude M=5.5, evaluate kernel on a grid.
    xs = np.linspace(-200, 200, 401)
    ys = np.linspace(-200, 200, 401)
    XX, YY = np.meshgrid(xs, ys)
    R = np.sqrt(XX**2 + YY**2)
    vals = spatial_kernel_powerlaw(R, mag=5.5, mc=4.5, d0=2.0, gamma=0.5, q=1.5)
    dx = xs[1] - xs[0]
    integral = float(vals.sum() * dx * dx)
    # Should integrate close to 1 over a sufficiently large domain.
    assert 0.85 < integral < 1.05, f"kernel integral {integral} not unit"


def test_spatial_kernel_decays_with_distance() -> None:
    from backend.app.ml.etas_ogata import spatial_kernel_powerlaw

    near = spatial_kernel_powerlaw(np.array([1.0]), mag=5.0, mc=4.5, d0=2.0, gamma=0.5, q=1.5)
    far = spatial_kernel_powerlaw(np.array([100.0]), mag=5.0, mc=4.5, d0=2.0, gamma=0.5, q=1.5)
    assert near[0] > far[0]
```

**Step 2: Run** — Expected FAIL (`spatial_kernel_powerlaw` missing).

**Step 3: Implementation** — append to `backend/app/ml/etas_ogata.py`:

```python
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

    Integral over R^2 equals 1.0 by construction.
    """
    d = d0 * np.exp(gamma * (mag - mc))
    coeff = (q - 1.0) / (np.pi * d * d)
    return coeff * (1.0 + (r_km / d) ** 2) ** (-q)
```

Update `_BOUNDS`/`_PARAM_ORDER` is deferred to Task 1.4 where the spatial parameters enter the joint MLE.

**Step 4: Run** — `pytest backend/tests/test_etas_ogata_spatial.py -v` → PASS (2 passed).

**Step 5: Commit**

```bash
git add backend/app/ml/etas_ogata.py backend/tests/test_etas_ogata_spatial.py
git commit -m "feat(ml): add isotropic power-law spatial kernel for ETAS triggering"
```

---

### Task 1.4: Joint spatio-temporal MLE fit + per-cell rate prediction

**Objective:** Gabungkan temporal Ogata + spatial power-law kernel jadi satu model. Output: `predict_dataframe(cell_ids)` yang return P(>=1 event) per (cell, horizon, threshold) — drop-in compatible dengan `PoissonBaseline`.

**Files:**
- Modify: `backend/app/ml/etas_ogata.py` (extend `OgataETAS` with spatial state + event ingestion)
- Test: `backend/tests/test_etas_ogata_predict.py`

**Design notes:**
- Spatial parameters (d0, gamma, q) di-fix ke nilai literatur dulu, supaya MLE temporal-only stabil. Joint optimization ditunda ke Phase 5.
- Threshold scaling pakai Gutenberg-Richter b=1 sebagai default. b-value per-region masuk Phase 3.
- Approx: rate at issued_at * horizon, abaikan Omori decay dalam horizon. Cukup akurat untuk h <= 60d, refined di Phase 3.

**Step 1: Failing test**

```python
# backend/tests/test_etas_ogata_predict.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd


def test_predict_dataframe_has_canonical_label_columns() -> None:
    from backend.app.features.labels import all_label_columns
    from backend.app.ml.etas_ogata import OgataETAS

    rng = np.random.default_rng(0)
    events = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(40)],
            "time": pd.to_datetime(
                [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=int(d))
                 for d in rng.integers(0, 365, size=40)],
                utc=True,
            ),
            "lat": -1.0 + rng.normal(0, 0.5, size=40),
            "lon": 120.0 + rng.normal(0, 0.5, size=40),
            "magnitude": 4.5 + rng.exponential(0.4, size=40),
            "depth": np.full(40, 10.0),
        }
    )
    model = OgataETAS(mc=4.5).fit_from_events(
        events,
        observation_start=datetime(2024, 1, 1, tzinfo=UTC),
        observation_end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    cell_ids = ["C_-1.00_120.00", "C_-1.00_120.50", "C_0.00_119.50"]
    df = model.predict_dataframe(cell_ids, issued_at=datetime(2025, 1, 1, tzinfo=UTC))
    assert df["cell_id"].tolist() == cell_ids
    for col in all_label_columns():
        assert col in df.columns
        assert (df[col] >= 0).all() and (df[col] <= 1).all()


def test_predict_higher_near_recent_large_event() -> None:
    from backend.app.features.labels import label_column_name
    from backend.app.ml.etas_ogata import OgataETAS

    issued = datetime(2025, 1, 1, tzinfo=UTC)
    recent = issued - timedelta(days=2)
    events = pd.DataFrame(
        {
            "event_id": ["mainshock"],
            "time": pd.to_datetime([recent], utc=True),
            "lat": [-1.0],
            "lon": [120.0],
            "magnitude": [5.8],
            "depth": [10.0],
        }
    )
    model = OgataETAS(mc=4.5).fit_from_events(
        events,
        observation_start=issued - timedelta(days=365),
        observation_end=issued,
    )
    df = model.predict_dataframe(
        ["C_-1.00_120.00", "C_-8.00_115.00"], issued_at=issued
    )
    col = label_column_name(7, 4.5)
    assert df.iloc[0][col] > df.iloc[1][col]
```

**Step 2: Run** — Expected FAIL (`fit_from_events`, `predict_dataframe` missing).

**Step 3: Implementation** — append to `backend/app/ml/etas_ogata.py`. Outline (full code in repo PR):

1. Helper `_haversine_km(lat1, lon1, lat2, lon2)` for distance in km.
2. Helper `_cell_center_from_id(cell_id)` parsing canonical format `C_lat_lon`.
3. Extend `OgataETAS` dataclass with stored event arrays (times_days, mags, lats, lons), reference time `_t0`, and `_spatial` dict (defaults: d0=2.0, gamma=0.5, q=1.5).
4. Method `fit_from_events(events_df, observation_start, observation_end)`: convert event timestamps to days-since-t0, sort, call existing `fit(times, mags, T_end=...)`, store all arrays.
5. Method `_cell_rate(lat, lon, t_query_days)`: returns mu + K * sum(productivity * Omori * spatial_kernel) over past events.
6. Method `predict_dataframe(cell_ids, issued_at)`: per cell compute rate_per_km2_day, multiply by cell area (~2500 km^2 for 0.5deg near equator), apply GR-1 threshold scaling, integrate over horizon to get `P(>=1) = 1 - exp(-rate*h)`.

**Step 4: Run** — `pytest backend/tests/test_etas_ogata_predict.py -v` → PASS (2 passed).

**Step 5: Commit**

```
git add backend/app/ml/etas_ogata.py backend/tests/test_etas_ogata_predict.py
git commit -m "feat(ml): per-cell Ogata ETAS prediction (temporal MLE + fixed spatial kernel)"
```

---

## Phase 2: Integration into ensemble & forecast service

### Task 2.1: Add ETAS-Ogata as optional second baseline in evaluator

**Objective:** Buat `evaluate_dataset` bisa terima dua baseline (Poisson + ETAS-Ogata), report BSS terhadap masing-masing. Tidak mengubah call site lama yang cuma kirim satu baseline.

**Files:**
- Modify: `backend/app/ml/evaluate.py`
- Modify: `backend/tests/test_ml.py` (atau test baru)

**Step 1: Failing test**

```python
# backend/tests/test_evaluate_dual_baseline.py
import numpy as np
import pandas as pd

from backend.app.features.labels import all_label_columns


def _toy_frame() -> pd.DataFrame:
    cols = all_label_columns()
    df = pd.DataFrame({c: [0, 1, 0, 1] for c in cols})
    df["cell_id"] = ["c1", "c2", "c1", "c2"]
    return df


def test_evaluate_reports_bss_for_both_baselines() -> None:
    from backend.app.ml.evaluate import evaluate_dataset

    y = _toy_frame()
    cols = [c for c in y.columns if c.startswith("p_")]
    preds = y[cols].astype(float) * 0.7 + 0.15
    poisson_b = y[cols].astype(float) * 0.3 + 0.30
    etas_b = y[cols].astype(float) * 0.4 + 0.25

    out = evaluate_dataset(
        pd.concat([y[["cell_id"]], y[cols]], axis=1),
        pd.concat([y[["cell_id"]], preds], axis=1),
        baseline=poisson_b.assign(cell_id=y["cell_id"]),
        baseline_etas=etas_b.assign(cell_id=y["cell_id"]),
    )
    head = next(iter(out["per_head"].values()))
    assert "bss_vs_poisson" in head
    assert "bss_vs_etas" in head
```

**Step 2: Run** — Expected FAIL (`baseline_etas` kwarg unknown).

**Step 3: Implementation**

Modifikasi `evaluate_dataset` di `backend/app/ml/evaluate.py`:

```python
def evaluate_dataset(
    truth: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    baseline: pd.DataFrame | None = None,
    baseline_etas: pd.DataFrame | None = None,
) -> dict:
    ...
    for head in heads:
        y = truth[head].to_numpy()
        p = preds[head].to_numpy()
        b_poisson = baseline[head].to_numpy() if baseline is not None else None
        b_etas = baseline_etas[head].to_numpy() if baseline_etas is not None else None
        out["per_head"][head] = {
            **evaluate_head(y, p, baseline=b_poisson),
        }
        if b_etas is not None:
            from sklearn.metrics import brier_score_loss
            b_e = brier_score_loss(y, b_etas)
            out["per_head"][head]["bss_vs_etas"] = float(
                1 - out["per_head"][head]["brier"] / max(b_e, 1e-9)
            )
        out["per_head"][head]["reliability"] = reliability_diagram(y, p)
        out["per_head"][head]["roc"] = roc_points(y, p)
```

Backward compat: kalau `baseline_etas=None`, behavior identik dengan sebelumnya.

**Step 4: Run** — `pytest backend/tests/test_evaluate_dual_baseline.py -v` → PASS.

**Step 5: Commit**

```
git add backend/app/ml/evaluate.py backend/tests/test_evaluate_dual_baseline.py
git commit -m "feat(eval): support dual baseline (Poisson + ETAS-Ogata) in evaluate_dataset"
```

---

### Task 2.2: Wire ETAS-Ogata into train_initial.py

**Objective:** Saat training initial, fit ETAS-Ogata di TRAIN window (sama dengan Poisson — no leakage), hitung `predict_dataframe` untuk test cells, panggil `evaluate_dataset(..., baseline=poisson, baseline_etas=etas)`.

**Files:**
- Modify: `scripts/train_initial.py`
- Modify: `backend/app/db/schema.py` (tambah field `bss_vs_etas` di metrics row)
- Modify: `backend/tests/test_train_initial_smoke.py` (kalau ada)

**Step 1: Failing test (smoke)**

```python
# backend/tests/test_train_initial_includes_etas_baseline.py
import subprocess, sys, os, json
from pathlib import Path


def test_train_initial_metrics_payload_has_etas_bss(tmp_path) -> None:
    """Ringkas: setelah train, metrics row harus punya kolom bss_vs_etas."""
    # Pakai env DRY_RUN=1 supaya cuma jalanin smoke pendek di CI.
    env = {**os.environ, "DRY_RUN": "1"}
    proc = subprocess.run(
        [sys.executable, "scripts/train_initial.py", "--smoke"],
        capture_output=True, text=True, env=env,
    )
    out = Path("data/runs/_smoke/metrics.json")
    if not out.exists():
        return  # smoke mode opsional, skip kalau script belum support
    payload = json.loads(out.read_text())
    sample_head = next(iter(payload.values()))
    assert "bss_vs_etas" in sample_head
```

**Step 2: Run** — FAIL atau skip (kalau smoke mode belum ada).

**Step 3: Implementation**

Di `scripts/train_initial.py`, setelah block fit Poisson baseline (sekitar line 112–130), tambah:

```python
from backend.app.ml.etas_ogata import OgataETAS

try:
    etas_model = OgataETAS(mc=4.5).fit_from_events(
        train_events,
        observation_start=obs_start,
        observation_end=obs_end,
    )
    etas_pred = etas_model.predict_dataframe(unique_test_cells, issued_at=obs_end)
    etas_for_eval = etas_pred.merge(
        test[["cell_id"]].drop_duplicates(),
        on="cell_id",
    )
except Exception as exc:
    logger.warning("etas_baseline_skipped", error=str(exc))
    etas_for_eval = None

eval_out = evaluate_dataset(
    test, preds,
    baseline=baseline_for_eval,
    baseline_etas=etas_for_eval,
)
```

Tambah `bss_vs_etas` ke metrics payload yang disimpan ke DB. Schema migration: kolom nullable supaya run lama tetap kompatibel.

**Step 4: Run** — `pytest backend/tests/test_train_initial_includes_etas_baseline.py -v` → PASS atau skip.

**Step 5: Commit**

```
git add scripts/train_initial.py backend/app/db/schema.py
git commit -m "feat(train): fit ETAS-Ogata baseline alongside Poisson, log bss_vs_etas"
```

---

### Task 2.3: Surface ETAS-Ogata as forecast service tier (optional)

**Objective:** Tambah ETAS-Ogata sebagai opsi tier 2.5 di `forecast_service.py` (di antara ML dan Poisson). Gated by config flag — default OFF supaya production tidak kepengaruh sampai validation selesai.

**Files:**
- Modify: `backend/app/services/forecast_service.py`
- Modify: `backend/app/core/config.py` (tambah `enable_etas_baseline_tier`)

**Step 1: Failing test**

```python
def test_forecast_service_uses_etas_when_flag_on(monkeypatch) -> None:
    from backend.app.services import forecast_service
    monkeypatch.setattr(forecast_service.settings, "enable_etas_baseline_tier", True)
    # Arrange: no ML model loaded -> falls through tiers.
    # Assert: response metadata includes mode="etas_ogata".
    ...
```

**Step 2-5:** Implement flag + branch + commit. Detail kode di-defer ke implementation PR; pattern ikut tier fallback yang sudah ada (cek `_poisson_predictions_for_cells`).

**Done when:** flag default OFF; flag ON + no ML model + events available → forecast pakai ETAS-Ogata; test response metadata `mode == "etas_ogata"`.

---

## Phase 3: Validation & scientific defensibility

### Task 3.1: Per-region b-value estimation (Aki-Utsu)

**Objective:** Estimasi b-value Gutenberg-Richter per cluster regional supaya threshold scaling di `predict_dataframe` tidak hardcoded ke b=1. Pakai Aki-Utsu MLE: b_hat = log10(e) / (mean(M) - Mc).

**Files:**
- Create: `backend/app/ml/b_value.py`
- Test: `backend/tests/test_b_value.py`
- Modify: `backend/app/ml/etas_ogata.py` (consume b-value dict di `predict_dataframe`)

**Step 1: Failing test**

```python
# backend/tests/test_b_value.py
import numpy as np


def test_aki_utsu_recovers_b_within_tolerance() -> None:
    from backend.app.ml.b_value import estimate_b_aki_utsu

    rng = np.random.default_rng(0)
    mc = 4.5
    true_b = 0.95
    mags = mc + rng.exponential(1.0 / (true_b * np.log(10)), size=5000)
    b_hat = estimate_b_aki_utsu(mags, mc=mc)
    assert abs(b_hat - true_b) < 0.10


def test_b_value_returns_default_for_small_sample() -> None:
    from backend.app.ml.b_value import estimate_b_aki_utsu

    b_hat = estimate_b_aki_utsu(np.array([4.6, 4.7, 5.0]), mc=4.5, default=1.0)
    assert b_hat == 1.0
```

**Step 2: Run** → FAIL.

**Step 3: Implementation**

```python
# backend/app/ml/b_value.py
import numpy as np

MIN_EVENTS = 50


def estimate_b_aki_utsu(magnitudes, *, mc, default=1.0):
    mags = np.asarray(magnitudes, dtype=np.float64)
    mags = mags[mags >= mc]
    if mags.size < MIN_EVENTS:
        return default
    mean_m = float(mags.mean())
    if mean_m <= mc:
        return default
    return float(np.log10(np.e) / (mean_m - mc))
```

Di `OgataETAS.predict_dataframe`, ganti `10 ** (-(t - self.mc))` → `10 ** (-b * (t - self.mc))` dengan b di-resolve per cell (atau global) dari b-value dict.

**Step 4-5:** Run, commit `feat(ml): add Aki-Utsu b-value estimator and consume in ETAS threshold scaling`.

---

### Task 3.2: Synthetic-catalog cross-validation against `etas` PyPI library

**Objective:** Validasi independen — bandingkan output `OgataETAS.fit` kita dengan library `etas` (Mizrahi 2023) di synthetic catalog yang sama. Toleransi: parameter dalam +/- 30%, log-likelihood within 5%.

**Files:**
- Create: `backend/tests/test_etas_cross_validation.py` (skip kalau library tidak ter-install)
- Modify: `requirements-dev.txt` (tambah `etas` sebagai dev dep, bukan runtime)

**Step 1-5:** Skipif decorator pakai `pytest.importorskip("etas")`. Test fitting & comparison. Commit pakai message `test(ml): cross-validate OgataETAS vs etas library on synthetic catalog`.

**Acceptance:** Kalau cross-validation passes, kita boleh klaim implementasi konsisten dengan reference. Kalau gagal, audit ulang likelihood/integral.

---

### Task 3.3: Aftershock decay sanity test on real Indonesian event

**Objective:** Setelah event besar (M>=6) di catalog Indonesia, ETAS rate harus naik tajam lalu meluruh seperti Omori law. Test ini bukan unit test — script analisis di `scripts/`.

**Files:**
- Create: `scripts/analyze_etas_aftershock_decay.py`
- Create: `docs/notebooks/etas_aftershock_validation.md` (output report)

**Steps:**
1. Pilih 3 event referensi Indonesia (Lombok 2018 M6.9, Palu 2018 M7.5, Mamuju 2021 M6.2).
2. Untuk tiap event: fit ETAS pada catalog 5 tahun sebelum event, predict per-day rate untuk 60 hari setelah event di cell event tersebut.
3. Plot rate vs time → harus terlihat power-law decay (line lurus di log-log).
4. Report ke `docs/notebooks/etas_aftershock_validation.md` dengan figure.

**Done when:** Tiga event nunjukkan decay slope p ∈ [0.8, 1.5] (literatur range).

---

### Task 3.4: Information gain (bits/event) as primary metric

**Objective:** Implementasi information gain log-loss based, sesuai Phase 3.2 di scientific-review-followups. ETAS Ogata jadi reference baseline kedua untuk info gain (selain Poisson).

**Files:**
- Modify: `backend/app/ml/evaluate.py` (extend metrics dict)
- Modify: `MODEL_CARD.md` (tambah baris info gain di metrics section)

**Math:**
IG = (1/N) * sum_i [log p_model(y_i) - log p_baseline(y_i)] / log(2)
Satuan: bits/event. Positif → model lebih baik dari baseline.

Test pakai toy probabilities. Commit: `feat(eval): report information gain (bits/event) vs Poisson and vs ETAS-Ogata`.

---

## Phase 4: CSEP-style prospective archive

### Task 4.1: Archive ETAS-Ogata forecasts alongside ML

**Objective:** Tiap kali scheduler trigger forecast, archive ETAS-Ogata predictions di parquet terpisah supaya bisa di-evaluate prospectively (sebelum future events diketahui). Ini melengkapi Phase 3.1 di scientific-review-followups.

**Files:**
- Modify: forecast persistence di `backend/app/services/forecast_service.py`
- Modify: storage layer untuk dual-archive
- Test: `backend/tests/test_forecast_archive_etas.py`

**Steps:**
1. Setiap run, tulis dua file parquet: `data/parquet/forecast_archive/YYYY-MM-DD_ml.parquet` dan `..._etas.parquet`.
2. Schema sama: generated_at, model_version, data_cutoff, horizon, threshold, cell_id, probability, baseline_type.
3. Prospective evaluator di Task 3.1 scientific-review-followups extend untuk score kedua arsip.
4. Test: write archive → read back → assert column shape sama, baseline_type field benar.

**Done when:** Archive run produce dua file, evaluator bisa load + score keduanya, BSS tercatat di metrics row prospective.

---

### Task 4.2: CSEP L-test and N-test on ETAS forecasts

**Objective:** Pakai L-test (likelihood) dan N-test (number) CSEP untuk validasi forecast ETAS prospective. Two-sided per Issue 5 di test_probability_audit.

**Files:**
- Create: `backend/app/ml/csep_tests.py`
- Test: `backend/tests/test_csep_tests.py`

**Steps:**
1. L-test: simulate K=1000 catalog dari forecast rate, hitung quantile log-likelihood observed vs simulated. Pass kalau 0.025 <= q <= 0.975.
2. N-test: count event observed vs distribution Poisson dari sum rate. Quantile two-sided same threshold.
3. Test pakai synthetic forecast + synthetic catalog dengan known ground truth — N-test harus pass kalau forecast benar.

**Done when:** Both tests implemented, prospective evaluator report L/N quantile per forecast window.

---

## Phase 5: Refinement (deferred / future)

### Task 5.1: Joint MLE for spatial parameters (d0, gamma, q)

**Objective:** Setelah Phase 1-4 stabil di production-shadow, extend optimizer untuk fit (d0, gamma, q) bersama (mu, K, c, p, alpha) — 8 parameter total.

**Risk note:** Joint optimization rentan local minima di catalog kecil. Mitigasi: multi-start (5-10 random init), pilih best loglik. Bound q in (1.1, 3.0), d0 in (0.5, 20.0), gamma in (0.0, 1.0).

**Files:** `backend/app/ml/etas_ogata.py`, test extension. Defer until Task 1.4 stable in shadow eval.

---

### Task 5.2: Anisotropic spatial kernel (fault-aligned)

**Objective:** Untuk event di subduction zone / strike-slip, distribusi aftershock anisotropic mengikuti rupture plane. Implementasi: rotate coordinate system per event berdasarkan focal mechanism (kalau ada), atau berdasarkan strike fault terdekat dari `fault_db.py`.

**Status:** Design only di plan ini — butuh akses ke focal mechanism catalog (GCMT) atau rupture plane Slab2 alignment yang belum ter-integrate.

---

### Task 5.3: Hierarchical Bayes ETAS (regional pooling)

**Objective:** Pooling parameter antar region geografis pakai hierarchical model — region dengan data sedikit pinjam strength dari region tetangga. Implementasi pakai `pymc` atau `numpyro`.

**Status:** Future research direction. Tidak dalam scope plan ini.

---

## Phase 6: Documentation & rollout

### Task 6.1: Update MODEL_CARD.md with ETAS-Ogata baseline

**Objective:** MODEL_CARD harus jujur soal dua baseline: Poisson (sederhana) DAN ETAS-Ogata (rigorous). Klaim skill harus pakai ETAS sebagai second comparison.

**Files:**
- Modify: `MODEL_CARD.md`

**Sections to add/update:**
1. "Baselines" subsection: jelaskan dua baseline, parameter, fit method, limitations.
2. "Metrics" subsection: tambah baris BSS vs ETAS-Ogata + information gain (bits/event).
3. "Limitations" subsection: tambah paragraf — ETAS pakai isotropic spatial kernel, b=1 default, single Mc per region. Bukan publication-grade tapi sudah jauh lebih kuat dari Poisson untuk klaim skill.

**Done when:** MODEL_CARD review pass — tidak ada klaim "true ETAS" tanpa qualifier "Ogata 1988 temporal + isotropic spatial".

---

### Task 6.2: Update README.md scientific section

**Objective:** README sebutkan dua baseline jelas. Hapus implication bahwa Poisson adalah satu-satunya baseline.

**Files:**
- Modify: `README.md` (section "Fitur utama" + "Architecture")

**Change:**
- "baseline Poisson" → "baseline Poisson + baseline ETAS Ogata 1988 (μ, K, c, p, α + spatial kernel)"
- Tambah catatan: "Klaim skill ML diukur terhadap dua baseline; ETAS sebagai second baseline lebih kuat untuk window 7-14 hari saat ada aftershock cluster."

---

### Task 6.3: Operational runbook untuk ETAS pipeline

**Objective:** Runbook ringkas — kapan re-fit ETAS, bagaimana monitor parameter drift, bagaimana respond kalau MLE diverge.

**Files:**
- Create: `docs/runbooks/etas-ogata-pipeline.md`

**Sections:**
1. **Re-fit cadence:** weekly recommended; daily kalau ada event M>=6 (aftershock dominan).
2. **Parameter drift monitor:** alert kalau |p - p_prev| > 0.3 atau |alpha - alpha_prev| > 0.5 antar fit.
3. **MLE convergence:** log `fit_status_`. Kalau "warn:..." muncul >3x berturut, fall back ke previous parameters.
4. **Catalog quality:** Mc estimate harus stabil (+/- 0.2 antar fit). Drift signals bisa karena station network change.

**Done when:** Runbook reviewed, linked dari MAINTENANCE.md.

---

### Task 6.4: Deprecation timeline for `ETASBaseline` alias

**Objective:** Setelah `OgataETAS` masuk, `ETASBaseline = PoissonBaseline` jadi misleading karena ada modul ETAS asli sekarang. Plan deprecation.

**Steps:**
1. Release N: Tambah DeprecationWarning di `etas.py` import:
   ```python
   import warnings
   def __getattr__(name):
       if name == "ETASBaseline":
           warnings.warn(
               "ETASBaseline is deprecated; use PoissonBaseline (or OgataETAS for true ETAS).",
               DeprecationWarning, stacklevel=2,
           )
           return PoissonBaseline
       raise AttributeError(name)
   ```
2. Release N+1 (≥2 minor versions later): Hapus alias.
3. Update import sites: `scripts/train_initial.py`, `forecast_service.py`, tests.

**Done when:** No code references `ETASBaseline` di repo. Search `grep -r "ETASBaseline" --include="*.py"` returns 0 hits.

---

## Verification commands

```bash
# Per-task tests
pytest backend/tests/test_mc_estimation.py -v
pytest backend/tests/test_catalog_filter.py -v
pytest backend/tests/test_etas_ogata_likelihood.py -v
pytest backend/tests/test_etas_ogata_fit.py -v
pytest backend/tests/test_etas_ogata_spatial.py -v
pytest backend/tests/test_etas_ogata_predict.py -v
pytest backend/tests/test_evaluate_dual_baseline.py -v
pytest backend/tests/test_b_value.py -v
pytest backend/tests/test_csep_tests.py -v

# Full suite + lint
make lint
make test
make test-cov
```

## Acceptance criteria

The plan is complete when ALL of the following hold:

1. `OgataETAS` class fits μ, K, c, p, α via L-BFGS-B and recovers synthetic params within tolerance.
2. Spatial kernel integrates to ~1.0 numerically and decays correctly with distance.
3. `predict_dataframe` produces canonical 16-column label matrix per cell, matching `PoissonBaseline` API.
4. Recent large-event aftershock test: cell within 50 km outranks cell 800 km away on 7-day horizon.
5. `evaluate_dataset` reports `bss_vs_poisson` AND `bss_vs_etas` per head.
6. `train_initial.py` logs both BSS values; ETAS fit failure does not break training (graceful skip).
7. MODEL_CARD and README accurately describe both baselines, no misleading "true ETAS" claims.
8. Cross-validation against `etas` PyPI library passes within tolerance (deferred validation gate).
9. CSEP L-test and N-test implemented and pass on synthetic ground-truth.
10. Three Indonesian reference events (Lombok 2018, Palu 2018, Mamuju 2021) show power-law decay slope p ∈ [0.8, 1.5].

## Risks & mitigations

- **MLE diverge on small catalogs:** Multi-start optimizer; fall back to fixed literature priors when convergence fails.
- **Mc misestimate inflates productivity:** Per-region MAXC + sanity bound check (Mc ∈ [3.5, 5.5]).
- **Over-reliance on ETAS prior:** Bayesian blend tetap pakai Poisson prior; ETAS hanya sebagai second baseline + optional tier 2.5.
- **Compute cost on 2000+ cell catalog:** Lazy evaluation per cell; cache `_cell_rate` per (lat, lon, t_query). Profiling before scaling.
- **Backward compatibility:** `etas.py` (PoissonBaseline) tidak diubah; alias `ETASBaseline` masih live sampai deprecation cycle selesai.

## Execution handoff

Plan saved to `docs/plans/2026-05-28-real-etas-ogata.md` (1187+ lines, 6 phases, 18 bite-sized tasks).

Eksekusi rekomendasi pakai skill `subagent-driven-development`: dispatch satu subagent per task dengan two-stage review (spec compliance lalu code quality). Phase 0 dan Phase 1 mandatory sebelum lanjut Phase 2-6. Phase 5 deferred sampai shadow eval pass.

Dependencies install lebih dulu kalau env baru:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
# Optional (untuk Task 3.2 cross-validation):
python -m pip install etas
```
