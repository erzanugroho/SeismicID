"""Tests for probability audit fixes (P0/P1).

Covers:
- Issue 1: PoissonBaseline must derive cell_id from lat/lon when missing.
- Issue 2: Bayesian blend must not collapse to ~0 when prior is missing/zero or
  when a cell is absent from cell_event_counts.
- Issue 3: Monotonicity helper enforces longer-horizon and lower-threshold
  monotonicity over the 16 multi-output forecast columns.
- Issue 4: Forecast archive is immutable per run (one file per call) and uses UTC.
- Issue 5: CSEP L/S tests are two-sided (pass requires 0.025 ≤ q ≤ 0.975).
- Issue 6: ``label_column_name`` is robust for unusual thresholds (e.g. 5.25)
  while remaining backward compatible for the canonical thresholds.
- Issue 7: Physics-informed static features (nearest_fault_km, fault_type_int,
  slab_depth_km, fault_slip_rate) are exposed by the feature builder.
- Issue 8: Post-hoc recalibration is applied per-head only when that head's
  calibrator is the IdentityCalibrator (no global side-effects).
- Issue 9-12 are UI/docs fixes; we add a smoke test for monotonicity wiring
  inside the forecast service so persisted forecasts respect the constraints.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from backend.app.core.grid import generate_grid
from backend.app.features.labels import (
    HORIZONS,
    THRESHOLDS,
    all_label_columns,
    label_column_name,
)

# ---------------------------------------------------------------------------
# Issue 1: Poisson baseline must work for events lacking ``cell_id``
# ---------------------------------------------------------------------------


def test_poisson_baseline_assigns_cell_id_from_lat_lon() -> None:
    """``PoissonBaseline.fit`` must populate per-cell rates when events only
    carry lat/lon (no ``cell_id`` column)."""
    from backend.app.ml.etas import PoissonBaseline

    rng = np.random.default_rng(0)
    events = pd.DataFrame(
        [
            {
                "event_id": f"e{i}",
                "time": datetime(2022, 1, 1, tzinfo=UTC)
                + timedelta(days=int(rng.integers(0, 700))),
                "lat": -0.9 + rng.normal(0, 0.05),
                "lon": 119.87 + rng.normal(0, 0.05),
                "magnitude": 5.0 + rng.exponential(0.3),
                "depth": 10.0,
            }
            for i in range(40)
        ]
    )
    baseline = PoissonBaseline()
    baseline.fit(
        events,
        observation_start=datetime(2022, 1, 1, tzinfo=UTC),
        observation_end=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert len(baseline.rates) > 0


def test_poisson_predictions_for_cells_produces_nonzero() -> None:
    """``forecast_service._poisson_predictions_for_cells`` must return at
    least some non-zero predictions when active events exist for the area."""
    from backend.app.services.forecast_service import _poisson_predictions_for_cells

    cells = [c for c in generate_grid() if -2 <= c.lat <= 2 and 119 <= c.lon <= 121]
    cell_ids = [c.cell_id for c in cells]

    rng = np.random.default_rng(0)
    events = pd.DataFrame(
        [
            {
                "event_id": f"e{i}",
                "time": datetime(2022, 1, 1, tzinfo=UTC)
                + timedelta(days=int(rng.integers(0, 700))),
                "lat": -0.9 + rng.normal(0, 0.05),
                "lon": 119.87 + rng.normal(0, 0.05),
                "magnitude": 5.0 + rng.exponential(0.3),
                "depth": 10.0,
            }
            for i in range(40)
        ]
    )

    df = _poisson_predictions_for_cells(events, cell_ids)
    arr = df.drop(columns=["cell_id"]).to_numpy(dtype=float)
    assert (arr > 0).any(), "Expected at least one non-zero Poisson prediction"


def test_poisson_baseline_global_smoothing_for_unknown_cells() -> None:
    """Cells without per-cell history should still receive a small global
    smoothed rate so they don't all sit at zero probability."""
    from backend.app.ml.etas import PoissonBaseline

    rng = np.random.default_rng(0)
    events = pd.DataFrame(
        [
            {
                "event_id": f"e{i}",
                "time": datetime(2022, 1, 1, tzinfo=UTC)
                + timedelta(days=int(rng.integers(0, 700))),
                "lat": -0.9 + rng.normal(0, 0.05),
                "lon": 119.87 + rng.normal(0, 0.05),
                "magnitude": 4.6 + rng.exponential(0.2),
                "depth": 10.0,
            }
            for i in range(60)
        ]
    )
    baseline = PoissonBaseline()
    baseline.fit(
        events,
        observation_start=datetime(2022, 1, 1, tzinfo=UTC),
        observation_end=datetime(2024, 1, 1, tzinfo=UTC),
    )
    # Probe a cell that almost certainly has no history
    p = baseline.predict_probability("Cm999_p9999", 30, 5.0)
    assert p > 0.0, "Unknown cells should fall back to a positive global smoothed rate"


# ---------------------------------------------------------------------------
# Issue 2: Bayesian blend must not collapse to ~0
# ---------------------------------------------------------------------------


class _FakeBooster:
    def __init__(self, p_value: float) -> None:
        self.p_value = p_value

    def predict_proba(self, x):
        n = len(x)
        return np.column_stack([1 - np.full(n, self.p_value), np.full(n, self.p_value)])


def _make_head(head_name: str, calibrator, p_value: float = 0.4):
    from backend.app.ml.train import HeadModel

    return HeadModel(
        head=head_name,
        booster_xgb=_FakeBooster(p_value),
        booster_lgbm=_FakeBooster(p_value),
        calibrator=calibrator,
        feature_names=["x0"],
        metrics={},
    )


def test_bayesian_blend_does_not_collapse_when_prior_missing() -> None:
    from backend.app.ml.calibration import IsotonicCalibrator
    from backend.app.ml.ensemble import predict_ensemble

    iso = IsotonicCalibrator()
    iso.iso.fit(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    head = "label_h30_m50"
    hm = _make_head(head, iso, p_value=0.10)

    features = pd.DataFrame({"cell_id": ["c1", "c2"], "x0": [0.0, 0.0]})

    out = predict_ensemble(
        {head: hm},
        features,
        cell_ids=["c1", "c2"],
        poisson_predictions=None,
        cell_event_counts=None,
    )
    # Calibrated value is ~0.10/(weights). Should not collapse to the 1e-6 floor.
    probs = out[head].to_numpy()
    assert (probs > 1e-3).all(), f"Bayesian blend collapsed: {probs}"


def test_bayesian_blend_unknown_cell_with_dict_evidence_does_not_collapse() -> None:
    """Cell missing from ``cell_event_counts`` must not be treated as n=0
    when there is no Poisson prior — that combination forced posterior=0."""
    from backend.app.ml.calibration import IsotonicCalibrator
    from backend.app.ml.ensemble import predict_ensemble

    iso = IsotonicCalibrator()
    iso.iso.fit(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    head = "label_h30_m50"
    hm = _make_head(head, iso, p_value=0.10)

    features = pd.DataFrame({"cell_id": ["c_known", "c_unknown"], "x0": [0.0, 0.0]})
    out = predict_ensemble(
        {head: hm},
        features,
        cell_ids=["c_known", "c_unknown"],
        poisson_predictions=None,
        cell_event_counts={"c_known": 25},
    )
    probs = out.set_index("cell_id")[head]
    assert probs.loc["c_unknown"] > 1e-3, (
        f"Unknown cell collapsed to floor: {probs.loc['c_unknown']}"
    )


def test_bayesian_blend_uses_prior_when_valid() -> None:
    """When a Poisson prior is present, the posterior should sit between
    the calibrated value and the prior (Bayesian shrinkage works)."""
    from backend.app.ml.calibration import IsotonicCalibrator
    from backend.app.ml.ensemble import predict_ensemble

    iso = IsotonicCalibrator()
    iso.iso.fit(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    head = "label_h30_m50"
    hm = _make_head(head, iso, p_value=0.30)

    features = pd.DataFrame({"cell_id": ["c1"], "x0": [0.0]})
    poisson_pred = pd.DataFrame({"cell_id": ["c1"], head: [0.05]})

    out = predict_ensemble(
        {head: hm},
        features,
        cell_ids=["c1"],
        poisson_predictions=poisson_pred,
        cell_event_counts={"c1": 5},
    )
    p = out.loc[0, head]
    # Calibrated (~0.225 due to weighted ensemble) and prior=0.05; posterior in between.
    assert 0.05 - 1e-6 <= p <= 0.30 + 1e-6


# ---------------------------------------------------------------------------
# Issue 3: monotonicity helper
# ---------------------------------------------------------------------------


def test_enforce_monotonicity_horizon() -> None:
    from backend.app.ml.ensemble import enforce_probability_monotonicity

    df = pd.DataFrame(
        [
            {
                "cell_id": "c1",
                "label_h7_m50": 0.05,
                "label_h14_m50": 0.10,
                "label_h30_m50": 0.08,
                "label_h60_m50": 0.06,
            }
        ]
    )
    out = enforce_probability_monotonicity(df)
    cols = [label_column_name(h, 5.0) for h in sorted(HORIZONS)]
    vals = out.loc[0, cols].tolist()
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_enforce_monotonicity_threshold() -> None:
    from backend.app.ml.ensemble import enforce_probability_monotonicity

    df = pd.DataFrame(
        [
            {
                "cell_id": "c1",
                "label_h30_m45": 0.05,
                "label_h30_m50": 0.10,
                "label_h30_m55": 0.06,
                "label_h30_m60": 0.04,
            }
        ]
    )
    out = enforce_probability_monotonicity(df)
    cols = [label_column_name(30, t) for t in sorted(THRESHOLDS)]
    vals = out.loc[0, cols].tolist()
    # Lower threshold → higher probability
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))


def test_enforce_monotonicity_random_grid() -> None:
    """Random violations across all 16 columns are corrected."""
    from backend.app.ml.ensemble import enforce_probability_monotonicity

    rng = np.random.default_rng(123)
    rows = []
    for cid in ["c1", "c2", "c3"]:
        row: dict[str, object] = {"cell_id": cid}
        for h in HORIZONS:
            for t in THRESHOLDS:
                row[label_column_name(h, t)] = float(rng.uniform(0, 0.5))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = enforce_probability_monotonicity(df)

    for _, row in out.iterrows():
        for t in THRESHOLDS:
            seq = [row[label_column_name(h, t)] for h in sorted(HORIZONS)]
            assert all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))
        for h in HORIZONS:
            seq = [row[label_column_name(h, t)] for t in sorted(THRESHOLDS)]
            assert all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))


def test_enforce_monotonicity_preserves_extremes() -> None:
    """Monotonicity correction should never lower a value that already
    dominates its constraints."""
    from backend.app.ml.ensemble import enforce_probability_monotonicity

    df = pd.DataFrame(
        [
            {
                "cell_id": "c1",
                "label_h7_m50": 0.10,
                "label_h14_m50": 0.20,
                "label_h30_m50": 0.30,
                "label_h60_m50": 0.40,
                "label_h30_m45": 0.40,
                "label_h30_m55": 0.20,
                "label_h30_m60": 0.10,
            }
        ]
    )
    out = enforce_probability_monotonicity(df)
    # All originally well-ordered values stay the same (cummax/cummin idempotent)
    for col in df.columns:
        if col == "cell_id":
            continue
        assert float(out.loc[0, col]) == float(df.loc[0, col])


# ---------------------------------------------------------------------------
# Issue 4: forecast archive immutability + UTC
# ---------------------------------------------------------------------------


def test_archive_forecast_immutable_per_run() -> None:
    """Two ``archive_forecast`` calls in the same UTC day should produce two
    distinct files (immutable per run)."""
    from backend.app.data.catalog import (
        archive_forecast,
        list_forecast_archive_runs,
    )

    df1 = pd.DataFrame([{"cell_id": "c1", "horizon_days": 30, "probability": 0.05}])
    df2 = pd.DataFrame([{"cell_id": "c1", "horizon_days": 30, "probability": 0.07}])
    day = date(2025, 1, 15)

    p1 = archive_forecast(
        df1,
        day=day,
        model_version="v1",
        issued_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC),
    )
    p2 = archive_forecast(
        df2,
        day=day,
        model_version="v2",
        issued_at=datetime(2025, 1, 15, 11, 0, 0, tzinfo=UTC),
    )
    assert p1 != p2
    assert p1.exists() and p2.exists()

    runs = list_forecast_archive_runs(day)
    assert len(runs) == 2


def test_archive_forecast_writes_run_metadata_columns() -> None:
    from backend.app.data.catalog import archive_forecast

    df = pd.DataFrame([{"cell_id": "c1", "probability": 0.05}])
    issued = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
    p = archive_forecast(df, model_version="v_test_42", issued_at=issued)
    out = pd.read_parquet(p)
    assert "forecast_run_id" in out.columns
    assert "issued_at_utc" in out.columns
    assert "model_version" in out.columns
    assert (out["model_version"] == "v_test_42").all()
    assert (out["issued_at_utc"] == issued.isoformat()).all()


def test_archive_forecast_uses_utc_date_by_default() -> None:
    from backend.app.data.catalog import archive_forecast, list_forecast_archive_days

    archive_forecast(pd.DataFrame([{"cell_id": "c1", "p": 0.05}]))
    today_utc = datetime.now(UTC).date()
    assert today_utc in list_forecast_archive_days()


def test_archive_forecast_backward_compat_legacy_file() -> None:
    """If a legacy ``YYYY-MM-DD.parquet`` exists, ``read_forecast_archive``
    must still return it for that day."""
    from backend.app.config import get_settings
    from backend.app.data.catalog import read_forecast_archive

    settings = get_settings()
    archive_dir = settings.parquet_path / "forecast_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    day = date(2024, 6, 1)
    legacy = archive_dir / f"{day.isoformat()}.parquet"
    legacy_df = pd.DataFrame([{"cell_id": "legacy", "p": 0.01}])
    legacy_df.to_parquet(legacy, index=False)

    out = read_forecast_archive(day)
    assert len(out) == 1
    assert out.iloc[0]["cell_id"] == "legacy"


# ---------------------------------------------------------------------------
# Issue 5: CSEP L/S two-sided
# ---------------------------------------------------------------------------


def test_csep_l_test_high_quantile_fails_two_sided() -> None:
    """If the observed log-likelihood is at the very top of the simulation
    distribution (q>0.975), a two-sided test must report ``fail``."""
    from backend.app.ml.evaluate import run_l_test

    n = 100
    p = np.full(n, 0.001)
    y = np.zeros(n, dtype=int)
    out = run_l_test(y, p, n_sim=1000)
    # All-zeros + tiny p means observed LL is the maximum achievable (~0).
    # Most simulations will draw 0 ones; quantile sits at 1.0.
    assert out["quantile"] >= 0.97
    assert out["status"] == "fail"


def test_csep_l_test_passes_in_normal_range() -> None:
    """When predictions are well-calibrated, the L-test passes."""
    from backend.app.ml.evaluate import run_l_test

    rng = np.random.default_rng(7)
    n = 500
    p = rng.uniform(0.05, 0.3, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    out = run_l_test(y, p, n_sim=400)
    assert out["status"] == "pass"


def test_csep_s_test_high_quantile_fails_two_sided() -> None:
    """S-test should also fail when the spatial log-likelihood is at the top
    of the simulated distribution."""
    from backend.app.ml.evaluate import run_s_test

    # Predictions concentrate weight on a single cell, and observed events
    # all land there too. The observed spatial LL is the maximum.
    n = 50
    y = np.zeros(n, dtype=int)
    y[0] = 5
    p = np.full(n, 0.01)
    p[0] = 1.0
    out = run_s_test(y, p, n_sim=500)
    assert out["status"] in ("pass", "fail")
    if out["quantile"] > 0.975:
        assert out["status"] == "fail"


# ---------------------------------------------------------------------------
# Issue 6: label_column_name robustness
# ---------------------------------------------------------------------------


def test_label_column_name_canonical_thresholds_unchanged() -> None:
    assert label_column_name(7, 4.5) == "label_h7_m45"
    assert label_column_name(14, 5.0) == "label_h14_m50"
    assert label_column_name(30, 5.5) == "label_h30_m55"
    assert label_column_name(60, 6.0) == "label_h60_m60"


def test_label_column_name_full_grid_count() -> None:
    cols = all_label_columns()
    assert len(cols) == len(HORIZONS) * len(THRESHOLDS)
    assert len(set(cols)) == len(cols)


def test_label_column_name_decimal_threshold_unambiguous() -> None:
    """Decimal thresholds (e.g. 5.25) must not collide with neighbouring
    canonical names like ``m52`` or ``m53``."""
    a = label_column_name(30, 5.2)
    b = label_column_name(30, 5.25)
    c = label_column_name(30, 5.3)
    assert len({a, b, c}) == 3


def test_label_column_name_handles_floating_point_drift() -> None:
    """``threshold = 0.1*45`` should still produce ``m45``."""
    drifted = 0.1 * 45  # FP drift
    assert label_column_name(7, drifted) == "label_h7_m45"


# ---------------------------------------------------------------------------
# Issue 7: Physics features in builder
# ---------------------------------------------------------------------------


def test_feature_columns_includes_physics_static() -> None:
    from backend.app.features.builder import feature_columns

    cols = feature_columns()
    for name in ("nearest_fault_km", "fault_type_int", "slab_depth_km", "fault_slip_rate"):
        assert name in cols, f"feature_columns missing {name!r}"


def test_build_features_includes_physics_per_cell() -> None:
    from backend.app.features.builder import build_features_for_snapshots

    cells = [c for c in generate_grid() if -2 <= c.lat <= 2 and 119 <= c.lon <= 121]
    snap = datetime(2024, 1, 1, tzinfo=UTC)
    df = build_features_for_snapshots(
        pd.DataFrame(columns=["event_id", "time", "lat", "lon", "magnitude", "depth"]),
        [snap],
        cells=cells,
    )
    assert "nearest_fault_km" in df.columns
    assert df["nearest_fault_km"].notna().all()
    assert df["fault_type_int"].between(0, 4).all()
    assert df["slab_depth_km"].notna().all()


# ---------------------------------------------------------------------------
# Issue 8: Per-head posthoc recalibration trigger
# ---------------------------------------------------------------------------


def test_posthoc_recalibration_only_for_identity_head() -> None:
    """Heads with a fitted calibrator (Isotonic/Beta/Platt) must NOT be
    re-touched by the post-hoc compressor, even when sibling heads use
    IdentityCalibrator."""
    from backend.app.ml.calibration import IdentityCalibrator, IsotonicCalibrator
    from backend.app.ml.ensemble import predict_ensemble

    # Isotonic calibrator that maps p → p (identity-like fit).
    iso = IsotonicCalibrator()
    iso.iso.fit(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))

    head_a = _make_head("label_h30_m50", IdentityCalibrator(), p_value=0.40)
    head_b = _make_head("label_h30_m45", iso, p_value=0.40)

    features = pd.DataFrame({"cell_id": ["c1"], "x0": [0.0]})
    out = predict_ensemble(
        {"label_h30_m50": head_a, "label_h30_m45": head_b},
        features,
        cell_ids=["c1"],
        poisson_predictions=None,
        cell_event_counts=None,
    )
    p_identity = float(out.loc[0, "label_h30_m50"])
    p_isotonic = float(out.loc[0, "label_h30_m45"])

    # The Identity head should be compressed toward the realistic base rate
    # (≪ 0.40). The Isotonic head should be left close to its calibrated
    # ensemble output (much closer to 0.4 than to 0.005).
    assert p_identity < 0.10, f"Identity head not recalibrated; got {p_identity}"
    assert p_isotonic > 2 * p_identity, (
        f"Isotonic head got bulk-compressed too: {p_isotonic} vs {p_identity}"
    )


# ---------------------------------------------------------------------------
# Forecast service smoke: persisted forecasts respect monotonicity
# ---------------------------------------------------------------------------


def test_run_forecast_demo_seed_persists_monotonic_forecasts() -> None:
    from backend.app.db.sqlite import get_connection
    from backend.app.services.forecast_service import run_forecast

    out = run_forecast(force_demo=True)
    assert out["mode"] == "demo_seed"
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT cell_id, horizon_days, mag_threshold, probability
               FROM current_forecasts"""
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    assert not df.empty

    # Sample 50 random cells for fast monotonicity verification
    rng = np.random.default_rng(0)
    sample_cells = rng.choice(df["cell_id"].unique(), size=min(50, df["cell_id"].nunique()), replace=False)
    for cid in sample_cells:
        sub = df[df["cell_id"] == cid]
        # Horizon monotone for each threshold
        for t in THRESHOLDS:
            seq = (
                sub[sub["mag_threshold"] == t]
                .sort_values("horizon_days")["probability"]
                .tolist()
            )
            assert all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1)), (
                f"horizon monotonicity violated for cell {cid}, threshold {t}: {seq}"
            )
        # Threshold monotone for each horizon (lower threshold ≥ higher)
        for h in HORIZONS:
            seq = (
                sub[sub["horizon_days"] == h]
                .sort_values("mag_threshold")["probability"]
                .tolist()
            )
            assert all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1)), (
                f"threshold monotonicity violated for cell {cid}, horizon {h}: {seq}"
            )
