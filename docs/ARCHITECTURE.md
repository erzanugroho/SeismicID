# Architecture: Probability Pipeline

Dokumen ini menjelaskan alur perhitungan probabilitas dari data mentah hingga angka yang ditampilkan di UI.
Digunakan sebagai referensi untuk developer baru, reviewer, dan maintainer.

## Overview

```
┌────────────────┐  ┌──────────────────┐    ┌─────────────────────────────────────┐
│  Data Sources   │─▶│  Feature Builder  │───▶│  Ensemble Predictor                 │
│  USGS/EMSC/BMKG │  │  backend/app/     │    │  backend/app/ml/ensemble.py          │
│                 │  │  features/builder │    │                                     │
└────────────────┘  └──────────────────┘    │  ┌────────┐ ┌────────┐ ┌──────────┐ │
                                            │  │ XGBoost│ │LightGBM│ │ Poisson  │ │
                                            │  │  p_xgb │ │ p_lgbm │ │ p_poiss  │ │
                                            │  └────┬───┘ └───┬────┘ └────┬─────┘ │
                                            │       └─────────┼──────────┘       │
                                            │           ┌─────▼──────┐           │
                                            │           │  Weighted   │           │
                                            │           │  Average    │           │
                                            │           │  (0.4/0.4/  │           │
                                            │           │   0.2)      │           │
                                            │           └─────┬──────┘           │
                                            └─────────────────┼──────────────────┘
                                                              │
                    ┌─────────────────────────────────────────┼──────────────┐
                    │  Per-Head Calibration                    │              │
                    │  backend/app/ml/calibration.py           ▼              │
                    │                                                       │
                    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
                    │  │ IdentityCal  │  │ Platt/Isoton │  │ BetaCal  │ │
                    │  │ (no calib)   │  │              │  │              │ │
                    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
                    │         │                 │                 │         │
                    │         ▼                 │                 │         │
                    │  ┌──────────────┐         │                 │         │
                    │  │  Posthoc     │         │                 │         │
                    │  │  Compression │         │                 │         │
                    │  └──────┬───────┘         │                 │         │
                    │         │                 │                 │         │
                    └─────────┼─────────────────┼─────────────────┼─────────┘
                              └─────────┬───────┴────────┬────────┘
                                        ▼                ▼
                              ┌────────────────────────────────────┐
                              │  Bayesian Blend                    │
                              │  posterior = (n_evidence × calib + │
                              │              alpha × prior) /      │
                              │              (n_evidence + alpha)  │
                              │                                    │
                              │  * Only mixes when prior > 0 &     │
                              │    finite. Otherwise keeps calib.  │
                              └───────────────┬────────────────────┘
                                              │
                    ┌─────────────────────────▼──────────────────────────┐
                    │  Monotonicity Enforcement                           │
                    │  enforce_probability_monotonicity()                 │
                    │                                                     │
                    │  Pass 1: horizon → P(60d) ≥ P(30d) ≥ P(14d) ≥ P(7d)│
                    │  Pass 2: threshold → P(4.5) ≥ P(5.0) ≥ P(5.5) ≥ P(6.0)│
                    └─────────────────────────┬──────────────────────────┘
                                              │
                    ┌─────────────────────────▼──────────────────────────┐
                    │  Persist                                              │
                    │                                                       │
                    │  ┌─────────────────────┐  ┌─────────────────────────┐│
                    │  │ SQLite              │  │ Forecast Archive         ││
                    │  │ current_forecasts   │  │ data/parquet/            ││
                    │  │ (live, overwritten) │  │ forecast_archive/        ││
                    │  │                     │  │ YYYY-MM-DD/              ││
                    │  │ Untuk API / UI      │  │ HHMMSSZ_version.parquet ││
                    │  └─────────────────────┘  │ (immutable, prospective) ││
                    │                          └─────────────────────────┘│
                    └──────────────────────────────────────────────────────┘
```

## Key files and responsibilities

| File | Responsibility |
|---|---|
| `backend/app/main.py` | FastAPI app factory + lifespan (migrate DB, start scheduler for `worker`/`combined` roles, mount `/api/*` routers + static frontend). |
| `backend/app/services/forecast_service.py` | Orchestrator. Pilih mode (ML/Poisson/demo), bangun fitur, panggil ensemble, persist, archive. |
| `backend/app/features/builder.py` | Bangun feature vector per cell per snapshot. 20 temporal/spatial + 4 physics static. |
| `backend/app/features/labels.py` | Nama kolom label (16 target × 4 horizon × 4 threshold). `label_column_name()` + `THRESHOLDS`/`HORIZONS`. |
| `backend/app/data/sources/{usgs,emsc,bmkg}.py` | Catalog ingest adapters. EMSC depths arrive negative (sign-convention quirk) and are flipped on ingest; legacy rows fixed in `data/parquet/historical_events.parquet` on 2026-05-25. |
| `backend/app/data/catalog.py` | Read/write Parquet: training set, forecast archive (immutable per-run), list runs. |
| `backend/app/ml/ensemble.py` | `predict_ensemble()` — weighted average → kalibrasi → Bayesian blend. `enforce_probability_monotonicity()`. |
| `backend/app/ml/etas.py` | `PoissonBaseline` — rate per cell per threshold. Auto-assign `cell_id` dari lat/lon. `global_rates` fallback. |
| `backend/app/ml/evaluate.py` | CSEP L/N/S tests (two-sided). Brier, ROC-AUC, reliability. |
| `backend/app/ml/calibration.py` | `IdentityCalibrator`, `PlattCalibrator`, `IsotonicCalibrator`, `BetaCalibrator`. |
| `backend/app/ml/posthoc_calibration.py` | Posthoc compression untuk head dengan `IdentityCalibrator`. Per-head, bukan global. |
| `backend/app/data/catalog.py` | Read/write Parquet: training set, forecast archive (immutable per-run), list runs. |
| `backend/app/db/sqlite.py` + `db/schema.sql` | SQLite: `current_forecasts`, `area_labels`, `realtime_events`, `scheduler_runs`, `model_metadata`, `evaluation_results`, `app_metadata`. WAL mode, idempotent migrations. |
| `backend/tests/test_probability_audit.py` | 30+ test cases audit P0/P1: Poisson, Bayesian, monotonicity, archive, CSEP, labels, physics. |

## Three forecast modes

`forecast_service.run_forecast()` picks one of three modes:

### 1. ML Ensemble (`mode = "ml_ensemble"`)
- **Trigger**: trained model exists + recent events available.
- **Flow**: build features → `predict_ensemble()` → monotonicity → persist.

### 2. Poisson Baseline (`mode = "poisson_baseline"`)
- **Trigger**: events available but no trained model (or model prediction fails).
- **Flow**: `PoissonBaseline.fit()` → `predict_dataframe()` → monotonicity → persist.

### 3. Demo Seed (`mode = "demo_seed"`)
- **Trigger**: no events at all (empty DB) or `force_demo=True`.
- **Flow**: `_demo_seed_predictions()` → physics-aware placeholders → monotonicity → persist.
- **Important**: ini placeholder, bukan ML output. Performance page tidak menampilkan metrik palsu.

## Probability ensemble detail

Untuk setiap dari 16 head (4 horizon × 4 threshold):

```
p_ensemble = (0.4 * p_xgb + 0.4 * p_lgbm + 0.2 * p_poisson) / 1.0
```

Kemudian:

```
IF calibrator is IdentityCalibrator:
    p = posthoc_compress(p, base_rate)
ELSE:
    p = calibrator.predict_proba(p)

IF poisson_prior > 0 AND finite:
    posterior = (n_evidence * p + alpha * prior) / (n_evidence + alpha)
ELSE:
    posterior = p

p_final = clip(posterior, 1e-6, 1 - 1e-6)
```

Di mana:
- `n_evidence` = jumlah event historis M≥4.5 di cell tersebut (min 1)
- `alpha` = `bayesian_alpha` = 5.0 (pseudo-count toward prior)
- `prior` = prediksi Poisson untuk cell/threshold/horizon tersebut

## Design decisions (post-audit)

1. **Poisson `cell_id` auto-assign**: `read_historical_events()` tidak menambah `cell_id`. `PoissonBaseline.fit()` menanganinya dengan `_assign_cell_id()` dari lat/lon via grid kanonik. Ini menghindari duplikasi logika di semua caller.

2. **Symmetric `cell_event_counts` handling**: `None` vs `{}` keduanya → 10 sampel efektif. Dict tak-kosong dengan key missing → 1 sampel efektif. Tidak ada path yang collapse ke nol.

3. **Per-head posthoc**: `isinstance(hm.calibrator, IdentityCalibrator)` diperiksa PER HEAD. Head dengan Platt/Isotonic tidak disentuh.

4. **Prior-gated Bayesian blend**: `prior > 0 AND isfinite(prior)` sebelum blend. Kalau prior invalid, calibrated estimate langsung dipakai — tidak dipaksa ke floor.

5. **Monotonicity as post-processing**: Dua pass cumulative max (horizon ascending, threshold descending). Rank-preserving, diterapkan untuk SEMUA mode.

6. **Immutable archive**: Setiap run menulis file baru. `read_forecast_archive()` baca file terbaru. Backward compat: legacy `<date>.parquet` tetap bisa dibaca.

7. **Two-sided CSEP**: `0.025 ≤ quantile ≤ 0.975` — "terlalu bagus" sekarang fail.

8. **Physics features static**: 4 fitur physics dihitung sekali per cell, di-cache di `physics_per_cell` dict. Biaya tambahan di feature builder ~0.

9. **Backward compat model**: `predict_ensemble` memilih kolom via `hm.feature_names`. Model lama (20 fitur) tetap jalan. Model baru (24 fitur) otomatis akan pakai physics features saat retrain.

10. **UI floor indicator**: `PROB_FLOOR = 1e-6` dibagi ke frontend via `api.js`. `formatPct()` menampilkan "data minim" untuk nilai di floor. Ini mencegah misinterpretasi "aman" dari angka sangat kecil.

## Test structure

```
backend/tests/
├── test_probability_audit.py   ← Audit-specific: Poisson, Bayesian, monotonicity, archive, CSEP, labels, physics
├── test_features.py            ← Feature builder tests (termasuk physics features)
├── test_ensemble.py            ← Ensemble blending + calibration
├── test_etas.py                ← Poisson baseline
├── test_evaluate.py            ← CSEP + metrics
├── test_forecast_service.py    ← run_forecast integrasi
└── ...                         ← Lainnya (API, DB, grid, dll)
```

112 tests, ~48 detik.

## Adding new features or thresholds

1. Feature baru: tambahkan di `builder.py` → `all_feature_names()` → model akan pakai saat retrain.
2. Threshold baru: tambahkan di `labels.py` → `THRESHOLDS` tuple. `label_column_name()` akan auto-handle naming.
3. Horizon baru: tambahkan di `labels.py` → `HORIZONS` tuple.
4. Jangan lupa update test expectation.