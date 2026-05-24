# Changelog

Semua perubahan signifikan pada SeismicID didokumentasikan di sini.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versi: [Semantic Versioning](https://semver.org/).

## [Unreleased] — 2026-05-22 (Probability Audit Sprint)

### Added

- **Physics features** (`nearest_fault_km`, `fault_type_int`, `fault_slip_rate`, `slab_depth_km`) masuk ke `FeatureBuilder`.
  Backward compat: model lama (20 fitur) tetap jalan karena `predict_ensemble` memilih kolom via `hm.feature_names`.
- **`enforce_probability_monotonicity()`** di `backend/app/ml/ensemble.py`.
  Menjamin dua properti:
  - Horizon: `P(60d) ≥ P(30d) ≥ P(14d) ≥ P(7d)` (threshold sama)
  - Threshold: `P(M≥4.5) ≥ P(M≥5.0) ≥ P(M≥5.5) ≥ P(M≥6.0)` (horizon sama)
- **`list_forecast_archive_runs(day)`** di catalog — mengembalikan semua file archive per run untuk satu hari.
- **`PROB_FLOOR`** constant (`1e-6`) di ensemble — diexport ke frontend sebagai batas "data minim".
- **`formatPct()` indikator "data minim"** — probabilitas di floor `1e-6` tidak lagi ditampilkan seolah-olah aman.
- **Empty state di Performance page** — jika belum ada `evaluation_results`, tampil pesan eksplisit, bukan angka demo palsu.
- **Legend note di peta** — "Skala warna absolut… bandingkan ranking relatif, jangan baca angka sebagai jaminan."
- **Test file baru**: `backend/tests/test_probability_audit.py` (596 baris, 30+ test cases).

### Fixed

- **P0: Poisson baseline selalu 0** (`backend/app/ml/etas.py`)
  - `PoissonBaseline.fit()` sekarang auto-assign `cell_id` dari lat/lon jika event belum punya `cell_id`.
  - `global_rates` smoothed (Laplace-style) untuk cell tanpa histori lokal.
  - Root cause: `read_historical_events()` tidak memberi kolom `cell_id`.

- **P0: Bayesian blend collapse ke floor** (`backend/app/ml/ensemble.py`)
  - Prior Poisson yang missing/zero/non-finite → fallback ke calibrated estimate.
  - `cell_event_counts=None` dan `{}` diperlakukan simetris (10 sampel efektif default).
  - Cell yang absen dari dict mendapat minimal 1 sampel efektif.

- **P1: Posthoc recalibration menghantam semua head** (`backend/app/ml/ensemble.py`)
  - Sebelumnya: jika SATU head `IdentityCalibrator`, SEMUA head (termasuk Platt/Isotonic/Beta) ikut terkompresi.
  - Sekarang: kompresi hanya untuk head dengan `IdentityCalibrator`.

- **P1: Forecast archive overwrite** (`backend/app/data/catalog.py`)
  - Layout lama: satu file `YYYY-MM-DD.parquet` di-overwrite tiap run → hindsight leakage.
  - Layout baru: `forecast_archive/<UTC-date>/<HHMMSSZ>_<model_version>.parquet` — immutable per run.
  - Kolom metadata: `forecast_run_id`, `issued_at_utc`, `model_version`.
  - Backward compat: `read_forecast_archive()` tetap bisa baca file legacy.

- **P1: CSEP L/S test one-sided** (`backend/app/ml/evaluate.py`)
  - Dulu: `pass` jika `quantile ≥ 0.025` → hasil "terlalu bagus" lolos.
  - Sekarang: two-sided, `pass` jika `0.025 ≤ quantile ≤ 0.975`.

- **P2: `label_column_name` fragile untuk non-tenths** (`backend/app/features/labels.py`)
  - Threshold 5.25 → dulu ambigu, sekarang `m525` (hundredths).
  - Threshold kanonik tetap backward compat: 4.5→`m45`, 5.0→`m50`, 5.5→`m55`, 6.0→`m60`.

### Changed

- **Performance page**: angka demo palsu (ROC=0.78, Brier=0.07, dll) dihapus. Empty state eksplisit sekarang.
- **Cells page**: "120 grid cell" hardcoded → dynamic dari API `data.count`.
- **Peta legend**: tambah note "warna absolut sensitif horizon/threshold".
- **README & MODEL_CARD**: wording lebih konservatif — "experimental relative-risk ranking", "bukan early warning resmi".
- **Forecast service**: `enforce_probability_monotonicity()` dipanggil sebelum persist untuk SEMUA mode (ML/Poisson/demo).

### Not Yet Done

- Retrain model dengan 24 fitur (physics features — model aktif masih 20 fitur).
- Fault/slab data dari PUSGEN/Slab2.0 shapefile asli (saat ini approximation).
- Recompute `current_forecasts` DB dengan kode baru.

---