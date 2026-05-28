# Changelog

Semua perubahan signifikan pada SeismicID didokumentasikan di sini.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versi: [Semantic Versioning](https://semver.org/).

## [Unreleased] — 2026-05-25 (EMSC depth + Active model evaluation)

### Fixed

- **EMSC catalog depth sign convention** (data layer)
  - EMSC FDSN feed mengembalikan `depth` negatif (konvensi geofisika "below sea level"); USGS/BMKG positif.
  - Akibat: `mean_depth_30d` / `std_depth_30d` di feature builder bias ke negatif untuk window yang berisi event EMSC.
  - Fix:
    1. SQLite `realtime_events`: 270 baris EMSC dengan `depth < 0` di-flip ke positif (script ad-hoc, idempotent).
    2. Parquet `data/parquet/historical_events.parquet`: 270 baris EMSC di-flip; backup di `historical_events.parquet.bak` (2.3 MB).
    3. `audit_parquet_depth.py` + `backfill_parquet_depth.py` dihapus setelah verifikasi (idempotent, bisa direkonstruksi dari changelog ini bila perlu).
  - Verifikasi: `min(depth) = 0.00 km`, `mean(depth) = 85.47 km`, no negative depths remaining across 57,664 events.
  - Dampak ke model produksi `v20260524_141104_5d3d40` (16 ROC-AUC heads 0.73–0.91): minimal — EMSC hanya 0.47% catalog (270/57,664), bias depth feature di posisi tengah importance ranking. Effective drift ROC-AUC <0.001.

### Added

- **Evaluation pipeline untuk active model** (`evaluate_active_model.py` runner)
  - End-to-end evaluation script yang baca `models/active.json`, load test split, hitung 16 head metrics (ROC-AUC, Brier, log-loss, reliability bins), tulis ke `evaluation_results` table.
  - Dipakai untuk validasi `v20260524_141104_5d3d40` setelah retrain Minggu 24 Mei.

### Known Issues

- `GET /api/events?source=emsc` ditolak — route validator masih `pattern="^(usgs|bmkg)$"`. Catatan untuk patch berikutnya: tambah `emsc` ke regex di `backend/app/api/routes/events.py`.

### Docs

- ARCHITECTURE.md: tambah baris untuk `main.py`, `data/sources/{usgs,emsc,bmkg}.py`; perbarui SQLite tables list jadi lengkap (7 tabel termasuk `model_metadata`, `evaluation_results`, `app_metadata`); pipeline diagram menampilkan EMSC sebagai sumber.
- README_OLD.md dihapus (legacy, drift indicator).
- API.md baru: full endpoint table + payload schema (lihat `docs/API.md`).
- DATA.md baru: parquet column dictionary, SQLite table inventory, forecast archive layout (lihat `docs/DATA.md`).

---

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