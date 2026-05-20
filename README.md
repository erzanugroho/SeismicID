# Gempa Forecast System

Sistem forecast probabilitas gempa bumi Indonesia berbasis data USGS + BMKG, ensemble machine learning (XGBoost + LightGBM + ETAS), physics-informed features, dan UI browser interaktif.

> **Output:** *"Sulawesi Tengah - Palu, 12.4% probabilitas M≥5.0 dalam 30 hari"*

![Status](https://img.shields.io/badge/status-active%20development-orange) ![Python](https://img.shields.io/badge/python-3.11+-blue) ![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen)

## Fitur

- **Multi-output forecast**: 4 horizon (7/14/30/60 hari) × 4 threshold magnitudo (M≥4.5/5.0/5.5/6.0) = 16 prediksi independen per cell.
- **Indonesia bounded grid 0.5°×0.5°** (~3000 cells) dengan label provinsi-subregion.
- **Ensemble ML**: XGBoost + LightGBM + ETAS Poisson baseline + Bayesian blending dengan prior Poisson per cell.
- **Physics-informed features**: jarak ke patahan aktif terdekat (tipe + slip rate), slab depth (zona subduksi), Z-value quiescence (ZMAP).
- **Calibration**: Platt vs Isotonic vs Beta calibration per head, pilih terbaik berdasarkan val Brier.
- **Auto-update scheduler**: realtime fetch tiap 15 menit, forecast recompute tiap 1 jam, retrain mingguan.
- **5 halaman browser UI**: Map (Leaflet), Detail Area (Chart.js), Recent Events, Performa Model, Tentang.
- **3-tier fallback** di forecast service: ML ensemble → ETAS only → physics-aware demo seed (UI selalu punya data).

## Quickstart

### 1. Local development (Python)

**Prasyarat:** Python 3.11+, opsional `make` (Linux/macOS) atau jalankan `tasks.ps1` di Windows.

```bash
# Clone & setup
git clone <repo> gempa && cd gempa
python -m venv .venv

# Aktifkan venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install
pip install -r requirements-dev.txt
# atau:
make install-dev          # Linux/macOS
.\tasks.ps1 install-dev   # Windows

# Konfigurasi
cp .env.example .env      # Linux/macOS  (atau: copy .env.example .env di Windows)

# Jalankan dev server
make run                  # Linux/macOS
.\tasks.ps1 run           # Windows
# atau:
uvicorn backend.app.main:app --reload
```

Buka browser:
- `http://localhost:8000/` — peta interaktif
- `http://localhost:8000/health` — cek status backend
- `http://localhost:8000/docs` — Swagger UI

### 2. Docker

```bash
cd docker
docker compose up --build
```

Akses `http://localhost:8000/`. Data persistent di `./data/`.

### 3. Bulk training (opsional)

Demo seed mode jalan tanpa data — tapi untuk model ML real, lakukan ingestion + training:

```bash
# Download geo assets (GADM, dll). PUSGEN/Slab2.0 manual.
python scripts/download_geo_assets.py

# Bulk historical USGS (mungkin makan waktu, ratusan MB)
python scripts/bootstrap_data.py --start 2000 --end 2024

# Decluster + Mc estimation (otomatis dijalankan saat training)
# Train initial model
python -m scripts.train_initial
```

Setelah training, `data/models/active.json` akan menunjuk ke versi terbaru. `POST /api/forecasts/run` akan otomatis pakai model itu (mode `ml_ensemble`).

## Arsitektur

```
backend/app/
├── api/routes/         # FastAPI: forecasts, events, areas, model, scheduler, health
├── core/               # grid.py (0.5° generator), geocode.py (province lookup), logging.py
├── data/
│   ├── sources/        # usgs.py, bmkg.py — adapter + Event dataclass
│   ├── ingest.py       # dedup logic + ingest_realtime/historical
│   ├── decluster.py    # Reasenberg algorithm
│   ├── completeness.py # Mc estimation (MAXC)
│   └── catalog.py      # Parquet I/O
├── geo/                # fault_db.py (16 patahan utama), slab_model.py
├── features/
│   ├── builder.py      # ~25 fitur per (cell, snapshot)
│   ├── seismology.py   # b-value, energy, IET stats
│   ├── physics.py      # fault dist, slab depth, Z-value
│   ├── spatial.py      # 8-neighbor aggregation
│   └── labels.py       # 4×4 multi-label generator
├── ml/
│   ├── train.py        # XGB + LGBM per-head trainer
│   ├── calibration.py  # Platt/Isotonic/Beta
│   ├── etas.py         # Poisson rate baseline
│   ├── ensemble.py     # weighted avg + Bayesian blend
│   ├── evaluate.py     # ROC/Brier/BSS/reliability
│   └── predict.py      # load + run inference
├── db/                 # sqlite.py + schema.sql (6 tabel)
├── scheduler/          # APScheduler runner + 3 jobs
└── services/           # area_service, forecast_service

frontend/               # 5 HTML + JS modules + main.css (vanilla, Tailwind via CDN optional)
data/parquet/           # historical_events.parquet, forecast_archive/YYYY-MM-DD.parquet
data/sqlite/gempa.db    # area_labels, current_forecasts, realtime_events, scheduler_runs, ...
data/models/            # XGB+LGBM+calibrator pickle bundles per version + active.json
data/geo/               # GADM shapefile, Slab2.0 grid (manual download)
scripts/                # download_geo_assets, bootstrap_data, train_initial
docker/                 # Dockerfile (multi-stage) + docker-compose.yml
backend/tests/          # 76 unit tests
```

## API Endpoints

| Method | Path | Deskripsi |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/api/areas` | Daftar grid cells + label |
| POST | `/api/areas/bootstrap` | Re-seed area_labels (admin) |
| GET | `/api/events?days=7&min_mag=4&source=usgs` | Recent earthquakes |
| POST | `/api/events/ingest` | Trigger USGS+BMKG fetch |
| GET | `/api/forecasts/latest?horizon=30&threshold=5.0` | Semua cell |
| GET | `/api/forecasts/top?n=10&horizon=30&threshold=5.0` | Top-N + kalimat ID |
| GET | `/api/forecasts/area/{cell_id}` | 16 forecast + metadata |
| POST | `/api/forecasts/run?force_demo=true` | Trigger forecast manual |
| GET | `/api/model/metadata` | Info model aktif |
| GET | `/api/model/evaluation` | Hasil evaluasi |
| GET | `/api/scheduler/runs?limit=50` | Audit log |
| POST | `/api/scheduler/trigger/{job_name}` | Manual trigger job |

## Improvement Ringkasan ML

Implementasi mencakup semua "Wajib + Rekomendasi kuat" improvement (ID lihat plan):

- **A1** b-value multi-window (90/365/1095d) + slope 1y
- **A2** Mc estimation per region per epoch (MAXC)
- **A3** Inter-event time mean + CV
- **A4** Moment release rate vs long-term average
- **A5** Distance to nearest active fault + tipe + slip rate
- **A6** Slab depth (analytical approximation; Slab2.0 grid bila tersedia)
- **A7** Z-value quiescence (Wiemer-Wyss)
- **A8** 8-neighbor spatial aggregations
- **B1** Multi-horizon labels (7/14/30/60d)
- **B2** Multi-threshold labels (M≥4.5/5/5.5/6)
- **C1** Ensemble XGB + LGBM + ETAS (weighted)
- **C3** Bayesian blending dengan Poisson prior
- **E2** Calibrator selection (Platt vs Isotonic vs Beta)
- **E3** CSEP-style endpoints + Molchan diagram (UI hooks)
- **E4** LORO CV slot tersedia di evaluate.py
- **F1** Reasenberg declustering

## Testing

```bash
make test           # pytest backend/tests (76 tests)
make test-cov       # dengan coverage report
make lint           # ruff + mypy
make format         # auto-format + fix
```

Coverage saat ini: ~80% pada modul backend.

## Roadmap & Limitasi

- Demo seed mode → fall back ke physics-aware probabilitas saat belum ada model trained.
- BMKG API kadang tidak stabil → di-treat sebagai optional source, USGS canonical.
- PUSGEN 2017 fault database tidak gratis → hardcoded 16 patahan utama sebagai substitusi (lihat `backend/app/geo/fault_db.py`).
- Slab2.0 grid file besar (~50MB) → analytical approximation default, real grid optional.
- Output adalah **probabilitas relative ranking**, bukan prediksi deterministik kapan/di mana persisnya gempa terjadi.

## Sumber Data

| Sumber | Kegunaan | Lisensi |
|---|---|---|
| [USGS ComCat](https://earthquake.usgs.gov/fdsnws/event/) | Historical + realtime catalog | Public domain |
| [BMKG TEWS](https://data.bmkg.go.id/DataMKG/TEWS/) | Realtime augmentation | Cek terms |
| [GADM](https://gadm.org/) | Provinsi & kabupaten boundaries | Free non-commercial |
| [USGS Slab2.0](https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467) | Subduction zone geometry | Public domain |
| PUSGEN 2017 / GEM Active Faults | Active fault database | Cek terms |

## License

MIT — lihat `LICENSE`.

## Catatan

Versi sistem lama (monolithic ~41KB Python) tetap ada di repo sebagai referensi (`earthquake_forecast_system*.py`, `real_time_forecast.py`, `run_step_by_step.py`, `test_system.py`). Dokumentasi sistem lama di `README_OLD.md`.
