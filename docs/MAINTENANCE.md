# Maintenance Guide

Panduan operasional untuk maintainer SeismicID — cara menjalankan, test, deploy, troubleshoot, dan meng-upgrade sistem.

## Quick reference

```bash
# Test everything
make test

# Test with coverage
make test-cov

# Lint + type check
make lint

# Format code
make format

# Run dev server
make run
```

## Development environment

```bash
# Clone + setup
git clone <repo-url>
cd gempa
python -m venv .venv-wsl
source .venv-wsl/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
make test
# Expected: 112 passed (48s)
```

### WSL-specific notes

Jika menggunakan WSL:
- Gunakan `.venv-wsl` (bukan `.venv` biasa) untuk kompatibilitas filesystem.
- `pip install -r requirements.txt` — beberapa paket (lightgbm, xgboost) mungkin butuh build tools.
- Data parquet di `data/parquet/` — pastikan I/O dari `/mnt/` tidak bottleneck.

## Testing

### Menjalankan test

```bash
# Semua test
pytest backend/tests -v

# Hanya audit-specific
pytest backend/tests/test_probability_audit.py -v

# Skip test lambat
pytest backend/tests -v -m "not slow"

# Coverage
pytest backend/tests --cov=backend/app --cov-report=term-missing
```

### Struktur test yang penting

| File | Cakupan |
|---|---|
| `test_probability_audit.py` | **P0/P1 audit**: Poisson cell_id, Bayesian collapse, monotonicity, archive immutability, CSEP two-sided, labels, physics features |
| `test_features.py` | Feature builder + physics static features |
| `test_ensemble.py` | Ensemble blending, calibration, Bayesian blend |
| `test_etas.py` | Poisson baseline rate, global smoothing |
| `test_evaluate.py` | CSEP L/N/S tests, metrics |
| `test_forecast_service.py` | `run_forecast` integrasi |

### Menambah test baru

1. Letakkan di `backend/tests/test_<module>.py`
2. Gunakan naming `test_<function_name>_<scenario>`.
3. Import dari `backend.app.*` — root project sudah di `pythonpath`.
4. Jalankan `make lint` + `make test` untuk memastikan tidak ada regresi.

## Linting & type checking

```bash
# Ruff (lint + format)
ruff check backend
ruff format backend --check

# Mypy (type check)
mypy backend/app

# Auto-fix
make format
```

Konfigurasi di `pyproject.toml`:
- Ruff: `E, F, W, I, B, UP, N, C4, SIM`
- Line length: 100
- Mypy: `strict=false`, `ignore_missing_imports=true`

## Training model

### Bootstrap data awal

```bash
python scripts/bootstrap_data.py
```
Ini akan:
1. Fetch data historis dari USGS (dan BMKG jika tersedia).
2. Generate grid cells + area labels.
3. Menyimpan training set dalam format Parquet.

### Training model

```bash
python scripts/train_initial.py
```
Ini akan:
1. Membaca training set dari `data/parquet/`.
2. Training 16 head model (4 horizon × 4 threshold).
3. Memilih calibrator terbaik per head (Platt/Isotonic/Beta/Identity).
4. Menyimpan model bundle + metadata.

### Setelah training baru

Setelah training, verifikasi:
```bash
# 1. Test suite harus tetap hijau
make test

# 2. Jalankan satu forecast
python scripts/run_forecast.py

# 3. Cek ada 16 kolom per cell + monotonicity
python -c "
from backend.app.services.forecast_service import run_forecast
result = run_forecast()
print(result['mode'], result['cells'], result['rows_written'])
"

# 4. Verifikasi archive immutable
ls data/parquet/forecast_archive/$(date +%Y-%m-%d)/
```

## Deployment

### Railway (production)

Service name: `SeismicID`
Project: `hopeful-dedication`
Endpoint: `https://seismicid-production.up.railway.app`

```bash
# Deploy
railway up --service SeismicID

# Cek status
railway status --service SeismicID

# Lihat log
railway logs --service SeismicID
```

### Scheduler

Scheduler berjalan di Railway via `scripts/worker.py`:
- **Fetch events**: tiap 15 menit (USGS + BMKG).
- **Run forecast**: tiap 1 jam, atau trigger-based jika ada event baru.
- **Retrain**: mingguan.

### Monitoring

```bash
# Cek forecast status
curl https://seismicid-production.up.railway.app/api/forecasts/status

# Response:
# {
#   "trigger_mode": "...",
#   "forecast_last_computed_at": "2026-05-22T09:07:00Z",
#   "forecast_mode": "ml_ensemble",
#   "forecast_model_version": "v20260521_124306_84b5ad",
#   "latest_event": {...},
#   "realtime_event_count": 1234,
#   ...
# }
```

Health check endpoint: `GET /api/health`

### Forecast recompute manual

```bash
# Di Railway
railway run --service SeismicID python scripts/run_forecast.py

# Atau langsung di server
python scripts/run_forecast.py
```

Ini akan:
1. Membaca `current_forecasts` DB.
2. Menjalankan `run_forecast()`.
3. Menulis `current_forecasts` baru + immutable archive.

## Database

SQLite di `data/seismicid.db`:

| Tabel | Isi |
|---|---|
| `current_forecasts` | Forecast terbaru per (cell_id, horizon, threshold). Di-overwrite tiap run. Live data untuk API. |
| `area_labels` | Metadata grid cell: lat/lon, provinsi, subregion, fault info. |
| `realtime_events` | Event real-time dari USGS/BMKG. |
| `scheduler_runs` | Log scheduler run. |
| `metadata` | Key-value: `last_forecast_at`, `last_forecast_mode`, dll. |

### Reset database

```bash
rm data/seismicid.db
python scripts/bootstrap_data.py
```

### Backup

```bash
cp data/seismicid.db "data/seismicid_$(date -Iseconds).db"
```

## Forecast archive (immutable)

Lokasi: `data/parquet/forecast_archive/`

Struktur:
```
forecast_archive/
├── 2026-05-22/
│   ├── 090700Z_v20260521_124306_84b5ad.parquet
│   └── 100700Z_v20260521_124306_84b5ad.parquet
├── 2026-05-21/
│   └── 150000Z_v20260521_124306_84b5ad.parquet
└── 2026-05-20.parquet   ← legacy (backward compat)
```

Setiap file memiliki kolom:
- `forecast_run_id` — `<date>T<HHMMSS>Z_<model_version>`
- `issued_at_utc` — ISO-8601 timestamp
- `model_version` — versi model aktif

### Membaca archive

```python
from datetime import date
from backend.app.data.catalog import read_forecast_archive, list_forecast_archive_runs

# Baca forecast terbaru untuk hari ini
df = read_forecast_archive(date.today())

# List semua run untuk hari tertentu
runs = list_forecast_archive_runs(date.today())
for r in runs:
    print(r.name)
```

## Upgrade checklist

Saat meng-upgrade model atau kode:

- [ ] `make lint` — no issues
- [ ] `make test` — all passing (112+ tests)
- [ ] `make test-cov` — coverage tidak turun
- [ ] Retrain model jika ada fitur baru (`python scripts/train_initial.py`)
- [ ] Jalankan forecast baru (`python scripts/run_forecast.py`)
- [ ] Verifikasi monotonicity di DB
- [ ] Cek archive immutable — file baru terbuat
- [ ] Deploy ke Railway (`railway up`)
- [ ] Cek health endpoint (`/api/health`)
- [ ] Cek forecast status (`/api/forecasts/status`)
- [ ] Update `CHANGELOG.md`
- [ ] Update `MODEL_CARD.md` jika ada perubahan signifikan
- [ ] Commit + push

## Adding a new feature

1. **Feature engineering**: tambahkan di `backend/app/features/builder.py`.
   - Static features: hitung sekali di `_physics_features_for_cell()` atau fungsi serupa.
   - Temporal features: tambahkan di per-snapshot loop.
   - Update `all_feature_names()`.

2. **Retrain model**: `python scripts/train_initial.py`.
   - Model akan otomatis memakai semua fitur dari `all_feature_names()`.
   - `predict_ensemble` memilih kolom via `hm.feature_names` → backward compat.

3. **Test**: tambahkan test di `test_features.py` untuk memastikan fitur baru ada di output.

## Adding a new threshold

1. Tambahkan ke `THRESHOLDS` tuple di `backend/app/features/labels.py`.
2. `label_column_name()` akan auto-handle naming (backward compat untuk tenths, hundredths untuk fine-grained).
3. Update `HORIZONS` jika perlu.
4. Retrain model — 16 head → (len(HORIZONS) × len(THRESHOLDS)) head.

## Troubleshooting

### Poisson predictions all zero

**Sebab**: `read_historical_events()` tidak memberi `cell_id`. `PoissonBaseline.fit()` harus auto-assign.

**Fix**: Sudah diimplementasikan pasca-audit. Jika regresi, cek `PoissonBaseline._assign_cell_id()`.

**Cek**:
```bash
pytest backend/tests/test_probability_audit.py::test_poisson_baseline_assigns_cell_id_from_lat_lon -v
```

### Probabilities all ~1e-6 (floor)

**Sebab**: Bayesian blend collapse ke floor. Prior Poisson missing/zero.

**Fix**: Sudah diimplementasikan — prior-gated blend. Jika regresi, cek `predict_ensemble()` → `prior_valid` mask.

**Cek**:
```bash
pytest backend/tests/test_probability_audit.py -k "bayesian" -v
```

### Monotonicity violations

**Sebab**: `enforce_probability_monotonicity()` tidak dipanggil, atau model menghasilkan anomali.

**Cek**:
```bash
pytest backend/tests/test_probability_audit.py -k "monotonic" -v
```

**Fix manual**: `run_forecast()` selalu memanggil `enforce_probability_monotonicity()` sebelum persist. Jika manual, panggil langsung.

### Forecast mode stuck di "demo_seed"

**Sebab**: Tidak ada event di DB, atau model gagal load.

**Cek**:
```bash
curl https://seismicid-production.up.railway.app/api/forecasts/status | jq .forecast_mode
```

**Fix**: `python scripts/bootstrap_data.py` untuk refetch event, lalu `python scripts/run_forecast.py`.

### Archive file lama hilang (overwrite)

**Sebab**: Bug regression — `archive_forecast()` menulis ke path yang sama.

**Fix**: Pasca-audit, setiap run menulis file baru. Jika terjadi overwrite, cek `_path_archive_legacy` tidak dipanggil untuk write.

**Cek**:
```bash
ls data/parquet/forecast_archive/$(date +%Y-%m-%d)/
# Harus ada multiple files dengan timestamp berbeda
```

---

*Untuk detail arsitektur: `docs/ARCHITECTURE.md`.*
*Untuk interpretasi probabilitas: `docs/PROBABILITY_GUIDE.md`.*
*Untuk changelog: `CHANGELOG.md`.*