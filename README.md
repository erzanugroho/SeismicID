# SeismicID

SeismicID adalah aplikasi web eksperimental untuk memantau dan memvisualisasikan **probabilitas risiko gempa Indonesia** berdasarkan katalog USGS/BMKG/EMSC, fitur seismologi, model ensemble, dan cache forecast realtime.

> **Disclaimer:** SeismicID bukan sistem peringatan dini. Untuk keputusan keselamatan, gunakan informasi resmi BMKG dan otoritas terkait.

**Live demo:** https://seismicid.erzanugroho.xyz  
**Repository:** https://github.com/erzanugroho/SeismicID

---

## Ringkasan

SeismicID menjawab pertanyaan praktis:

> “Wilayah mana yang relatif lebih berisiko untuk gempa M ≥ 5.0 dalam 30 hari ke depan?”

Output berupa ranking probabilistik per grid 0.5°×0.5°, bukan prediksi pasti kapan/di mana gempa terjadi.

Contoh output:

```text
Lepas Pantai DKI Jakarta - dekat Jakarta
0.18% probabilitas · 30 hari · M ≥ 4.5
Cell: Cm48_p1072 · data sedang
```

---

## Fitur Utama

### Forecast risiko

- Grid Indonesia 0.5°×0.5°.
- 4 horizon waktu:
  - 7 hari
  - 14 hari
  - 30 hari
  - 60 hari
- 4 threshold magnitudo:
  - M ≥ 4.5
  - M ≥ 5.0
  - M ≥ 5.5
  - M ≥ 6.0
- Top risk cell dan cluster.
- Data quality badge:
  - `data kuat`
  - `data sedang`
  - `data terbatas`
  - `data minim`

### UI publik

- Peta risiko interaktif berbasis Leaflet.
- Detail area/cell.
- Recent events.
- Performa model.
- Tentang/cara kerja.
- Mobile UI dengan FAB `kontrol peta`, bottom sheet, dan panel Wilayah Saya.
- Animasi risiko horizon 7d → 14d → 30d → 60d.
- GPS default wilayah user bila izin lokasi diberikan.

### Telegram bot

- Laporan pagi 1x sehari jam 07:00 WIB.
- Alert hanya bila perubahan risiko signifikan.
- Tidak spam tiap forecast jika probabilitas tidak berubah.
- Isi alert: top 5 area risiko tertinggi.

### Admin / scheduler

- Tab scheduler dilindungi password via `ADMIN_TOKEN`.
- Manual trigger job admin.
- Audit log scheduler runs.
- Forecast recompute berjalan dari cache/worker, bukan dari refresh user publik.

---

## Arsitektur Singkat

```text
Frontend static HTML/CSS/JS
        │
        ▼
FastAPI backend
        │
        ├── REST API public cache
        ├── Admin scheduler endpoints
        ├── Telegram alert service
        └── Forecast service
              │
              ├── SQLite runtime DB
              ├── Parquet archive
              ├── ML model bundle
              └── realtime event ingest
```

Struktur utama:

```text
backend/app/
├── api/routes/          FastAPI routes
├── data/                ingest, catalog, sources
├── db/                  SQLite schema + helpers
├── features/            seismology + spatial features
├── geo/                 fault/slab/geographic helpers
├── ml/                  train, calibration, ensemble, predict
├── scheduler/           APScheduler jobs
└── services/            forecast, area, Telegram alerts

frontend/
├── index.html           map UI
├── area.html            detail cell
├── events.html          recent events
├── model.html           model/performance page
├── scheduler.html       protected scheduler admin
└── static/              CSS + JS

docs/                    project documentation
scripts/                 data/bootstrap/worker scripts
```

---

## Quickstart Lokal

### 1. Clone dan setup

```bash
git clone https://github.com/erzanugroho/SeismicID.git
cd SeismicID
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env
```

### 2. Jalankan server

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Buka:

```text
http://localhost:8000/
http://localhost:8000/docs
http://localhost:8000/health
```

### 3. Jalankan scheduler tick lokal

```bash
python scripts/scheduler_tick.py
```

### 4. Jalankan test

```bash
python -m compileall backend
pytest backend/tests
```

Jika pakai Makefile:

```bash
make test
make lint
make format
```

---

## Environment Variables

Lihat `.env.example` untuk template lengkap.

Minimal lokal:

```env
APP_ENV=development
SQLITE_PATH=data/sqlite/gempa_runtime.db
ADMIN_TOKEN=change-me-dev-token
```

Production/Railway penting:

```env
APP_ENV=production
DATA_DIR=/data
PARQUET_DIR=/data/parquet
SQLITE_PATH=/data/sqlite/gempa_runtime.db
MODELS_DIR=/data/models
GEO_DIR=/data/geo
ADMIN_TOKEN=<secret>
```

Telegram optional:

```env
TELEGRAM_BOT_TOKEN=<bot-token>
TELEGRAM_CHAT_ID=<chat-id>
TELEGRAM_ALERT_MIN_PROBABILITY=0.03
TELEGRAM_SIGNIFICANT_ABS_DELTA=0.005
TELEGRAM_SIGNIFICANT_REL_DELTA=0.25
TELEGRAM_DAILY_REPORT_HOUR_UTC=0
```

`TELEGRAM_DAILY_REPORT_HOUR_UTC=0` berarti laporan dikirim jam 07:00 WIB.

---

## Scheduler & Alert Policy

Default worker policy:

- Check realtime events berkala.
- Forecast recompute jika ada event baru dan debounce lewat.
- Fallback recompute jika forecast terlalu lama tidak diperbarui.
- Telegram tidak dikirim setiap forecast.

Telegram dikirim jika:

1. Laporan pagi harian jam 07:00 WIB, atau
2. Perubahan signifikan terjadi:
   - top #1 cell berubah,
   - probabilitas berubah ≥ 0.5 percentage point,
   - perubahan relatif ≥ 25%,
   - risiko melewati ambang alert.

---

## API Ringkas

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/api/forecast/status` | status freshness forecast |
| GET | `/api/events` | recent earthquakes |
| GET | `/api/forecasts/latest` | forecast cell terbaru |
| GET | `/api/forecasts/top` | top risk cell |
| GET | `/api/forecasts/top-clusters` | top risk cluster |
| GET | `/api/forecasts/area/{cell_id}` | detail forecast area |
| POST | `/api/forecasts/run` | recompute forecast admin |
| GET | `/api/scheduler/runs` | audit log scheduler |
| POST | `/api/scheduler/auth` | cek admin token |
| POST | `/api/scheduler/trigger/{job_name}` | manual trigger job admin |

Endpoint admin butuh header:

```http
Authorization: Bearer <ADMIN_TOKEN>
```

---

## Dokumentasi

| Dokumen | Isi |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | arsitektur backend, data flow, design decision |
| [`docs/API.md`](docs/API.md) | reference REST API |
| [`docs/DATA.md`](docs/DATA.md) | struktur data, SQLite, Parquet, archive |
| [`docs/MAINTENANCE.md`](docs/MAINTENANCE.md) | operasi, testing, deploy, troubleshooting |
| [`docs/PROBABILITY_GUIDE.md`](docs/PROBABILITY_GUIDE.md) | cara membaca probabilitas |
| [`docs/TELEGRAM_ALERTS.md`](docs/TELEGRAM_ALERTS.md) | kebijakan bot Telegram |
| [`docs/UI_GUIDE.md`](docs/UI_GUIDE.md) | panduan fitur UI desktop/mobile |
| [`MODEL_CARD.md`](MODEL_CARD.md) | model card dan batasan |
| [`CHANGELOG.md`](CHANGELOG.md) | riwayat perubahan |

---

## Sumber Data

| Sumber | Kegunaan |
|---|---|
| USGS ComCat | historical + realtime catalog |
| BMKG TEWS / realtime page | realtime augmentation Indonesia |
| EMSC FDSN | alternative realtime source |
| GADM | label administrasi |
| Slab2.0 / fault datasets | fitur geologi bila tersedia |

---

## Batasan

- Output adalah ranking risiko probabilistik, bukan prediksi deterministik.
- Probabilitas rendah bukan berarti aman.
- Probabilitas tinggi bukan berarti pasti terjadi.
- Model sangat bergantung pada kualitas katalog event dan fitur geologi.
- Demo/fallback mode bukan pengganti model trained.
- BMKG tetap sumber resmi untuk informasi gempa dan peringatan dini.

---

## Deployment

Production saat ini berjalan di Railway dengan domain:

```text
https://seismicid.erzanugroho.xyz
```

Deployment flow:

```text
git push origin main → Railway auto deploy
```

Gunakan Railway Volume untuk runtime data:

```text
/data/sqlite/gempa_runtime.db
/data/parquet
/data/models
```

---

## License

MIT — lihat [`LICENSE`](LICENSE).
