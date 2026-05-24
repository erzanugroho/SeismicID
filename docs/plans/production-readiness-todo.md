# SeismicID Production Readiness TODO

Dokumen ini merangkum pekerjaan lanjutan untuk menaikkan SeismicID dari **public demo** menjadi **production-ready public service**.

Status saat dokumen dibuat:

- Railway service: `SeismicID`
- Public URL: <https://seismicid-production.up.railway.app>
- Environment: `production`
- Persistent volume: `/data`
- Historical data: USGS Indonesia 2000–2024 sudah dibootstrap
- Active model: ML ensemble `v20260521_124306_84b5ad`
- Forecast cache: aktif, `ml_ensemble`
- Scheduler: internal APScheduler aktif di web service, `scheduler_tick` tiap 10 menit

---

## P0 — Wajib sebelum publik serius

### 1. Fix runtime dependency Railway/Docker

**Masalah**

Runtime Railway saat ini pernah membutuhkan `LD_LIBRARY_PATH` hardcoded ke Nix store agar dependency native seperti `numpy`, `pandas`, `lightgbm`, dan `xgboost` bisa import dengan benar.

Ini brittle karena path Nix store bisa berubah saat rebuild.

**Task**

- Tambahkan `nixpacks.toml` atau custom `docker/Dockerfile` production.
- Pastikan library native tersedia secara eksplisit:
  - `gcc` / `libstdc++`
  - `zlib`
  - dependency runtime untuk `numpy`, `pandas`, `lightgbm`, `xgboost`
- Hilangkan kebutuhan env `LD_LIBRARY_PATH` hardcoded dari Railway.
- Verifikasi di container Railway:

```bash
python -c "import numpy, pandas, lightgbm, xgboost; print('ok')"
```

- Redeploy Railway.
- Test endpoint:

```bash
curl https://seismicid-production.up.railway.app/health
curl https://seismicid-production.up.railway.app/api/forecast/status
```

**Done when**

- App jalan tanpa `LD_LIBRARY_PATH` manual.
- Scheduler tick tetap sukses setelah redeploy.
- Forecast tetap `ml_ensemble`.

---

### 2. Tambahkan CI GitHub Actions

**Masalah**

Repo sudah punya tests, pre-commit, Makefile, dan pyproject, tetapi belum ada `.github/workflows/ci.yml`.

**Task**

- Buat workflow GitHub Actions:
  - checkout
  - setup Python
  - install deps
  - run lint
  - run tests
- Minimal command:

```bash
make lint
make test
```

- Jika `make lint` terlalu strict, pisahkan dulu:

```bash
ruff check .
pytest
```

- Tambahkan badge CI di README.
- Update README metrics supaya tidak hardcoded, misalnya jangan lagi menyebut angka stale seperti `76 tests passing` jika sudah tidak akurat.

**Done when**

- GitHub Actions hijau di main branch.
- README tidak punya angka test/coverage stale.

---

### 3. Tambahkan public disclaimer di UI

**Masalah**

User publik bisa mengira sistem ini adalah prediksi pasti atau early warning resmi.

**Task**

- Tambahkan banner di UI:

```text
Eksperimental — bukan sistem peringatan dini. Gunakan BMKG untuk informasi resmi.
```

- Tambahkan tooltip/info:
  - probabilitas rendah bukan berarti aman
  - probabilitas tinggi bukan berarti pasti terjadi
  - model probabilistik dan masih experimental
  - bukan pengganti BMKG/otoritas resmi
- Tambahkan link ke BMKG resmi.
- Pastikan terlihat jelas di mobile.

**Done when**

- Disclaimer terlihat di desktop dan mobile.
- Tidak mengganggu kontrol peta.
- Pengguna publik tidak diberi kesan bahwa ini sistem peringatan dini resmi.

---

### 4. Hardening admin endpoints

**Masalah**

Endpoint admin bisa menjalankan job berat seperti forecast recompute dan retrain.

**Task**

- Audit endpoint:
  - `POST /api/forecasts/run`
  - `POST /api/scheduler/trigger/{job_name}`
- Pastikan semua endpoint berat butuh `ADMIN_TOKEN`.
- Tambahkan cooldown/rate limit sederhana untuk job berat.
- Tambahkan guard agar `retrain` tidak mudah dipicu dari web publik.
- Tambahkan response jelas jika job sedang berjalan.
- Jangan pernah log token.

**Done when**

- Endpoint admin aman dari akses tanpa token.
- Job berat tidak bisa dipicu paralel berulang.
- Error/blocked response jelas untuk operator.

---

## P1 — Arsitektur produksi

### 5. Pisahkan web service dan worker service

**Masalah**

Scheduler sekarang berjalan di process FastAPI web service. Ini cukup untuk demo, tetapi kurang ideal untuk produksi karena job fetch/forecast/training bisa mengganggu latency publik.

**Task**

- Tambahkan role runtime:

```env
APP_ROLE=web
APP_ROLE=worker
```

- Web role:
  - serve UI/API
  - scheduler disabled
- Worker role:
  - menjalankan scheduler/fetch/forecast
  - tidak expose public UI/API, atau expose health minimal
- Update `backend/app/main.py` agar scheduler hanya start saat role worker atau saat internal scheduler enabled.
- Buat start command:

```bash
# Web
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT

# Worker
python scripts/worker.py
```

- Buat `scripts/worker.py` yang menjalankan APScheduler foreground.

**Done when**

- Web tetap responsif.
- Worker menjalankan tick otomatis.
- `/api/forecast/status` tetap update.

---

### 6. Migrasi SQLite cache ke Postgres

**Masalah**

SQLite + Railway Volume cukup untuk demo kecil, tetapi kurang ideal untuk traffic publik karena forecast write bisa menyentuh puluhan ribu row dan berpotensi lock saat request publik membaca.

**Task**

- Tambahkan Postgres support untuk tabel:
  - `events`
  - `current_forecasts`
  - `scheduler_runs`
  - `model_registry`
  - `evaluation_results`
- Buat abstraction DB layer:
  - SQLite untuk local/dev
  - Postgres untuk production
- Tambahkan env:

```env
DATABASE_URL=...
DB_BACKEND=postgres
```

- Migrasi query `INSERT OR REPLACE` ke Postgres `ON CONFLICT`.
- Tambahkan migration script.
- Add Postgres service di Railway.
- Update README deployment.

**Done when**

- Production pakai Postgres.
- SQLite tetap bisa dipakai local.
- Forecast write tidak block public read secara signifikan.

---

### 7. Forecast archive dan prospective evaluation

**Masalah**

Evaluasi retrospective bisa leak. Untuk klaim ilmiah, perlu prospective evaluation: forecast disimpan sebelum event terjadi, lalu dievaluasi setelah horizon lewat.

**Task**

- Pastikan setiap forecast disimpan immutable:

```text
data/parquet/forecast_archive/YYYY-MM-DD.parquet
```

- Tambahkan job evaluasi harian/mingguan:
  - ambil forecast lama yang horizon-nya sudah lewat
  - compare dengan event aktual
  - hitung metrik:
    - Information Gain vs Poisson baseline
    - CSEP N-test
    - CSEP L-test
    - CSEP S-test
    - Brier Skill Score
    - reliability
- Simpan hasil ke `evaluation_results` dengan `eval_type=prospective`.
- Tambahkan endpoint:

```text
GET /api/model/prospective-evaluation
```

**Done when**

- Ada metrik prospective yang terus bertambah seiring waktu.
- README menjadikan Information Gain vs Poisson baseline sebagai metrik utama.

---

## P1 — Data ilmiah/geologi

### 8. Replace hardcoded fault DB dengan GEM Active Faults Database

**Masalah**

`fault_db.py` masih hardcoded dengan jumlah patahan sangat terbatas dan polyline kasar. Ini menghasilkan fitur `nearest_fault_km` yang kurang defensible dan bisa punya artefak spasial.

**Task**

- Download GEM Active Faults Database.
- Simpan di:

```text
data/geo/gem_active_faults/
```

- Tambahkan loader shapefile/GeoJSON.
- Gunakan hook existing seperti `has_real_pusgen()` atau rename menjadi hook yang lebih general.
- Compute distance-to-fault dari geometri real.
- Tambahkan cache spatial index.
- Update README attribution dan lisensi CC-BY.

**Done when**

- Feature `nearest_fault_km` pakai data real jika tersedia.
- Fallback hardcoded hanya untuk demo/dev.
- README menjelaskan sumber data fault.

---

### 9. Integrasi Slab2.0 grid

**Masalah**

Slab depth analytical approximation kurang defensible untuk subduksi Indonesia.

**Task**

- Download Slab2.0 Hayes et al. grid untuk region Indonesia.
- Simpan di:

```text
data/geo/slab2/
```

- Tambahkan loader/interpolator grid.
- Replace analytical approximation bila grid tersedia.
- Tambahkan README attribution.
- Tambahkan test interpolasi sederhana.

**Done when**

- `slab_depth_km` berasal dari Slab2.0 bila tersedia.
- Approximation hanya fallback.

---

### 10. Rename/clarify ETAS baseline

**Masalah**

Modul/class ETAS saat ini sebenarnya homogeneous Poisson approximation, bukan ETAS proper Ogata dengan fitting parameter via MLE.

**Task**

- Rename class ke salah satu:

```python
PoissonBaseline
SimplifiedEtasBaseline
```

- Update docstring, README, dan UI wording.
- Pastikan tidak disebut ETAS proper kecuali sudah implement Ogata MLE.
- Tambahkan milestone proper ETAS:
  - fit μ, K, c, p, α
  - Omori decay
  - likelihood-based fitting
  - information gain vs baseline

**Done when**

- Tidak ada klaim misleading “ETAS proper”.
- Baseline dijelaskan jujur sebagai Poisson/simplified baseline.

---

## P2 — API/UI performance

### 11. Tambahkan endpoint top-risk/filter

**Masalah**

UI saat ini mengambil dan memproses semua cell forecast. Untuk mobile/traffic publik, lebih baik ada endpoint server-side filtering.

**Task**

- Tambahkan endpoint:

```text
GET /api/forecasts/top-risk?horizon=30&threshold=5.0&limit=10
GET /api/forecasts/latest?min_probability=0.001
```

- Server-side filter probability.
- Tambahkan index DB untuk:
  - `horizon_days`
  - `mag_threshold`
  - `probability`
  - `computed_at`
- Update UI agar Top 10 tidak perlu scan semua cells client-side.

**Done when**

- Top 10 cepat.
- Mobile render lebih ringan.

---

### 12. Improve map rendering

**Masalah**

Semua cell polygon dirender di Leaflet. Masih oke untuk demo, tapi bisa berat di mobile dan traffic publik.

**Task**

- Tambahkan salah satu atau kombinasi:
  - render only visible bounds
  - server-side threshold
  - simplify polygon
  - vector tile / `leaflet.vectorgrid` later
- Tambahkan debounce refresh UI.
- Tambahkan cache response.
- Tambahkan ETag atau `Cache-Control`.

**Done when**

- Mobile smooth.
- Refresh tidak terasa berat.
- Payload publik lebih kecil untuk use case umum.

---

### 13. Rework color scale

**Masalah**

Absolute probability gempa per grid cell kecil bisa terlihat flat hijau, meskipun relatif ada area yang lebih tinggi risikonya.

**Task**

- Tambahkan mode warna:
  - Absolute probability
  - Percentile / relative risk
- Default publik bisa menggunakan percentile/relative risk, tetapi popup tetap menampilkan probability asli.
- Legend jelas, misalnya:

```text
Top 1%
Top 5%
Top 10%
Normal
```

**Done when**

- Map informatif tanpa melebih-lebihkan risiko.
- Popup tetap transparan dengan angka probabilitas asli.

---

## P2 — Observability & ops

### 14. Tambahkan health/readiness lebih lengkap

**Task**

- Tambahkan endpoint:

```text
GET /api/health/readiness
```

- Check:
  - DB reachable
  - latest forecast age
  - scheduler last success
  - active model exists
  - event count
- Return degraded kalau stale.

**Done when**

- Monitoring bisa tahu app “hidup tapi stale”.
- `/health` tetap liveness ringan, `/api/health/readiness` untuk readiness/quality.

---

### 15. Tambahkan alert stale forecast

**Task**

- Buat script/cron monitoring:
  - cek `/api/forecast/status`
  - alert kalau forecast age > 6 jam
  - alert kalau scheduler error berulang
- Kirim alert ke Telegram/Hermes atau platform ops lain.
- Bisa pakai Hermes cronjob.

**Done when**

- Kalau scheduler mati/stale, ada notifikasi otomatis.

---

### 16. Backup data Railway

**Task**

- Backup selama masih SQLite:
  - `/data/sqlite/gempa.db`
  - `/data/models`
  - `/data/parquet/forecast_archive`
- Jika migrasi Postgres:
  - scheduled `pg_dump`
- Simpan ke storage eksternal.

**Done when**

- Redeploy/kerusakan volume tidak menghapus model/data penting.
- Ada prosedur restore yang terdokumentasi.

---

## P2 — Documentation

### 17. Update README deployment section

**Task**

- Jelaskan production mode:
  - Railway env
  - Volume `/data`
  - scheduler
  - admin token
  - bootstrap/train commands
- Jelaskan demo vs real model lebih tegas.
- Hapus klaim Tailwind jika tidak dipakai konsisten.
- Update test/coverage metrics.
- Tambahkan badge CI setelah workflow ada.

**Done when**

- Reviewer tidak salah paham antara demo seed dan real trained model.
- README tidak berisi klaim stale atau misleading.

---

### 18. Tambahkan model card

**Task**

- Buat:

```text
MODEL_CARD.md
```

- Isi:
  - data source
  - target labels
  - horizons/thresholds
  - limitations
  - intended use
  - not intended use
  - evaluation
  - disclaimer keselamatan

**Done when**

- Publik tahu batasan model.
- Reviewer punya dokumen ringkas untuk menilai klaim model.

---

## Urutan kerja yang disarankan

1. Fix runtime Docker/Nixpacks.
2. Tambah CI GitHub Actions.
3. Tambah disclaimer UI.
4. Hardening admin endpoints.
5. Tambah top-risk endpoint dan color scale percentile.
6. Tambah readiness/observability dan alert stale forecast.
7. Pisahkan worker atau migrasi Postgres.
8. Tambah prospective evaluation.
9. Integrasi GEM Active Faults Database.
10. Integrasi Slab2.0.
11. Update README dan buat `MODEL_CARD.md`.

---

## Quick verification commands

```bash
curl https://seismicid-production.up.railway.app/health
curl https://seismicid-production.up.railway.app/api/forecast/status
curl https://seismicid-production.up.railway.app/api/model/metadata
curl https://seismicid-production.up.railway.app/api/model/evaluation
curl "https://seismicid-production.up.railway.app/api/scheduler/runs?limit=10"
```

Expected production indicators:

```text
/health env = production
/api/forecast/status forecast_mode = ml_ensemble
/api/model/metadata version != null
/api/scheduler/runs latest scheduler_tick status = success
```
