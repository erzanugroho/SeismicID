# API Reference

REST API SeismicID — endpoint, parameter, schema response. Semua endpoint dilayani oleh FastAPI di `backend/app/main.py`, mounted di prefix `/api`. Health endpoint juga tersedia tanpa prefix untuk probe k8s/docker.

Base URL lokal: `http://127.0.0.1:8000`

## Auth model

- **Public** (read): tidak butuh token. Semua `GET` di tabel di bawah.
- **Admin** (write/trigger): butuh header `X-Admin-Token: <token>` yang cocok dengan `Settings.admin_token`. Dipakai oleh: `POST /api/forecasts/run`, `POST /api/events/ingest`, `POST /api/areas/bootstrap`, `POST /api/scheduler/trigger/{job_name}`.
- Admin token disimpan lokal di `~/.hermes/secrets/seismicid_admin_token.txt` (Railway production: env `ADMIN_TOKEN`).

## Quick reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET  | `/health` (and `/api/health`) | public | Liveness probe (uptime, version, env) |
| GET  | `/api/health/readiness` | public | Readiness: DB, model, forecast freshness, scheduler |
| GET  | `/api/areas` | public | Grid cell list with labels (lazy bootstrap) |
| POST | `/api/areas/bootstrap` | admin | Repopulate `area_labels` |
| GET  | `/api/events` | public | Recent realtime events (last N days) |
| POST | `/api/events/ingest` | admin | Trigger ingest from USGS/BMKG |
| GET  | `/api/forecasts/latest` | public | All cells, given (horizon, threshold) |
| GET  | `/api/forecasts/top` | public | Top-N cells by probability |
| GET  | `/api/forecasts/top-risk` | public | Alias of `/top` (limit-based) |
| GET  | `/api/forecasts/area/{cell_id}` | public | All 16 heads for one cell |
| GET  | `/api/forecasts/top-clusters` | public | Top-N subregion clusters |
| GET  | `/api/forecasts/clusters-latest` | public | All clusters, filterable by macro/province |
| GET  | `/api/forecasts/status` | public | Forecast freshness + active model |
| GET  | `/api/forecast/status` | public | Singular alias (legacy) |
| POST | `/api/forecasts/run` | admin | Trigger forecast recompute |
| GET  | `/api/model/metadata` | public | Active model metadata payload |
| GET  | `/api/model/evaluation` | public | All stored evaluation results |
| GET  | `/api/scheduler/runs` | public | Scheduler audit log (last N runs) |
| POST | `/api/scheduler/trigger/{job_name}` | admin | Trigger named scheduler job |

## Canonical input domains

Hardcoded di `backend/app/features/labels.py`:

```
HORIZONS    = (7, 14, 30, 60)              days
THRESHOLDS  = (4.5, 5.0, 5.5, 6.0)         magnitude floor
SOURCES     = {usgs, emsc, bmkg}           catalog adapters
REGIONS     = Sumatera | Jawa | BaliNusa | Kalimantan | Sulawesi | MalukuPapua
SORT_BY     = top3_mean | max | any_cell | mean
```

Endpoint validator mereject (HTTP 400) horizon/threshold/sort_by di luar domain ini.

> **Note**: `/api/events?source=` saat ini hanya menerima `usgs|bmkg` (regex validator belum di-update untuk EMSC). Untuk filter EMSC, ambil `source=null` dan filter client-side, atau tunggu patch berikutnya.

## Endpoints (detail)

### Health

```
GET /health
GET /api/health
```

Liveness probe. Tidak menyentuh DB.

```json
{
  "status": "ok",
  "name": "SeismicID",
  "version": "0.x.y",
  "env": "production",
  "role": "combined",
  "uptime_seconds": 1234.56
}
```

```
GET /api/health/readiness
```

Readiness probe: cek DB, active model, forecast freshness (≤ `forecast_fallback_hours * 2`, min 6h), model dir ada.

```json
{
  "status": "ready" | "degraded",
  "ok": true,
  "checks": {
    "db": {"ok": true, "event_count": 57664},
    "active_model": {"ok": true, "version": "v20260524_141104_5d3d40"},
    "scheduler_last_success": {"job_name": "...", "started_at": "...", "finished_at": "...", "status": "success"},
    "forecast": {"ok": true, "age_hours": 0.5, "mode": "ml_ensemble", "model_version": "v20260524_141104_5d3d40"},
    "model_dir": {"ok": true, "path": "/.../models"}
  }
}
```

### Areas

```
GET /api/areas?province=&region_macro=
```

Grid cell list (~3000 cells, 0.5°×0.5°). Lazy-bootstraps `area_labels` jika kosong.

```json
{
  "count": 3120,
  "items": [
    {
      "cell_id": "ID-01-NN",
      "lat": -6.25, "lon": 106.75,
      "lat_min": -6.5, "lat_max": -6.0,
      "lon_min": 106.5, "lon_max": 107.0,
      "province": "DKI Jakarta",
      "subregion": "Jakarta Pusat",
      "full_label": "DKI Jakarta — Jakarta Pusat",
      "is_offshore": 0,
      "region_macro": "Jawa",
      "nearest_fault_km": 12.3,
      "fault_type": "transform",
      "fault_slip_rate": 5.2,
      "slab_depth_km": 95.0
    }
  ]
}
```

```
POST /api/areas/bootstrap?force=false   [admin]
```

Repopulate `area_labels`. `force=true` untuk overwrite.

### Events

```
GET /api/events?days=7&min_mag=&source=&limit=500
```

- `days`: 1–3650
- `min_mag`: 0–10
- `source`: `usgs` | `bmkg` (EMSC: pending fix)
- `limit`: 1–5000

```json
{
  "count": 42,
  "items": [
    {
      "event_id": "us7000pxxx",
      "time": "2026-05-25T14:32:11Z",
      "lat": -1.45, "lon": 122.34,
      "depth": 35.2, "magnitude": 5.1, "mag_type": "Mw",
      "source": "usgs",
      "place": "Sulawesi"
    }
  ]
}
```

```
POST /api/events/ingest?fetch_usgs=true&fetch_bmkg=true&lookback_hours=24   [admin]
```

### Forecasts

All forecast endpoints accept `horizon` ∈ HORIZONS dan `threshold` ∈ THRESHOLDS (default dari settings). Return 400 untuk nilai di luar domain.

```
GET /api/forecasts/latest?horizon=30&threshold=5.0&min_probability=&limit=
```

```json
{
  "horizon_days": 30,
  "mag_threshold": 5.0,
  "count": 3120,
  "items": [
    {
      "cell_id": "ID-01-NN",
      "lat": -6.25, "lon": 106.75,
      "province": "DKI Jakarta",
      "subregion": "Jakarta Pusat",
      "probability": 0.083,
      "model_version": "v20260524_141104_5d3d40",
      "computed_at": "2026-05-25T14:36:09Z"
    }
  ]
}
```

```
GET /api/forecasts/top?n=10&horizon=30&threshold=5.0
GET /api/forecasts/top-risk?limit=10&horizon=30&threshold=5.0   # alias
```

Same payload as `/latest` plus a `sentences[]` array of pre-formatted Indonesian narration:

```json
{
  "...": "...",
  "sentences": [
    "Sulawesi Tengah - Palu, 12.4% probabilitas M≥5.0 dalam 30 hari"
  ]
}
```

```
GET /api/forecasts/area/{cell_id}
```

All 16 heads (4 horizons × 4 thresholds) for one cell. 404 jika cell tidak ada.

```
GET /api/forecasts/top-clusters?n=10&horizon=30&threshold=5.0&sort_by=top3_mean
```

Top-N subregion clusters. `sort_by`:
- `top3_mean` (default) — mean of 3 highest cells in the cluster
- `max` — single worst cell
- `any_cell` — `1 - Π(1 - pᵢ)`; probability ≥1 cell exceeds threshold
- `mean` — mean of all cells (rarely correct)

```
GET /api/forecasts/clusters-latest?horizon=&threshold=&sort_by=&region_macro=&province=&min_probability=&limit=
```

All clusters with optional filters.

```
GET /api/forecasts/status
GET /api/forecast/status   # singular alias
```

```json
{
  "forecast_last_computed_at": "2026-05-25T14:36:09Z",
  "forecast_mode": "ml_ensemble",
  "forecast_model_version": "v20260524_141104_5d3d40",
  "n_cells": 3120
}
```

```
POST /api/forecasts/run?force_demo=false   [admin]
```

Triggers `run_forecast()`. Returns mode, n_cells, model_version, archive path.

### Model

```
GET /api/model/metadata
```

Returns full metadata JSON of active model (training_date, dataset_size, feature_count, feature_list, per-head metrics, feature_importance, calibrators). `{"version": null, "status": "no_active_model"}` jika belum ada.

```
GET /api/model/evaluation
```

```json
{
  "count": 16,
  "items": [
    {
      "model_version": "v20260524_141104_5d3d40",
      "eval_type": "reliability",
      "computed_at": "2026-05-25T15:00:00Z",
      "payload_json": { "...": "head-specific metrics" }
    }
  ]
}
```

### Scheduler

```
GET /api/scheduler/runs?limit=50
```

```json
{
  "count": 50,
  "items": [
    {
      "id": 12345,
      "job_name": "forecast_recompute",
      "started_at": "2026-05-25T14:35:00Z",
      "finished_at": "2026-05-25T14:36:09Z",
      "status": "success",
      "items_processed": 3120,
      "metadata_json": "{...}"
    }
  ]
}
```

```
POST /api/scheduler/trigger/{job_name}   [admin]
```

Job names registered di `backend/app/scheduler/runner.py` (cek `register_jobs()`): `forecast_recompute`, `realtime_ingest`, `weekly_retrain`, dst.

## Error responses

FastAPI standard:

```json
{ "detail": "horizon must be one of [7, 14, 30, 60]" }
```

| Status | When |
|---|---|
| 400 | Invalid horizon/threshold/sort_by, parameter out of range |
| 401 | Missing/wrong `X-Admin-Token` header (admin endpoints) |
| 404 | `/forecasts/area/{cell_id}` for unknown cell |
| 409 | Admin job already running (via `guarded_admin_job`) |
| 500 | Unhandled (logged with `logger.error`, payload obscure) |

## CORS

`CORSMiddleware` mengizinkan origins dari `Settings.cors_origins`. Default lokal: `http://localhost:8000`, `http://127.0.0.1:8000`. Production: tambah domain frontend di `.env`.

## Versioning

Tidak ada API versioning di path (no `/v1/`). Breaking change → bump `app_version` di `pyproject.toml`, dokumentasikan di `CHANGELOG.md`. Pertimbangkan `/v2/` jika respons schema berubah inkompatibel.
