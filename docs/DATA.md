# Data Layer Reference

Inventaris semua data store SeismicID — Parquet, SQLite, model artifacts, forecast archive. Gunakan dokumen ini untuk audit, backfill, migrasi, atau debug data drift.

> Semua path relatif ke project root (`E:\Project\gempa\` di Windows host, `/mnt/e/Project/gempa/` di WSL).
> Path resolution di code: `Settings.parquet_path`, `Settings.sqlite_full_path`, `Settings.models_path` (lihat `backend/app/config.py`).

## Layout pohon

```
data/
├── parquet/
│   ├── .gitkeep
│   ├── historical_events.parquet            ← truth of training (~57k rows)
│   ├── historical_events.parquet.bak        ← pre-2026-05-25 (sebelum EMSC depth fix)
│   ├── declustered_events.parquet           ← optional; dihasilkan declustering job
│   ├── training_features.parquet            ← features+labels join, output trainer
│   └── forecast_archive/
│       ├── 2026-05-24/
│       │   └── HHMMSSZ_<model_version>.parquet  ← per-run, immutable
│       ├── 2026-05-25/
│       └── 2026-05-26/
│       └── 2026-05-NN.parquet               ← legacy single-file layout (backward compat)
├── sqlite/
│   ├── README.md
│   ├── gempa_runtime.db                     ← active DB (WAL mode)
│   ├── gempa_runtime.db-wal
│   ├── gempa_runtime.db-shm
│   ├── gempa.db                             ← old DB, kept for reference (WAL bisa stale)
│   └── *.{broken,preversion,corruptwal}_*  ← backup snapshots
└── logs/
    └── uvicorn.log

models/
├── active.json                              ← {"version": "v..."}
├── metadata_<version>.json                  ← per-head metrics, feature_list, calibrators
├── model_<version>.pkl                      ← XGBoost+LightGBM bundle
└── poisson_<version>.json                   ← Poisson baseline params
```

## Parquet stores

### `historical_events.parquet`

Single source of truth untuk training. Append-only, deduplikasi by `event_id`. Sumber: USGS FDSN, EMSC FDSN, BMKG katalog (sekarang dominasi USGS).

```
shape           : (57,664+, 9)
file size       : ~2.3 MB (snappy)
partitioning    : none (monolithic)
sort order      : append order, not guaranteed
duplicate key   : event_id (dedup on append in catalog.append_historical_events)
```

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `event_id` | string | no | `<source>_<external_id>`, e.g. `usgs_us7000pxxx`, `emsc_1234567`, `bmkg_xxx` |
| `time` | datetime64[ns, UTC] | no | Origin time, always UTC |
| `lat` | float64 | no | Decimal degrees, range Indonesia ±11° |
| `lon` | float64 | no | Decimal degrees, range Indonesia 95–141° |
| `depth` | float64 | yes | km, **always non-negative after 2026-05-25 fix**. EMSC sebelum fix: negatif (sign convention) |
| `magnitude` | float64 | no | Mw/Mb/Ml (mixed) |
| `mag_type` | string | yes | `Mw`, `mb`, `ml`, `mwp`, ... |
| `source` | string | no | One of `usgs`, `emsc`, `bmkg` |
| `place` | string | yes | Free-text location ("114 km SE of Gorontalo, Indonesia") |

**Source breakdown** (per 2026-05-25):
- `usgs`: 57,243 (99.27%)
- `emsc`: 270 (0.47%) ← all depth fixed positive
- `bmkg`: 151 (0.26%)

**Data quirks**:
- BMKG `event_id` sometimes overlaps semantically with USGS (different IDs, same physical event). Declustering doesn't dedupe across sources — consider this if you re-introduce strict cross-source deduplication.
- EMSC depth was historically negative (geophysics convention). Fixed in `historical_events.parquet` on 2026-05-25; backup at `historical_events.parquet.bak`. Ingest adapter `emsc.py` now flips sign on insert.
- `mag_type` mixed: don't assume Mw. Convert before doing energy/moment calculations (`backend/app/features/seismology.py` already handles this).

### `declustered_events.parquet`

Output of declustering job (Reasenberg/Gardner-Knopoff). Same columns as historical + 2 extras:

| Column | Type | Notes |
|---|---|---|
| `is_mainshock` | int (0/1) | Identifies independent events |
| `cluster_id` | string | Group id for aftershock chains |

Used as input to b-value / Poisson rate estimation downstream.

### `training_features.parquet`

Features + labels join. Output of training pipeline, consumed by `train_model.py`.

- Rows: `(cell_id, snapshot_date)` pairs (~3000 cells × N snapshots)
- Columns:
  - `cell_id`, `snapshot` (key)
  - 24 feature columns (lihat `backend/app/features/builder.py` docstring untuk daftar lengkap)
  - 16 label columns (`label_h{horizon}_m{threshold_x10}` untuk semua kombinasi horizon × threshold)

**Feature inventory** (24):

```
Temporal (20):
  event_count_30d, event_count_90d, event_count_365d
  max_mag_30d, max_mag_90d
  mean_depth_30d, std_depth_30d
  log_energy_30d
  moment_release_ratio_30d_vs_365d
  b_value_90d, b_value_365d, b_value_1095d
  b_value_slope_1y
  iet_mean_30d, iet_cv_30d
  time_since_last_M4_days, time_since_last_M5_days
  activity_trend_90d
  neighbor_event_count_30d_mean, neighbor_max_mag_30d_max

Physics static (4):
  nearest_fault_km, fault_type_int, fault_slip_rate, slab_depth_km
```

Static physics features dihitung sekali per cell, di-cache di `physics_per_cell` dict.

### `forecast_archive/`

**Layout baru (2026-05-22+, immutable per-run)**:

```
forecast_archive/
└── 2026-05-25/
    ├── 080746Z_v20260524_141104_5d3d40.parquet
    ├── 082746Z_v20260524_141104_5d3d40.parquet
    └── ...
```

- One file per `run_forecast()` call, named `HHMMSSZ_<safe_model_version>.parquet`
- Never overwritten (collision on same-second runs → suffix `_01`, `_02`, ...)
- Three metadata columns appended:
  - `forecast_run_id` — `<date>T<HHMMSSZ>_<model_version>`
  - `issued_at_utc` — ISO-8601 UTC
  - `model_version` — e.g. `v20260524_141104_5d3d40` or `unknown`

**Layout lama (legacy, still readable)**:

```
forecast_archive/
└── 2026-05-18.parquet     ← single file overwritten daily (deprecated)
```

`read_forecast_archive(day)` returns the most recent per-run file; falls back to legacy single-file layout if no per-run dir exists.

## SQLite

Database aktif: `data/sqlite/gempa_runtime.db` (WAL mode, `synchronous=NORMAL`, `foreign_keys=ON`).

Schema: `backend/app/db/schema.sql` (idempotent `CREATE IF NOT EXISTS`, dijalankan tiap startup via `migrate()`).

### Tables

| Table | Rows ~ | Purpose |
|---|---|---|
| `area_labels` | ~3,000 | Grid cells + province/subregion + physics metadata. Bootstrap idempotent. |
| `current_forecasts` | ~50,000 | Latest forecast per `(cell_id, horizon, threshold)`. PK = (`cell_id`, `horizon_days`, `mag_threshold`). Overwritten each run. |
| `realtime_events` | ~50,000 | Last ~30 days of events (older → Parquet only). |
| `app_metadata` | <50 | Lightweight key-value (forecast freshness, last ingest, dst). |
| `scheduler_runs` | grows | Audit log of every scheduled job (`forecast_recompute`, `realtime_ingest`, `weekly_retrain`, dll). |
| `model_metadata` | ~10 | Trained model registry. Only one row with `is_active=1`. |
| `evaluation_results` | grows | CSEP, Molchan, LORO, reliability per bin payloads. |

### Key columns

`area_labels`:
- PK `cell_id`. `lat_min/lat_max/lon_min/lon_max` define cell bounds (0.5° × 0.5°).
- `region_macro` ∈ {Sumatera, Jawa, BaliNusa, Kalimantan, Sulawesi, MalukuPapua}
- `is_offshore` = 1 untuk cells dominasi laut.
- Physics: `nearest_fault_km`, `fault_type` ∈ {subduction, transform, normal, reverse}, `fault_slip_rate`, `slab_depth_km`.

`current_forecasts`:
- Composite PK `(cell_id, horizon_days, mag_threshold)`.
- Indexes: `idx_forecasts_horizon`, `idx_forecasts_risk` (composite for top-N queries).
- `model_version` traces back to `model_metadata.version`.

`realtime_events`:
- PK `event_id`. Same field semantics as Parquet.
- `raw_json` keeps original API response for debugging.
- Older rows pruned periodically (>30d → only in Parquet).

`model_metadata`:
- `metrics_json`, `feature_list_json`, `feature_importance_json`, `calibrator_json` — all stringified JSON.
- `is_active=1` enforced di runtime, bukan constraint DB. Cek via app code (sebelum patch SQL CHECK constraint, hati-hati race condition).

`evaluation_results`:
- `eval_type` ∈ {reliability, roc, csep, molchan, loro, skill}
- `payload_json` — schema bervariasi per `eval_type`.

### Operations

```bash
# Checkpoint WAL (sebelum backup)
python -c "from backend.app.db.sqlite import checkpoint; print(checkpoint(truncate=True))"

# Integrity check
python -c "from backend.app.db.sqlite import integrity_check; print(integrity_check())"

# Run migrations manually
python -c "from backend.app.db.sqlite import migrate; migrate()"
```

### Stale files

Folder `data/sqlite/` punya backup historis:
- `gempa.db.corruptwal_*` — broken WAL recovery dari interrupted runs (boleh dihapus jika sudah recovered)
- `gempa.db.new`, `gempa.db.broken_*`, `gempa.db.preversion_*` — backup manual sebelum schema migration
- Hanya `gempa_runtime.db*` (3 file: db, wal, shm) yang aktif.

Pertimbangkan cleanup: simpan satu `*.preversion_*` terakhir saja; sisanya bisa dihapus untuk hemat disk (~70 MB).

## Model artifacts (`models/`)

```
models/
├── active.json                 # {"version": "v20260524_141104_5d3d40"}
├── metadata_<version>.json     # mirrors model_metadata table row
├── model_<version>.pkl         # serialized {head_name: HeadModel}
└── poisson_<version>.json      # PoissonBaseline.{rates_per_cell, global_rates, alpha}
```

**`HeadModel`** (di `model_<version>.pkl`):

```python
@dataclass
class HeadModel:
    head_name: str               # e.g. "h30_m50"
    feature_names: list[str]     # untuk backward compat (24 fitur baru, 20 fitur lama)
    xgb: XGBClassifier
    lgbm: LGBMClassifier
    calibrator: Calibrator       # Identity | Platt | Isotonic | Beta
    base_rate: float             # train positive rate
```

`predict_ensemble()` selects feature columns per-head via `hm.feature_names`, sehingga model lama (20 fitur) tetap bisa predict di code baru (24 fitur).

## Backup & migration

| What | Frequency | Where |
|---|---|---|
| `historical_events.parquet` | Manual before destructive ops | `*.bak` siblings |
| SQLite | Manual (use `checkpoint()` first) | `*.preversion_*` siblings |
| Forecast archive | Never (immutable per-run) | n/a |
| Model artifacts | Per training | `models/*_<version>.*` accumulate |

Untuk full project backup: tar + gzip seluruh `data/` + `models/`. Skip `data/sqlite/*.{broken,corruptwal}_*` dan `data/sqlite/*.db-{wal,shm}` (transient).

## Audit checklist (data drift)

Quick script-friendly checks:

```python
import pandas as pd

df = pd.read_parquet("data/parquet/historical_events.parquet")
assert (df["depth"] >= 0).all(), "negative depth — EMSC sign convention regression?"
assert df["source"].isin(["usgs", "emsc", "bmkg"]).all(), "unknown source"
assert df["lat"].between(-11, 6).all(), "lat outside Indonesia"
assert df["lon"].between(95, 141).all(), "lon outside Indonesia"
assert df["event_id"].is_unique, "duplicate event_id"
print(df["source"].value_counts())
```

```bash
sqlite3 data/sqlite/gempa_runtime.db "SELECT COUNT(*) FROM realtime_events;"
sqlite3 data/sqlite/gempa_runtime.db "SELECT version, training_date, dataset_size, is_active FROM model_metadata ORDER BY training_date DESC LIMIT 5;"
sqlite3 data/sqlite/gempa_runtime.db "SELECT COUNT(*), MAX(computed_at) FROM current_forecasts;"
```
