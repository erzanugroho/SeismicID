-- Earthquake Probability Forecast System — SQLite schema
-- All tables. Migrations applied idempotently via CREATE TABLE IF NOT EXISTS.

-- Grid cells with labels and physics metadata.
CREATE TABLE IF NOT EXISTS area_labels (
    cell_id              TEXT PRIMARY KEY,
    lat                  REAL NOT NULL,
    lon                  REAL NOT NULL,
    lat_min              REAL NOT NULL,
    lat_max              REAL NOT NULL,
    lon_min              REAL NOT NULL,
    lon_max              REAL NOT NULL,
    province             TEXT,
    subregion            TEXT,
    full_label           TEXT NOT NULL,
    is_offshore          INTEGER NOT NULL DEFAULT 0,
    region_macro         TEXT,                -- Sumatera|Jawa|Sulawesi|MalukuPapua|...
    nearest_fault_km     REAL,
    fault_type           TEXT,                -- subduction|transform|normal|reverse
    fault_slip_rate      REAL,
    slab_depth_km        REAL,
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_area_province ON area_labels(province);
CREATE INDEX IF NOT EXISTS idx_area_region   ON area_labels(region_macro);

-- Latest forecast per (cell, horizon, threshold). Multi-output.
-- ``probability`` is the public-facing value (calibrated + display-capped +
-- monotonicity-enforced). ``raw_probability`` is the model output BEFORE the
-- public probability cap and shrinkage blend (apply_public_probability_calibration)
-- but AFTER per-head Platt/Isotonic/Beta calibration. Storing both keeps the
-- audit trail intact: skill scoring should use ``raw_probability``, while UI
-- surfaces should use ``probability``.
CREATE TABLE IF NOT EXISTS current_forecasts (
    cell_id          TEXT NOT NULL,
    horizon_days     INTEGER NOT NULL,
    mag_threshold    REAL NOT NULL,
    probability      REAL NOT NULL,
    raw_probability  REAL,
    computed_at      TEXT NOT NULL,
    model_version    TEXT,
    PRIMARY KEY (cell_id, horizon_days, mag_threshold),
    FOREIGN KEY (cell_id) REFERENCES area_labels(cell_id)
);

-- Idempotent backfill of the new column for existing databases. SQLite has no
-- ``ADD COLUMN IF NOT EXISTS`` clause; the duplicate-column error is caught
-- and ignored at runtime in db/sqlite.py:migrate.
ALTER TABLE current_forecasts ADD COLUMN raw_probability REAL;

CREATE INDEX IF NOT EXISTS idx_forecasts_cell      ON current_forecasts(cell_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_horizon   ON current_forecasts(horizon_days, mag_threshold);
CREATE INDEX IF NOT EXISTS idx_forecasts_computed  ON current_forecasts(computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_risk ON current_forecasts(horizon_days, mag_threshold, probability DESC, computed_at DESC);

-- Realtime events buffer (last ~30 days; older is in Parquet).
CREATE TABLE IF NOT EXISTS realtime_events (
    event_id     TEXT PRIMARY KEY,
    time         TEXT NOT NULL,            -- ISO8601 UTC
    lat          REAL NOT NULL,
    lon          REAL NOT NULL,
    depth        REAL,
    magnitude    REAL NOT NULL,
    mag_type     TEXT,
    source       TEXT NOT NULL,            -- usgs|bmkg
    place        TEXT,
    raw_json     TEXT,
    inserted_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_time   ON realtime_events(time DESC);
CREATE INDEX IF NOT EXISTS idx_events_mag    ON realtime_events(magnitude);
CREATE INDEX IF NOT EXISTS idx_events_source ON realtime_events(source);

-- Lightweight app/status metadata.
CREATE TABLE IF NOT EXISTS app_metadata (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Scheduler audit log.
CREATE TABLE IF NOT EXISTS scheduler_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name          TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL,        -- running|success|error
    error             TEXT,
    items_processed   INTEGER,
    metadata_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_started ON scheduler_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_job     ON scheduler_runs(job_name, started_at DESC);

-- Trained model registry. Only one row with is_active=1 at a time.
CREATE TABLE IF NOT EXISTS model_metadata (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    version                  TEXT NOT NULL UNIQUE,
    training_date            TEXT NOT NULL,
    dataset_size             INTEGER,
    feature_count            INTEGER,
    feature_list_json        TEXT,
    metrics_json             TEXT,            -- per-head ROC/Brier/etc.
    feature_importance_json  TEXT,
    calibrator_json          TEXT,            -- per-head best calibrator
    is_active                INTEGER NOT NULL DEFAULT 0,
    notes                    TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_active ON model_metadata(is_active);

-- Evaluation result payloads (CSEP, Molchan, LORO, reliability per bin).
CREATE TABLE IF NOT EXISTS evaluation_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version  TEXT NOT NULL,
    eval_type      TEXT NOT NULL,            -- reliability|roc|csep|molchan|loro|skill
    payload_json   TEXT NOT NULL,
    computed_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_eval_model_type ON evaluation_results(model_version, eval_type);

-- AI response cache for safe, low-cost public summaries.
CREATE TABLE IF NOT EXISTS ai_cache (
    cache_key     TEXT PRIMARY KEY,
    payload_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ai_cache_expires ON ai_cache(expires_at);

-- Telegram bot user area selection.
CREATE TABLE IF NOT EXISTS telegram_regions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    level       TEXT NOT NULL, -- province|regency|district
    parent_id   INTEGER,
    lat         REAL,
    lon         REAL,
    cell_id     TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (parent_id) REFERENCES telegram_regions(id),
    FOREIGN KEY (cell_id) REFERENCES area_labels(cell_id)
);

CREATE INDEX IF NOT EXISTS idx_telegram_regions_parent ON telegram_regions(parent_id, name);
CREATE INDEX IF NOT EXISTS idx_telegram_regions_level ON telegram_regions(level);

CREATE TABLE IF NOT EXISTS telegram_user_locations (
    chat_id          TEXT PRIMARY KEY,
    username         TEXT,
    first_name       TEXT,
    province         TEXT,
    regency          TEXT,
    district         TEXT,
    lat_rounded      REAL,
    lon_rounded      REAL,
    nearest_cell_id  TEXT NOT NULL,
    area_label       TEXT NOT NULL,
    radius_km        INTEGER NOT NULL DEFAULT 50,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
    stopped_at       TEXT,
    FOREIGN KEY (nearest_cell_id) REFERENCES area_labels(cell_id)
);

ALTER TABLE telegram_user_locations ADD COLUMN stopped_at TEXT;

CREATE INDEX IF NOT EXISTS idx_telegram_user_locations_cell ON telegram_user_locations(nearest_cell_id);

CREATE TABLE IF NOT EXISTS telegram_bot_opt_outs (
    chat_id     TEXT PRIMARY KEY,
    stopped_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
