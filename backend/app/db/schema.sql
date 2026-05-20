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
CREATE TABLE IF NOT EXISTS current_forecasts (
    cell_id        TEXT NOT NULL,
    horizon_days   INTEGER NOT NULL,
    mag_threshold  REAL NOT NULL,
    probability    REAL NOT NULL,
    computed_at    TEXT NOT NULL,
    model_version  TEXT,
    PRIMARY KEY (cell_id, horizon_days, mag_threshold),
    FOREIGN KEY (cell_id) REFERENCES area_labels(cell_id)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_cell      ON current_forecasts(cell_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_horizon   ON current_forecasts(horizon_days, mag_threshold);
CREATE INDEX IF NOT EXISTS idx_forecasts_computed  ON current_forecasts(computed_at DESC);

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
