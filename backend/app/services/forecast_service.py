"""Forecast service: build features → predict → write current_forecasts.

Has 3 modes:
1. Full ML mode: active trained model + recent events available.
2. Poisson-baseline mode: no ML model but recent events for rate baseline.
3. Demo seed mode: no events at all → synthetic-but-physics-aware probabilities
   based on fault distance + slab depth so the UI has something to render.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.data.catalog import (
    archive_forecast,
    read_historical_events,
)
from backend.app.db.metadata import get_metadata_values, set_metadata_value
from backend.app.db.sqlite import get_connection, migrate
from backend.app.features.builder import build_features_for_snapshots
from backend.app.features.labels import HORIZONS, THRESHOLDS, label_column_name
from backend.app.ml.ensemble import enforce_probability_monotonicity
from backend.app.ml.etas import PoissonBaseline
from backend.app.ml.predict import predict_all

logger = get_logger(__name__)


def _all_area_rows() -> list[dict]:
    migrate()
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM area_labels")
        return [dict(r) for r in cur.fetchall()]


def _compute_cell_event_counts(
    events: pd.DataFrame,
    cells: list,
) -> dict[str, int]:
    """Count M≥4.5 events per cell from historical data for Bayesian evidence."""
    if events.empty:
        return {}
    from backend.app.features.labels import assign_cell_id_vec

    df = assign_cell_id_vec(events[events["magnitude"] >= 4.5], cells)
    df = df.dropna(subset=["cell_id"])
    if df.empty:
        return {}
    counts = df.groupby("cell_id").size().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _persist_forecasts(predictions: pd.DataFrame, model_version: str | None) -> int:
    """Upsert into current_forecasts (cell_id, horizon, threshold)."""
    migrate()
    rows: list[tuple[Any, ...]] = []
    now = datetime.now(UTC).isoformat()
    for _, row in predictions.iterrows():
        cid = row["cell_id"]
        for h in HORIZONS:
            for t in THRESHOLDS:
                col = label_column_name(h, t)
                if col not in row:
                    continue
                p = float(row[col])
                rows.append((cid, h, t, p, now, model_version or "demo"))
    if not rows:
        return 0
    with get_connection() as conn:
        conn.execute("BEGIN")
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO current_forecasts
                   (cell_id, horizon_days, mag_threshold, probability, computed_at, model_version)
                   VALUES (?,?,?,?,?,?)""",
                rows,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return len(rows)


def _demo_seed_predictions(area_rows: list[dict]) -> pd.DataFrame:
    """Physics-aware synthetic probabilities so UI has data without a model.

    Higher base rate near subduction trenches and active transform faults.
    Decays exponentially with fault distance. Different scaling per
    horizon and per threshold to mirror realistic class imbalance.
    """
    base_rates: dict[tuple[int, float], float] = {}
    # Reference annual rate of M≥5 at 50km from a major fault: ~0.30 events
    for h in HORIZONS:
        for t in THRESHOLDS:
            # Annual baseline scales by exp(-(t-4.5)/0.3) per Gutenberg-Richter
            annual_baseline = 0.30 * math.exp(-(t - 4.5) / 0.30)
            base_rates[(h, t)] = annual_baseline * (h / 365.0)

    rows: list[dict[str, Any]] = []
    for area in area_rows:
        row: dict[str, Any] = {"cell_id": area["cell_id"]}
        nf = area.get("nearest_fault_km") or 200.0
        ftype = (area.get("fault_type") or "").lower()
        # Distance decay (e-folding 75 km)
        decay = math.exp(-nf / 75.0)
        # Subduction zones get a 1.5× boost
        type_factor = 1.5 if ftype == "subduction" else 1.0 if ftype == "transform" else 0.7
        for h in HORIZONS:
            for t in THRESHOLDS:
                rate = base_rates[(h, t)] * decay * type_factor
                p = 1 - math.exp(-rate)
                row[label_column_name(h, t)] = float(min(p, 0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def _poisson_predictions_for_cells(events: pd.DataFrame, cell_ids: list[str]) -> pd.DataFrame:
    """Fit the recent 5-year Poisson baseline for ML prior compatibility."""
    if events.empty:
        return pd.DataFrame({"cell_id": cell_ids})
    end = datetime.now(UTC)
    return _poisson_predictions_for_window(events, cell_ids, end=end, days=365 * 5)


def _poisson_predictions_for_window(
    events: pd.DataFrame,
    cell_ids: list[str],
    *,
    end: datetime,
    days: int | None,
) -> pd.DataFrame:
    """Fit a Poisson baseline over a specific historical window.

    ``days=None`` uses the full available catalog. This is used for public
    calibration so recent swarms do not dominate the displayed probability.
    """
    if events.empty:
        return pd.DataFrame({"cell_id": cell_ids})
    baseline = PoissonBaseline()
    df = events.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    start = df["time"].min().to_pydatetime() if days is None else end - pd.Timedelta(days=days)
    baseline.fit(df, observation_start=start, observation_end=end)
    return baseline.predict_dataframe(cell_ids)


def _tectonic_prior_predictions(area_rows: list[dict]) -> pd.DataFrame:
    """Create a conservative tectonic prior from fault proximity and type."""
    rows: list[dict[str, Any]] = []
    annual_base = {4.5: 0.90, 5.0: 0.28, 5.5: 0.08, 6.0: 0.025}
    for area in area_rows:
        row: dict[str, Any] = {"cell_id": area["cell_id"]}
        nf = float(area.get("nearest_fault_km") or 250.0)
        ftype = (area.get("fault_type") or "").lower()
        slab_depth = area.get("slab_depth_km")
        distance_factor = 0.25 + 0.75 * math.exp(-nf / 90.0)
        type_factor = 1.35 if ftype == "subduction" else 1.05 if ftype == "transform" else 0.75
        slab_factor = 1.10 if slab_depth is not None and float(slab_depth) <= 80.0 else 1.0
        offshore_factor = 0.95 if area.get("is_offshore") else 1.0
        hazard_factor = distance_factor * type_factor * slab_factor * offshore_factor
        for h in HORIZONS:
            for t in THRESHOLDS:
                rate_year = annual_base.get(float(t), 0.01) * hazard_factor
                p = 1.0 - math.exp(-rate_year * (h / 365.0))
                row[label_column_name(h, t)] = float(max(1e-6, min(p, 0.80)))
        rows.append(row)
    return pd.DataFrame(rows)


def _gamma_poisson_smoothed_predictions(
    recent: pd.DataFrame,
    long_term: pd.DataFrame,
    *,
    prior_weight: float = 0.45,
) -> pd.DataFrame:
    """Shrink recent Poisson rates toward long-term catalog rates."""
    out = recent[["cell_id"]].copy()
    long_idx = long_term.set_index("cell_id")
    for h in HORIZONS:
        for t in THRESHOLDS:
            col = label_column_name(h, t)
            if col not in recent.columns or col not in long_term.columns:
                continue
            p_recent = recent[col].clip(1e-9, 1 - 1e-9)
            p_long = recent["cell_id"].map(long_idx[col]).fillna(0.0).clip(1e-9, 1 - 1e-9)
            rate_recent = -p_recent.map(lambda p: math.log1p(-float(p))) / h
            rate_long = -p_long.map(lambda p: math.log1p(-float(p))) / h
            rate = (1.0 - prior_weight) * rate_recent + prior_weight * rate_long
            out[col] = rate.map(lambda r: 1.0 - math.exp(-float(r) * h))
    return out


def _public_probability_cap(horizon: int, threshold: float) -> float:
    """Caps prevent raw swarm/aftershock overconfidence in public UI/API."""
    base_30d = {4.5: 0.65, 5.0: 0.35, 5.5: 0.16, 6.0: 0.07}.get(float(threshold), 0.20)
    scaled = 1.0 - math.exp(-(-math.log1p(-base_30d)) * (horizon / 30.0))
    return float(min(scaled, 0.85))


def apply_public_probability_calibration(
    predictions: pd.DataFrame,
    *,
    events: pd.DataFrame,
    area_rows: list[dict],
    issued_at: datetime,
) -> pd.DataFrame:
    """Blend raw ML, recent activity, long-term catalog, and tectonic prior."""
    if predictions.empty or events.empty:
        return predictions
    cell_ids = predictions["cell_id"].astype(str).tolist()
    cell_set = set(cell_ids)
    recent = _poisson_predictions_for_window(events, cell_ids, end=issued_at, days=365 * 5)
    long_term = _poisson_predictions_for_window(events, cell_ids, end=issued_at, days=None)
    smoothed = _gamma_poisson_smoothed_predictions(recent, long_term)
    tectonic = _tectonic_prior_predictions([a for a in area_rows if a["cell_id"] in cell_set])
    recent_idx = recent.set_index("cell_id")
    long_idx = long_term.set_index("cell_id")
    smooth_idx = smoothed.set_index("cell_id")
    tect_idx = tectonic.set_index("cell_id")

    out = predictions.copy()
    for h in HORIZONS:
        for t in THRESHOLDS:
            col = label_column_name(h, t)
            if col not in out.columns:
                continue
            ids = out["cell_id"].astype(str)
            raw = out[col].astype(float).clip(1e-6, 1 - 1e-6)
            p_recent = ids.map(recent_idx[col]).fillna(0.0).astype(float)
            p_long = ids.map(long_idx[col]).fillna(0.0).astype(float)
            p_smooth = ids.map(smooth_idx[col]).fillna(p_long).astype(float)
            p_tect = ids.map(tect_idx[col]).fillna(p_long).astype(float)

            if float(t) <= 4.5:
                weights = (0.25, 0.15, 0.45, 0.15)
            elif float(t) <= 5.0:
                weights = (0.22, 0.13, 0.45, 0.20)
            elif float(t) <= 5.5:
                weights = (0.18, 0.10, 0.42, 0.30)
            else:
                weights = (0.15, 0.08, 0.37, 0.40)
            p = weights[0] * raw + weights[1] * p_recent + weights[2] * p_smooth + weights[3] * p_tect
            out[col] = p.clip(1e-6, _public_probability_cap(h, float(t)))
    logger.info("public_probability_calibration_applied", cells=len(out))
    return out


def run_forecast(*, force_demo: bool = False) -> dict:
    """Compute and persist current forecasts for all cells.

    Returns summary dict.
    """
    area_rows = _all_area_rows()
    if not area_rows:
        from backend.app.services.area_service import bootstrap_area_labels

        bootstrap_area_labels()
        area_rows = _all_area_rows()

    cell_ids = [a["cell_id"] for a in area_rows]
    events = read_historical_events()
    has_events = not events.empty
    has_model = False
    predictions: pd.DataFrame
    model_version: str | None = None
    mode: str
    issued_at = datetime.now(UTC)

    if force_demo or (not has_events and not has_model):
        predictions = _demo_seed_predictions(area_rows)
        mode = "demo_seed"
    else:
        # Build features for current snapshot
        snap = issued_at
        # Try ML prediction first
        try:
            from backend.app.core.grid import generate_grid

            cells = generate_grid()
            features = build_features_for_snapshots(events, [snap], cells=cells)
            features = features[features["cell_id"].isin(cell_ids)]
            poisson_pred = _poisson_predictions_for_cells(events, cell_ids) if has_events else None

            # Compute per-cell event counts for Bayesian evidence weighting
            cell_event_counts = _compute_cell_event_counts(events, cells)

            # Compute empirical base rates for post-hoc recalibration
            from backend.app.ml.posthoc_calibration import compute_base_rates

            base_rates = compute_base_rates(events, n_cells=len(cell_ids))

            predictions, model_version = predict_all(
                features,
                poisson_predictions=poisson_pred,
                cell_event_counts=cell_event_counts,
                base_rates=base_rates,
            )
            if predictions.empty or model_version is None:
                # Fall back to Poisson-baseline or demo
                if has_events:
                    predictions = _poisson_predictions_for_cells(events, cell_ids)
                    mode = "poisson_baseline"
                else:
                    predictions = _demo_seed_predictions(area_rows)
                    mode = "demo_seed"
            else:
                mode = "ml_ensemble"
                has_model = True
        except Exception as e:  # noqa: BLE001
            logger.warning("forecast_ml_path_failed", error=str(e))
            predictions = _demo_seed_predictions(area_rows)
            mode = "demo_seed"

    if has_events and mode in {"ml_ensemble", "poisson_baseline"}:
        predictions = apply_public_probability_calibration(
            predictions,
            events=events,
            area_rows=area_rows,
            issued_at=issued_at,
        )
        mode = f"{mode}_public_calibrated"

    # Enforce monotonic probability constraints across horizons and thresholds
    # after public calibration. Blending/capping is monotone in most cases, but
    # this final pass guarantees every persisted/API value respects the basic
    # probability ordering.
    predictions = enforce_probability_monotonicity(predictions)

    n = _persist_forecasts(predictions, model_version)
    archive_forecast(
        predictions,
        day=issued_at.date(),
        model_version=model_version or mode,
        issued_at=issued_at,
    )
    computed_at = issued_at.isoformat()
    summary = {
        "mode": mode,
        "model_version": model_version,
        "cells": len(cell_ids),
        "rows_written": n,
        "computed_at": computed_at,
    }
    set_metadata_value("last_forecast_at", computed_at)
    set_metadata_value("last_forecast_mode", mode)
    set_metadata_value("last_forecast_model_version", model_version or "demo")
    logger.info("forecast_run_done", **summary)
    return summary


def get_latest_forecasts(
    *,
    horizon_days: int,
    mag_threshold: float,
    min_probability: float | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Return latest forecasts for given (horizon, threshold), joined with area labels."""
    migrate()
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT a.cell_id, a.lat, a.lon, a.lat_min, a.lat_max, a.lon_min, a.lon_max,
                   a.full_label, a.province, a.subregion, a.region_macro,
                   a.is_offshore, a.nearest_fault_km, a.fault_type, a.slab_depth_km,
                   f.probability, f.computed_at, f.model_version
            FROM area_labels a
            LEFT JOIN current_forecasts f
              ON f.cell_id = a.cell_id
              AND f.horizon_days = ?
              AND f.mag_threshold = ?
            WHERE (? IS NULL OR f.probability >= ?)
            ORDER BY f.probability DESC NULLS LAST
            LIMIT ?
            """,
            (horizon_days, mag_threshold, min_probability, min_probability, limit or 1000000),
        )
        return [dict(r) for r in cur.fetchall()]


def get_top_forecasts(
    *,
    horizon_days: int,
    mag_threshold: float,
    n: int = 10,
) -> list[dict]:
    rows = get_latest_forecasts(horizon_days=horizon_days, mag_threshold=mag_threshold)
    rows = [r for r in rows if r.get("probability") is not None]
    return rows[:n]


def get_area_forecasts(cell_id: str) -> dict:
    """Return all 16 forecasts for a single cell."""
    migrate()
    with get_connection() as conn:
        area = conn.execute("SELECT * FROM area_labels WHERE cell_id = ?", (cell_id,)).fetchone()
        if area is None:
            return {}
        forecasts = conn.execute(
            """SELECT horizon_days, mag_threshold, probability, computed_at, model_version
               FROM current_forecasts WHERE cell_id = ?
               ORDER BY horizon_days, mag_threshold""",
            (cell_id,),
        ).fetchall()
    return {
        "area": dict(area),
        "forecasts": [dict(r) for r in forecasts],
    }


def _metadata_int(metadata: dict[str, str | None], key: str, default: int = 0) -> int:
    try:
        return int(metadata.get(key) or default)
    except (TypeError, ValueError):
        return default


def get_forecast_status() -> dict[str, Any]:
    """Return public-safe metadata about cached forecasts and worker freshness."""
    settings = get_settings()
    migrate()
    metadata = get_metadata_values()
    with get_connection() as conn:
        forecast_row = conn.execute(
            """SELECT computed_at, model_version
               FROM current_forecasts
               ORDER BY computed_at DESC
               LIMIT 1"""
        ).fetchone()
        event_row = conn.execute(
            """SELECT event_id, time, magnitude, place, source
               FROM realtime_events
               ORDER BY time DESC
               LIMIT 1"""
        ).fetchone()
        event_count = conn.execute("SELECT COUNT(*) AS n FROM realtime_events").fetchone()["n"]
        run_row = conn.execute(
            """SELECT job_name, started_at, finished_at, status, error, metadata_json
               FROM scheduler_runs
               ORDER BY started_at DESC
               LIMIT 1"""
        ).fetchone()

    return {
        "trigger_mode": settings.forecast_trigger_mode,
        "fetch_interval_minutes": settings.forecast_fetch_interval_minutes,
        "debounce_minutes": settings.forecast_debounce_minutes,
        "fallback_hours": settings.forecast_fallback_hours,
        "last_checked_at": metadata.get("last_checked_at"),
        "forecast_last_computed_at": metadata.get("last_forecast_at") or (forecast_row["computed_at"] if forecast_row else None),
        "forecast_mode": metadata.get("last_forecast_mode"),
        "forecast_model_version": metadata.get("last_forecast_model_version") or (forecast_row["model_version"] if forecast_row else None),
        "latest_event": dict(event_row) if event_row else None,
        "realtime_event_count": int(event_count),
        "new_events_since_last_forecast": _metadata_int(metadata, "new_events_since_last_forecast"),
        "last_run": dict(run_row) if run_row else None,
    }


def format_sentence(area: dict, probability: float, *, horizon_days: int, mag_threshold: float) -> str:
    """Build the canonical user-facing sentence."""
    pct = round(probability * 100, 1)
    return f"{area['full_label']}, {pct}% probabilitas M≥{mag_threshold} dalam {horizon_days} hari"
