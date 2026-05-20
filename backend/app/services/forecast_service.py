"""Forecast service: build features → predict → write current_forecasts.

Has 3 modes:
1. Full ML mode: active trained model + recent events available.
2. ETAS-only mode: no ML model but recent events for Poisson rate baseline.
3. Demo seed mode: no events at all → synthetic-but-physics-aware probabilities
   based on fault distance + slab depth so the UI has something to render.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.data.catalog import (
    archive_forecast,
    read_historical_events,
)
from backend.app.db.sqlite import get_connection, migrate
from backend.app.features.builder import build_features_for_snapshots
from backend.app.features.labels import HORIZONS, THRESHOLDS, label_column_name
from backend.app.ml.etas import ETASBaseline
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
    now = datetime.now(timezone.utc).isoformat()
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


def _etas_predictions_for_cells(events: pd.DataFrame, cell_ids: list[str]) -> pd.DataFrame:
    """Fit ETAS baseline and produce per-(cell, horizon, threshold) predictions."""
    if events.empty:
        return pd.DataFrame({"cell_id": cell_ids})
    et = ETASBaseline()
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=365 * 5)
    et.fit(events, observation_start=start, observation_end=end)
    return et.predict_dataframe(cell_ids)


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

    if force_demo or (not has_events and not has_model):
        predictions = _demo_seed_predictions(area_rows)
        mode = "demo_seed"
    else:
        # Build features for current snapshot
        snap = datetime.now(timezone.utc)
        # Try ML prediction first
        try:
            from backend.app.core.grid import generate_grid

            cells = generate_grid()
            features = build_features_for_snapshots(events, [snap], cells=cells)
            features = features[features["cell_id"].isin(cell_ids)]
            etas_pred = _etas_predictions_for_cells(events, cell_ids) if has_events else None

            # Compute per-cell event counts for Bayesian evidence weighting
            cell_event_counts = _compute_cell_event_counts(events, cells)

            # Compute empirical base rates for post-hoc recalibration
            from backend.app.ml.posthoc_calibration import compute_base_rates

            base_rates = compute_base_rates(events, n_cells=len(cell_ids))

            predictions, model_version = predict_all(
                features,
                etas_predictions=etas_pred,
                cell_event_counts=cell_event_counts,
                base_rates=base_rates,
            )
            if predictions.empty or model_version is None:
                # Fall back to ETAS-only or demo
                if has_events:
                    predictions = _etas_predictions_for_cells(events, cell_ids)
                    mode = "etas_only"
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

    n = _persist_forecasts(predictions, model_version)
    archive_forecast(predictions, day=date.today())
    summary = {
        "mode": mode,
        "model_version": model_version,
        "cells": len(cell_ids),
        "rows_written": n,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("forecast_run_done", **summary)
    return summary


def get_latest_forecasts(
    *,
    horizon_days: int,
    mag_threshold: float,
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
            ORDER BY f.probability DESC NULLS LAST
            """,
            (horizon_days, mag_threshold),
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


def format_sentence(area: dict, probability: float, *, horizon_days: int, mag_threshold: float) -> str:
    """Build the canonical user-facing sentence."""
    pct = round(probability * 100, 1)
    return f"{area['full_label']}, {pct}% probabilitas M≥{mag_threshold} dalam {horizon_days} hari"
