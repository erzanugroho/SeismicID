"""Forecast service: build features → predict → write current_forecasts.

Has 3 modes:
1. Full ML mode: active trained model + recent events available.
2. Poisson-baseline mode: no ML model but recent events for rate baseline.
3. Demo seed mode: no events at all → synthetic-but-physics-aware probabilities
   based on fault distance + slab depth so the UI has something to render.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import UTC, date, datetime
from pathlib import Path
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


def _persist_forecasts(
    predictions: pd.DataFrame,
    model_version: str | None,
    *,
    raw_predictions: pd.DataFrame | None = None,
) -> int:
    """Upsert into current_forecasts (cell_id, horizon, threshold).

    ``raw_predictions`` carries the model output BEFORE the public probability
    cap and shrinkage blend; we persist it next to ``probability`` so skill
    scoring on archived forecasts can audit the calibrated value, not the
    UI-capped value.
    """
    migrate()
    rows: list[tuple[Any, ...]] = []
    now = datetime.now(UTC).isoformat()
    raw_index = (
        raw_predictions.set_index("cell_id")
        if raw_predictions is not None and not raw_predictions.empty
        else None
    )
    for _, row in predictions.iterrows():
        cid = row["cell_id"]
        raw_row = (
            raw_index.loc[cid]
            if raw_index is not None and cid in raw_index.index
            else None
        )
        for h in HORIZONS:
            for t in THRESHOLDS:
                col = label_column_name(h, t)
                if col not in row:
                    continue
                p = float(row[col])
                raw_p = (
                    float(raw_row[col])
                    if raw_row is not None and col in raw_row and pd.notna(raw_row[col])
                    else None
                )
                rows.append((cid, h, t, p, raw_p, now, model_version or "demo"))
    if not rows:
        return 0
    with get_connection() as conn:
        conn.execute("BEGIN")
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO current_forecasts
                   (cell_id, horizon_days, mag_threshold, probability,
                    raw_probability, computed_at, model_version)
                   VALUES (?,?,?,?,?,?,?)""",
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
    horizon and per threshold, with the threshold ratio governed by the
    Gutenberg-Richter law: log10 N ∝ -b*M, i.e. rate(M) ∝ 10**(-b*(M-mc)).
    Using b=1.0 gives the canonical 10× drop per magnitude unit.
    """
    b_value = 1.0
    mc = 4.5
    base_rates: dict[tuple[int, float], float] = {}
    # Reference annual rate of M>=mc at 50km from a major fault: ~0.30 events
    annual_at_mc = 0.30
    for h in HORIZONS:
        for t in THRESHOLDS:
            annual_baseline = annual_at_mc * (10 ** (-b_value * (t - mc)))
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


def _etas_predictions_for_cells(
    events: pd.DataFrame, cell_ids: list[str]
) -> pd.DataFrame:
    """ETAS-Ogata per-cell forecast — optional Phase 2 tier.

    Fits over the recent 5-year window mirroring the Poisson baseline so the
    two are directly comparable. Empty / unfittable catalogs return an empty
    cell_id-only frame, matching the Poisson helper's contract.
    """
    if events.empty:
        return pd.DataFrame({"cell_id": cell_ids})
    from backend.app.ml.etas_ogata import OgataETAS

    end = datetime.now(UTC)
    start = end - pd.Timedelta(days=365 * 5)
    df = events.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    model = OgataETAS(mc=4.5).fit_from_events(
        df, observation_start=start, observation_end=end
    )
    return model.predict_dataframe(cell_ids, issued_at=end)


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
    """UI display cap to prevent occasional swarm/aftershock spikes from showing
    extreme values. This is a *display* policy — the underlying calibrated
    probability is preserved upstream and used for skill metrics. The cap is
    applied last, only to the value sent to public surfaces.
    """
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
    """Light shrinkage of calibrated ML probabilities toward stable priors.

    This is NOT recalibration of the model. Upstream probabilities are already
    calibrated by the per-head Platt/Isotonic/Beta calibrator (see
    ml/calibration.py). What this function does is a small, intentional
    James-Stein-style shrinkage toward stable references so isolated swarms
    or short-lived spikes don't dominate the public UI.

    Weights (sum to 1.0): the trained ML stays the dominant signal. Higher
    thresholds get slightly more shrinkage because per-cell M>=6 evidence is
    much sparser and the variance of the raw ML output is correspondingly
    larger relative to what the data can support.

        threshold      ML    recent  long-term  tectonic
        M>=4.5        0.70    0.10    0.15       0.05
        M>=5.0        0.65    0.10    0.18       0.07
        M>=5.5        0.55    0.10    0.22       0.13
        M>=6.0        0.45    0.10    0.25       0.20

    The display cap (`_public_probability_cap`) is applied separately and
    documented as a UI ceiling, not a probability transformation.
    """
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

            # ML is the primary signal; priors provide gentle shrinkage.
            if float(t) <= 4.5:
                weights = (0.70, 0.10, 0.15, 0.05)
            elif float(t) <= 5.0:
                weights = (0.65, 0.10, 0.18, 0.07)
            elif float(t) <= 5.5:
                weights = (0.55, 0.10, 0.22, 0.13)
            else:
                weights = (0.45, 0.10, 0.25, 0.20)
            assert abs(sum(weights) - 1.0) < 1e-9, "blend weights must sum to 1"
            p = weights[0] * raw + weights[1] * p_recent + weights[2] * p_smooth + weights[3] * p_tect
            out[col] = p.clip(1e-6, _public_probability_cap(h, float(t)))
    logger.info("public_probability_calibration_applied", cells=len(out))
    return out


def run_forecast(*, force_demo: bool = False) -> dict:
    """Compute and persist current forecasts for all cells.

    Returns summary dict.
    """
    import time as _time

    _t0 = _time.monotonic()
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
                # Fall back: ETAS-Ogata (if flag on) → Poisson → demo seed.
                if has_events:
                    if get_settings().enable_etas_baseline_tier:
                        try:
                            predictions = _etas_predictions_for_cells(events, cell_ids)
                            mode = "etas_ogata"
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "etas_tier_fit_failed_fallback_poisson",
                                error=str(exc),
                            )
                            predictions = _poisson_predictions_for_cells(events, cell_ids)
                            mode = "poisson_baseline"
                    else:
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

    # Snapshot the calibrated, pre-public-cap predictions for audit/skill use
    # before ``apply_public_probability_calibration`` mutates them. Persisted
    # alongside the displayed value as ``raw_probability``.
    raw_predictions = predictions.copy()

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
    raw_predictions = enforce_probability_monotonicity(raw_predictions)

    n = _persist_forecasts(predictions, model_version, raw_predictions=raw_predictions)
    # Phase 4 Task 4.1: tag the archive with the forecast tier so prospective
    # evaluators can score ML and ETAS-Ogata runs separately.
    if mode.startswith("etas_ogata"):
        baseline_type = "etas"
    elif mode.startswith("poisson_baseline"):
        baseline_type = "poisson"
    elif mode.startswith("ml_ensemble"):
        baseline_type = "ml"
    else:
        baseline_type = mode
    archive_forecast(
        predictions,
        day=issued_at.date(),
        model_version=model_version or mode,
        issued_at=issued_at,
        raw_df=raw_predictions,
        baseline_type=baseline_type,
    )
    computed_at = issued_at.isoformat()
    latency_ms = round((_time.monotonic() - _t0) * 1000.0, 1)
    summary = {
        "mode": mode,
        "model_version": model_version,
        "cells": len(cell_ids),
        "rows_written": n,
        "computed_at": computed_at,
        "baseline_type": baseline_type,
        "latency_ms": latency_ms,
    }
    set_metadata_value("last_forecast_at", computed_at)
    set_metadata_value("last_forecast_mode", mode)
    set_metadata_value("last_forecast_baseline_type", baseline_type)
    set_metadata_value("last_forecast_model_version", model_version or "demo")
    # Structured tier observability — one log line per run that downstream
    # log aggregators (or a simple grep) can use to count tier distribution.
    logger.info(
        "forecast_run_done",
        tier=baseline_type,
        mode=mode,
        cells=len(cell_ids),
        rows_written=n,
        latency_ms=latency_ms,
        model_version=model_version,
    )
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
        "forecast_baseline_type": metadata.get("last_forecast_baseline_type"),
        "forecast_model_version": metadata.get("last_forecast_model_version") or (forecast_row["model_version"] if forecast_row else None),
        "etas_baseline_tier_enabled": bool(getattr(settings, "enable_etas_baseline_tier", False)),
        "latest_event": dict(event_row) if event_row else None,
        "realtime_event_count": int(event_count),
        "new_events_since_last_forecast": _metadata_int(metadata, "new_events_since_last_forecast"),
        "last_run": dict(run_row) if run_row else None,
    }


def get_tier_distribution(*, hours: int = 24) -> dict[str, Any]:
    """Count archive runs by ``baseline_type`` over the last ``hours`` hours.

    Walks ``forecast_archive/<UTC-day>/<HHMMSSZ>_<model_version>.parquet`` and
    reads each run's parquet to extract ``baseline_type`` + ``forecast_run_id``.
    Used by /admin/health to surface how often each tier is actually firing —
    without this, the ETAS flag is invisible to operators.
    """
    from backend.app.data.catalog import (
        list_forecast_archive_days,
        list_forecast_archive_runs,
    )

    cutoff = datetime.now(UTC) - pd.Timedelta(hours=hours)
    counts: dict[str, int] = {}
    runs_meta: list[dict[str, Any]] = []

    days = list_forecast_archive_days()
    # Walk newest day first so we can short-circuit once we cross the cutoff.
    for day in sorted(days, reverse=True):
        if datetime.combine(day, datetime.min.time(), tzinfo=UTC) + pd.Timedelta(days=1) < cutoff:
            break
        try:
            run_paths = list_forecast_archive_runs(day)
        except Exception:  # noqa: BLE001
            continue
        # Iterate newest-first within the day too.
        for path in sorted(run_paths, reverse=True):
            issued_dt = _parse_run_issued_at(path, day)
            if issued_dt is None or issued_dt < cutoff:
                continue
            try:
                df = pd.read_parquet(path, columns=["baseline_type", "forecast_run_id"])
            except Exception:  # noqa: BLE001
                # Older archives may lack one of these columns — read full file.
                try:
                    df = pd.read_parquet(path)
                except Exception:  # noqa: BLE001
                    continue
            if df.empty:
                continue
            tier = (
                str(df["baseline_type"].iloc[0])
                if "baseline_type" in df.columns
                else "unknown"
            )
            run_id = (
                str(df["forecast_run_id"].iloc[0])
                if "forecast_run_id" in df.columns
                else path.stem
            )
            counts[tier] = counts.get(tier, 0) + 1
            runs_meta.append(
                {
                    "run_id": run_id,
                    "issued_at": issued_dt.isoformat(),
                    "baseline_type": tier,
                    "path": str(path),
                }
            )
    return {
        "hours": hours,
        "total_runs": sum(counts.values()),
        "by_tier": counts,
        "runs": runs_meta,
    }


def _parse_run_issued_at(path: Path, day: date) -> datetime | None:
    """Parse ``HHMMSSZ_modelversion.parquet`` → tz-aware datetime in UTC."""
    stem = path.stem
    # Legacy single-file layout (``<day>.parquet``) has no time component;
    # treat it as midnight UTC of the day.
    if not stem or "_" not in stem:
        return datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    head = stem.split("_", 1)[0]
    if not head.endswith("Z") or len(head) < 7:
        return datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    hms = head[:-1]
    if len(hms) != 6 or not hms.isdigit():
        return datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    h, m, s = int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
    return datetime(day.year, day.month, day.day, h, m, s, tzinfo=UTC)


def format_sentence(area: dict, probability: float, *, horizon_days: int, mag_threshold: float) -> str:
    """Build the canonical user-facing sentence."""
    pct = round(probability * 100, 1)
    return f"{area['full_label']}, {pct}% probabilitas M≥{mag_threshold} dalam {horizon_days} hari"



# ============================================================================
# Cluster aggregation (subregion-based)
# ============================================================================
#
# Cells are grouped by `full_label` (e.g. "Sulawesi Tengah - Palu" or
# "Lepas Pantai Sumatera Barat - dekat Padang"). A cluster is therefore a
# subregion as labelled in `area_labels`. Three aggregation metrics are
# exposed so the UI can rank by whichever interpretation the user wants:
#
#   - prob_max        : worst cell in the cluster  (worst-case ranking)
#   - prob_top3_mean  : mean of the 3 highest cells (DEFAULT — balances size
#                       bias and single-outlier dominance)
#   - prob_any_cell   : 1 - Π(1 - pᵢ), assuming independent cells
#                       (probability that AT LEAST ONE cell sees ≥M
#                        threshold; mathematically area-wise hazard but
#                        biased upward by cluster size)
#
# A side metric `prob_mean` is also returned for completeness; it is rarely
# the right ranking choice because large clusters with one hot cell look
# misleadingly low.
#
# All four metrics satisfy:
#   prob_any_cell ≥ prob_max ≥ prob_top3_mean ≥ prob_mean
# (proof: see test_clusters.py::test_cluster_aggregation_metric_chain).

ALLOWED_CLUSTER_SORTS: tuple[str, ...] = ("top3_mean", "max", "any_cell", "mean")

_CLUSTER_SORT_FIELDS: dict[str, str] = {
    "top3_mean": "prob_top3_mean",
    "max": "prob_max",
    "any_cell": "prob_any_cell",
    "mean": "prob_mean",
}

_CLUSTER_SORT_DESCRIPTORS_ID: dict[str, str] = {
    "top3_mean": "rata-rata 3 cell tertinggi",
    "max": "cell tertinggi",
    "any_cell": "minimal 1 cell",
    "mean": "rata-rata seluruh cell",
}


def _cluster_id_from_label(label: str) -> str:
    """Stable URL-safe slug from a cluster's full_label.

    e.g. "Sulawesi Tengah - Palu"            → "sulawesi-tengah-palu"
         "Lepas Pantai Aceh - dekat Meulaboh" → "lepas-pantai-aceh-dekat-meulaboh"
    """
    s = unicodedata.normalize("NFKD", label or "")
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-")
    return s[:80] or "unknown"


def _aggregate_cluster(full_label: str, members: list[dict]) -> dict:
    """Build a single cluster dict from N member rows of `get_latest_forecasts`.

    Each member must have at minimum: cell_id, probability (non-null),
    province, subregion, region_macro, lat, lon, lat_min/max, lon_min/max.
    """
    probs = [float(m["probability"]) for m in members if m.get("probability") is not None]
    if not probs:
        # Defensive: caller is expected to filter, but don't crash on stray nulls.
        probs = [0.0]

    probs_sorted = sorted(probs, reverse=True)
    top3 = probs_sorted[: min(3, len(probs_sorted))]
    prob_top3_mean = sum(top3) / len(top3)
    prob_max = probs_sorted[0]
    prob_mean = sum(probs) / len(probs)

    # Cumulative ("any-cell") probability under independence assumption.
    # Clip each p to (0, 1-eps) before log to avoid -inf when p saturates.
    log_complement = 0.0
    for p in probs:
        p_clipped = min(max(p, 0.0), 1.0 - 1e-9)
        log_complement += math.log1p(-p_clipped)
    prob_any_cell = 1.0 - math.exp(log_complement)

    members_sorted = sorted(
        members,
        key=lambda m: float(m.get("probability") or 0.0),
        reverse=True,
    )
    top_cells = [
        {
            "cell_id": m["cell_id"],
            "full_label": m.get("full_label"),
            "lat": float(m["lat"]) if m.get("lat") is not None else None,
            "lon": float(m["lon"]) if m.get("lon") is not None else None,
            "probability": float(m["probability"]) if m.get("probability") is not None else None,
        }
        for m in members_sorted[:3]
    ]

    lats = [float(m["lat"]) for m in members if m.get("lat") is not None]
    lons = [float(m["lon"]) for m in members if m.get("lon") is not None]
    lat_mins = [float(m["lat_min"]) for m in members if m.get("lat_min") is not None]
    lat_maxs = [float(m["lat_max"]) for m in members if m.get("lat_max") is not None]
    lon_mins = [float(m["lon_min"]) for m in members if m.get("lon_min") is not None]
    lon_maxs = [float(m["lon_max"]) for m in members if m.get("lon_max") is not None]

    offshore_count = sum(1 for m in members if m.get("is_offshore"))
    computed_ats = [m["computed_at"] for m in members if m.get("computed_at")]
    latest_computed = max(computed_ats) if computed_ats else None
    first = members_sorted[0]

    return {
        "cluster_id": _cluster_id_from_label(full_label),
        "cluster_label": full_label,
        "province": first.get("province"),
        "subregion": first.get("subregion"),
        "region_macro": first.get("region_macro"),
        "is_offshore": offshore_count > len(members) / 2,
        "n_cells": len(members),
        "n_offshore_cells": offshore_count,
        "prob_max": prob_max,
        "prob_top3_mean": prob_top3_mean,
        "prob_any_cell": prob_any_cell,
        "prob_mean": prob_mean,
        "lat": sum(lats) / len(lats) if lats else None,
        "lon": sum(lons) / len(lons) if lons else None,
        "lat_min": min(lat_mins) if lat_mins else None,
        "lat_max": max(lat_maxs) if lat_maxs else None,
        "lon_min": min(lon_mins) if lon_mins else None,
        "lon_max": max(lon_maxs) if lon_maxs else None,
        "top_cells": top_cells,
        "cell_ids": [m["cell_id"] for m in members_sorted],
        "computed_at": latest_computed,
    }


def get_cluster_forecasts(
    *,
    horizon_days: int,
    mag_threshold: float,
    sort_by: str = "top3_mean",
    region_macro: str | None = None,
    province: str | None = None,
    min_probability: float | None = None,
    min_cells: int = 1,
) -> list[dict]:
    """Aggregate cell-level forecasts into subregion clusters.

    Returns a list of cluster dicts (see `_aggregate_cluster` for keys),
    sorted descending by the metric corresponding to ``sort_by``.

    ``min_probability`` filters clusters whose ranking metric is below the
    threshold (after aggregation), not the underlying cells. Use it for
    "show only meaningful risk clusters" UX.
    """
    if sort_by not in _CLUSTER_SORT_FIELDS:
        raise ValueError(f"sort_by must be one of {ALLOWED_CLUSTER_SORTS}, got {sort_by!r}")

    rows = get_latest_forecasts(horizon_days=horizon_days, mag_threshold=mag_threshold)

    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("probability") is None:
            continue
        if region_macro and r.get("region_macro") != region_macro:
            continue
        if province and r.get("province") != province:
            continue
        key = r.get("full_label") or "(tanpa label)"
        groups.setdefault(key, []).append(r)

    if not groups:
        return []

    clusters = [_aggregate_cluster(label, members) for label, members in groups.items()]

    if min_cells > 1:
        clusters = [c for c in clusters if c["n_cells"] >= min_cells]

    sort_field = _CLUSTER_SORT_FIELDS[sort_by]
    clusters.sort(key=lambda c: c.get(sort_field) or 0.0, reverse=True)

    if min_probability is not None:
        clusters = [c for c in clusters if (c.get(sort_field) or 0.0) >= min_probability]

    return clusters


def get_top_clusters(
    *,
    horizon_days: int,
    mag_threshold: float,
    n: int = 10,
    sort_by: str = "top3_mean",
) -> list[dict]:
    """Return the top-N clusters sorted by the chosen metric."""
    items = get_cluster_forecasts(
        horizon_days=horizon_days,
        mag_threshold=mag_threshold,
        sort_by=sort_by,
    )
    return items[: max(0, int(n))]


def format_cluster_sentence(
    cluster: dict,
    *,
    horizon_days: int,
    mag_threshold: float,
    sort_by: str = "top3_mean",
) -> str:
    """Build the canonical Indonesian summary sentence for a cluster."""
    field = _CLUSTER_SORT_FIELDS.get(sort_by, "prob_top3_mean")
    descriptor = _CLUSTER_SORT_DESCRIPTORS_ID.get(sort_by, "rata-rata 3 cell tertinggi")
    label = cluster.get("cluster_label") or "(tanpa label)"
    n_cells = int(cluster.get("n_cells") or 0)
    pct = round(float(cluster.get(field) or 0.0) * 100.0, 1)
    plural = "cell"  # Bahasa Indonesia: tidak menjamakkan kata benda
    return (
        f"{label} ({n_cells} {plural}), {pct}% {descriptor}, "
        f"M≥{mag_threshold} dalam {horizon_days} hari"
    )
