"""Multi-output training: XGBoost + LightGBM trained per head, calibrated, saved."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from backend.app.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.features.labels import all_label_columns
from backend.app.ml.calibration import fit_best_calibrator

logger = get_logger(__name__)


@dataclass
class HeadModel:
    head: str
    booster_xgb: Any
    booster_lgbm: Any
    calibrator: Any
    feature_names: list[str]
    metrics: dict[str, float]


def _xgb_train(x_tr, y_tr, x_val, y_val):
    import xgboost as xgb
    pos = max(int(y_tr.sum()), 1)
    neg = max(len(y_tr) - pos, 1)
    spw = neg / pos
    use_gpu = get_settings().use_gpu
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        scale_pos_weight=spw, eval_metric="logloss",
        tree_method="hist", device="cuda" if use_gpu else "cpu",
        verbosity=0, n_jobs=2,
    )
    if len(np.unique(y_tr)) < 2:
        return None
    clf.fit(x_tr, y_tr, eval_set=[(x_val, y_val)] if len(y_val) > 0 else None, verbose=False)
    return clf


def _lgbm_train(x_tr, y_tr, x_val, y_val):
    import lightgbm as lgb
    pos = max(int(y_tr.sum()), 1)
    neg = max(len(y_tr) - pos, 1)
    spw = neg / pos
    use_gpu = get_settings().use_gpu
    def _make(device: str):
        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            scale_pos_weight=spw, verbose=-1, n_jobs=2,
            device=device,
            # Robustness against extreme class imbalance / sparse splits
            num_leaves=15,
            min_data_in_leaf=100,
            min_sum_hessian_in_leaf=1e-3,
            min_gain_to_split=0.0,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
        )
    if len(np.unique(y_tr)) < 2:
        return None
    clf = _make("gpu" if use_gpu else "cpu")
    try:
        clf.fit(x_tr, y_tr)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if use_gpu and "OpenCL" in msg:
            logger.warning("lgbm_gpu_unavailable_fallback_cpu", error=msg)
            clf = _make("cpu")
            try:
                clf.fit(x_tr, y_tr)
                return clf
            except Exception as exc2:  # noqa: BLE001
                logger.warning("lgbm_train_failed_xgb_only", error=str(exc2))
                return None
        # Known LightGBM split-failure on tiny/imbalanced labels: skip LGBM, keep XGB
        logger.warning("lgbm_train_failed_xgb_only", error=msg)
        return None
    return clf


def _safe_proba(clf, x: np.ndarray) -> np.ndarray:
    if clf is None:
        return np.full(len(x), 0.0)
    p = clf.predict_proba(x)
    if p.shape[1] == 1:
        return p[:, 0]
    return p[:, 1]


def train_heads(
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
) -> dict[str, HeadModel]:
    """Train one (XGB, LGBM, calibrator) per label head."""
    label_cols = all_label_columns()
    feat_cols = feature_cols or [c for c in train.columns if c not in label_cols and c not in ("cell_id", "snapshot")]

    x_tr = train[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
    x_val = val[feat_cols].fillna(0.0).to_numpy(dtype=np.float32) if len(val) else x_tr[:0]

    heads: dict[str, HeadModel] = {}
    for head in label_cols:
        if head not in train.columns:
            continue
        y_tr = train[head].to_numpy(dtype=np.int32)
        y_val = val[head].to_numpy(dtype=np.int32) if len(val) and head in val.columns else np.array([], dtype=np.int32)

        xgb_clf = _xgb_train(x_tr, y_tr, x_val, y_val)
        lgbm_clf = _lgbm_train(x_tr, y_tr, x_val, y_val)

        # Calibrate using val predictions averaged across both bases
        if len(y_val) > 0 and len(np.unique(y_val)) > 1:
            p_xgb = _safe_proba(xgb_clf, x_val)
            p_lgbm = _safe_proba(lgbm_clf, x_val)
            p_avg = (p_xgb + p_lgbm) / 2.0
            calib, scores = fit_best_calibrator(p_avg, y_val)
            metrics = {"val_brier_uncalib": float(brier_score_loss(y_val, p_avg)), **scores}
            try:
                metrics["val_roc_auc"] = float(roc_auc_score(y_val, p_avg))
            except ValueError:
                metrics["val_roc_auc"] = float("nan")
        elif len(y_tr) > 50 and len(np.unique(y_tr)) > 1:
            # Fallback: Cross-validation calibration on training set
            # Split training data into 3 folds, fit calibrator on held-out predictions
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            cv_probs = np.zeros(len(y_tr))
            for train_idx, cal_idx in kf.split(x_tr):
                xgb_cv = _xgb_train(x_tr[train_idx], y_tr[train_idx], x_tr[cal_idx], y_tr[cal_idx])
                lgbm_cv = _lgbm_train(x_tr[train_idx], y_tr[train_idx], x_tr[cal_idx], y_tr[cal_idx])
                p_xgb_cv = _safe_proba(xgb_cv, x_tr[cal_idx])
                p_lgbm_cv = _safe_proba(lgbm_cv, x_tr[cal_idx])
                cv_probs[cal_idx] = (p_xgb_cv + p_lgbm_cv) / 2.0
            calib, scores = fit_best_calibrator(cv_probs, y_tr)
            metrics = {"cv_brier_uncalib": float(brier_score_loss(y_tr, cv_probs)), **scores}
            try:
                metrics["cv_roc_auc"] = float(roc_auc_score(y_tr, cv_probs))
            except ValueError:
                metrics["cv_roc_auc"] = float("nan")
            logger.info("head_calibrated_via_cv", head=head, calib=calib.name)
        else:
            from backend.app.ml.calibration import IdentityCalibrator
            calib = IdentityCalibrator()
            metrics = {}

        heads[head] = HeadModel(
            head=head,
            booster_xgb=xgb_clf,
            booster_lgbm=lgbm_clf,
            calibrator=calib,
            feature_names=feat_cols,
            metrics=metrics,
        )
        logger.info("head_trained", head=head, metrics={k: round(v, 4) for k, v in metrics.items()})
    return heads


def register_model_in_db(
    version: str,
    training_date: str,
    dataset_size: int | None,
    feature_count: int,
    feature_names: list[str],
    metrics: dict[str, dict[str, float]],
    calibrators: dict[str, str],
) -> None:
    """Register trained model in model_metadata table and set as active."""
    from backend.app.db.sqlite import get_connection
    with get_connection() as conn, conn:  # Transaction block
        # Deactivate any active models
        conn.execute("UPDATE model_metadata SET is_active = 0")
        # Insert the new model
        conn.execute(
            """
            INSERT OR REPLACE INTO model_metadata (
                version, training_date, dataset_size, feature_count,
                feature_list_json, metrics_json, calibrator_json, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                version,
                training_date,
                dataset_size,
                feature_count,
                json.dumps(feature_names),
                json.dumps(metrics),
                json.dumps(calibrators),
            )
        )


def save_evaluation_results(model_version: str, eval_type: str, payload: dict) -> None:
    """Save evaluation payload (ROC, reliability, Molchan, CSEP) for a model version to DB."""
    from backend.app.db.sqlite import get_connection
    with get_connection() as conn, conn:
        # Delete old evaluation of same type for this version if any
        conn.execute(
            "DELETE FROM evaluation_results WHERE model_version = ? AND eval_type = ?",
            (model_version, eval_type)
        )
        conn.execute(
            """
            INSERT INTO evaluation_results (model_version, eval_type, payload_json, computed_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                model_version,
                eval_type,
                json.dumps(payload),
                datetime.now(UTC).isoformat()
            )
        )


def save_models(
    heads: dict[str, HeadModel],
    *,
    version: str,
    models_dir: Path | None = None,
    dataset_size: int | None = None,
) -> Path:
    settings = get_settings()
    out_dir = models_dir or settings.models_path
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / f"models_{version}.pkl"
    with bundle_path.open("wb") as f:
        pickle.dump(heads, f)

    # Extract feature list, metrics, calibrators
    first_head = next(iter(heads.values())) if heads else None
    feature_names = first_head.feature_names if first_head else []
    feature_count = len(feature_names)

    calibrators = {h: hm.calibrator.__class__.__name__ for h, hm in heads.items()}
    metrics = {h: hm.metrics for h, hm in heads.items()}
    training_date = datetime.now(UTC).isoformat()

    metadata = {
        "version": version,
        "training_date": training_date,
        "heads": metrics,
        "feature_count": feature_count,
    }
    (out_dir / f"metadata_{version}.json").write_text(json.dumps(metadata, indent=2))
    # Also write/update active.json to point to this version
    (out_dir / "active.json").write_text(json.dumps({"version": version}))
    logger.info("models_saved", path=str(bundle_path), version=version)

    # Register model in DB. A failed registration leaves ``active.json``
    # pointing at a model that has no metadata row, which silently breaks the
    # /api/model/info endpoint, the performance frontend, and any rollback
    # tooling. We log the error AND raise so callers know the artifact pair
    # (pickle + DB row) is incomplete and must be fixed before serving it.
    try:
        register_model_in_db(
            version=version,
            training_date=training_date,
            dataset_size=dataset_size,
            feature_count=feature_count,
            feature_names=feature_names,
            metrics=metrics,
            calibrators=calibrators,
        )
        logger.info("model_registered_in_db", version=version)
    except Exception as e:
        logger.error("model_registration_failed", version=version, error=str(e))
        raise

    return bundle_path


def load_active_models(models_dir: Path | None = None) -> tuple[dict[str, HeadModel], str] | tuple[None, None]:
    settings = get_settings()
    out_dir = models_dir or settings.models_path
    active_file = out_dir / "active.json"
    if not active_file.exists():
        return None, None
    version = json.loads(active_file.read_text())["version"]
    bundle_path = out_dir / f"models_{version}.pkl"
    if not bundle_path.exists():
        return None, None
    with bundle_path.open("rb") as f:
        heads = pickle.load(f)
    return heads, version
