"""Prediction interface: load active model + ensemble + Bayesian blend."""

from __future__ import annotations

import pandas as pd

from backend.app.core.logging import get_logger
from backend.app.ml.ensemble import EnsembleConfig, predict_ensemble
from backend.app.ml.train import load_active_models

logger = get_logger(__name__)


def predict_all(
    features: pd.DataFrame,
    *,
    poisson_predictions: pd.DataFrame | None = None,
    cell_event_counts: dict[str, int] | None = None,
    config: EnsembleConfig | None = None,
    base_rates: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Run prediction for all (cell, horizon, threshold) combinations.

    Returns (predictions_df, model_version) or (empty_df, None) if no model.
    """
    heads, version = load_active_models()
    if heads is None or not heads:
        logger.warning("predict_no_active_model")
        return pd.DataFrame(), None
    cell_ids = features["cell_id"].tolist()
    pred = predict_ensemble(
        heads,
        features,
        cell_ids=cell_ids,
        poisson_predictions=poisson_predictions,
        cell_event_counts=cell_event_counts,
        config=config,
        base_rates=base_rates,
    )
    return pred, version
