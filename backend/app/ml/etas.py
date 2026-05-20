"""ETAS-style statistical baseline.

Simplified ETAS for forecast: per-cell Poisson rate from historical event count
divided by observation duration. P(>=1 event in horizon) = 1 - exp(-rate*horizon).

True ETAS (Ogata 1988) requires fitting μ, K, c, p, α via MLE. Here we use the
homogeneous-Poisson approximation which serves as the baseline that more
sophisticated models must beat to demonstrate skill.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.app.features.labels import HORIZONS, THRESHOLDS, label_column_name


class ETASBaseline:
    """Per-(cell, threshold) Poisson rate model."""

    def __init__(self) -> None:
        self.rates: dict[tuple[str, float], float] = {}  # (cell_id, threshold) → events/day
        self.fit_at: datetime | None = None

    def fit(self, events: pd.DataFrame, *, observation_start: datetime, observation_end: datetime) -> "ETASBaseline":
        if events.empty:
            self.rates = {}
            self.fit_at = observation_end
            return self
        df = events.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        mask = (df["time"] >= observation_start) & (df["time"] <= observation_end)
        df = df[mask]
        days = max(1.0, (observation_end - observation_start).total_seconds() / 86400)
        for thresh in THRESHOLDS:
            sub = df[df["magnitude"] >= thresh]
            if "cell_id" not in sub.columns:
                continue
            counts = sub.groupby("cell_id").size()
            for cid, n in counts.items():
                self.rates[(cid, thresh)] = float(n) / days
        self.fit_at = observation_end
        return self

    def predict_probability(
        self,
        cell_id: str,
        horizon_days: int,
        threshold: float,
        *,
        global_rate: float = 0.0,
    ) -> float:
        rate = self.rates.get((cell_id, threshold), global_rate)
        return float(1 - np.exp(-rate * horizon_days))

    def predict_dataframe(self, cell_ids: list[str]) -> pd.DataFrame:
        """Return predictions for all (cell, horizon, threshold) combos as DataFrame."""
        rows = []
        for cid in cell_ids:
            row = {"cell_id": cid}
            for h in HORIZONS:
                for t in THRESHOLDS:
                    row[label_column_name(h, t)] = self.predict_probability(cid, h, t)
            rows.append(row)
        return pd.DataFrame(rows)
