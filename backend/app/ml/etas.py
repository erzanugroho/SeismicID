"""Homogeneous-Poisson statistical baseline.

Per-cell Poisson rate from historical event count
divided by observation duration. P(>=1 event in horizon) = 1 - exp(-rate*horizon).

This is not true ETAS. True ETAS (Ogata 1988) requires fitting μ, K, c, p, α
via MLE and models Omori aftershock decay. This module intentionally provides
a transparent Poisson baseline that more sophisticated models must beat to
demonstrate skill.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from backend.app.features.labels import (
    HORIZONS,
    THRESHOLDS,
    label_column_name,
)


class PoissonBaseline:
    """Per-(cell, threshold) Poisson rate model."""

    def __init__(self) -> None:
        self.rates: dict[tuple[str, float], float] = {}  # (cell_id, threshold) → events/day
        # Global smoothed daily rate per threshold, used as a fallback for
        # cells that have no per-cell history. Without this, every "unknown"
        # cell collapses to zero probability — which is what triggered the
        # all-zero Poisson predictions reported in the audit.
        self.global_rates: dict[float, float] = {}
        self.fit_at: datetime | None = None

    def fit(
        self,
        events: pd.DataFrame,
        *,
        observation_start: datetime,
        observation_end: datetime,
    ) -> PoissonBaseline:
        if events.empty:
            self.rates = {}
            self.global_rates = {t: 0.0 for t in THRESHOLDS}
            self.fit_at = observation_end
            return self

        df = events.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        mask = (df["time"] >= observation_start) & (df["time"] <= observation_end)
        df = df[mask]
        days = max(1.0, (observation_end - observation_start).total_seconds() / 86400)

        # If the events frame doesn't carry ``cell_id`` (the canonical case
        # for ``read_historical_events`` output), assign one on the fly using
        # the same vectorised lat/lon → cell mapping the labeller uses. This
        # was the root cause of the "Poisson rates always 0" audit finding.
        if "cell_id" not in df.columns or df["cell_id"].isna().all():
            df = self._assign_cell_id(df)

        df = df.dropna(subset=["cell_id"])

        # Pre-compute a *weak* global smoothed daily rate per threshold for
        # cells with zero local history. The raw catalogue-wide average across
        # Indonesia is not an appropriate prior for aseismic/low-history cells:
        # it made almost every cell show ~2% for M>=4.5/30d. Treat the global
        # value as a small backoff only; observed cells still use their own
        # empirical per-cell rates.
        try:
            from backend.app.core.grid import generate_grid

            n_cells = max(1, len(generate_grid()))
        except Exception:  # noqa: BLE001
            n_cells = max(1, df["cell_id"].nunique() or 1)
        zero_history_backoff = 0.05

        for thresh in THRESHOLDS:
            sub = df[df["magnitude"] >= thresh]
            global_count = int(len(sub))
            # Global rate per cell-day with Laplace-style smoothing
            # so a single rare event still leaves a tiny positive baseline.
            self.global_rates[thresh] = ((global_count + 1) / (n_cells * days * 1.0)) * zero_history_backoff

            if "cell_id" in sub.columns and not sub.empty:
                counts = sub.groupby("cell_id").size()
                for cid, n in counts.items():
                    self.rates[(cid, thresh)] = float(n) / days

        self.fit_at = observation_end
        return self

    @staticmethod
    def _assign_cell_id(df: pd.DataFrame) -> pd.DataFrame:
        """Fall back lat/lon → cell_id mapping using the canonical grid."""
        if df.empty or "lat" not in df.columns or "lon" not in df.columns:
            return df
        # Local imports keep this module light and avoid a circular dep.
        from backend.app.core.grid import generate_grid
        from backend.app.features.labels import assign_cell_id_vec

        cells = generate_grid()
        return assign_cell_id_vec(df, cells)

    def predict_probability(
        self,
        cell_id: str,
        horizon_days: int,
        threshold: float,
        *,
        global_rate: float | None = None,
    ) -> float:
        """Return P(≥1 event) for the given cell/horizon/threshold.

        Cells without per-cell history fall back to the smoothed global rate
        captured during ``fit`` so the baseline is never identically zero.
        """
        if (cell_id, threshold) in self.rates:
            rate = self.rates[(cell_id, threshold)]
        else:
            rate = global_rate if global_rate is not None else self.global_rates.get(threshold, 0.0)
        return float(1 - np.exp(-rate * horizon_days))

    def predict_dataframe(self, cell_ids: list[str]) -> pd.DataFrame:
        """Return predictions for all (cell, horizon, threshold) combos as DataFrame."""
        rows = []
        for cid in cell_ids:
            row: dict[str, str | float] = {"cell_id": cid}
            for h in HORIZONS:
                for t in THRESHOLDS:
                    row[label_column_name(h, t)] = self.predict_probability(cid, h, t)
            rows.append(row)
        return pd.DataFrame(rows)


# Backward-compatible alias for existing imports/model bundles. Prefer
# PoissonBaseline in new code and docs to avoid implying a true ETAS model.
ETASBaseline = PoissonBaseline
