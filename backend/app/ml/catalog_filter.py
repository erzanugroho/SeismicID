"""Helpers to filter event catalogs prior to ETAS / Poisson fitting.

The ETAS likelihood assumes the catalog is complete above Mc — sub-completeness
events depress observed productivity and bias parameter estimates. This module
provides a thin filter that records audit attrs so the calling code (and
later the model card) can report exactly how many events were dropped and at
what threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def filter_below_mc(events: pd.DataFrame, *, mc: float) -> pd.DataFrame:
    """Drop events with magnitude < mc. Preserves audit attrs.

    Parameters
    ----------
    events : DataFrame with at least a ``magnitude`` column. Empty / missing
        column inputs are returned unchanged so callers can chain freely.
    mc : magnitude threshold; events strictly below this are dropped.

    Returns
    -------
    DataFrame
        Filtered copy. ``df.attrs`` carries:
          * ``mc_value``           — the threshold applied
          * ``mc_filter_dropped``  — int count of dropped events
    """
    if events.empty:
        return events
    if "magnitude" not in events.columns:
        return events
    n_before = len(events)
    mags = events["magnitude"]
    out = events[mags.notna() & (mags >= mc)].copy()
    out.attrs["mc_value"] = float(mc)
    out.attrs["mc_filter_dropped"] = int(n_before - len(out))
    return out
