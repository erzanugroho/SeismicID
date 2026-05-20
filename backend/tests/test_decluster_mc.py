"""Tests for declustering + Mc + b-value primitives."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from backend.app.data.completeness import build_mc_lookup, estimate_mc_maxc, lookup_mc
from backend.app.data.decluster import decluster
from backend.app.features.seismology import compute_b_value, iet_stats, seismic_energy


# -- seismology ----------------------------------------------------------------

def test_seismic_energy_known_values() -> None:
    # log10(E) for M=5 is 12.3
    assert abs(seismic_energy(5.0) - 12.3) < 1e-9


def test_b_value_synthetic_population() -> None:
    """Synthetic G-R sample with b=1.0 should recover b ≈ 1.0."""
    rng = np.random.default_rng(42)
    mc = 4.0
    # Inverse CDF of exponential(b*ln10) translates to magnitudes
    b_true = 1.0
    n = 5000
    mags = mc + rng.exponential(scale=1.0 / (b_true * np.log(10.0)), size=n)
    b, se = compute_b_value(mags, mc=mc)
    assert abs(b - b_true) < 0.15  # finite-sample bias tolerance


def test_b_value_returns_nan_for_few_events() -> None:
    b, se = compute_b_value(np.array([4.0, 4.5]), mc=4.0)
    assert np.isnan(b)


def test_iet_stats_constant_arrival() -> None:
    times = np.array(["2024-01-01T00:00", "2024-01-02T00:00", "2024-01-03T00:00"], dtype="datetime64[s]")
    mu, cv = iet_stats(times)
    assert mu == 86400
    assert cv == 0.0


# -- completeness Mc -----------------------------------------------------------

def test_estimate_mc_maxc_synthetic() -> None:
    """G-R sample with true Mc=4.0 → MAXC + 0.2 correction ≈ 4.0-4.3."""
    rng = np.random.default_rng(1)
    true_mc = 4.0
    mags_above = true_mc + rng.exponential(scale=1.0 / np.log(10), size=2000)
    # Add some incomplete tail below Mc
    mags_below = rng.uniform(2.5, true_mc, size=200)
    all_mags = np.concatenate([mags_above, mags_below])
    mc = estimate_mc_maxc(all_mags)
    assert 3.8 <= mc <= 4.5  # allow correction range


def test_build_mc_lookup_indonesia_regions() -> None:
    """Build Mc lookup for synthetic Indonesia events; expect entries per active region."""
    rng = np.random.default_rng(0)
    rows = []
    # Sumatera
    for _ in range(500):
        rows.append(
            {
                "time": datetime(2010, 1, 1, tzinfo=timezone.utc) + timedelta(days=int(rng.integers(0, 1500))),
                "lat": rng.uniform(-3, 3),
                "lon": rng.uniform(95, 105),
                "magnitude": 4.0 + rng.exponential(0.4),
            }
        )
    # Sulawesi
    for _ in range(500):
        rows.append(
            {
                "time": datetime(2012, 1, 1, tzinfo=timezone.utc) + timedelta(days=int(rng.integers(0, 1500))),
                "lat": rng.uniform(-2, 2),
                "lon": rng.uniform(120, 124),
                "magnitude": 4.0 + rng.exponential(0.4),
            }
        )
    df = pd.DataFrame(rows)
    table = build_mc_lookup(df)
    assert len(table) >= 2
    # Lookup
    mc_palu = lookup_mc(table, lat=-0.9, lon=119.87, when=datetime(2013, 6, 1))
    assert mc_palu > 0


# -- declustering --------------------------------------------------------------

def test_decluster_separates_aftershock_sequence() -> None:
    """Mainshock + 3 aftershocks within 10km, 5 days → 1 mainshock, 3 aftershocks."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    events = pd.DataFrame(
        [
            {"event_id": "main", "time": base, "lat": -0.9, "lon": 119.87, "magnitude": 6.5, "depth": 10.0, "source": "usgs"},
            {"event_id": "after1", "time": base + timedelta(hours=2), "lat": -0.91, "lon": 119.88, "magnitude": 5.0, "depth": 10.0, "source": "usgs"},
            {"event_id": "after2", "time": base + timedelta(days=1), "lat": -0.85, "lon": 119.85, "magnitude": 4.5, "depth": 10.0, "source": "usgs"},
            {"event_id": "after3", "time": base + timedelta(days=4), "lat": -0.92, "lon": 119.90, "magnitude": 4.2, "depth": 10.0, "source": "usgs"},
        ]
    )
    out = decluster(events)
    assert int(out["is_mainshock"].sum()) == 1
    assert out.iloc[0]["is_mainshock"] is np.True_ or out.iloc[0]["is_mainshock"]
    assert int((~out["is_mainshock"]).sum()) == 3


def test_decluster_keeps_distant_events_independent() -> None:
    """Two large events 1000+ km apart, days apart → both mainshocks."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    events = pd.DataFrame(
        [
            {"event_id": "a", "time": base, "lat": -0.9, "lon": 119.87, "magnitude": 6.0, "depth": 10.0, "source": "usgs"},
            {"event_id": "b", "time": base + timedelta(days=2), "lat": 5.5, "lon": 95.3, "magnitude": 6.0, "depth": 10.0, "source": "usgs"},
        ]
    )
    out = decluster(events)
    assert int(out["is_mainshock"].sum()) == 2
