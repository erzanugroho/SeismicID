"""End-to-end smoke test: when ML model returns empty AND the ETAS tier flag
is on, ``run_forecast`` must pick the ETAS-Ogata branch and archive the run
with ``baseline_type='etas'``.

This is the integration counterpart to ``test_forecast_service_etas_tier.py``
which only exercised the tier helper function. Here we drive the full
``run_forecast`` pipeline with a monkeypatched ``predict_all`` to simulate
"ML model unavailable" without needing a trained artifact on disk.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest


def _seed_events(n: int = 60) -> None:
    """Seed the historical parquet that ``run_forecast`` reads from.

    ``read_historical_events`` reads parquet, not the realtime sqlite table,
    so we go through ``append_historical_events`` to drop a synthetic ~5-year
    catalog onto disk.
    """
    from backend.app.data.catalog import append_historical_events

    rng = np.random.default_rng(11)
    base = datetime(2022, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(n):
        t = base + timedelta(days=float(rng.uniform(0, 365 * 4)))
        rows.append(
            {
                "event_id": f"smoke_{i:04d}",
                "time": t,
                "lat": float(rng.uniform(-8, 6)),
                "lon": float(rng.uniform(95, 141)),
                "depth": float(rng.uniform(5, 80)),
                "magnitude": float(rng.uniform(4.5, 6.0)),
                "mag_type": "mw",
                "source": "usgs",
                "place": "Smoke Test",
            }
        )
    append_historical_events(pd.DataFrame(rows))


def test_run_forecast_falls_back_to_etas_when_flag_on_and_ml_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ML returns empty + flag ON ⇒ tier must be ETAS-Ogata, archive tagged 'etas'."""
    from backend.app.config import get_settings
    from backend.app.services import forecast_service

    settings = get_settings()
    monkeypatch.setattr(settings, "enable_etas_baseline_tier", True, raising=False)

    # Bootstrap area labels (run_forecast does this lazily but ensure rows exist).
    from backend.app.services.area_service import bootstrap_area_labels

    bootstrap_area_labels()

    _seed_events(60)

    # Force ML path to return empty so the fallback ladder kicks in.
    def _fake_predict_all(features, **kwargs):  # noqa: ANN001
        return pd.DataFrame(), None

    monkeypatch.setattr(forecast_service, "predict_all", _fake_predict_all)

    summary = forecast_service.run_forecast()

    assert summary["mode"] == "etas_ogata", (
        f"expected ETAS tier to fire, got mode={summary['mode']!r}"
    )
    assert summary["rows_written"] > 0

    # Verify the archive parquet recorded baseline_type='etas'.
    from backend.app.data.catalog import read_forecast_archive

    today = datetime.now(UTC).date()
    archive = read_forecast_archive(today)
    assert "baseline_type" in archive.columns
    assert (archive["baseline_type"] == "etas").all(), (
        f"archive baseline_type values: {archive['baseline_type'].unique().tolist()}"
    )


def test_run_forecast_falls_back_to_poisson_when_flag_off_and_ml_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag OFF ⇒ tier must skip ETAS and go straight to Poisson baseline."""
    from backend.app.config import get_settings
    from backend.app.services import forecast_service

    settings = get_settings()
    monkeypatch.setattr(settings, "enable_etas_baseline_tier", False, raising=False)

    from backend.app.services.area_service import bootstrap_area_labels

    bootstrap_area_labels()
    _seed_events(40)

    def _fake_predict_all(features, **kwargs):  # noqa: ANN001
        return pd.DataFrame(), None

    monkeypatch.setattr(forecast_service, "predict_all", _fake_predict_all)

    summary = forecast_service.run_forecast()

    # When ETAS flag is OFF, tier collapses to Poisson which then gets
    # public-calibrated, so the final mode is the calibrated variant.
    assert summary["mode"].startswith("poisson_baseline"), (
        f"expected Poisson tier with flag OFF, got mode={summary['mode']!r}"
    )

    from backend.app.data.catalog import read_forecast_archive

    today = datetime.now(UTC).date()
    archive = read_forecast_archive(today)
    assert (archive["baseline_type"] == "poisson").all()
