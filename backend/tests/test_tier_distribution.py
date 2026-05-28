"""Observability: ``get_tier_distribution`` counts archive runs per tier.

Drops a few synthetic per-run parquet files into the configured archive
directory and asserts the helper buckets them by ``baseline_type`` over
the requested time window.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_run(
    archive_root: Path,
    *,
    issued_at: datetime,
    baseline_type: str,
    model_version: str = "v_test",
    n_cells: int = 4,
) -> Path:
    """Write a minimal per-run archive parquet matching the production layout."""
    day_dir = archive_root / issued_at.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{issued_at:%H%M%S}Z_{model_version}.parquet"
    path = day_dir / fname

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "cell_id": [f"C{i:03d}" for i in range(n_cells)],
            "label_h7_m45": rng.uniform(0, 0.1, size=n_cells),
            "computed_at": [issued_at.isoformat()] * n_cells,
            "model_version": [model_version] * n_cells,
            "baseline_type": [baseline_type] * n_cells,
            "forecast_run_id": [f"{issued_at:%Y-%m-%dT%H%M%S}Z_{model_version}"] * n_cells,
        }
    )
    df.to_parquet(path)
    return path


@pytest.fixture
def archive_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point Settings.parquet_path at tmp_path and return the archive root."""
    monkeypatch.setenv("PARQUET_DIR", str(tmp_path))
    from backend.app.config import get_settings

    get_settings.cache_clear()
    root = tmp_path / "forecast_archive"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_tier_distribution_counts_runs_by_baseline_type(archive_root: Path) -> None:
    from backend.app.services.forecast_service import get_tier_distribution

    now = datetime.now(UTC)
    _write_run(archive_root, issued_at=now - timedelta(hours=1), baseline_type="ml")
    _write_run(archive_root, issued_at=now - timedelta(hours=2), baseline_type="ml")
    _write_run(archive_root, issued_at=now - timedelta(hours=3), baseline_type="etas")
    _write_run(archive_root, issued_at=now - timedelta(hours=5), baseline_type="poisson")

    out = get_tier_distribution(hours=24)
    assert out["total_runs"] == 4
    assert out["by_tier"] == {"ml": 2, "etas": 1, "poisson": 1}
    assert len(out["runs"]) == 4
    # Newest first (within day).
    assert out["runs"][0]["baseline_type"] == "ml"


def test_tier_distribution_excludes_runs_outside_window(archive_root: Path) -> None:
    from backend.app.services.forecast_service import get_tier_distribution

    now = datetime.now(UTC)
    _write_run(archive_root, issued_at=now - timedelta(hours=1), baseline_type="etas")
    _write_run(archive_root, issued_at=now - timedelta(days=3), baseline_type="ml")

    out = get_tier_distribution(hours=24)
    assert out["total_runs"] == 1
    assert out["by_tier"] == {"etas": 1}


def test_tier_distribution_empty_archive_returns_zero(archive_root: Path) -> None:
    from backend.app.services.forecast_service import get_tier_distribution

    out = get_tier_distribution(hours=24)
    assert out["total_runs"] == 0
    assert out["by_tier"] == {}
    assert out["runs"] == []
