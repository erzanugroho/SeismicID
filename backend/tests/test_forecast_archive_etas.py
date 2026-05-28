"""Tests for ETAS forecast archive (Phase 4 Task 4.1)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.features.labels import all_label_columns


def _toy_forecast_frame(n_cells: int = 6) -> pd.DataFrame:
    cols = all_label_columns()
    cells = [f"C+0{i}_p{i:03d}" for i in range(n_cells)]
    df = pd.DataFrame(
        {c: np.full(n_cells, 0.05, dtype=float) for c in cols}
    )
    df["cell_id"] = cells
    return df


def test_archive_writes_baseline_type_etas(tmp_path: Path) -> None:
    from backend.app.config import Settings
    from backend.app.data.catalog import archive_forecast, read_forecast_archive

    settings = Settings(parquet_path=str(tmp_path))

    df = _toy_forecast_frame()
    issued = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    path = archive_forecast(
        df,
        day=issued.date(),
        settings=settings,
        model_version="etas_v1",
        issued_at=issued,
        baseline_type="etas",
    )
    assert path.exists()
    out = read_forecast_archive(issued.date(), settings=settings)
    assert "baseline_type" in out.columns
    assert (out["baseline_type"] == "etas").all()


def test_archive_default_baseline_type_is_ml(tmp_path: Path) -> None:
    """Backward compat: legacy callers without baseline_type get 'ml'."""
    from backend.app.config import Settings
    from backend.app.data.catalog import archive_forecast, read_forecast_archive

    settings = Settings(parquet_path=str(tmp_path))

    df = _toy_forecast_frame()
    issued = datetime(2024, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    archive_forecast(
        df, day=issued.date(), settings=settings,
        model_version="ml_v1", issued_at=issued,
    )
    out = read_forecast_archive(issued.date(), settings=settings)
    assert "baseline_type" in out.columns
    assert (out["baseline_type"] == "ml").all()
