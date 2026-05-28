"""Shared pytest fixtures."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_data_dirs(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Per-test isolated data directory so tests never touch real data/."""
    with tempfile.TemporaryDirectory(prefix="gempa-test-") as td:
        tmp = Path(td)
        for sub in ("parquet", "parquet/forecast_archive", "sqlite", "models", "geo"):
            (tmp / sub).mkdir(parents=True, exist_ok=True)

        monkeypatch.setenv("DATA_DIR", str(tmp))
        monkeypatch.setenv("PARQUET_DIR", str(tmp / "parquet"))
        monkeypatch.setenv("SQLITE_PATH", str(tmp / "sqlite" / "gempa.db"))
        monkeypatch.setenv("MODELS_DIR", str(tmp / "models"))
        monkeypatch.setenv("GEO_DIR", str(tmp / "geo"))
        # Disable scheduler in tests so test client doesn't spin up jobs
        monkeypatch.setenv("DISABLE_SCHEDULER", "1")
        monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
        monkeypatch.setenv("FORECAST_TRIGGER_MODE", "any_new_event")
        monkeypatch.setenv("FORECAST_FETCH_INTERVAL_MINUTES", "10")
        monkeypatch.setenv("FORECAST_DEBOUNCE_MINUTES", "5")
        monkeypatch.setenv("FORECAST_FALLBACK_HOURS", "3")
        monkeypatch.setenv("USE_GPU", "false")

        # Clear cached settings so override takes effect
        from backend.app.config import get_settings

        get_settings.cache_clear()

        # Initialize the SQLite schema so tests that touch the DB directly
        # (without going through FastAPI startup) see the expected tables.
        from backend.app.db.sqlite import migrate as _migrate

        _migrate()

        # Make project root point inside tmp for tests writing relative paths
        old_cwd = os.getcwd()
        try:
            yield tmp
        finally:
            os.chdir(old_cwd)
            get_settings.cache_clear()
