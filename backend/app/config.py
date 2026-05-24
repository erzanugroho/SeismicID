"""Application configuration loaded from environment variables.

All paths are resolved relative to project root (parent of `backend/`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # App
    app_name: str = "gempa-forecast"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: str = "INFO"
    app_role: str = "web"  # web|worker|combined

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Data paths (relative to project root, resolved to absolute on access)
    data_dir: str = "data"
    parquet_dir: str = "data/parquet"
    sqlite_path: str = "data/sqlite/gempa.db"
    models_dir: str = "data/models"
    geo_dir: str = "data/geo"

    # Indonesia bounding box
    grid_lat_min: float = -11.0
    grid_lat_max: float = 6.0
    grid_lon_min: float = 95.0
    grid_lon_max: float = 141.0
    grid_step: float = 0.5

    # External APIs
    usgs_base_url: str = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    bmkg_autogempa_url: str = "https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"
    bmkg_terkini_url: str = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"
    bmkg_dirasakan_url: str = "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json"

    # Admin / protected actions
    admin_token: str | None = None
    admin_job_cooldown_seconds: int = 300
    admin_retrain_cooldown_seconds: int = 86400

    # Scheduler
    sched_realtime_fetch_min: int = 10
    sched_forecast_recompute_min: int = 180
    sched_retrain_cron_day: str = "sun"
    sched_retrain_cron_hour: int = 2

    # Forecast worker policy
    forecast_trigger_mode: str = "any_new_event"
    forecast_fetch_interval_minutes: int = 10
    forecast_debounce_minutes: int = 5
    forecast_fallback_hours: int = 3

    # GPU
    use_gpu: bool = False

    # Forecast defaults
    default_horizon_days: int = 30
    default_mag_threshold: float = 5.0

    # CORS
    cors_allow_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # Resolved paths (absolute)
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_path(self) -> Path:
        return self._resolve(self.data_dir)

    @property
    def parquet_path(self) -> Path:
        return self._resolve(self.parquet_dir)

    @property
    def sqlite_full_path(self) -> Path:
        return self._resolve(self.sqlite_path)

    @property
    def models_path(self) -> Path:
        return self._resolve(self.models_dir)

    @property
    def geo_path(self) -> Path:
        return self._resolve(self.geo_dir)

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def ensure_dirs(self) -> None:
        """Create all data directories if missing."""
        for path in (
            self.data_path,
            self.parquet_path,
            self.parquet_path / "forecast_archive",
            self.sqlite_full_path.parent,
            self.models_path,
            self.geo_path,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
