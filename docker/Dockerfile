# Multi-stage build for SeismicID (gempa-forecast).
# Replaces Nixpacks — see railway-deployments skill for rationale:
# Nixpacks /nix/store/<hash> paths rot on every rebuild, breaking
# LD_LIBRARY_PATH for numpy/lightgbm/xgboost native libs. Docker pins
# everything reproducibly.
#
# rasterio/pyproj/shapely ship manylinux wheels with bundled GDAL/PROJ/GEOS,
# so no apt GDAL is required — only libgomp1 for the OpenMP runtime that
# lightgbm/xgboost/scipy link against.

FROM python:3.11.10-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts
# Volume-backed dirs are created at runtime too, but make them exist for
# local/no-volume runs so ensure_dirs() never races.
RUN mkdir -p /app/data/parquet/forecast_archive /app/data/sqlite \
             /app/data/models /app/data/geo
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1
# Railway injects $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
