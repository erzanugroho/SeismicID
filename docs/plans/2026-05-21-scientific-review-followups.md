# Scientific Review Follow-ups Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn the review notes into concrete code, documentation, data, and evaluation improvements without overstating earthquake forecast skill.

**Architecture:** Prioritize low-risk correctness first (README, CI, vectorized cell assignment), then scientific data upgrades (GEM faults, Slab2.0), then evaluation/prospective archive and UI scalability. Keep backward compatibility for public APIs/model bundles while renaming misleading internal concepts toward Poisson baseline.

**Tech Stack:** Python/FastAPI, pandas/geopandas/xarray, SQLite/Postgres, pytest/ruff/mypy, GitHub Actions, Leaflet frontend.

---

## Phase 0: Done in this pass

### Task 0.1: Fix stale README test count and coverage wording

**Files:**
- Modified: `README.md`

**Verification:**
- AST count found 84 `test_*` functions across 14 files.
- README now says to measure coverage with `make test-cov` instead of hardcoding `~80%`.

### Task 0.2: Remove row-wise `df.apply` cell assignment path

**Files:**
- Modified: `backend/app/features/builder.py`

**Verification:**
- `assign_cell_id()` now delegates to shared `assign_cell_id_vec()`.
- No `df.apply(to_id, axis=1)` remains in builder.

### Task 0.3: Add minimal CI workflow

**Files:**
- Created: `.github/workflows/ci.yml`

**Verification:**
- CI runs dependency install, `make lint`, and `make test` on push/PR to `main`.

### Task 0.4: Make Poisson baseline naming explicit

**Files:**
- Modified: `backend/app/ml/etas.py`
- Modified: `README.md`

**Verification:**
- Added `PoissonBaseline` class with backward-compatible `ETASBaseline = PoissonBaseline` alias.
- README now calls C1 a Poisson baseline instead of ETAS.

---

## Phase 1: Finish naming cleanup without breaking APIs

### Task 1.1: Rename internal variables from `etas_*` to `poisson_*`

**Files:**
- Modify: `backend/app/services/forecast_service.py`
- Modify: `backend/app/ml/ensemble.py`
- Modify: `backend/app/ml/predict.py`
- Modify tests in `backend/tests/test_ml.py`

**Steps:**
1. Keep public response mode `etas_only` temporarily if UI/API depends on it, but add a new mode alias `poisson_only` in docs.
2. Rename comments/docstrings to Poisson baseline.
3. Keep config key `weight_etas` for backward compatibility, add `weight_poisson` only if model config migration is handled.
4. Run `make test`.

### Task 1.2: Add deprecation note for old `ETASBaseline` import

**Files:**
- Modify: `backend/app/ml/etas.py`
- Test: `backend/tests/test_ml.py`

**Steps:**
1. Add docstring/comment that `ETASBaseline` is deprecated alias.
2. Add a test that `ETASBaseline is PoissonBaseline`.
3. Run `pytest backend/tests/test_ml.py -q`.

---

## Phase 2: Scientific data assets

### Task 2.1: Replace hardcoded active faults with GEM Active Faults loader

**Files:**
- Modify: `backend/app/geo/fault_db.py`
- Modify: `scripts/download_geo_assets.py`
- Create: `backend/tests/test_fault_db_gem.py`

**Steps:**
1. Add config path `data/geo/gem_active_faults.*`.
2. If shapefile/GeoJSON exists, load it with geopandas, normalize fields: name, fault_type, slip_rate_mm_yr, geometry.
3. Fall back to existing hardcoded faults when missing.
4. Keep `has_real_pusgen()` or rename to `has_real_fault_db()` with compatibility wrapper.
5. Test fallback and real-file path using a tiny synthetic GeoJSON fixture.

### Task 2.2: Make Slab2.0 grid the preferred path and analytical model fallback-only

**Files:**
- Modify: `backend/app/geo/slab_model.py`
- Modify: `scripts/download_geo_assets.py`
- Create/update tests in `backend/tests/test_physics.py`

**Steps:**
1. Support local Slab2 `.grd`/NetCDF via xarray where available.
2. Cover Indonesia subduction zones: Sunda, Banda, Philippines/Molucca as applicable.
3. Return `source='slab2'` vs `source='analytical_fallback'` in metadata if possible.
4. README should state analytical fallback is not scientifically defensible for production.

---

## Phase 3: Prospective evaluation and metrics

### Task 3.1: Make forecast archive immutable enough for CSEP-style prospective evaluation

**Files:**
- Modify: forecast persistence code in `backend/app/services/forecast_service.py` or storage layer
- Modify: `backend/app/ml/evaluate.py`
- Create tests for archive write/read behavior

**Steps:**
1. Store every scheduled forecast under `data/parquet/forecast_archive/YYYY-MM-DD.parquet` before future events are known.
2. Include generated_at, model_version, data_cutoff, horizon, threshold, cell_id, probability.
3. Prevent overwrite unless admin explicitly passes `force=true` and log it.
4. Add evaluator that only scores archives generated before the target event window.

### Task 3.2: Add information gain vs declustered Poisson as primary metric

**Files:**
- Modify: `backend/app/ml/evaluate.py`
- Modify: model metadata endpoint/UI if metrics are surfaced
- Add tests with toy probabilities

**Steps:**
1. Compute log-likelihood model and Poisson baseline with epsilon clipping.
2. Report information gain per event and aggregate bits/event.
3. Keep AUC/Brier as secondary diagnostics.
4. Add README note explaining why AUC is not primary for rare earthquake labels.

---

## Phase 4: UI/API scalability

### Task 4.1: Add server-side probability threshold for latest forecasts

**Files:**
- Modify: `backend/app/api/routes/forecasts.py`
- Modify: storage query layer if needed
- Modify: `frontend/index.html` / JS fetch code
- Add tests in `backend/tests/test_forecasts.py`

**Steps:**
1. Add query parameter `min_probability` defaulting to `0` for backward compatibility.
2. In frontend, request a low threshold for map rendering and keep top-N endpoint for high-risk list.
3. Add UI copy indicating hidden low-probability cells at current zoom/threshold.

### Task 4.2: Plan vector tiles or zoom aggregation

**Files:**
- Create a separate technical design before implementation.

**Steps:**
1. Benchmark current GeoJSON payload size and mobile render time.
2. Choose between `leaflet.vectorgrid`, server-side simplified polygons, or zoom-level aggregation.
3. Implement only after benchmark confirms bottleneck.

---

## Phase 5: Storage scaling

### Task 5.1: Add Postgres option before public traffic grows

**Files:**
- Modify DB abstraction/config.
- Add Railway deployment docs.

**Steps:**
1. Keep SQLite as local/demo default.
2. Add `DATABASE_URL` Postgres path with migrations.
3. Test current forecast write/read under concurrent requests.

---

## Verification commands

```bash
make lint
make test
make test-cov
```

If local dependencies are missing, first run:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```
