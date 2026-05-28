# ETAS-Ogata Pipeline Runbook

## Scope

Operational notes for the dual-baseline pipeline introduced in plan
`docs/plans/2026-05-28-real-etas-ogata.md`. Covers when to refit, how to
monitor parameter drift, and how to respond when the MLE diverges. Does
**not** repeat product details — see `MODEL_CARD.md` for the scientific
description and `backend/app/ml/etas_ogata.py` for the implementation.

## When ETAS is fit

1. **Each retrain run** (`scripts/train_initial.py`) — fits ETAS over the
   same TRAIN window as the Poisson baseline (no leakage). Skill metrics
   `bss_vs_etas` and `info_gain_vs_etas` land in `evaluation_results` next
   to the Poisson scores.
2. **Each forecast tier fall-through** (`backend/app/services/forecast_service.py`)
   — only when `enable_etas_baseline_tier=True` AND no ML model is loaded
   AND events are available. Default OFF in production.

## Monitor

- `model_metadata.metrics_json` and `evaluation_results` rows where
  `eval_type='skill'` carry `bss_vs_poisson` and `bss_vs_etas` per head.
  Healthy: `bss_vs_etas > 0` on at least the M≥4.5 horizons. Negative
  consistently means the ensemble is trailing a clustering-aware baseline
  — investigate calibration, not the ETAS code.
- ETAS parameters logged at fit time. Plausible ranges (Indonesia
  catalog, 5y window):
  - μ: 0.05 – 1.0 events/day catalog-wide
  - K: 0.01 – 1.0
  - c: 0.001 – 0.5 days
  - p: 0.8 – 1.5
  - α: 0.5 – 2.5
  Any param at a bound for two consecutive runs ⇒ check catalog quality
  before widening bounds.
- Parameter drift session-over-session > 50% on μ or p without a
  concurrent catalog change is a smell. Inspect with
  `scripts/analyze_etas_aftershock_decay.py`.

## When the MLE diverges

1. Check L-BFGS-B `success` flag in logs — the wrapper reports
   `mle_status`. `warn:Maximum number of iterations` usually means
   the catalog is too small (< ~50 events above Mc).
2. Bump `mc` higher (5.0) if catalog completeness is the issue. The
   Aki-Utsu `b` estimator falls back to default 1.0 when there are
   < 50 events.
3. If still failing, the temporal-only fit can be sanity-checked against
   the synthetic catalog test (`backend/tests/test_etas_ogata_fit.py`).
   That suite must remain green; if it breaks, treat as a regression
   rather than a data issue.
4. Reference cross-validation: `pip install etas` then run
   `pytest backend/tests/test_etas_cross_validation.py -v` to compare
   fits against the Mizrahi 2023 reference library on synthetic data.
   Tolerance is ±30% on parameters; failures suggest a likelihood bug.

## Rollout flag

`backend/app/config.py::Settings.enable_etas_baseline_tier` (default
`False`). To enable in a deployment, set the env var
`ENABLE_ETAS_BASELINE_TIER=1` and restart the uvicorn process —
`get_settings()` is `lru_cache`-decorated.

When enabled, forecast archives carry `baseline_type='etas'` rows so
prospective evaluators can score ETAS runs separately.

## Known limitations

See `MODEL_CARD.md` "Limitations" — isotropic spatial kernel, single Mc
per region, no anisotropic faulting, no joint spatial MLE. Phase 5 in the
plan covers the deferred work.
