# SeismicID Model Card

## Model summary

SeismicID estimates per-grid-cell earthquake probabilities for Indonesia across four horizons (7/14/30/60 days) and four magnitude thresholds (M≥4.5/5.0/5.5/6.0). The production path uses an ensemble of XGBoost, LightGBM, and a Poisson rate baseline with optional post-hoc compression for heads that ship with an `IdentityCalibrator`. Skill is reported against **two baselines**: a flat Poisson rate model and a temporal Ogata 1988 ETAS model with isotropic spatial kernel. **All output is treated as experimental relative-risk ranking, not an official early-warning probability.**

## Intended use

- Public educational/research exploration of probabilistic earthquake-risk ranking.
- Internal monitoring of cached model freshness and prospective evaluation over time.
- Comparing **relative spatial risk patterns** across cells; not deterministic event prediction.

## Not intended use

- Emergency alerting, evacuation decisions, public warning issuance, or any official early-warning workflow.
- Claiming certainty about exact earthquake time, location, or magnitude.
- Replacing BMKG or any other competent authority.

## Data sources

- USGS earthquake catalog for historical and realtime events.
- BMKG public feeds as an additional Indonesian realtime source where available.
- Geologic features include hardcoded/fallback fault and slab approximations unless real GEM Active Faults / Slab2.0 assets are installed locally. The fallback covers only ~16 major regional faults; slab depth is an analytical approximation outside Slab2.0 grid coverage.

## Targets and outputs

Each grid cell has 16 binary targets — for every (horizon, magnitude threshold) combination — capturing whether at least one event above that threshold occurs within the horizon. The service exposes cached probabilities, the active model version, and the run timestamp via API endpoints. Probabilities are constrained to be:

- Monotone in horizon for fixed threshold: `P(60d) ≥ P(30d) ≥ P(14d) ≥ P(7d)`.
- Monotone (decreasing) in threshold for fixed horizon: `P(M≥4.5) ≥ P(M≥5.0) ≥ P(M≥5.5) ≥ P(M≥6.0)`.

Both constraints are enforced post-prediction by `enforce_probability_monotonicity`.

## Baselines

Skill claims compare the ensemble against two baselines, not just one:

1. **Poisson** — flat per-cell rate fit on the training window, scaled by horizon and threshold via Gutenberg-Richter (b=1 default). Captures only background seismicity. Cheap to fit, easy to beat, but a weak benchmark on its own.
2. **ETAS-Ogata** — temporal Ogata 1988 likelihood (μ, K, c, p, α) fit by L-BFGS-B MLE, paired with an isotropic power-law spatial kernel and per-region b-value (Aki-Utsu MLE) when available. Captures aftershock clustering, so beating it on horizons 7-14 days and on cells with recent activity is a meaningful claim.

The ETAS-Ogata implementation here is honest about its scope: it is the temporal Ogata likelihood plus an isotropic spatial kernel, **not** a publication-grade ETAS with anisotropic faulting, joint spatial MLE, or hierarchical pooling (those live in deferred Phase 5 work). It is, however, materially stronger than the Poisson baseline for any window where aftershock clustering matters.

## Calibration and Bayesian blending

- Per-head calibrators (Platt / Isotonic / Beta) are picked by validation Brier score where labelled validation data is available.
- Heads that fall back to `IdentityCalibrator` are individually compressed against an empirical base rate (per-head, never globally — this was an explicit audit fix).
- The Bayesian blend uses the Poisson baseline as a prior **only when the prior is positive and finite**. Cells with missing/zero priors retain the calibrated estimate instead of being shrunk toward zero.

## Forecast archive (immutable, prospective)

Every successful forecast run writes a fresh Parquet file under `data/parquet/forecast_archive/<YYYY-MM-DD>/<HHMMSSZ>_<model_version>.parquet` and tags every row with `forecast_run_id`, `issued_at_utc`, and `model_version`. The legacy single-file-per-day layout is still readable for backward compatibility but is no longer written. This protects prospective evaluation from hindsight-leakage caused by the previous overwriting behaviour.

## Evaluation

Retrospective metrics include Brier score, ROC-AUC, calibration/reliability, BSS vs Poisson **and** BSS vs ETAS-Ogata, information gain in bits/event vs each baseline, CSEP-style L/N/S tests (now two-sided: pass requires `0.025 ≤ quantile ≤ 0.975`), and Molchan diagrams. **Production claims should always lean on prospective evaluation across the immutable archive** rather than retrospective metrics on the training period.

## Limitations

- Earthquake occurrence is highly uncertain; per-cell probabilities are small and easily misinterpreted as guarantees.
- The PUSGEN/GEM fault database and Slab2.0 grid are not bundled; without them feature quality is reduced.
- Retrospective validation can overstate performance if not strictly separated from model development.
- Cached forecasts can become stale if the scheduler/worker fails; readiness probes report this state.
- Demo-seed mode emits physics-aware placeholders, not ML predictions.
- The ETAS-Ogata baseline uses an isotropic spatial kernel and a single completeness magnitude per region. Anisotropic faulting, joint spatial MLE, and hierarchical regional pooling are deferred (Phase 5). It is materially stronger than the Poisson baseline for clustering windows but is not a publication-grade ETAS reference.

## Safety disclaimer

This is experimental research software, not an official early-warning system. Low probability does **not** mean safe; high probability does **not** mean an earthquake will occur. Use BMKG and other competent authorities for safety-critical information.
