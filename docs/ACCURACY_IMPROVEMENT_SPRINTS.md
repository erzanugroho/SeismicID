# SeismicID Accuracy Improvement Sprints

Safe point before work:

- Git commit: `c0d3b0e`
- Git tag: `safe-pre-accuracy-sprints-20260609`
- Railway DB backup: `/data/backups/manual/gempa_safe_pre_accuracy_20260609T102212Z.db`
- DB integrity at backup time: `ok`

Rollback:

```bash
git checkout main
git reset --hard safe-pre-accuracy-sprints-20260609
git push --force-with-lease origin main
# Railway volume DB restore, if needed:
# cp /data/backups/manual/gempa_safe_pre_accuracy_20260609T102212Z.db /data/sqlite/gempa.db
```

## Sprint 0 — Baseline lock

Goal: freeze current metrics before accuracy experiments.

Done:

- Recovery tag created.
- Railway DB backup created.
- Live metrics captured.

## Sprint 1 — Pre-event evaluation v2

Goal: make evaluation more scientific before changing training/model logic.

Scope:

- Extend `/api/model/pre-event-backtest`.
- Add thresholds: M≥5.0, M≥5.5, M≥6.0.
- Add metrics:
  - Top10/25/50/100 hit rate
  - MRR
  - NDCG@10
  - exact cell rank
  - neighbor ring 1 / ring 2 best rank
  - cluster 100 km / 300 km best rank
- Update backtest UI with these metrics.

Conflict risk: low. Evaluation-only.

## Sprint 2 — Canonical event dedupe

Goal: clean input event data before feature/model changes.

Scope:

- Add `canonical_events` table.
- Dedupe BMKG/USGS/EMSC using time ±10 minutes, distance ≤100 km, magnitude difference ≤0.5.
- Keep `realtime_events` untouched.
- Use canonical layer in evaluation first, not live forecast.

Conflict risk: medium but additive.

## Sprint 3 — Canonical evaluation

Goal: switch evaluation endpoints to canonical events with fallback.

Scope:

- Update `pre-event-backtest`, `performance-v2`, `/backtest`.
- Fallback to `realtime_events` if canonical table empty.
- Compare before/after metrics.

## Sprint 4 — Cluster probability/rank

Goal: make regional signal official.

Scope:

- Add cluster 100 km / 300 km probability and rank in API.
- Add exact vs cluster rank to UI and Telegram M≥6 context.

## Sprint 5 — Short-term seismicity features

Goal: add rate acceleration and activity features from clean canonical events.

Scope:

- counts by magnitude/radius/window
- energy release
- rate ratios
- nearest M5/M6 distance/time

Feature flag: `FEATURE_SET=v1|v2`.

## Sprint 6 — Spatial label smoothing

Goal: train model to learn regional risk, not only exact cell.

Feature flag: `LABEL_MODE=exact|spatial_smooth`.

## Sprint 7 — ETAS component

Goal: add aftershock point-process signal.

Output:

- `etas_intensity`
- `etas_rank`
- parent event metadata

## Sprint 8 — Baseline tectonic prior

Goal: stabilize rare-event M≥6 and H30/H60.

Start with:

- historical M5/M6 density
- historical max magnitude
- time since last M6

## Sprint 9 — Hybrid ensemble

Goal: combine ML + baseline + ETAS.

Feature flag: `FORECAST_MODEL_MODE=v1|v2_hybrid`.

## Sprint 10 — Calibration per horizon+threshold

Goal: improve probability quality.

Metric:

- Brier score
- ECE
- reliability curve

## Sprint 11 — Ranking objective + hard negatives

Goal: improve TopK ranking and reduce false alarms.

Metric:

- MRR
- NDCG@10
- TopK recall
- false alarm top cells

## Sprint 12 — Uncertainty interval

Goal: add probability low/mid/high + confidence.

## Sprint 13 — UI/Telegram wording

Goal: use “regional risk” and exact/cluster language to avoid overclaim.

## Sprint 14 — A/B shadow

Goal: compare v1 vs v2 for several days.

## Sprint 15 — Safe switch

Goal: switch live only if v2 beats v1 across key metrics and rollback stays ready.
