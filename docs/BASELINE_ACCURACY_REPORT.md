# SeismicID Baseline Accuracy Report

Captured: 2026-06-09 UTC

## Recovery point

- Git commit: `c0d3b0e`
- Git tag: `safe-pre-accuracy-sprints-20260609`
- Railway DB backup: `/data/backups/manual/gempa_safe_pre_accuracy_20260609T102212Z.db`
- DB integrity: `ok`
- `current_forecasts`: 50,048 rows
- `realtime_events`: 2,543 rows
- `area_labels`: 3,128 cells
- `telegram_user_locations`: 2 users

## Operational proxy hit rate — H30 Top 10% cells

These compare current high-risk map against recent observed events. They are operational proxy metrics, not full prospective CSEP validation.

| Threshold | Hit rate 30d | Hit rate 365d | False alarm rate | Precision | Observed 30d | Observed 365d | Hit cells 365d |
|---|---:|---:|---:|---:|---:|---:|---:|
| M≥5.0 | 83.33% | 81.19% | 92.95% | 7.05% | 90 | 101 | 22 |
| M≥5.5 | 100.00% | 100.00% | 96.47% | 3.53% | 29 | 32 | 11 |
| M≥6.0 | 100.00% | 100.00% | 98.08% | 1.92% | 12 | 14 | 6 |

Interpretation:

- Top-10%-area hit rate is high.
- False alarm is also high because earthquakes are sparse and high-risk cells are many.
- These metrics are useful for public awareness coverage, not exact prediction skill.

## Pre-event rank baseline — last snapshot before event

### M≥5.0

Valid events with pre-event snapshots: 26.

| Horizon | Exact Top10 | Exact Top50 | Exact Top100 | Cluster Top10 300km | Median exact rank |
|---|---:|---:|---:|---:|---:|
| H7 | 38.46% | 50.00% | 57.69% | 88.46% | #86 |
| H14 | 30.77% | 42.31% | 61.54% | 88.46% | #70 |
| H30 | 26.92% | 42.31% | 46.15% | 92.31% | #164 |
| H60 | 38.46% | 65.38% | 69.23% | 92.31% | #19 |

### M≥5.5

Valid events with pre-event snapshots: 10.

| Horizon | Exact Top10 | Exact Top50 | Exact Top100 | Cluster Top10 300km | Median exact rank |
|---|---:|---:|---:|---:|---:|
| H7 | 30.00% | 50.00% | 70.00% | 80.00% | #79 |
| H14 | 30.00% | 60.00% | 70.00% | 80.00% | #46 |
| H30 | 30.00% | 50.00% | 50.00% | 80.00% | #255 |
| H60 | 30.00% | 40.00% | 50.00% | 90.00% | #105 |

### M≥6.0

Valid events with pre-event snapshots: 3.

| Horizon | Exact Top10 | Exact Top50 | Exact Top100 | Cluster Top10 300km | Median exact rank |
|---|---:|---:|---:|---:|---:|
| H7 | 0.00% | 33.33% | 33.33% | 100.00% | #118 |
| H14 | 0.00% | 33.33% | 33.33% | 66.67% | #116 |
| H30 | 0.00% | 33.33% | 66.67% | 100.00% | #89 |
| H60 | 0.00% | 33.33% | 33.33% | 33.33% | #285 |

## Main conclusions

1. Exact-cell ranking is weak for M≥6.
2. Cluster/regional signal is strong, especially 300 km cluster.
3. Operational hit rate is high but false alarm remains high.
4. Future work should improve:
   - canonical event dedupe
   - cluster rank/probability
   - rate acceleration features
   - spatial label smoothing
   - ETAS + tectonic prior hybrid model
   - calibration per horizon/magnitude

## Rollback

Code rollback:

```bash
git reset --hard safe-pre-accuracy-sprints-20260609
git push --force-with-lease origin main
```

Railway DB rollback, if needed:

```bash
cp /data/backups/manual/gempa_safe_pre_accuracy_20260609T102212Z.db /data/sqlite/gempa.db
```
