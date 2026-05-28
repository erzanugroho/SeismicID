"""Initial training pipeline.

Steps:
1. Read declustered events (or fall back to historical).
2. Build feature dataset across snapshots.
3. Generate multi-horizon multi-threshold labels.
4. Time-based split (train/val/test).
5. Train XGBoost+LightGBM heads, calibrate, save.

Usage:
    python -m scripts.train_initial
    python -m scripts.train_initial --snap-freq 14   # change snapshot cadence
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd  # noqa: E402  (used in eval block below)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.grid import generate_grid  # noqa: E402
from backend.app.core.logging import configure_logging, get_logger  # noqa: E402
from backend.app.data.catalog import (  # noqa: E402
    read_declustered_events,
    read_historical_events,
)
from backend.app.features.builder import build_features_for_snapshots, default_snapshots  # noqa: E402
from backend.app.features.labels import build_labels, join_features_and_labels, time_split  # noqa: E402
from backend.app.ml.train import save_models, train_heads  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap-freq", type=int, default=14, help="snapshot cadence in days")
    parser.add_argument("--start-year", type=int, default=None, help="First snapshot year, e.g. 1990")
    parser.add_argument("--lookback-years", type=int, default=5, help="Used only when --start-year is omitted")
    parser.add_argument("--min-train-mag", type=float, default=None, help="Optional event filter before feature/label build")
    args = parser.parse_args()

    configure_logging("INFO")
    events = read_declustered_events()
    if events.empty:
        events = read_historical_events()
    if args.min_train_mag is not None and not events.empty:
        events = events[events["magnitude"] >= args.min_train_mag].copy()
    if events.empty:
        logger.error("no_events_run_bootstrap_first", hint="python scripts/bootstrap_data.py")
        return 1

    cells = generate_grid()
    end = datetime.now(timezone.utc)
    if args.start_year is not None:
        start = datetime(args.start_year, 1, 1, tzinfo=timezone.utc)
    else:
        start = end - timedelta(days=365 * args.lookback_years)
    snaps = default_snapshots(start, end, freq_days=args.snap_freq)
    logger.info(
        "training_init",
        n_events=len(events),
        n_snapshots=len(snaps),
        n_cells=len(cells),
        start=start.isoformat(),
        end=end.isoformat(),
    )

    feats = build_features_for_snapshots(events, snaps, cells=cells)
    labs = build_labels(events, snaps, cells=cells)
    dataset = join_features_and_labels(feats, labs)
    train, val, test = time_split(dataset)
    logger.info("split_done", train=len(train), val=len(val), test=len(test))

    if len(train) == 0 or len(val) == 0:
        logger.error("not_enough_data_after_split")
        return 1

    heads = train_heads(train, val)
    version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    # Save the model
    save_models(heads, version=version, dataset_size=len(train) + len(val) + len(test))
    
    # Evaluate model on test set if not empty
    if len(test) > 0:
        logger.info("evaluating_on_test_set", test_size=len(test))
        from backend.app.ml.ensemble import predict_ensemble
        from backend.app.ml.etas import PoissonBaseline
        from backend.app.ml.evaluate import evaluate_dataset
        from backend.app.ml.posthoc_calibration import compute_base_rates
        from backend.app.ml.train import save_evaluation_results

        try:
            # Build Poisson baseline + per-cell event counts on the TRAIN window
            # only — using the test window would leak future evidence into both
            # the prior and the BSS denominator. The baseline must be a
            # genuine alternative forecaster, not an oracle.
            train_events_mask = (
                events["time"] <= pd.to_datetime(train["snapshot"].max(), utc=True)
                if "snapshot" in train.columns and len(train) and "time" in events.columns
                else None
            )
            train_events = events.loc[train_events_mask] if train_events_mask is not None else events

            baseline_model = PoissonBaseline()
            obs_start = pd.to_datetime(train_events["time"], utc=True).min().to_pydatetime()
            obs_end = pd.to_datetime(train_events["time"], utc=True).max().to_pydatetime()
            baseline_model.fit(
                train_events,
                observation_start=obs_start,
                observation_end=obs_end,
            )
            test_cell_ids = test["cell_id"].astype(str).tolist()
            unique_test_cells = sorted(set(test_cell_ids))
            poisson_pred = baseline_model.predict_dataframe(unique_test_cells)

            # Per-cell event counts from the TRAIN window only — same rationale
            # as above. Mirrors what production sees (cell_event_counts is
            # computed from the historical catalog at forecast time).
            from backend.app.features.labels import assign_cell_id_vec

            train_evt_for_counts = train_events[train_events["magnitude"] >= 4.5]
            counts_df = assign_cell_id_vec(train_evt_for_counts, cells)
            counts_df = counts_df.dropna(subset=["cell_id"])
            cell_event_counts = (
                counts_df.groupby("cell_id").size().astype(int).to_dict()
                if not counts_df.empty
                else {}
            )

            # Empirical base rates for posthoc identity-calibrator fallback,
            # also from the train window only.
            base_rates = compute_base_rates(train_events, n_cells=len(cells))

            preds = predict_ensemble(
                heads,
                test,
                cell_ids=test_cell_ids,
                snapshots=test["snapshot"].tolist() if "snapshot" in test.columns else None,
                poisson_predictions=poisson_pred,
                cell_event_counts=cell_event_counts,
                base_rates=base_rates,
            )

            # Build a per-row baseline frame aligned to the test snapshots so
            # ``evaluate_dataset`` can compute Brier-Skill-Score per head.
            baseline_for_eval = poisson_pred.merge(
                test[["cell_id"]].drop_duplicates(),
                on="cell_id",
                how="right",
            )

            eval_out = evaluate_dataset(test, preds, baseline=baseline_for_eval)

            # Save metrics to DB
            skill_payload = {}
            roc_payload = {}
            reliability_payload = {}
            molchan_payload = {}
            csep_payload = {}

            for head, metrics in eval_out["per_head"].items():
                skill_payload[head] = {
                    "roc_auc": metrics.get("roc_auc", 0.5),
                    "brier": metrics.get("brier", 0.0),
                    "bss_vs_poisson": metrics.get("bss_vs_poisson", 0.0),
                }
                roc_payload[head] = metrics.get("roc", {})
                reliability_payload[head] = metrics.get("reliability", {})
                molchan_payload[head] = metrics.get("molchan", {})
                csep_payload[head] = metrics.get("csep", {})

            save_evaluation_results(version, "skill", skill_payload)
            save_evaluation_results(version, "roc", roc_payload)
            save_evaluation_results(version, "reliability", reliability_payload)
            save_evaluation_results(version, "molchan", molchan_payload)
            save_evaluation_results(version, "csep", csep_payload)

            # Sanity: every head should have produced a real BSS value (not the
            # silent 0.0 fallback that previously hid the missing-baseline bug).
            zero_bss = [h for h, m in skill_payload.items() if m["bss_vs_poisson"] == 0.0]
            if len(zero_bss) == len(skill_payload) and skill_payload:
                logger.warning("all_bss_zero_baseline_may_be_broken", heads=len(zero_bss))

            logger.info("evaluation_results_saved_to_db", version=version)
        except Exception as e:
            logger.error("evaluation_failed", version=version, error=str(e))
    else:
        logger.warning("test_set_empty_skipping_evaluation")

    logger.info("training_complete", version=version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
