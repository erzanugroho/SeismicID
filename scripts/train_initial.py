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
        from backend.app.ml.evaluate import evaluate_dataset
        from backend.app.ml.train import save_evaluation_results
        
        try:
            preds = predict_ensemble(heads, test, cell_ids=test["cell_id"].tolist())
            eval_out = evaluate_dataset(test, preds)
            
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
                    "bss_vs_poisson": metrics.get("bss_vs_poisson", 0.0)
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
            
            logger.info("evaluation_results_saved_to_db", version=version)
        except Exception as e:
            logger.error("evaluation_failed", version=version, error=str(e))
    else:
        logger.warning("test_set_empty_skipping_evaluation")

    logger.info("training_complete", version=version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
