"""Re-run evaluation for the active model and persist results to DB.

The original training run on 2026-05-24 saved the model bundle but the
``evaluate_dataset`` call OOM'd (308 GiB allocation) — that path is now
fixed in ``backend/app/ml/evaluate.py`` (chunked L-test, dual-key merge to
avoid Cartesian explosion). This script reuses the saved model so we don't
need a full retrain.

Strategy:
1. Load active heads from data/models/active.json.
2. Rebuild the SAME dataset training used: events from declustered (or
   historical) parquet, snapshots(start=2000-01-01, freq=14d, end=now),
   features + labels.
3. Apply the SAME time_split (train<=2020 / val=2021 / test>2021).
4. Run predict_ensemble on the test slice with snapshots threaded through.
5. Run evaluate_dataset → save reliability/roc/skill/molchan/csep payloads.
"""

from __future__ import annotations

import sys
import time
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
from backend.app.features.builder import (  # noqa: E402
    build_features_for_snapshots,
    default_snapshots,
)
from backend.app.features.labels import (  # noqa: E402
    build_labels,
    join_features_and_labels,
    time_split,
)
from backend.app.ml.ensemble import predict_ensemble  # noqa: E402
from backend.app.ml.evaluate import evaluate_dataset  # noqa: E402
from backend.app.ml.train import (  # noqa: E402
    load_active_models,
    save_evaluation_results,
)


def main() -> int:
    configure_logging("INFO")
    logger = get_logger(__name__)

    t0 = time.time()
    logger.info("eval_active_start")

    heads, version = load_active_models()
    if heads is None or version is None:
        logger.error("no_active_model")
        return 1
    logger.info("loaded_active_model", version=version, head_count=len(heads))

    # 1. Events (mirrors scripts/train_initial.py)
    events = read_declustered_events()
    if events.empty:
        events = read_historical_events()
    if events.empty:
        logger.error("no_events_run_bootstrap_first")
        return 1

    # 2. Snapshots: same defaults as training (start-year=2000, snap-freq=14)
    cells = generate_grid()
    end = datetime.now(timezone.utc)
    start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    snaps = default_snapshots(start, end, freq_days=14)
    logger.info(
        "rebuilding_dataset",
        n_events=len(events),
        n_snapshots=len(snaps),
        n_cells=len(cells),
    )

    feats = build_features_for_snapshots(events, snaps, cells=cells)
    labs = build_labels(events, snaps, cells=cells)
    dataset = join_features_and_labels(feats, labs)
    train, val, test = time_split(dataset)
    logger.info(
        "split_done",
        train=len(train),
        val=len(val),
        test=len(test),
        elapsed_s=round(time.time() - t0, 1),
    )

    if len(test) == 0:
        logger.error("test_set_empty")
        return 1

    # 3. Predict on test set, threading snapshot through so evaluate_dataset
    #    merges on (cell_id, snapshot) — avoids the original 308 GiB blowup.
    t1 = time.time()
    preds = predict_ensemble(
        heads,
        test,
        cell_ids=test["cell_id"].tolist(),
        snapshots=test["snapshot"].tolist() if "snapshot" in test.columns else None,
    )
    logger.info("predict_done", n_rows=len(preds), elapsed_s=round(time.time() - t1, 1))

    # 4. Evaluate
    t2 = time.time()
    eval_out = evaluate_dataset(test, preds)
    logger.info("evaluate_done", elapsed_s=round(time.time() - t2, 1))

    # 5. Decompose into the same eval_type buckets train_initial.py uses
    skill_payload: dict = {}
    roc_payload: dict = {}
    reliability_payload: dict = {}
    molchan_payload: dict = {}
    csep_payload: dict = {}

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

    logger.info(
        "eval_active_complete",
        version=version,
        total_elapsed_s=round(time.time() - t0, 1),
        n_heads=len(eval_out["per_head"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
