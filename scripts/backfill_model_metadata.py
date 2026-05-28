"""Backfill missing model_metadata rows from on-disk model bundles.

Background: ``register_model_in_db`` used to silently swallow exceptions, so a
training run could finish, write the pickle bundle and ``active.json``, and
still leave ``model_metadata`` empty. This script reads every
``data/models/metadata_<version>.json`` plus the matching pickle bundle and
inserts the missing rows. The model whose version matches ``active.json`` is
marked active.

Usage:
    python -m scripts.backfill_model_metadata          # dry run, list gaps
    python -m scripts.backfill_model_metadata --apply  # write rows
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.core.logging import configure_logging, get_logger  # noqa: E402
from backend.app.db.sqlite import get_connection, migrate  # noqa: E402

logger = get_logger(__name__)


def _existing_versions() -> set[str]:
    migrate()
    with get_connection() as conn:
        rows = conn.execute("SELECT version FROM model_metadata").fetchall()
    return {r["version"] for r in rows}


def _bundle_versions(models_dir: Path) -> list[str]:
    return sorted(
        p.stem.removeprefix("models_")
        for p in models_dir.glob("models_*.pkl")
    )


def _read_active(models_dir: Path) -> str | None:
    active_file = models_dir / "active.json"
    if not active_file.exists():
        return None
    try:
        return json.loads(active_file.read_text())["version"]
    except (json.JSONDecodeError, KeyError):
        return None


def _build_row(models_dir: Path, version: str) -> dict | None:
    metadata_file = models_dir / f"metadata_{version}.json"
    bundle_file = models_dir / f"models_{version}.pkl"
    if not bundle_file.exists():
        logger.warning("backfill_skip_no_bundle", version=version)
        return None

    metadata = (
        json.loads(metadata_file.read_text())
        if metadata_file.exists()
        else {}
    )

    # Reconstruct feature_list/calibrators by loading the pickle (cheap;
    # reading metadata.json alone is not enough because feature_list_json was
    # never written there historically).
    with bundle_file.open("rb") as f:
        heads = pickle.load(f)

    if not heads:
        logger.warning("backfill_skip_empty_bundle", version=version)
        return None

    first_head = next(iter(heads.values()))
    feature_names = list(first_head.feature_names or [])
    feature_count = len(feature_names)
    calibrators = {h: hm.calibrator.__class__.__name__ for h, hm in heads.items()}
    metrics = {h: hm.metrics for h, hm in heads.items()}
    training_date = metadata.get("training_date") or ""

    return {
        "version": version,
        "training_date": training_date,
        "dataset_size": metadata.get("dataset_size"),
        "feature_count": feature_count,
        "feature_list_json": json.dumps(feature_names),
        "metrics_json": json.dumps(metrics),
        "calibrator_json": json.dumps(calibrators),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually write rows. Default: dry-run.")
    args = parser.parse_args()

    configure_logging("INFO")

    settings = get_settings()
    models_dir = settings.models_path
    if not models_dir.exists():
        logger.error("backfill_no_models_dir", path=str(models_dir))
        return 1

    bundles = _bundle_versions(models_dir)
    if not bundles:
        logger.info("backfill_no_bundles_found", path=str(models_dir))
        return 0

    existing = _existing_versions()
    missing = [v for v in bundles if v not in existing]
    active_version = _read_active(models_dir)

    logger.info(
        "backfill_inventory",
        bundles=len(bundles),
        registered=len(existing),
        missing=len(missing),
        active=active_version,
    )

    if not missing:
        logger.info("backfill_nothing_to_do")
        return 0

    rows = []
    for version in missing:
        row = _build_row(models_dir, version)
        if row:
            rows.append(row)

    if not args.apply:
        logger.info(
            "backfill_dry_run",
            would_insert=[r["version"] for r in rows],
            would_activate=active_version if active_version in [r["version"] for r in rows] else None,
        )
        print("\n--- Dry run. Pass --apply to write the rows. ---")
        for r in rows:
            print(f"  insert {r['version']}  features={r['feature_count']}  date={r['training_date']}")
        return 0

    migrate()
    with get_connection() as conn, conn:
        # If we are about to register the active version, deactivate any
        # currently-active row first so the unique-active invariant holds.
        if active_version and any(r["version"] == active_version for r in rows):
            conn.execute("UPDATE model_metadata SET is_active = 0")
        for r in rows:
            is_active = 1 if r["version"] == active_version else 0
            conn.execute(
                """
                INSERT OR REPLACE INTO model_metadata (
                    version, training_date, dataset_size, feature_count,
                    feature_list_json, metrics_json, calibrator_json, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["version"],
                    r["training_date"],
                    r["dataset_size"],
                    r["feature_count"],
                    r["feature_list_json"],
                    r["metrics_json"],
                    r["calibrator_json"],
                    is_active,
                ),
            )
            logger.info(
                "backfill_inserted",
                version=r["version"],
                is_active=bool(is_active),
                features=r["feature_count"],
            )
    logger.info("backfill_complete", inserted=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
