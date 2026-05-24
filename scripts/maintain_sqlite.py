"""SQLite maintenance helpers for WAL/checkpoint/integrity.

Usage:
    python scripts/maintain_sqlite.py --checkpoint --integrity
    python scripts/maintain_sqlite.py --vacuum
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import get_settings  # noqa: E402
from backend.app.db.sqlite import checkpoint, integrity_check  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="store_true", help="Run WAL checkpoint truncate")
    parser.add_argument("--integrity", action="store_true", help="Run PRAGMA integrity_check")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after checkpoint")
    args = parser.parse_args()

    db = get_settings().sqlite_full_path
    print(f"db={db}")

    if args.checkpoint or args.vacuum:
        try:
            print("checkpoint", checkpoint(db, truncate=True))
        except sqlite3.Error as exc:
            print(f"checkpoint_error={exc}")
            return 2

    if args.integrity:
        try:
            print("integrity", integrity_check(db))
        except sqlite3.Error as exc:
            print(f"integrity_error={exc}")
            return 3

    if args.vacuum:
        try:
            with sqlite3.connect(str(db), timeout=60.0, isolation_level=None) as conn:
                conn.execute("VACUUM")
            print("vacuum ok")
        except sqlite3.Error as exc:
            print(f"vacuum_error={exc}")
            return 4

    if not (args.checkpoint or args.integrity or args.vacuum):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
