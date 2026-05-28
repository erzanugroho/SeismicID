"""SQLite connection helper + migration runner.

WAL mode for concurrent read/write. Migrations are idempotent (CREATE IF NOT EXISTS).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from backend.app.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        detect_types=sqlite3.PARSE_DECLTYPES,
        timeout=30.0,
        isolation_level=None,  # autocommit; transactions managed explicitly
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA wal_autocheckpoint = 1000")
    except sqlite3.OperationalError as exc:
        # WSL/Windows mounts can leave stale WAL/SHM files after interrupted runs.
        # Keep the app readable rather than failing every connection attempt.
        logger.warning("sqlite_wal_init_failed", db=str(db_path), error=str(exc))
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def checkpoint(db_path: Path | None = None, *, truncate: bool = True) -> dict[str, int | str]:
    """Checkpoint WAL and optionally truncate it.

    Returns SQLite's (busy, log, checkpointed) counters plus mode. This is safe
    to call from maintenance scripts before long reads/backups.
    """
    path = db_path or get_settings().sqlite_full_path
    mode = "TRUNCATE" if truncate else "PASSIVE"
    with get_connection(path) as conn:
        busy, log, checkpointed = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
    return {"mode": mode, "busy": int(busy), "log": int(log), "checkpointed": int(checkpointed)}


def integrity_check(db_path: Path | None = None) -> str:
    """Run PRAGMA integrity_check."""
    path = db_path or get_settings().sqlite_full_path
    with get_connection(path) as conn:
        return str(conn.execute("PRAGMA integrity_check").fetchone()[0])


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Context manager yielding a SQLite connection (auto-closed)."""
    path = db_path or get_settings().sqlite_full_path
    conn = _connect(path)
    try:
        yield conn
    finally:
        conn.close()


def migrate(db_path: Path | None = None) -> None:
    """Apply schema.sql idempotently.

    The schema deliberately uses ``ALTER TABLE ... ADD COLUMN`` for additive
    migrations on existing tables (SQLite has no ``ADD COLUMN IF NOT EXISTS``).
    We split the script on ``;`` and tolerate duplicate-column / duplicate-index
    errors so a fresh DB and a previously-migrated DB both succeed.
    """
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    path = db_path or get_settings().sqlite_full_path
    logger.info("sqlite_migrate", db=str(path))
    # Strip ``--`` line comments before splitting on ``;`` — schema.sql contains
    # semicolons inside comments (e.g. "clause; the duplicate-column...") which
    # would otherwise break a naive split.
    cleaned_lines = []
    for line in sql.splitlines():
        idx = line.find("--")
        cleaned_lines.append(line if idx < 0 else line[:idx])
    cleaned = "\n".join(cleaned_lines)
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    with get_connection(path) as conn:
        for stmt in statements:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "duplicate column" in msg or "already exists" in msg:
                    logger.debug("sqlite_migrate_skip_existing", stmt=stmt[:60], error=str(exc))
                    continue
                raise
