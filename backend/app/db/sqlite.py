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
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


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
    """Apply schema.sql idempotently."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    path = db_path or get_settings().sqlite_full_path
    logger.info("sqlite_migrate", db=str(path))
    with get_connection(path) as conn:
        conn.executescript(sql)
