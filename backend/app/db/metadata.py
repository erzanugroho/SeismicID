"""Small key/value metadata helpers backed by SQLite."""

from __future__ import annotations

from datetime import UTC, datetime

from backend.app.db.sqlite import get_connection, migrate


def set_metadata_value(key: str, value: str | None) -> None:
    """Set a metadata value, replacing an existing key."""
    migrate()
    with get_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO app_metadata (key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, value, datetime.now(UTC).isoformat()),
        )


def get_metadata_value(key: str) -> str | None:
    """Return a metadata value or None."""
    migrate()
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM app_metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def get_metadata_values() -> dict[str, str | None]:
    """Return all metadata key/value pairs."""
    migrate()
    with get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM app_metadata").fetchall()
    return {row["key"]: row["value"] for row in rows}
