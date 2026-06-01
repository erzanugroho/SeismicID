"""Privacy-safe usage analytics."""

from __future__ import annotations

import hashlib
import ipaddress
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.app.config import get_settings
from backend.app.db.sqlite import get_connection, migrate

_SKIP_PREFIXES = (
    "/api/health",
    "/favicon.ico",
    "/static/",
)


def _client_ip(request: Any) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return getattr(request.client, "host", "") or "unknown"


def _ip_bucket(raw_ip: str) -> str:
    try:
        ip = ipaddress.ip_address(raw_ip)
        if isinstance(ip, ipaddress.IPv4Address):
            network = ipaddress.ip_network(f"{ip}/24", strict=False)
        else:
            network = ipaddress.ip_network(f"{ip}/48", strict=False)
        return str(network.network_address)
    except ValueError:
        return "unknown"


def _visitor_hash(day: str, ip_bucket: str, user_agent: str) -> str:
    settings = get_settings()
    salt = settings.admin_token or settings.telegram_webhook_secret or settings.app_name
    payload = f"{day}|{ip_bucket}|{user_agent[:160]}|{salt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def should_track_path(path: str) -> bool:
    return not any(path.startswith(prefix) for prefix in _SKIP_PREFIXES)


def track_daily_active_user(request: Any) -> None:
    if not should_track_path(str(request.url.path)):
        return
    now = datetime.now(UTC)
    day = now.date().isoformat()
    visitor_id = _visitor_hash(day, _ip_bucket(_client_ip(request)), request.headers.get("user-agent", ""))
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO daily_active_users(day, visitor_id, first_seen, last_seen, hits)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(day, visitor_id) DO UPDATE SET
                last_seen = excluded.last_seen,
                hits = daily_active_users.hits + 1
            """,
            (day, visitor_id, now.isoformat(), now.isoformat()),
        )


def daily_active_users(days: int = 14) -> dict[str, Any]:
    days = max(1, min(int(days), 90))
    start = (datetime.now(UTC).date() - timedelta(days=days - 1)).isoformat()
    migrate()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT day, COUNT(*) AS dau, SUM(hits) AS hits
            FROM daily_active_users
            WHERE day >= ?
            GROUP BY day
            ORDER BY day DESC
            """,
            (start,),
        ).fetchall()
    return {
        "days": days,
        "privacy": "visitor_id is SHA-256(day + coarse IP bucket + user-agent + server salt); raw IP/user-agent not stored",
        "items": [dict(row) for row in rows],
    }
