"""GET /api/model/metadata and /api/model/evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.db.sqlite import get_connection, migrate

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metadata")
def metadata() -> dict:
    settings = get_settings()
    active = settings.models_path / "active.json"
    if not active.exists():
        return {"version": None, "status": "no_active_model"}
    version = json.loads(active.read_text())["version"]
    meta_path = settings.models_path / f"metadata_{version}.json"
    if not meta_path.exists():
        return {"version": version, "status": "metadata_missing"}
    return json.loads(meta_path.read_text())


@router.get("/evaluation")
def evaluation() -> dict:
    """Return latest stored evaluation payload from DB. Empty if none yet."""
    migrate()
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT model_version, eval_type, payload_json, computed_at FROM evaluation_results ORDER BY computed_at DESC"
        )
        items = [dict(r) for r in cur.fetchall()]
    return {"count": len(items), "items": [{**it, "payload_json": json.loads(it["payload_json"])} for it in items]}
