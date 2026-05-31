"""AI MVP v1 routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from backend.app.api.deps import rate_limit_ai
from backend.app.services.ai_guardrails import validate_public_text
from backend.app.services.ai_service import auto_changelog, cell_explanation, daily_briefing

router = APIRouter(prefix="/ai", tags=["ai"], dependencies=[Depends(rate_limit_ai)])


class GuardrailRequest(BaseModel):
    text: str


@router.get("/daily-briefing")
def get_daily_briefing(
    horizon: int = Query(30, ge=1, le=365),
    threshold: float = Query(5.0, ge=4.0, le=9.0),
    force: bool = False,
) -> dict:
    return daily_briefing(horizon=horizon, threshold=threshold, force=force)


@router.get("/cell-explanation")
def get_cell_explanation(
    cell_id: str,
    horizon: int = Query(30, ge=1, le=365),
    threshold: float = Query(5.0, ge=4.0, le=9.0),
    force: bool = False,
) -> dict:
    return cell_explanation(cell_id=cell_id, horizon=horizon, threshold=threshold, force=force)


@router.get("/changelog")
def get_ai_changelog(limit: int = Query(12, ge=1, le=50), force: bool = False) -> dict:
    return auto_changelog(limit=limit, force=force)


@router.post("/guardrail")
def post_guardrail(payload: GuardrailRequest) -> dict:
    return validate_public_text(payload.text)
