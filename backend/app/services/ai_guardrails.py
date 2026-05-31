"""Safety guardrails for AI-generated public seismic text."""
from __future__ import annotations

FORBIDDEN_PHRASES = [
    "pasti terjadi",
    "akan terjadi gempa",
    "akan gempa",
    "segera evakuasi",
    "harus evakuasi",
    "abaikan bmkg",
    "aman dari gempa",
    "tidak akan gempa",
]
REQUIRED_HINTS = ["bukan", "bmkg"]


def safe_fallback(text: str) -> str:
    base = (text or "").strip()
    if not base:
        base = "Ringkasan belum tersedia."
    note = "\n\nCatatan: ini analisis probabilistik eksperimental, bukan sistem peringatan dini dan bukan pengganti BMKG/otoritas resmi."
    low = base.lower()
    if "bukan" in low and "bmkg" in low:
        return base
    return base + note


def validate_public_text(text: str) -> dict[str, object]:
    low = (text or "").lower()
    blocked = [p for p in FORBIDDEN_PHRASES if p in low]
    missing = [p for p in REQUIRED_HINTS if p not in low]
    ok = not blocked
    return {"ok": ok, "blocked_phrases": blocked, "missing_hints": missing}


def guard_public_text(text: str) -> str:
    checked = validate_public_text(text)
    if not checked["ok"]:
        return safe_fallback("Ringkasan AI diblokir karena mengandung klaim gempa yang terlalu pasti.")
    return safe_fallback(text)
