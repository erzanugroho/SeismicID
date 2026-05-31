"""Tiny AI provider wrapper with safe fallback behavior."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from backend.app.config import get_settings


def ai_enabled() -> bool:
    s = get_settings()
    return bool(s.ai_enabled and s.openai_api_key)


def generate_text(system: str, prompt: str, *, max_tokens: int = 420) -> str | None:
    """Return model text or None if AI disabled / unavailable."""
    s = get_settings()
    if not ai_enabled():
        return None
    payload = {
        "model": s.ai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        s.openai_chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {s.openai_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=s.ai_timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
        return None
