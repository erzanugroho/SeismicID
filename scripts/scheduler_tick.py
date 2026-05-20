"""One-shot local/Railway worker tick.

Fetches latest events, detects any new earthquake event (including small ones),
runs forecast when needed, applies debounce/fallback policy, and exits.
"""

from __future__ import annotations

import json

from backend.app.scheduler.jobs import scheduler_tick


if __name__ == "__main__":
    print(json.dumps(scheduler_tick(), indent=2, ensure_ascii=False, default=str))
