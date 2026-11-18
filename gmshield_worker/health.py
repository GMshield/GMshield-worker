"""Health as a file, because the worker opens no port.

The supervisor writes the file only after the backend answered with a compatible contract
version, refreshes it on every re-check (every 30 s), and removes it as soon as a re-check fails
or reports an older version. `python -m gmshield_worker.health` (the image's HEALTHCHECK) exits 0
when the file exists and is fresh, 1 otherwise.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# A folder of the image owned by the worker (mode 0700), not a world-writable one; a VOLUME, so it
# stays writable with a read-only root filesystem. GMSHIELD_HEALTH_FILE moves it, for tests that run
# the worker outside the image (the worker and the HEALTHCHECK read the same environment).
HEALTH_FILE = Path(os.environ.get("GMSHIELD_HEALTH_FILE") or "/run/gmshield-worker/healthy")
# Three missed re-checks (every 30 s) and the worker counts as hung.
MAX_AGE_SECONDS = 90.0


def mark_healthy(path: Path = HEALTH_FILE) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(f"{time.time():.0f}\n", encoding="ascii")
    os.replace(temporary, path)


def mark_unhealthy(path: Path = HEALTH_FILE) -> None:
    path.unlink(missing_ok=True)


def is_healthy(path: Path = HEALTH_FILE, *, now: float | None = None, max_age: float = MAX_AGE_SECONDS) -> bool:
    try:
        modified = path.stat().st_mtime
    except OSError:
        return False
    current = time.time() if now is None else now
    # A future time (the clock stepped back) is not fresh: it would never go stale.
    return 0 <= current - modified <= max_age


def main() -> int:
    return 0 if is_healthy(HEALTH_FILE) else 1


if __name__ == "__main__":
    sys.exit(main())
