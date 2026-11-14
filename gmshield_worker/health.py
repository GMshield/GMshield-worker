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

HEALTH_FILE = Path("/tmp/gmshield-worker.healthy")  # noqa: S108 - /tmp is the only writable place
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
    return current - modified <= max_age


def main() -> int:
    return 0 if is_healthy(HEALTH_FILE) else 1


if __name__ == "__main__":
    sys.exit(main())
