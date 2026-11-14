"""The work folder: where decrypted material lives while one step runs.

At start the supervisor empties it: every entry inside is removed, the folder itself stays, and a
symbolic link is removed as a link, never followed. A removal that fails is logged by error type
and the start continues.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

from gmshield_worker.log import log_event

UNIT = "worker.workdir"


def empty_work_dir(work_dir: str, trace_id: str) -> int:
    """Empties the folder (creating it when missing). Returns the number of failed removals."""
    path = Path(work_dir)
    failures = 0

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_event("INFO", UNIT, "workdir.created", trace_id)
        return 0
    if not stat.S_ISDIR(mode):
        # A link or a file in place of the folder: never follow it, never clean through it.
        log_event("ERROR", UNIT, "workdir.not_a_directory", trace_id)
        return 1

    def failed(_function: Callable[..., object], _path: str, error: BaseException) -> None:
        nonlocal failures
        failures += 1
        log_event("WARN", UNIT, "workdir.removal_failed", trace_id, errorType=type(error).__name__)

    removed = 0
    with os.scandir(path) as entries:
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path, onexc=failed)
                else:
                    # Files, symbolic links (to anything, inside or outside) and special files.
                    os.unlink(entry.path)
                removed += 1
            except OSError as error:
                failed(os.unlink, entry.path, error)
    log_event("INFO", UNIT, "workdir.emptied", trace_id, entries=removed, failures=failures)
    return failures
