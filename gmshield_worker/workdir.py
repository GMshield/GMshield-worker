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


class WorkDirUnusable(Exception):
    """The work folder cannot be used: it is a link or a file, or it cannot be read or created."""


def empty_work_dir(work_dir: str, trace_id: str) -> int:
    """Empties the folder (creating it when missing). Returns the number of failed removals.

    Raises `WorkDirUnusable` when the folder itself cannot be used; the start must then stop.
    A failed removal of one entry is logged and the rest continues.
    """
    path = Path(work_dir)
    failures = 0

    def unusable(error: OSError) -> WorkDirUnusable:
        log_event("ERROR", UNIT, "workdir.unusable", trace_id, errorType=type(error).__name__)
        return WorkDirUnusable(type(error).__name__)

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as error:
            raise unusable(error) from error
        log_event("INFO", UNIT, "workdir.created", trace_id)
        return 0
    except OSError as error:
        raise unusable(error) from error
    if not stat.S_ISDIR(mode):
        # A link or a file in place of the folder: never follow it, never clean through it.
        log_event("ERROR", UNIT, "workdir.not_a_directory", trace_id)
        raise WorkDirUnusable("NotADirectory")

    def failed(_function: Callable[..., object], _path: str, error: BaseException) -> None:
        nonlocal failures
        failures += 1
        log_event("WARN", UNIT, "workdir.removal_failed", trace_id, errorType=type(error).__name__)

    removed = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                failures_before = failures
                try:
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path, onexc=failed)
                    else:
                        # Files, symbolic links (to anything, inside or outside) and special files.
                        os.unlink(entry.path)
                except OSError as error:
                    failed(os.unlink, entry.path, error)
                if failures == failures_before:
                    removed += 1
    except OSError as error:
        raise unusable(error) from error
    log_event("INFO", UNIT, "workdir.emptied", trace_id, removed=removed, failures=failures)
    return failures
