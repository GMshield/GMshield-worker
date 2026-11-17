"""The work folder: where decrypted material lives while one step runs (AD-21).

At start the supervisor empties it: every entry inside is removed, the folder itself stays, and a
symbolic link is removed as a link, never followed. A removal that fails is logged by error type
and the start continues.

Since Story 1.15 each ingestion step works in a folder of its own, `<work>/<jobId>/`
(`step_folder`), erased when the step ends, whatever way it ends. A sweep thread (`WorkDirSweep`)
removes, every 2 minutes, any entry older than 10 minutes that no running step holds: what a
crash between steps left behind.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from gmshield_worker.log import log_event

UNIT = "worker.workdir"
SWEEP_EVERY_SECONDS = 120.0
SWEEP_OLDER_THAN_SECONDS = 600.0

# The step folders in use, by name: the sweep never removes one of them.
_active: set[str] = set()
_active_lock = threading.Lock()


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


def _remove(path: Path) -> None:
    """Removes a file, a link (never followed) or a folder with everything inside it."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


@contextlib.contextmanager
def step_folder(work_dir: str, name: str, trace_id: str) -> Iterator[Path]:
    """A new, private folder `<work>/<name>/` for one step, erased when the step ends, also when it fails.

    Whatever an earlier attempt of the same job left there goes first. While it is in use, the sweep leaves it.
    """
    if not name or "/" in name or name in (".", ".."):
        raise ValueError("a step folder is one plain name")
    folder = Path(work_dir) / name
    with _active_lock:
        if name in _active:
            raise RuntimeError("step folder already in use")
        _active.add(name)
    try:
        if folder.exists() or folder.is_symlink():
            _remove(folder)
        folder.mkdir(mode=0o700)
        yield folder
    finally:
        try:
            if folder.exists() or folder.is_symlink():
                _remove(folder)
        except OSError as error:
            log_event("WARN", UNIT, "workdir.step_removal_failed", trace_id, errorType=type(error).__name__)
        finally:
            with _active_lock:
                _active.discard(name)


def sweep_once(work_dir: str, trace_id: str, *, older_than: float = SWEEP_OLDER_THAN_SECONDS) -> int:
    """Removes every entry last changed more than `older_than` seconds ago that no step holds; how many it removed."""
    removed = 0
    limit = time.time() - older_than
    try:
        entries = list(os.scandir(work_dir))
    except OSError as error:
        log_event("WARN", UNIT, "workdir.sweep_failed", trace_id, errorType=type(error).__name__)
        return 0
    for entry in entries:
        with _active_lock:
            held = entry.name in _active
        try:
            if held or entry.stat(follow_symlinks=False).st_mtime > limit:
                continue
            _remove(Path(entry.path))
            removed += 1
        except OSError as error:
            log_event("WARN", UNIT, "workdir.removal_failed", trace_id, errorType=type(error).__name__)
    if removed:
        log_event("INFO", UNIT, "workdir.swept", trace_id, removed=removed)
    return removed


class WorkDirSweep:
    """A daemon thread that runs `sweep_once` every `every` seconds until stopped."""

    def __init__(
        self,
        work_dir: str,
        trace_id: str,
        *,
        every: float = SWEEP_EVERY_SECONDS,
        older_than: float = SWEEP_OLDER_THAN_SECONDS,
    ) -> None:
        self._work_dir = work_dir
        self._trace_id = trace_id
        self._every = every
        self._older_than = older_than
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="workdir-sweep", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._every):
            sweep_once(self._work_dir, self._trace_id, older_than=self._older_than)
