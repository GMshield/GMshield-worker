"""The worker's supervisor: `python -m gmshield_worker.main`.

Start-up order:
  0. Make the process non-dumpable (`PR_SET_DUMPABLE 0`, Story 1.15): no child it starts can read
     its memory or `/proc/<pid>/environ`, which holds the worker's credential. Refused, the start
     stops with exit status 78.
  1. Check the settings. Outside test mode a missing secret or setting stops the start with exit
     status 78 and one `startup.refused` line that names the settings, never a value.
  2. Empty the work folder (its contents, never the folder; links are not followed). A folder
     that is a link or a file, or that cannot be read or created, stops the start with exit
     status 78; a failed removal of one entry is logged and the start continues.
  3. Ask the backend for its internal contract version, with the worker's credential, retrying
     with exponential backoff (1 s up to 30 s, with jitter, 2 s per attempt) until it answers.
  4. After the first compatible answer, start both work lanes (Story 1.12): the interactive lane,
     then the ingestion lane (`gmshield_worker.lanes`).

From step 2 on, a sweep thread removes, every 2 minutes, what no running step holds in the work
folder and was last changed more than 10 minutes ago (`workdir.WorkDirSweep`).

The worker is healthy only after a compatible answer, and re-checks every 30 s. Against an older
backend it logs both versions once, stays up, unhealthy, and keeps checking. SIGTERM (or SIGINT)
stops the lanes (a running job is abandoned unreported: the backend takes it back when its claim
lapses) and ends the process with exit status 0. Nothing here opens a listening port or calls a
model, Qdrant or Ollama.
"""

from __future__ import annotations

import contextlib
import os
import random
import select
import signal
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from gmshield_worker import health
from gmshield_worker.backend.client import VersionAnswer, fetch_contract_version
from gmshield_worker.backend.contract import MINIMUM_BACKEND_CONTRACT_VERSION, meets_minimum, parse_version
from gmshield_worker.ingestion.sandbox import SandboxUnavailable, make_non_dumpable
from gmshield_worker.lanes import build_lanes
from gmshield_worker.log import log_event, new_trace_id
from gmshield_worker.models import select_model_client
from gmshield_worker.settings import WORK_DIR, Settings, SettingsRefused, check_settings
from gmshield_worker.workdir import WorkDirSweep, WorkDirUnusable, empty_work_dir

UNIT = "worker.supervisor"
# Exit status of a refused start (EX_CONFIG), the same as the backend's and the frontend's.
REFUSED_EXIT_STATUS = 78

BACKOFF_INITIAL_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
RECHECK_SECONDS = 30.0

type Fetch = Callable[[str, str], VersionAnswer]


class Startable(Protocol):
    """What the supervisor starts after the first compatible answer and stops at the end: the lanes."""

    def start(self) -> None: ...

    def stop(self) -> None: ...


type LanesFactory = Callable[[Settings, str], Startable]
type SweepFactory = Callable[[str, str], Startable]


class Stop(Protocol):
    """What the supervisor needs from a stop request (`threading.Event` fits too)."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class SignalStop:
    """A stop request that a signal handler may set.

    `threading.Event.set()` takes a lock that the interrupted main thread may already hold inside
    `wait()`, so calling it from a signal handler can deadlock. This one only writes a byte to a
    pipe; `wait()` sleeps in `select()`, which wakes as soon as that byte arrives.
    """

    def __init__(self) -> None:
        self._read, self._write = os.pipe()
        os.set_blocking(self._write, False)
        self._set = False

    def set(self) -> None:
        self._set = True
        # A full pipe already holds a wake-up byte.
        with contextlib.suppress(BlockingIOError):
            os.write(self._write, b"x")

    def is_set(self) -> bool:
        return self._set

    def wait(self, timeout: float | None = None) -> bool:
        if not self._set:
            select.select([self._read], [], [], timeout)
        return self._set


def backoff_delay(
    failures: int,
    rng: Callable[[], float] = random.random,
    *,
    initial: float = BACKOFF_INITIAL_SECONDS,
    maximum: float = BACKOFF_MAX_SECONDS,
) -> float:
    """The wait after the n-th failure in a row: doubling from `initial`, jittered by +/-20%,
    never below `initial` nor above `maximum`."""
    base = initial * 2.0 ** min(max(failures - 1, 0), 30)
    jittered = base * (0.8 + 0.4 * rng())
    return max(initial, min(maximum, jittered))


@dataclass
class Supervisor:
    backend_url: str
    trace_id: str
    stop: Stop
    health_file: Path = health.HEALTH_FILE
    fetch: Fetch = field(default=fetch_contract_version)
    rng: Callable[[], float] = random.random
    backoff_initial: float = BACKOFF_INITIAL_SECONDS
    backoff_max: float = BACKOFF_MAX_SECONDS
    recheck_seconds: float = RECHECK_SECONDS
    # Called once, after the first compatible answer: starts the lanes.
    on_ready: Callable[[], None] | None = None
    ready: bool = field(default=False, init=False)
    _started: bool = field(default=False, init=False)
    _failures: int = field(default=0, init=False)
    _incompatible: str | None = field(default=None, init=False)

    def _unhealthy(self) -> None:
        health.mark_unhealthy(self.health_file)
        if self.ready:
            log_event("WARN", UNIT, "worker.unhealthy", self.trace_id)
        self.ready = False

    def _failed(self, event: str, **fields: Any) -> float:
        self._unhealthy()
        self._failures += 1
        log_event("WARN", UNIT, event, self.trace_id, attempt=self._failures, **fields)
        return backoff_delay(self._failures, self.rng, initial=self.backoff_initial, maximum=self.backoff_max)

    def check_once(self) -> float:
        """One check of the backend. Returns how long to wait before the next one."""
        try:
            answer = self.fetch(self.backend_url, self.trace_id)
        except Exception as error:  # logged by type only, then retried
            return self._failed("backend.unreachable", errorType=type(error).__name__)
        if answer.status != 200 or answer.version is None:
            return self._failed("backend.unexpected_answer", status=answer.status)

        self._failures = 0
        if meets_minimum(answer.version):
            health.mark_healthy(self.health_file)
            if not self.ready:
                log_event(
                    "INFO",
                    UNIT,
                    "worker.ready",
                    self.trace_id,
                    backendContractVersion=answer.version,
                    minimumContractVersion=MINIMUM_BACKEND_CONTRACT_VERSION,
                )
            self.ready = True
            self._incompatible = None
            if not self._started and self.on_ready is not None:
                self._started = True
                self.on_ready()
            return self.recheck_seconds

        self._unhealthy()
        # A version that is not major.minor.patch is never echoed.
        reported = answer.version if parse_version(answer.version) is not None else "invalid"
        if reported != self._incompatible:
            log_event(
                "WARN",
                UNIT,
                "backend.incompatible",
                self.trace_id,
                backendContractVersion=reported,
                minimumContractVersion=MINIMUM_BACKEND_CONTRACT_VERSION,
            )
        self._incompatible = reported
        return self.recheck_seconds

    def run(self) -> None:
        while not self.stop.is_set():
            delay = self.check_once()
            self.stop.wait(delay)
        health.mark_unhealthy(self.health_file)


def run(
    environment: Mapping[str, str],
    stop: Stop,
    *,
    health_file: Path = health.HEALTH_FILE,
    fetch: Fetch | None = None,
    rng: Callable[[], float] = random.random,
    backoff_initial: float = BACKOFF_INITIAL_SECONDS,
    recheck_seconds: float = RECHECK_SECONDS,
    lanes_factory: LanesFactory = build_lanes,
    sweep_factory: SweepFactory = WorkDirSweep,
    non_dumpable: Callable[[], None] = make_non_dumpable,
) -> int:
    """Runs the worker until `stop` is set. Returns the exit status."""
    trace_id = new_trace_id()
    try:
        non_dumpable()
    except SandboxUnavailable, OSError:
        log_event("ERROR", UNIT, "startup.refused", trace_id, reason="process.dumpable", settings=[])
        return REFUSED_EXIT_STATUS
    health.mark_unhealthy(health_file)
    try:
        settings = check_settings(environment)
    except SettingsRefused as refused:
        log_event("ERROR", UNIT, "startup.refused", trace_id, reason=refused.reason, settings=refused.settings)
        return REFUSED_EXIT_STATUS

    model = select_model_client(settings)
    log_event(
        "INFO",
        UNIT,
        "startup.checked",
        trace_id,
        testMode=settings.test_mode,
        model=model.kind,
        minimumContractVersion=MINIMUM_BACKEND_CONTRACT_VERSION,
        interactiveCapacity=settings.interactive_capacity,
        redisTls=settings.redis.tls,
    )
    try:
        empty_work_dir(settings.work_dir, trace_id)
    except WorkDirUnusable:
        log_event("ERROR", UNIT, "startup.refused", trace_id, reason="workdir.unusable", settings=[WORK_DIR])
        return REFUSED_EXIT_STATUS

    sweep = sweep_factory(settings.work_dir, trace_id)
    sweep.start()
    lanes = lanes_factory(settings, trace_id)
    try:
        Supervisor(
            backend_url=settings.backend_url,
            trace_id=trace_id,
            stop=stop,
            health_file=health_file,
            fetch=fetch or partial(fetch_contract_version, credential=settings.worker_credential),
            rng=rng,
            backoff_initial=backoff_initial,
            recheck_seconds=recheck_seconds,
            on_ready=lanes.start,
        ).run()
    finally:
        lanes.stop()
        sweep.stop()
    log_event("INFO", UNIT, "worker.stopped", trace_id)
    return 0


def main() -> int:
    stop = SignalStop()

    def on_signal(_signum: int, _frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    return run(os.environ, stop)


if __name__ == "__main__":
    sys.exit(main())
