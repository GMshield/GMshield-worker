"""Test mode only: the `probe` job kind. It works `workMs` milliseconds (stopping early when
cancelled), fails its first `failTimes` attempts with `probe.failed`, and is done afterwards. It
drives the jobs tests until Story 1.15 adds the first real kind."""

from __future__ import annotations

import threading

from gmshield_worker.backend.client import ClaimedJob, Outcome

KIND = "probe"
MAX_WORK_MS = 3_600_000


def _whole(value: object, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= high:
        raise ValueError("probe input out of range")
    return value


def run(job: ClaimedJob, cancel: threading.Event, /) -> Outcome:
    work_ms = _whole(job.input.get("workMs", 0), MAX_WORK_MS)
    fail_times = _whole(job.input.get("failTimes", 0), 100)
    if cancel.wait(work_ms / 1000):
        return Outcome.failed("probe.cancelled")
    if job.attempt <= fail_times:
        return Outcome.failed("probe.failed")
    return Outcome.done({"workedMs": work_ms})
