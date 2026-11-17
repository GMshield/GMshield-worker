"""The handlers by job kind. A handler runs one claimed attempt and says how it ended; the lane
claims, renews and reports. `cancel` is set when the work must stop (the claim was refused or
lapsed, or the worker is stopping): a handler checks it and returns early, and the lane then
reports nothing.

`ingestion` (Story 1.15, `gmshield_worker.ingestion.step`) runs in every mode; `probe` in test
mode only.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Protocol

from gmshield_worker.backend.client import ClaimedJob, Outcome
from gmshield_worker.backend.files import JobFiles
from gmshield_worker.ingestion import step
from gmshield_worker.lanes import probe
from gmshield_worker.settings import Settings


class Handler(Protocol):
    def __call__(self, job: ClaimedJob, cancel: threading.Event, /) -> Outcome: ...


def handlers_for(settings: Settings) -> Mapping[str, Handler]:
    """The handlers this worker runs: ingestion steps always, the probe in test mode only."""
    found: dict[str, Handler] = {
        step.KIND: step.IngestionStep(JobFiles(settings.backend_url, settings.worker_credential), settings.work_dir)
    }
    if settings.test_mode:
        found[probe.KIND] = probe.run
    return found
