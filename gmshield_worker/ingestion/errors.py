"""The codes an ingestion step reports when it fails (category `ingestion`; the frontend has text for each).

- `PASSWORD_REQUIRED`: the PDF needs a user password; no attempt can change that (`retry: false`).
- `FILE_UNREADABLE`: qpdf cannot read the file, or it has no page (`retry: false`).
- `STEP_FAILED`: anything else (the child crashed or ran past its deadline, a transfer failed): the backend retries
  while attempts remain.
- `STAGE_UNKNOWN`: a stage or format this worker does not run (`retry: false`).
- `PIPELINE_CHANGED` (Story 1.16): the chain was started by another pipeline version than this worker's
  (`retry: false`); the backend restarts the chain from the cleaned original.

A records step (Story 4.2) fails with its model's code instead (`model.refused`, `model.truncated`,
`model.unavailable`, `model.output_invalid`, `model.input_invalid`, `model.provider_*`): the backend never fails the
version for it, and marks the step's pages unread once it ends.
"""

from __future__ import annotations

PASSWORD_REQUIRED = "ingestion.password_required"  # noqa: S105 - a failure code, not a password
FILE_UNREADABLE = "ingestion.file_unreadable"
STEP_FAILED = "ingestion.step_failed"
STAGE_UNKNOWN = "ingestion.stage_unknown"
PIPELINE_CHANGED = "ingestion.pipeline_changed"


class StepFailed(Exception):
    """A step that cannot finish: its code, and whether another attempt could. The message names nothing else."""

    def __init__(self, code: str, *, retry: bool = True) -> None:
        super().__init__(code)
        self.code = code
        self.retry = retry
