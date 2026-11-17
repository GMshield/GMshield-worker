"""The codes an ingestion step reports when it fails (category `ingestion`; the frontend has text for each).

- `PASSWORD_REQUIRED`: the PDF needs a user password; no attempt can change that (`retry: false`).
- `FILE_UNREADABLE`: qpdf cannot read the file, or it has no page (`retry: false`).
- `STEP_FAILED`: anything else (the child crashed or ran past its deadline, a transfer failed): the backend retries
  while attempts remain.
- `STAGE_UNKNOWN`: a stage or format this worker does not run (`retry: false`).
"""

from __future__ import annotations

PASSWORD_REQUIRED = "ingestion.password_required"  # noqa: S105 - a failure code, not a password
FILE_UNREADABLE = "ingestion.file_unreadable"
STEP_FAILED = "ingestion.step_failed"
STAGE_UNKNOWN = "ingestion.stage_unknown"


class StepFailed(Exception):
    """A step that cannot finish: its code, and whether another attempt could. The message names nothing else."""

    def __init__(self, code: str, *, retry: bool = True) -> None:
        super().__init__(code)
        self.code = code
        self.retry = retry
