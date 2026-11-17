"""One claimed step of the `ingestion` job kind (Story 1.15): dispatch on the claim's stage.

The clean stage (`clean`, a PDF) works in the step's own folder (`workdir.step_folder`, erased however the step ends):

1. it streams the step's input (`source`, the raw upload) from the backend into `source.pdf`;
2. it runs `clean.py` in the sandbox (no credential, no network, the folder as its working directory, killed at the
   deadline of 9 minutes 30 seconds or when the claim ends);
3. it streams the cleaned original and every batch back as outputs, each checked against its size and SHA-256, and
   reports `{pipelineVersion, pageTotal, original, batches: [{fileId, firstPage, lastPage}], next: []}`.

A PDF that needs a password, or that qpdf cannot read, fails with `retry: false`; anything else that goes wrong
fails with `ingestion.step_failed`, and the backend retries while attempts remain. Every line is written under the
job's trace id and names ids, sizes, counts, times and error types only: never a file's content, a name or the grant.
"""

from __future__ import annotations

import json
import re
import resource
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from gmshield_worker.backend.client import ClaimedJob, Outcome
from gmshield_worker.backend.files import FileTransferError, StoredOutput
from gmshield_worker.ingestion import PIPELINE_VERSION, clean, sandbox
from gmshield_worker.ingestion.errors import (
    FILE_UNREADABLE,
    PASSWORD_REQUIRED,
    STAGE_UNKNOWN,
    STEP_FAILED,
    StepFailed,
)
from gmshield_worker.log import TraceLog
from gmshield_worker.workdir import step_folder

KIND = "ingestion"
CLEAN = "clean"
UNIT = "worker.ingestion"
SOURCE_FILE = "source.pdf"
OUT_DIR = "out"
CLEAN_SCRIPT = Path(__file__).with_name("clean.py")
_INPUT_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}", re.ASCII)
_OUTPUT_FILE = re.compile(r"(original|batch-[0-9]{5})\.pdf", re.ASCII)
MAX_PAGES = 1_000_000


class FileRoutes(Protocol):
    """The internal API's file routes (`backend.files.JobFiles`, or a stand-in in tests)."""

    def download(self, job: ClaimedJob, name: str, target: Path) -> int: ...

    def upload(self, job: ClaimedJob, source: Path) -> StoredOutput: ...


type Runner = Callable[..., sandbox.ChildResult]


@dataclass(frozen=True)
class Batch:
    file: str
    first_page: int
    last_page: int


@dataclass(frozen=True)
class Manifest:
    page_total: int
    original: str
    batches: list[Batch]


def read_manifest(out: Path) -> Manifest:
    """The child's manifest, checked: plain file names of its own, page ranges that cover 1 to the total in order."""
    try:
        body = json.loads((out / clean.MANIFEST).read_text(encoding="utf-8"))
        total = body["pageTotal"]
        original = body["original"]
        batches = [Batch(item["file"], item["firstPage"], item["lastPage"]) for item in body["batches"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StepFailed(STEP_FAILED) from error
    expected_first = 1
    for batch in batches:
        if (
            not isinstance(batch.file, str)
            or not _OUTPUT_FILE.fullmatch(batch.file)
            or type(batch.first_page) is not int
            or type(batch.last_page) is not int
            or batch.first_page != expected_first
            or batch.last_page < batch.first_page
        ):
            raise StepFailed(STEP_FAILED)
        expected_first = batch.last_page + 1
    if (
        type(total) is not int
        or not 1 <= total <= MAX_PAGES
        or expected_first != total + 1
        or not isinstance(original, str)
        or not _OUTPUT_FILE.fullmatch(original)
    ):
        raise StepFailed(STEP_FAILED)
    return Manifest(total, original, batches)


def _output(out: Path, name: str) -> Path:
    """A file the child wrote, as a plain file inside `out`: never a link, never anywhere else."""
    path = out / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != out.resolve():
        raise StepFailed(STEP_FAILED)
    return path


@dataclass
class IngestionStep:
    """The `ingestion` handler: `(job, cancel) -> Outcome`, as the lanes call it."""

    files: FileRoutes
    work_dir: str
    deadline_seconds: float = sandbox.DEADLINE_SECONDS
    runner: Runner = field(default=sandbox.run_child)

    def __call__(self, job: ClaimedJob, cancel: threading.Event, /) -> Outcome:
        log = TraceLog(UNIT, job.trace_id)
        job_id = str(job.job_id)
        stage = job.input.get("stage")
        if stage != CLEAN or job.input.get("format") != "pdf":
            log.warn("ingestion.stage_unknown", jobId=job_id)
            return Outcome.failed(STAGE_UNKNOWN, retry=False)
        name = job.input.get("input", "source")
        if not isinstance(name, str) or not _INPUT_NAME.fullmatch(name):
            log.warn("ingestion.input_invalid", jobId=job_id)
            return Outcome.failed(STEP_FAILED)
        started = time.monotonic()
        log.info("ingestion.step_started", jobId=job_id, stage=CLEAN, attempt=job.attempt)
        try:
            with step_folder(self.work_dir, job_id, job.trace_id) as folder:
                result = self._clean(job, name, cancel, folder, log)
        except StepFailed as failed:
            log.warn("ingestion.step_failed", jobId=job_id, stage=CLEAN, code=failed.code, retry=failed.retry)
            return Outcome.failed(failed.code, retry=None if failed.retry else False)
        except (httpx.HTTPError, FileTransferError, OSError) as error:
            log.warn(
                "ingestion.step_failed", jobId=job_id, stage=CLEAN, code=STEP_FAILED, errorType=type(error).__name__
            )
            return Outcome.failed(STEP_FAILED)
        log.info(
            "ingestion.step_done",
            jobId=job_id,
            stage=CLEAN,
            pageTotal=result["pageTotal"],
            batches=len(result["batches"]),
            seconds=round(time.monotonic() - started, 3),
            workerMaxRssKib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        return Outcome.done(result)

    def _clean(
        self, job: ClaimedJob, name: str, cancel: threading.Event, folder: Path, log: TraceLog
    ) -> dict[str, Any]:
        job_id = str(job.job_id)
        source = folder / SOURCE_FILE
        size = self.files.download(job, name, source)
        log.info("ingestion.input_downloaded", jobId=job_id, size=size)
        if cancel.is_set():
            raise StepFailed(STEP_FAILED)
        child = self.runner(
            CLEAN_SCRIPT, [SOURCE_FILE, OUT_DIR], folder=folder, cancel=cancel, deadline_seconds=self.deadline_seconds
        )
        log.info(
            "ingestion.child_ended",
            jobId=job_id,
            returncode=child.returncode,
            timedOut=child.timed_out,
            cancelled=child.cancelled,
            seconds=round(child.seconds, 3),
            childMaxRssKib=child.max_rss_kib,
        )
        if child.returncode == clean.EXIT_PASSWORD:
            raise StepFailed(PASSWORD_REQUIRED, retry=False)
        if child.returncode == clean.EXIT_UNREADABLE:
            raise StepFailed(FILE_UNREADABLE, retry=False)
        if child.returncode != 0 or child.timed_out or child.cancelled:
            raise StepFailed(STEP_FAILED)
        source.unlink()
        out = folder / OUT_DIR
        manifest = read_manifest(out)
        original = self._send(job, out, manifest.original)
        batches = [
            {"fileId": str(self._send(job, out, batch.file).file_id), "firstPage": batch.first_page,
             "lastPage": batch.last_page}
            for batch in manifest.batches
        ]  # fmt: skip
        log.info("ingestion.outputs_stored", jobId=job_id, files=1 + len(batches))
        return {
            "pipelineVersion": PIPELINE_VERSION,
            "pageTotal": manifest.page_total,
            "original": {"fileId": str(original.file_id)},
            "batches": batches,
            "next": [],
        }

    def _send(self, job: ClaimedJob, out: Path, name: str) -> StoredOutput:
        path = _output(out, name)
        stored = self.files.upload(job, path)
        path.unlink()
        return stored
