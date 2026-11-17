"""One claimed step of the `ingestion` job kind (Stories 1.15 to 1.19): dispatch on the claim's stage.

Every stage works in the step's own folder (`workdir.step_folder`, erased however the step ends): it streams the step's
input (`source`) from the backend, runs its child in the sandbox (no credential, no network, the folder as its working
directory, killed at its deadline or when the claim ends), streams its files back as outputs, each checked against its
size and SHA-256, and reports. Ranges of pages are inclusive and numbered as in the version.

- `clean` (a PDF; the raw upload, or the stored original on a restart): `clean.py` makes it safe and splits it into
  batches; the report gives `pageTotal`, the `original`, the `batches`, one `triage` step per batch and, last, the
  `passages` step over every text file (`input: "texts"`), which the backend holds until every page has an outcome.
- `triage` (a batch): `triage.py` says which pages have a text layer; the report gives their `layers`, a `layout`
  step for each run of text pages, then an `ocr` step for each run of image pages, cut to at most `OCR_PAGES`.
- `layout` (a run of text pages of a batch): `layout.py` reads them with Docling and renders each one; the report
  gives each page `read_directly`, the pages' JSON as `texts` files and, from Story 1.19, a `pageViews` entry per
  page: its WebP picture (sent back as an output) with its size, and its text lines with their boxes.
- `ocr` (a run of at most `OCR_PAGES` image pages of a batch): `ocr.py` straightens each page and reads it with Docling
  and Tesseract; the report gives each page `read_by_ocr` or `unreadable`, the pages read as `texts` files, in
  layout's shape, and a page view per page whose child read it (a blank scan, `unreadable`, has its picture and no
  lines).
- `text` (a TXT or Markdown upload): `text.py` splits it into pages of bounded length; the report gives the file back
  unchanged as the `original`, `pageTotal`, one `read_directly` range, its `texts` files, a page view `{page, text}`
  per page (no picture) and the `passages` step.
- `passages` (Story 1.18; every text file of the version, named in the claim input's `texts` in page order): runs
  in-process, with no child, since it reads only our own JSON. It downloads one text file at a time and feeds its
  page records to `passages.py`, which carries the heading stack across files; the report gives the `passages`
  (page, section path, text, boxes or range) and the `outline`.

Before anything is downloaded, a step whose input names the chain's `pipelineVersion` checks it against this worker's:
another one fails the step with `ingestion.pipeline_changed` and `retry: false`, and the backend restarts the chain.
From the claim's second attempt on, triage, layout and OCR run one child per page, each with an equal share of
`PAGE_BUDGET_SECONDS`; a page whose child fails, crashes or runs out of time is reported `unreadable`, with no page
view.

A PDF that needs a password, or a file no stage can read, fails with `retry: false`; anything else that goes wrong
fails with `ingestion.step_failed`, and the backend retries while attempts remain. Every line is written under the
job's trace id and names ids, sizes, counts, times and error types only: never a file's content, a name or the grant.
"""

from __future__ import annotations

import json
import os
import re
import resource
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from gmshield_worker.backend.client import ClaimedJob, Outcome
from gmshield_worker.backend.files import FileTransferError, StoredOutput
from gmshield_worker.ingestion import PIPELINE_VERSION, clean, layout, ocr, passages, sandbox, text, triage
from gmshield_worker.ingestion.errors import (
    FILE_UNREADABLE,
    PASSWORD_REQUIRED,
    PIPELINE_CHANGED,
    STAGE_UNKNOWN,
    STEP_FAILED,
    StepFailed,
)
from gmshield_worker.log import TraceLog
from gmshield_worker.workdir import step_folder

KIND = "ingestion"
CLEAN = "clean"
TRIAGE = "triage"
LAYOUT = "layout"
OCR = "ocr"
TEXT = "text"
PASSAGES = "passages"
PDF = "pdf"
# The input of a step over every text file of the version (Story 1.18), and the one file a passages step holds.
TEXTS = "texts"
TEXTS_FILE = "texts.json"
UNIT = "worker.ingestion"
SOURCE_FILE = "source.pdf"
SOURCE_TEXT = "source.txt"
OUT_DIR = "out"
CLEAN_SCRIPT = Path(clean.__file__)
TRIAGE_SCRIPT = Path(triage.__file__)
LAYOUT_SCRIPT = Path(layout.__file__)
OCR_SCRIPT = Path(ocr.__file__)
TEXT_SCRIPT = Path(text.__file__)
# The stages this worker runs, by stage and format.
STAGES = frozenset(
    {
        (CLEAN, PDF),
        (TRIAGE, PDF),
        (LAYOUT, PDF),
        (OCR, PDF),
        (TEXT, "txt"),
        (TEXT, "markdown"),
        (PASSAGES, PDF),
        (PASSAGES, "txt"),
        (PASSAGES, "markdown"),
    }
)
# The most image pages one `ocr` step reads: the worker owns this size. Ten pages must stay well inside a step's
# 10 minutes on roberto-server; lower it if the timing check there shows a step over 8 minutes.
OCR_PAGES = 10
# The time a page-by-page retry shares among its pages: each child gets this divided by the step's page count.
PAGE_BUDGET_SECONDS = 540.0
# Docling's models, baked into the image (the Dockerfile's DOCLING_ARTIFACTS_PATH), and its threads.
DOCLING_ARTIFACTS = "/opt/docling-models"
LAYOUT_THREADS = 2
READ_DIRECTLY = "read_directly"
READ_BY_OCR = "read_by_ocr"
UNREADABLE = "unreadable"
_INPUT_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}", re.ASCII)
_OUTPUT_FILE = re.compile(r"(original|batch-[0-9]{5})\.pdf", re.ASCII)
_TEXT_FILE = re.compile(r"text-[0-9]{5}\.json", re.ASCII)
_UUID = re.compile(r"[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", re.ASCII)
_TEXT_INPUT = re.compile(r"text-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.ASCII)
MAX_PAGES = 1_000_000
# The backend's limits on a page view (Story 1.19): a picture's side in pixels, its lines and each line's code points,
# and a text page's code points. Lines past them are dropped or cut here, so a page never fails its version.
MAX_PICTURE_SIDE = 10_000
MAX_LINES = 10_000
MAX_LINE = 4_096
MAX_PAGE_TEXT = 16_384


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


def _page(value: object, total: int = MAX_PAGES) -> bool:
    return type(value) is int and 1 <= value <= total


def read_manifest(out: Path) -> Manifest:
    """The clean child's manifest, checked: plain file names of its own, page ranges that cover 1 to the total in
    order."""
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


def read_layers(out: Path, pages: list[int]) -> dict[int, str]:
    """The triage child's manifest, checked: exactly the pages asked for, each `text` or `image`."""
    try:
        body = json.loads((out / triage.MANIFEST).read_text(encoding="utf-8"))
        items = [(item["page"], item["layer"]) for item in body["pages"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StepFailed(STEP_FAILED) from error
    layers = dict(items)
    if (
        len(items) != len(pages)
        or sorted(layers) != pages
        or any(layer not in (triage.TEXT, triage.IMAGE) for layer in layers.values())
    ):
        raise StepFailed(STEP_FAILED)
    return layers


def read_text_manifest(out: Path) -> tuple[int, list[Batch]]:
    """The text child's manifest, checked: its own file names, ranges that cover 1 to the total in order."""
    try:
        body = json.loads((out / text.MANIFEST).read_text(encoding="utf-8"))
        total = body["pageTotal"]
        files = [Batch(item["file"], item["firstPage"], item["lastPage"]) for item in body["texts"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StepFailed(STEP_FAILED) from error
    expected_first = 1
    for item in files:
        if (
            not isinstance(item.file, str)
            or not _TEXT_FILE.fullmatch(item.file)
            or not _page(item.first_page)
            or not _page(item.last_page)
            or item.first_page != expected_first
            or item.last_page < item.first_page
        ):
            raise StepFailed(STEP_FAILED)
        expected_first = item.last_page + 1
    if not _page(total) or expected_first != total + 1:
        raise StepFailed(STEP_FAILED)
    return total, files


def cut(runs: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """Each run as consecutive runs of at most `size` pages."""
    return [(start, min(start + size - 1, last)) for first, last in runs for start in range(first, last + 1, size)]


def ranges(pages: Iterable[int]) -> list[tuple[int, int]]:
    """Sorted pages as inclusive runs of consecutive pages."""
    runs: list[tuple[int, int]] = []
    for page in sorted(pages):
        if runs and runs[-1][1] == page - 1:
            runs[-1] = (runs[-1][0], page)
        else:
            runs.append((page, page))
    return runs


def _output(out: Path, name: str) -> Path:
    """A file the child wrote, as a plain file inside `out`: never a link, never anywhere else."""
    path = out / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != out.resolve():
        raise StepFailed(STEP_FAILED)
    return path


@dataclass(frozen=True)
class TextInput:
    """One text file a passages step reads: its input name and the pages it holds."""

    name: str
    first_page: int
    last_page: int


@dataclass(frozen=True)
class StepInput:
    """What the claim's input says about the step, checked: the stage's pages and the input file's; for a step over
    every text file, those files in page order."""

    stage: str
    name: str
    page_total: int | None
    first_page: int | None
    last_page: int | None
    input_file_id: str | None
    input_first_page: int | None
    texts: tuple[TextInput, ...] = ()

    def pages(self) -> list[int]:
        if self.first_page is None or self.last_page is None:
            raise StepFailed(STEP_FAILED, retry=False)
        return list(range(self.first_page, self.last_page + 1))

    def batch(self) -> tuple[str, int]:
        """The input batch's file id and first page (for triage and layout)."""
        if self.input_file_id is None or self.input_first_page is None:
            raise StepFailed(STEP_FAILED, retry=False)
        return self.input_file_id, self.input_first_page


def _optional_page(values: dict[str, Any], name: str, total: int = MAX_PAGES) -> int | None:
    value = values.get(name)
    if value is None:
        return None
    if not _page(value, total):
        raise StepFailed(STEP_FAILED, retry=False)
    return int(value)


def step_input(values: dict[str, Any]) -> StepInput:
    """The claim's input, checked; a malformed page or file id is a backend fault no attempt changes."""
    name = values.get("input", "source")
    if not isinstance(name, str) or not _INPUT_NAME.fullmatch(name):
        raise StepFailed(STEP_FAILED)
    total = _optional_page(values, "pageTotal")
    first = _optional_page(values, "firstPage", total or 0)
    last = _optional_page(values, "lastPage", total or 0)
    input_first = _optional_page(values, "inputFirstPage")
    input_last = _optional_page(values, "inputLastPage")
    file_id = values.get("inputFileId")
    if file_id is not None and (not isinstance(file_id, str) or not _UUID.fullmatch(file_id)):
        raise StepFailed(STEP_FAILED, retry=False)
    if (first is None) != (last is None) or (input_first is None) != (input_last is None):
        raise StepFailed(STEP_FAILED, retry=False)
    if first is not None and last is not None and last < first:
        raise StepFailed(STEP_FAILED, retry=False)
    if input_first is not None and input_last is not None:
        if input_last < input_first:
            raise StepFailed(STEP_FAILED, retry=False)
        if first is not None and last is not None and not input_first <= first <= last <= input_last:
            raise StepFailed(STEP_FAILED, retry=False)
    texts = _texts(values.get("texts"), total) if name == TEXTS else ()
    return StepInput(str(values.get("stage")), name, total, first, last, file_id, input_first, texts)


def _texts(value: object, total: int | None) -> tuple[TextInput, ...]:
    """The text files of a step over every text file: names of their own kind, page ranges within the total, in
    page order and never overlapping."""
    if not isinstance(value, list) or total is None:
        raise StepFailed(STEP_FAILED, retry=False)
    found: list[TextInput] = []
    for item in value:
        if not isinstance(item, dict):
            raise StepFailed(STEP_FAILED, retry=False)
        name, first, last = item.get("name"), item.get("firstPage"), item.get("lastPage")
        if not isinstance(name, str) or not _TEXT_INPUT.fullmatch(name):
            raise StepFailed(STEP_FAILED, retry=False)
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= total:
            raise StepFailed(STEP_FAILED, retry=False)
        if found and first <= found[-1].last_page:
            raise StepFailed(STEP_FAILED, retry=False)
        found.append(TextInput(name, first, last))
    return tuple(found)


@dataclass
class IngestionStep:
    """The `ingestion` handler: `(job, cancel) -> Outcome`, as the lanes call it."""

    files: FileRoutes
    work_dir: str
    deadline_seconds: float = sandbox.DEADLINE_SECONDS
    page_budget_seconds: float = PAGE_BUDGET_SECONDS
    runner: Runner = field(default=sandbox.run_child)
    artifacts: str = field(default_factory=lambda: os.environ.get("DOCLING_ARTIFACTS_PATH", DOCLING_ARTIFACTS))

    def __call__(self, job: ClaimedJob, cancel: threading.Event, /) -> Outcome:
        log = TraceLog(UNIT, job.trace_id)
        job_id = str(job.job_id)
        stage = job.input.get("stage")
        # The chain's pipeline first: another pipeline's stage is a pipeline change (a restart), not an unknown stage.
        chain = job.input.get("pipelineVersion")
        if chain is not None and chain != PIPELINE_VERSION:
            log.warn("ingestion.pipeline_changed", jobId=job_id)
            return Outcome.failed(PIPELINE_CHANGED, retry=False)
        if (stage, job.input.get("format")) not in STAGES:
            log.warn("ingestion.stage_unknown", jobId=job_id)
            return Outcome.failed(STAGE_UNKNOWN, retry=False)
        try:
            checked = step_input(job.input)
        except StepFailed as failed:
            log.warn("ingestion.input_invalid", jobId=job_id)
            return Outcome.failed(failed.code, retry=None if failed.retry else False)
        started = time.monotonic()
        log.info("ingestion.step_started", jobId=job_id, stage=stage, attempt=job.attempt)
        run = {
            CLEAN: self._clean,
            TRIAGE: self._triage,
            LAYOUT: self._layout,
            OCR: self._ocr,
            TEXT: self._text,
            PASSAGES: self._passages,
        }[checked.stage]
        try:
            with step_folder(self.work_dir, job_id, job.trace_id) as folder:
                result, counts = run(job, checked, cancel, folder, log)
        except StepFailed as failed:
            log.warn("ingestion.step_failed", jobId=job_id, stage=stage, code=failed.code, retry=failed.retry)
            return Outcome.failed(failed.code, retry=None if failed.retry else False)
        except (httpx.HTTPError, FileTransferError, OSError) as error:
            log.warn(
                "ingestion.step_failed", jobId=job_id, stage=stage, code=STEP_FAILED, errorType=type(error).__name__
            )
            return Outcome.failed(STEP_FAILED)
        log.info(
            "ingestion.step_done",
            jobId=job_id,
            stage=stage,
            **counts,
            seconds=round(time.monotonic() - started, 3),
            workerMaxRssKib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        return Outcome.done(result)

    # --- the stages

    def _clean(
        self, job: ClaimedJob, checked: StepInput, cancel: threading.Event, folder: Path, log: TraceLog
    ) -> tuple[dict[str, Any], dict[str, int]]:
        source = self._download(job, checked, folder, target=SOURCE_FILE, log=log, cancel=cancel)
        child = self._child(
            job,
            CLEAN_SCRIPT,
            [SOURCE_FILE, OUT_DIR],
            folder=folder,
            log=log,
            cancel=cancel,
            deadline=self.deadline_seconds,
        )
        if child.returncode == clean.EXIT_PASSWORD:
            raise StepFailed(PASSWORD_REQUIRED, retry=False)
        if child.returncode == clean.EXIT_UNREADABLE:
            raise StepFailed(FILE_UNREADABLE, retry=False)
        if not _succeeded(child):
            raise StepFailed(STEP_FAILED)
        source.unlink()
        out = folder / OUT_DIR
        manifest = read_manifest(out)
        original = self._send(job, out, manifest.original)
        stored_batches = [
            (str(self._send(job, out, batch.file).file_id), batch.first_page, batch.last_page)
            for batch in manifest.batches
        ]
        batches = [{"fileId": file_id, "firstPage": first, "lastPage": last} for file_id, first, last in stored_batches]
        log.info("ingestion.outputs_stored", jobId=str(job.job_id), files=1 + len(batches))
        result = {
            "pipelineVersion": PIPELINE_VERSION,
            "pageTotal": manifest.page_total,
            "original": {"fileId": str(original.file_id)},
            "batches": batches,
            # Each batch's triage, then the passages step over every text file, held until every page has an outcome.
            "next": [
                {"stage": TRIAGE, "firstPage": first, "lastPage": last, "input": file_id}
                for file_id, first, last in stored_batches
            ]
            + [passages_step(manifest.page_total)],
        }
        return result, {"pageTotal": manifest.page_total, "batches": len(batches)}

    def _triage(
        self, job: ClaimedJob, checked: StepInput, cancel: threading.Event, folder: Path, log: TraceLog
    ) -> tuple[dict[str, Any], dict[str, int]]:
        pages = checked.pages()
        file_id, offset = checked.batch()
        self._download(job, checked, folder, target=SOURCE_FILE, log=log, cancel=cancel)

        def arguments(first: int, last: int, out: str) -> list[str]:
            return [SOURCE_FILE, out, str(offset), str(first), str(last)]

        layers: dict[int, str] = {}
        unreadable: list[int] = []
        if job.attempt <= 1:
            child = self._child(
                job,
                TRIAGE_SCRIPT,
                arguments(pages[0], pages[-1], OUT_DIR),
                folder=folder,
                log=log,
                cancel=cancel,
                deadline=self.deadline_seconds,
            )
            if not _succeeded(child):
                raise StepFailed(STEP_FAILED)
            layers = read_layers(folder / OUT_DIR, pages)
        else:
            for page in pages:
                out = f"{OUT_DIR}-{page}"
                child = self._child(
                    job,
                    TRIAGE_SCRIPT,
                    arguments(page, page, out),
                    folder=folder,
                    log=log,
                    cancel=cancel,
                    deadline=self.page_budget_seconds / len(pages),
                    page=page,
                )
                _stop_if_cancelled(cancel)
                try:
                    if not _succeeded(child):
                        raise StepFailed(STEP_FAILED)
                    layers.update(read_layers(folder / out, [page]))
                except StepFailed:
                    unreadable.append(page)
        text_pages = [page for page, layer in layers.items() if layer == triage.TEXT]
        image_pages = [page for page, layer in layers.items() if layer == triage.IMAGE]
        layer_ranges = sorted(
            (first, last, layer)
            for layer in (triage.TEXT, triage.IMAGE)
            for first, last in ranges(page for page, value in layers.items() if value == layer)
        )
        result = {
            "pipelineVersion": PIPELINE_VERSION,
            "pageTotal": checked.page_total,
            "layers": [{"firstPage": first, "lastPage": last, "layer": layer} for first, last, layer in layer_ranges],
            "pages": [
                {"firstPage": first, "lastPage": last, "outcome": UNREADABLE} for first, last in ranges(unreadable)
            ],
            # Text first, then OCR: within one owner, steps are offered oldest first.
            "next": [
                {"stage": LAYOUT, "firstPage": first, "lastPage": last, "input": file_id}
                for first, last in ranges(text_pages)
            ]
            + [
                {"stage": OCR, "firstPage": first, "lastPage": last, "input": file_id}
                for first, last in cut(ranges(image_pages), OCR_PAGES)
            ],
        }
        return result, {
            "textPages": len(text_pages),
            "imagePages": len(image_pages),
            "unreadablePages": len(unreadable),
        }

    def _layout(
        self, job: ClaimedJob, checked: StepInput, cancel: threading.Event, folder: Path, log: TraceLog
    ) -> tuple[dict[str, Any], dict[str, int]]:
        return self._read(job, checked, cancel, folder, log, script=LAYOUT_SCRIPT, outcome=READ_DIRECTLY)

    def _ocr(
        self, job: ClaimedJob, checked: StepInput, cancel: threading.Event, folder: Path, log: TraceLog
    ) -> tuple[dict[str, Any], dict[str, int]]:
        return self._read(job, checked, cancel, folder, log, script=OCR_SCRIPT, outcome=READ_BY_OCR)

    def _read(
        self,
        job: ClaimedJob,
        checked: StepInput,
        cancel: threading.Event,
        folder: Path,
        log: TraceLog,
        *,
        script: Path,
        outcome: str,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Layout or OCR over the step's pages: one child, or from the second attempt one child per page. A page the
        child calls unreadable (OCR's blank scans), or whose own child fails on a retry, is `unreadable`."""
        pages = checked.pages()
        _file_id, offset = checked.batch()
        self._download(job, checked, folder, target=SOURCE_FILE, log=log, cancel=cancel)

        def arguments(first: int, last: int, out: str) -> list[str]:
            return [SOURCE_FILE, out, str(offset), str(first), str(last), self.artifacts, str(LAYOUT_THREADS)]

        read: dict[int, Path] = {}
        pictures: dict[int, Picture] = {}
        if job.attempt <= 1:
            child = self._child(
                job,
                script,
                arguments(pages[0], pages[-1], OUT_DIR),
                folder=folder,
                log=log,
                cancel=cancel,
                deadline=self.deadline_seconds,
            )
            if not _succeeded(child):
                raise StepFailed(STEP_FAILED)
            read = read_pages(folder / OUT_DIR, pages)
            pictures = read_pictures(folder / OUT_DIR, pages)
        else:
            for page in pages:
                out = f"{OUT_DIR}-{page}"
                child = self._child(
                    job,
                    script,
                    arguments(page, page, out),
                    folder=folder,
                    log=log,
                    cancel=cancel,
                    deadline=self.page_budget_seconds / len(pages),
                    page=page,
                )
                _stop_if_cancelled(cancel)
                try:
                    if not _succeeded(child):
                        raise StepFailed(STEP_FAILED)
                    page_read = read_pages(folder / out, [page])
                    page_picture = read_pictures(folder / out, [page])
                except StepFailed:
                    continue
                read.update(page_read)
                pictures.update(page_picture)
        unreadable = [page for page in pages if page not in read]
        texts = []
        for first, last in ranges(read):
            combined = folder / f"text-{first:07d}.json"
            write_text_file(combined, [read[page] for page in range(first, last + 1)])
            stored = self.files.upload(job, combined)
            combined.unlink()
            texts.append({"fileId": str(stored.file_id), "firstPage": first, "lastPage": last})
        views = []
        for page, picture in sorted(pictures.items()):
            stored = self.files.upload(job, picture.path)
            picture.path.unlink()
            views.append(
                {
                    "page": page,
                    "picture": {"fileId": str(stored.file_id), "width": picture.width, "height": picture.height},
                    "lines": page_lines(read[page]) if page in read else [],
                }
            )
        log.info("ingestion.outputs_stored", jobId=str(job.job_id), files=len(texts) + len(views))
        outcomes = [(first, last, outcome) for first, last in ranges(read)]
        outcomes += [(first, last, UNREADABLE) for first, last in ranges(unreadable)]
        result = {
            "pipelineVersion": PIPELINE_VERSION,
            "pageTotal": checked.page_total,
            "pages": [
                {"firstPage": first, "lastPage": last, "outcome": outcome} for first, last, outcome in sorted(outcomes)
            ],
            "texts": texts,
            "pageViews": views,
            "next": [],
        }
        return result, {"readPages": len(read), "unreadablePages": len(unreadable), "pictures": len(views)}

    def _text(
        self, job: ClaimedJob, checked: StepInput, cancel: threading.Event, folder: Path, log: TraceLog
    ) -> tuple[dict[str, Any], dict[str, int]]:
        source = self._download(job, checked, folder, target=SOURCE_TEXT, log=log, cancel=cancel)
        child = self._child(
            job,
            TEXT_SCRIPT,
            [SOURCE_TEXT, OUT_DIR, str(job.input.get("format"))],
            folder=folder,
            log=log,
            cancel=cancel,
            deadline=self.deadline_seconds,
        )
        if child.returncode == text.EXIT_UNREADABLE:
            raise StepFailed(FILE_UNREADABLE, retry=False)
        if not _succeeded(child):
            raise StepFailed(STEP_FAILED)
        out = folder / OUT_DIR
        total, files = read_text_manifest(out)
        original = self.files.upload(job, source)
        source.unlink()
        texts = []
        views: list[dict[str, Any]] = []
        for item in files:
            views += text_views(_output(out, item.file), item.first_page, item.last_page)
            stored = self._send(job, out, item.file)
            texts.append({"fileId": str(stored.file_id), "firstPage": item.first_page, "lastPage": item.last_page})
        log.info("ingestion.outputs_stored", jobId=str(job.job_id), files=1 + len(texts))
        result = {
            "pipelineVersion": PIPELINE_VERSION,
            "pageTotal": total,
            "original": {"fileId": str(original.file_id)},
            "pages": [{"firstPage": 1, "lastPage": total, "outcome": READ_DIRECTLY}],
            "texts": texts,
            "pageViews": views,
            # Every page has its outcome in this report: the backend gives the passages step its job at once.
            "next": [passages_step(total)],
        }
        return result, {"pageTotal": total, "texts": len(texts)}

    def _passages(
        self, job: ClaimedJob, checked: StepInput, cancel: threading.Event, folder: Path, log: TraceLog
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Every text file of the version, one at a time, through one cutter: the heading stack crosses files."""
        pdf = job.input.get("format") == PDF
        cutter = passages.cutter(pdf)
        path = folder / TEXTS_FILE
        for item in checked.texts:
            size = self.files.download(job, item.name, path)
            log.info("ingestion.input_downloaded", jobId=str(job.job_id), size=size)
            _stop_if_cancelled(cancel)
            try:
                records = passages.read_records(path, item.first_page, item.last_page, pdf=pdf)
            except passages.RecordsInvalid as error:
                raise StepFailed(STEP_FAILED, retry=False) from error
            path.unlink()
            for record in records:
                cutter.page(record)
        found, outline = cutter.finish()
        result = {
            "pipelineVersion": PIPELINE_VERSION,
            "pageTotal": checked.page_total,
            "passages": found,
            "outline": outline,
            "next": [],
        }
        return result, {"texts": len(checked.texts), "passages": len(found), "sections": len(outline)}

    # --- shared

    def _download(
        self, job: ClaimedJob, checked: StepInput, folder: Path, *, target: str, log: TraceLog, cancel: threading.Event
    ) -> Path:
        path = folder / target
        size = self.files.download(job, checked.name, path)
        log.info("ingestion.input_downloaded", jobId=str(job.job_id), size=size)
        _stop_if_cancelled(cancel)
        return path

    def _child(
        self,
        job: ClaimedJob,
        script: Path,
        arguments: list[str],
        *,
        folder: Path,
        log: TraceLog,
        cancel: threading.Event,
        deadline: float,
        page: int | None = None,
    ) -> sandbox.ChildResult:
        child = self.runner(script, arguments, folder=folder, cancel=cancel, deadline_seconds=deadline)
        details = {} if page is None else {"page": page}
        log.info(
            "ingestion.child_ended",
            jobId=str(job.job_id),
            **details,
            returncode=child.returncode,
            timedOut=child.timed_out,
            cancelled=child.cancelled,
            seconds=round(child.seconds, 3),
            childMaxRssKib=child.max_rss_kib,
        )
        return child

    def _send(self, job: ClaimedJob, out: Path, name: str) -> StoredOutput:
        path = _output(out, name)
        stored = self.files.upload(job, path)
        path.unlink()
        return stored


def passages_step(total: int) -> dict[str, Any]:
    """The next step over every text file of the version (Story 1.18): its pages are all of them."""
    return {"stage": PASSAGES, "firstPage": 1, "lastPage": total, "input": TEXTS}


def _succeeded(child: sandbox.ChildResult) -> bool:
    return child.returncode == 0 and not child.timed_out and not child.cancelled


def _stop_if_cancelled(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise StepFailed(STEP_FAILED)


def _page_list(value: object) -> list[int]:
    if not isinstance(value, list) or any(type(page) is not int for page in value):
        raise StepFailed(STEP_FAILED)
    return sorted(value)


def read_pages(out: Path, pages: list[int]) -> dict[int, Path]:
    """The layout or OCR child's pages, checked: its manifest lists exactly the pages asked for, as pages read or (the
    OCR child's `unreadable`) pages it could not read, each once, and each page read has a plain file of its own that
    holds a JSON object for that page."""
    try:
        body = json.loads((out / layout.MANIFEST).read_text(encoding="utf-8"))
        listed = _page_list(body["pages"])
        unreadable = _page_list(body.get("unreadable", []))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise StepFailed(STEP_FAILED) from error
    if sorted(listed + unreadable) != pages:
        raise StepFailed(STEP_FAILED)
    found = {}
    for page in listed:
        path = _output(out, layout.page_file(page))
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise StepFailed(STEP_FAILED) from error
        if not isinstance(record, dict) or record.get("page") != page:
            raise StepFailed(STEP_FAILED)
        found[page] = path
    return found


def write_text_file(target: Path, pages: list[Path]) -> None:
    """One `texts` file, `{"pages": [...]}`, from page files read one at a time: never the whole step in memory."""
    with target.open("w", encoding="utf-8") as combined:
        combined.write('{"pages": [')
        for index, page in enumerate(pages):
            if index:
                combined.write(", ")
            combined.write(json.dumps(json.loads(page.read_text(encoding="utf-8")), ensure_ascii=False))
        combined.write("]}")


@dataclass(frozen=True)
class Picture:
    """A page's picture the layout or OCR child wrote (Story 1.19): its file and its size in pixels."""

    path: Path
    width: int
    height: int


def _side(value: object) -> bool:
    return type(value) is int and 1 <= value <= MAX_PICTURE_SIDE


def read_pictures(out: Path, pages: list[int]) -> dict[int, Picture]:
    """The pictures the layout or OCR child listed, checked: pages asked for, each once, each with a plain WebP file of
    its own and a size the backend takes. A manifest without pictures lists none."""
    try:
        body = json.loads((out / layout.MANIFEST).read_text(encoding="utf-8"))
        items = [(item["page"], item["width"], item["height"]) for item in body.get("pictures", [])]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise StepFailed(STEP_FAILED) from error
    found: dict[int, Picture] = {}
    for page, width, height in items:
        if type(page) is not int or page not in pages or page in found or not _side(width) or not _side(height):
            raise StepFailed(STEP_FAILED)
        found[page] = Picture(_output(out, layout.picture_file(page)), width, height)
    return found


def _box(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(type(side) in (int, float) and 0 <= side <= 1 for side in value)
    )


def page_lines(path: Path) -> list[dict[str, Any]]:
    """A page record's text lines as a page view shows them: those with text and a box of four fractions, each cut to
    `MAX_LINE` code points, at most `MAX_LINES`."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StepFailed(STEP_FAILED) from error
    lines: list[dict[str, Any]] = []
    for line in record.get("lines") or []:
        if not isinstance(line, dict):
            continue
        text_value, box = line.get("text"), line.get("box")
        if isinstance(text_value, str) and text_value and _box(box):
            lines.append({"text": text_value[:MAX_LINE], "box": box})
            if len(lines) == MAX_LINES:
                break
    return lines


def text_views(path: Path, first: int, last: int) -> list[dict[str, Any]]:
    """A text file's pages as page views, `{page, text}`, checked: exactly its pages, in order, each text within
    `MAX_PAGE_TEXT` code points."""
    try:
        records = json.loads(path.read_text(encoding="utf-8"))["pages"]
        views = [{"page": record["page"], "text": record["text"]} for record in records]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StepFailed(STEP_FAILED) from error
    if [view["page"] for view in views] != list(range(first, last + 1)) or any(
        not isinstance(view["text"], str) or len(view["text"]) > MAX_PAGE_TEXT for view in views
    ):
        raise StepFailed(STEP_FAILED)
    return views
