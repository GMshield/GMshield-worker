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
- `images` (Story 1.21; an image set, its images named in the claim input's `inputs` in their order): `images.py`
  assembles them into one image-only PDF, one page per image (per frame of a TIFF), and splits it into batches as
  clean does; the report gives what clean's gives. An image that did not decode is a blank page at its place, which
  OCR then reports `unreadable` like any blank scan (so Reprocessar finds the step that read it, as for a PDF). From
  the claim's second attempt on, one child decodes each image (a child that fails, crashes or runs out of time is one
  blank page) and a last one assembles. A restart reads the stored original instead (`input: "source"`), which is a
  PDF: it is cleaned again. The later stages read an image set's pages as a PDF's.
- `text` (a TXT or Markdown upload): `text.py` splits it into pages of bounded length; the report gives the file back
  unchanged as the `original`, `pageTotal`, one `read_directly` range, its `texts` files, a page view `{page, text}`
  per page (no picture) and the `passages` step.
- `passages` (Story 1.18; every text file of the version, named in the claim input's `texts` in page order): cuts
  in-process, since it reads only our own JSON. It downloads one text file at a time and feeds its page records to
  `passages.py`, which carries the section across files; the report gives the `passages` (page, section path, text,
  boxes or range) and the `outline`. From pipeline "7", a PDF's passages step whose claim input names the version's
  cleaned `original` first downloads it and reads its bookmarks with `clean.py bookmarks` in the sandbox, so section
  paths follow the file's own bookmarks; a child that fails or writes something else leaves no bookmarks (the paths
  then come from the headings alone), and the original is deleted before the text files come.

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
from gmshield_worker.ingestion import PIPELINE_VERSION, clean, images, layout, ocr, passages, sandbox, text, triage
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
IMAGES = "images"
PDF = "pdf"
# The format of an image set's version (Story 1.21): its pages are read as a PDF's once assembled.
IMAGE_SET = "images"
# The formats an image of a set may have, and the most images a set holds (the backend's limits).
IMAGE_FORMATS = frozenset({"jpeg", "png", "webp", "tiff", "bmp", "heic"})
MAX_IMAGES = 500
# The input of a step over every text file of the version (Story 1.18), and the one file a passages step holds.
TEXTS = "texts"
TEXTS_FILE = "texts.json"
# The file the clean child writes the original's bookmarks to, in a passages step (pipeline "7").
BOOKMARKS_FILE = "bookmarks.json"
UNIT = "worker.ingestion"
SOURCE_FILE = "source.pdf"
SOURCE_TEXT = "source.txt"
OUT_DIR = "out"
CLEAN_SCRIPT = Path(clean.__file__)
TRIAGE_SCRIPT = Path(triage.__file__)
LAYOUT_SCRIPT = Path(layout.__file__)
OCR_SCRIPT = Path(ocr.__file__)
TEXT_SCRIPT = Path(text.__file__)
IMAGES_SCRIPT = Path(images.__file__)
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
        (IMAGES, IMAGE_SET),
        (TRIAGE, IMAGE_SET),
        (LAYOUT, IMAGE_SET),
        (OCR, IMAGE_SET),
        (PASSAGES, IMAGE_SET),
    }
)
# The most image pages one `ocr` step reads: the worker owns this size. Ten pages must stay well inside a step's
# 10 minutes on roberto-server; lower it if the timing check there shows a step over 8 minutes.
OCR_PAGES = 10
# The time a page-by-page retry shares among its pages: each child gets this divided by the step's page count.
PAGE_BUDGET_SECONDS = 540.0
# A retried images step (Story 1.21): the time kept for its assembling child, and the least an image's decoding gets.
ASSEMBLE_RESERVE_SECONDS = 60.0
IMAGE_FLOOR_SECONDS = 2.0
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
_IMAGE_INPUT = re.compile(r"image-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.ASCII)
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
    # The images child's pages whose image did not decode (Story 1.21); never any from clean.
    unreadable: tuple[int, ...] = ()


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
    unreadable = body.get("unreadable", [])
    if (
        not isinstance(unreadable, list)
        or any(not _page(page, total) for page in unreadable)
        or unreadable != sorted(set(unreadable))
    ):
        raise StepFailed(STEP_FAILED)
    return Manifest(total, original, batches, tuple(unreadable))


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
class ImageInput:
    """One image an images step reads (Story 1.21): its input name and its format."""

    name: str
    format: str


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
    images: tuple[ImageInput, ...] = ()
    # A step over every text file: the input name of the version's cleaned original, when the backend names one.
    original: str | None = None

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
    inputs = _images(values.get("inputs")) if name == IMAGES else ()
    original = values.get("original") if name == TEXTS else None
    if original is not None and (not isinstance(original, str) or not _INPUT_NAME.fullmatch(original)):
        raise StepFailed(STEP_FAILED, retry=False)
    return StepInput(str(values.get("stage")), name, total, first, last, file_id, input_first, texts, inputs, original)


def _images(value: object) -> tuple[ImageInput, ...]:
    """The images of an images step (Story 1.21), in their order: 1 to `MAX_IMAGES`, names of their own kind, each
    once, each an image's format."""
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_IMAGES:
        raise StepFailed(STEP_FAILED, retry=False)
    found: list[ImageInput] = []
    for item in value:
        if not isinstance(item, dict):
            raise StepFailed(STEP_FAILED, retry=False)
        name, kind = item.get("name"), item.get("format")
        if not isinstance(name, str) or not _IMAGE_INPUT.fullmatch(name) or kind not in IMAGE_FORMATS:
            raise StepFailed(STEP_FAILED, retry=False)
        found.append(ImageInput(name, str(kind)))
    if len({item.name for item in found}) != len(found):
        raise StepFailed(STEP_FAILED, retry=False)
    return tuple(found)


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
    clock: Callable[[], float] = field(default=time.monotonic)
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
            IMAGES: self._images,
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

    def _images(
        self, job: ClaimedJob, checked: StepInput, cancel: threading.Event, folder: Path, log: TraceLog
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """An image set's images, in order, into one PDF (Story 1.21). A restart's input is the stored original, a PDF
        this stage assembled before: it is cleaned and split again, as clean does."""
        if checked.name != IMAGES:
            return self._clean(job, checked, cancel, folder, log)
        started = self.clock()
        sources: list[str] = []
        for index, item in enumerate(checked.images, start=1):
            target = f"image-{index:05d}"
            size = self.files.download(job, item.name, folder / target)
            log.info("ingestion.input_downloaded", jobId=str(job.job_id), size=size)
            _stop_if_cancelled(cancel)
            sources.append(target)
        deadline = self.deadline_seconds
        if job.attempt <= 1:
            arguments = [OUT_DIR, *sources]
        else:
            # A retry: one child per image, so an image that crashes its decoder is one blank page. The whole retry
            # keeps to one budget, `deadline_seconds` from the step's start: each decoding child gets the time left,
            # less a reserve for assembling, shared among the images left (never under `IMAGE_FLOOR_SECONDS`); an image
            # with no time left for it is not decoded (a blank page); the assembling child gets whatever is left.
            decoded: list[str] = []
            for index, source in enumerate(sources, start=1):
                target = f"decoded-{index:05d}"
                spare = self.deadline_seconds - (self.clock() - started) - ASSEMBLE_RESERVE_SECONDS
                if spare >= IMAGE_FLOOR_SECONDS:
                    share = max(IMAGE_FLOOR_SECONDS, spare / (len(sources) - index + 1))
                    self._child(
                        job,
                        IMAGES_SCRIPT,
                        ["--decode", source, target],
                        folder=folder,
                        log=log,
                        cancel=cancel,
                        deadline=min(share, spare),
                        page=index,
                    )
                    _stop_if_cancelled(cancel)
                (folder / source).unlink()
                decoded.append(target)
            arguments = ["--assemble", OUT_DIR, *decoded]
            deadline = max(self.deadline_seconds - (self.clock() - started), 0.0)
        child = self._child(job, IMAGES_SCRIPT, arguments, folder=folder, log=log, cancel=cancel, deadline=deadline)
        if not _succeeded(child):
            raise StepFailed(STEP_FAILED)
        for source in sources:
            (folder / source).unlink(missing_ok=True)
        out = folder / OUT_DIR
        manifest = read_manifest(out)
        original = self._send(job, out, manifest.original)
        stored_batches = [
            (str(self._send(job, out, batch.file).file_id), batch.first_page, batch.last_page)
            for batch in manifest.batches
        ]
        log.info("ingestion.outputs_stored", jobId=str(job.job_id), files=1 + len(stored_batches))
        result = {
            "pipelineVersion": PIPELINE_VERSION,
            "pageTotal": manifest.page_total,
            "original": {"fileId": str(original.file_id)},
            "batches": [
                {"fileId": file_id, "firstPage": first, "lastPage": last} for file_id, first, last in stored_batches
            ],
            # Each batch's triage, as clean asks it; the blank page of an image that did not decode goes to OCR,
            # which reports it unreadable. Then the passages step, held until every page has an outcome.
            "next": [
                {"stage": TRIAGE, "firstPage": first, "lastPage": last, "input": file_id}
                for file_id, first, last in stored_batches
            ]
            + [passages_step(manifest.page_total)],
        }
        return result, {
            "images": len(sources),
            "pageTotal": manifest.page_total,
            "batches": len(stored_batches),
            "undecodedPages": len(manifest.unreadable),
        }

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
        """Every text file of the version, one at a time, through one cutter: the section crosses files. A PDF's
        bookmarks come first, from its cleaned original."""
        # An image set's pages are a PDF's once assembled (Story 1.21); it has no bookmarks.
        pdf = job.input.get("format") in (PDF, IMAGE_SET)
        bookmarks = []
        if job.input.get("format") == PDF and checked.original is not None:
            bookmarks = self._bookmarks(job, checked, cancel, folder, log)
        cutter = passages.cutter(pdf, bookmarks=bookmarks)
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
        return result, {
            "texts": len(checked.texts),
            "bookmarks": len(bookmarks),
            "passages": len(found),
            "sections": len(outline),
        }

    def _bookmarks(
        self, job: ClaimedJob, checked: StepInput, cancel: threading.Event, folder: Path, log: TraceLog
    ) -> list[passages.Bookmark]:
        """The cleaned original's bookmarks, read by the clean child in the sandbox. They are optional: a download
        that fails, a sandbox that cannot start, a child that fails or a file that is not bookmarks each gives none
        (logged `ingestion.bookmarks_unread` with its `reason`), so the passages keep the paths their headings give.
        A cancel still ends the step."""
        job_id = str(job.job_id)
        source = folder / SOURCE_FILE
        reason = "download"
        try:
            size = self.files.download(job, str(checked.original), source)
            log.info("ingestion.input_downloaded", jobId=job_id, input="original", size=size)
            _stop_if_cancelled(cancel)
            reason = "sandbox"
            child = self._child(
                job,
                CLEAN_SCRIPT,
                [clean.BOOKMARKS, SOURCE_FILE, BOOKMARKS_FILE],
                folder=folder,
                log=log,
                cancel=cancel,
                deadline=self.deadline_seconds,
            )
            _stop_if_cancelled(cancel)
            reason = "child"
            if not _succeeded(child):
                raise StepFailed(STEP_FAILED)
            reason = "invalid"
            path = _output(folder, BOOKMARKS_FILE)
            found = passages.read_bookmarks(path, checked.page_total or 0)
            path.unlink()
        except (FileTransferError, httpx.HTTPError, OSError, sandbox.SandboxUnavailable, passages.RecordsInvalid,
                StepFailed) as error:  # fmt: skip
            if cancel.is_set():
                raise StepFailed(STEP_FAILED) from error
            log.warn("ingestion.bookmarks_unread", jobId=job_id, reason=reason, errorType=type(error).__name__)
            return []
        finally:
            source.unlink(missing_ok=True)
        return found

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
