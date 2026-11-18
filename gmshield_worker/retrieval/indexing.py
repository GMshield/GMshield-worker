"""The index job handlers (Story 2.1, AD-10): `index.change` writes a run of a version's passages into Qdrant,
`index.removal` deletes a version's points.

`index.change`: the claim input is `{source, owner, kind, language, tokenKey, passages}` (the version's id, its owner's
id, the material's kind and language or null, Base64 of the owner's 32-byte token key, and the name of the input that
holds the run). The run, at most 200 passages, is read whole from that input as JSON Lines `{id, ordinal, page, title,
text}`; then, `BATCH` passages at a time, each passage is embedded (`embedding.py`), its keyword vector computed under
the token key (`keywords.py`) and its point upserted (`store.py`), the claim's cancellation checked between batches.
The result is `{indexRecipe, points}`. A re-run upserts the same ids.

`index.removal`: the claim input is `{source, owner}`; every point of that owner's source is deleted by a filter.

Failures: an input that is not the backend's fails with `INPUT_INVALID` and `retry: false`; a passage still too long
after its splits with `index.passage_too_long` and `retry: false`; Ollama with `index.embedding_unavailable`, Qdrant
with `index.store_unavailable` and the backend's input route with `INPUT_UNAVAILABLE`, all three retried by the
backend. Every line names ids, counts, codes and times: never a text, a title, a token or the key, which the input
holds with `repr=False`.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from gmshield_worker.backend.client import ClaimedJob, Outcome
from gmshield_worker.backend.files import FileTransferError
from gmshield_worker.log import TraceLog
from gmshield_worker.retrieval.embedding import BATCH, Embedder, EmbeddingFailed
from gmshield_worker.retrieval.keywords import KEY_BYTES, keyword_vector
from gmshield_worker.retrieval.recipe import INDEX_RECIPE
from gmshield_worker.retrieval.store import STORE_UNAVAILABLE, PassageStore, Point, StoreFailed

CHANGE = "index.change"
REMOVAL = "index.removal"
UNIT = "worker.index"
INPUT_INVALID = "index.input_invalid"
INPUT_UNAVAILABLE = "index.input_unavailable"
CANCELLED = "index.cancelled"
# The backend's run is at most 200 passages of at most 16,384 code points: far below this.
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_PAGE = 1_000_000
_UUID = re.compile(r"[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", re.ASCII)
_INPUT_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}", re.ASCII)
# A material's kind (`rulebook`, `campaign_notes`, ...) or language (`pt-BR`, `en`), as the backend checks them.
_CODE = re.compile(r"[A-Za-z0-9_-]{1,16}", re.ASCII)


class InputInvalid(ValueError):
    """The claim's input or the passages are not what the backend sends. Names nothing."""


class PassageFiles(Protocol):
    """Reading a claimed job's input (`backend.files.JobFiles`, or a stand-in in tests)."""

    def read(self, job: ClaimedJob, name: str, limit: int) -> bytes: ...


type StoreFactory = Callable[[], AbstractContextManager[PassageStore]]


@dataclass(frozen=True)
class ChangeInput:
    source: str
    owner: str
    kind: str | None
    language: str | None
    passages: str
    token_key: bytes = field(repr=False)


@dataclass(frozen=True)
class Passage:
    passage_id: str
    ordinal: int
    page: int
    title: str = field(repr=False)
    text: str = field(repr=False)


def _uuid(value: object) -> str:
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        raise InputInvalid
    return value.lower()


def _code(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _CODE.fullmatch(value):
        raise InputInvalid
    return value


def change_input(raw: dict[str, Any]) -> ChangeInput:
    """The `index.change` input, checked; `InputInvalid` otherwise."""
    passages = raw.get("passages")
    if not isinstance(passages, str) or not _INPUT_NAME.fullmatch(passages):
        raise InputInvalid
    encoded = raw.get("tokenKey")
    if not isinstance(encoded, str):
        raise InputInvalid
    try:
        key = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise InputInvalid from error
    if len(key) != KEY_BYTES:
        raise InputInvalid
    return ChangeInput(
        source=_uuid(raw.get("source")),
        owner=_uuid(raw.get("owner")),
        kind=_code(raw.get("kind")),
        language=_code(raw.get("language")),
        passages=passages,
        token_key=key,
    )


def removal_input(raw: dict[str, Any]) -> tuple[str, str]:
    """The `index.removal` input, checked: `(owner, source)`; `InputInvalid` otherwise."""
    return _uuid(raw.get("owner")), _uuid(raw.get("source"))


def read_passages(body: bytes) -> list[Passage]:
    """The run's JSON Lines, each `{id, ordinal, page, title, text}`, in order with no id twice; `InputInvalid`
    otherwise."""
    try:
        lines = body.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise InputInvalid from error
    found: list[Passage] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as error:
            raise InputInvalid from error
        if not isinstance(item, dict):
            raise InputInvalid
        ordinal, page, title, text = item.get("ordinal"), item.get("page"), item.get("title"), item.get("text")
        if (
            type(ordinal) is not int
            or ordinal < 0
            or (found and ordinal <= found[-1].ordinal)
            or type(page) is not int
            or not 1 <= page <= MAX_PAGE
            or not isinstance(title, str)
            or not isinstance(text, str)
            or not text
        ):
            raise InputInvalid
        found.append(Passage(_uuid(item.get("id")), ordinal, page, title, text))
    if len({passage.passage_id for passage in found}) != len(found):
        raise InputInvalid
    return found


def _failed(code: str, *, retry: bool) -> Outcome:
    return Outcome.failed(code, retry=None if retry else False)


@dataclass
class IndexChange:
    """The `index.change` handler: `(job, cancel) -> Outcome`, as the lanes call it."""

    files: PassageFiles
    embedder: Embedder
    stores: StoreFactory
    clock: Callable[[], float] = field(default=time.monotonic)

    def __call__(self, job: ClaimedJob, cancel: threading.Event, /) -> Outcome:
        log = TraceLog(UNIT, job.trace_id)
        job_id = str(job.job_id)
        started = self.clock()
        try:
            checked = change_input(job.input)
            log.info("index.change_started", jobId=job_id, source=checked.source, attempt=job.attempt)
            passages = read_passages(self.files.read(job, checked.passages, MAX_INPUT_BYTES))
            points, split = self._index(checked, passages, cancel)
        except (InputInvalid, EmbeddingFailed, StoreFailed, httpx.HTTPError, FileTransferError, OSError) as error:
            return _failure(error, log, job_id, "index.change_failed")
        if points is None:
            log.warn("index.change_cancelled", jobId=job_id)
            return _failed(CANCELLED, retry=True)
        log.info(
            "index.change_done",
            jobId=job_id,
            source=checked.source,
            passages=len(passages),
            points=points,
            splitPassages=split,
            seconds=round(self.clock() - started, 3),
        )
        return Outcome.done({"indexRecipe": INDEX_RECIPE, "points": points})

    def _index(self, checked: ChangeInput, passages: list[Passage], cancel: threading.Event) -> tuple[int | None, int]:
        """Embeds and upserts the passages a batch at a time: the points written (None once cancelled) and how many
        passages were split for embedding."""
        points = split = 0
        with self.stores() as store:
            store.ensure_collection()
            for start in range(0, len(passages), BATCH):
                if cancel.is_set():
                    return None, split
                batch = passages[start : start + BATCH]
                embedded = self.embedder.embed([(passage.title, passage.text) for passage in batch])
                store.upsert(
                    [
                        Point(
                            passage_id=passage.passage_id,
                            dense=vector.vector,
                            sparse=keyword_vector(passage.text, checked.token_key),
                            owner=checked.owner,
                            source=checked.source,
                            kind=checked.kind,
                            page=passage.page,
                            language=checked.language,
                        )
                        for passage, vector in zip(batch, embedded, strict=True)
                    ]
                )
                points += len(batch)
                split += sum(1 for vector in embedded if vector.parts > 1)
        return points, split


@dataclass
class IndexRemoval:
    """The `index.removal` handler: deletes every point of the owner's source, by a filter."""

    stores: StoreFactory

    def __call__(self, job: ClaimedJob, cancel: threading.Event, /) -> Outcome:
        log = TraceLog(UNIT, job.trace_id)
        job_id = str(job.job_id)
        try:
            owner, source = removal_input(job.input)
            with self.stores() as store:
                store.ensure_collection()
                store.remove(owner, source)
        except (InputInvalid, StoreFailed) as error:
            return _failure(error, log, job_id, "index.removal_failed")
        log.info("index.removal_done", jobId=job_id, source=source)
        return Outcome.done({})


def _failure(error: Exception, log: TraceLog, job_id: str, event: str) -> Outcome:
    """The outcome of a failed index job, logged by code and error type only."""
    error_type = type(error).__name__
    if isinstance(error, InputInvalid):
        code, retry = INPUT_INVALID, False
    elif isinstance(error, EmbeddingFailed):
        code, retry = error.code, error.retry
    elif isinstance(error, StoreFailed):
        code, retry = STORE_UNAVAILABLE, True
        error_type = type(error.__cause__).__name__ if error.__cause__ is not None else error_type
    else:
        code, retry = INPUT_UNAVAILABLE, True
    log.warn(event, jobId=job_id, code=code, retry=retry, errorType=error_type)
    return _failed(code, retry=retry)
