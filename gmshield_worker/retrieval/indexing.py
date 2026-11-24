"""The index job handlers (Story 2.1, AD-10): `index.change` writes a run of a version's passages, or of its records,
into Qdrant; `index.removal` deletes a version's points.

`index.change`: the claim input is `{source, owner, kind, language, tokenKey}` (the version's id, its owner's id, the
material's kind and language or null, Base64 of the owner's 32-byte token key) and exactly one of `passages` or
`records`: the name of the input that holds the run, and what the run is.

- A passages run, at most 200 passages, is read whole from its input as JSON Lines `{id, ordinal, page, title, text}`
  (`read_passages`). Each passage is embedded with its title in the title slot (`embedding.py`) and its keyword vector
  computed over its text.
- A records run (Story 4.2 part B), about 200 records cut at a page's end by the backend, is read whole from its input
  as JSON Lines `{id, page, type, name, fields: [{name, value}]}` (`read_records`: a record's first page and its type's
  key, a core key or, from Story 4.4 part B, an approved type's). Each record is embedded with its name in the title
  slot and its fields as `name: value` lines as the text (its name alone when it has none), the two cut together to
  `RECORD_TEXT_LIMIT` (16,384) code points, name first; its keyword vector is computed over the same name and text.
  Its point's payload adds `record_type`.

Either way, `BATCH` items at a time, each point is upserted with the item's id as the point id (`store.py`), the
claim's cancellation checked between batches. The result is `{indexRecipe, points}`. A re-run upserts the same ids.

`index.removal`: the claim input is `{source, owner}`; every point of that owner's source is deleted by a filter.

Failures: an input that is not the backend's fails with `INPUT_INVALID` and `retry: false`; a passage or a record
still too long after its splits with `index.passage_too_long` and `retry: false`; Ollama with
`index.embedding_unavailable`, Qdrant with `index.store_unavailable` and the backend's input route with
`INPUT_UNAVAILABLE`, all three retried by the backend. Every line names ids, counts, codes and times: never a text, a
title, a record's name or field, a token or the key, which the input holds with `repr=False`.
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
from typing import Any, Protocol, TypeIs

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
# The backend's run is at most 200 passages of at most 16,384 code points, or about 200 records, each from one records
# step's bounded answer: far below this.
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_PAGE = 1_000_000
PASSAGES = "passages"
RECORDS = "records"
# A record's embedded text, its name and its fields' lines together, is cut to this many code points.
RECORD_TEXT_LIMIT = 16_384
# A record type's key as the backend stores it (Story 4.4 part B, internal contract 1.21.0): a core key (`rule`,
# `spell`, ...) or the key of a type the GM approved. One pattern for the index, the searches and the records step.
RECORD_TYPE_KEY = re.compile(r"[a-z][a-z0-9_]{0,31}", re.ASCII)
_UUID = re.compile(r"[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", re.ASCII)
_INPUT_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}", re.ASCII)
# A material's kind (`rulebook`, `campaign_notes`, ...) or language (`pt-BR`, `en`), as the backend checks them.
_CODE = re.compile(r"[A-Za-z0-9_-]{1,16}", re.ASCII)


class InputInvalid(ValueError):
    """The claim's input, the passages or the records are not what the backend sends. Names nothing."""


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
    # The name of the input that holds the run; `records` tells a records run from a passages run.
    input_name: str
    token_key: bytes = field(repr=False)
    records: bool = False


@dataclass(frozen=True)
class Passage:
    passage_id: str
    ordinal: int
    page: int
    title: str = field(repr=False)
    text: str = field(repr=False)


@dataclass(frozen=True)
class Record:
    """One record of a records run: its id, its first page, its core type's key, its name and its fields in order."""

    record_id: str
    page: int
    record_type: str
    name: str = field(repr=False)
    fields: tuple[tuple[str, str], ...] = field(repr=False)


@dataclass(frozen=True)
class _Item:
    """What one point is made of: its id, its page, the embedding's title and text, the keyword vector's text, and the
    record type of a record's point (None for a passage's)."""

    point_id: str
    page: int
    title: str = field(repr=False)
    text: str = field(repr=False)
    keyword_text: str = field(repr=False)
    record_type: str | None = None


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
    """The `index.change` input, checked: exactly one of `passages` and `records` names the input; `InputInvalid`
    otherwise."""
    named = [key for key in (PASSAGES, RECORDS) if key in raw]
    if len(named) != 1:
        raise InputInvalid
    input_name = raw[named[0]]
    if not isinstance(input_name, str) or not _INPUT_NAME.fullmatch(input_name):
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
        input_name=input_name,
        token_key=key,
        records=named[0] == RECORDS,
    )


def removal_input(raw: dict[str, Any]) -> tuple[str, str]:
    """The `index.removal` input, checked: `(owner, source)`; `InputInvalid` otherwise."""
    return _uuid(raw.get("owner")), _uuid(raw.get("source"))


def _lines(body: bytes) -> list[dict[str, Any]]:
    """The input's JSON Lines, each an object; `InputInvalid` otherwise. Lines end at `\n` only."""
    found: list[dict[str, Any]] = []
    for line in body.split(b"\n"):
        if not line.strip():
            continue
        try:
            item = json.loads(line.decode("utf-8"))
        except ValueError as error:  # UnicodeDecodeError is a ValueError
            raise InputInvalid from error
        if not isinstance(item, dict):
            raise InputInvalid
        found.append(item)
    return found


def read_passages(body: bytes) -> list[Passage]:
    """The run's JSON Lines, each `{id, ordinal, page, title, text}`, in order with no id twice; `InputInvalid`
    otherwise. Lines end at `\n` only: a text may hold U+2028, U+0085 and other Unicode line breaks, which JSON
    writes unescaped."""
    found: list[Passage] = []
    for item in _lines(body):
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


def _fields(value: object) -> tuple[tuple[str, str], ...]:
    """A record's fields: a list of `{name, value}`, both strings; `InputInvalid` otherwise."""
    if not isinstance(value, list):
        raise InputInvalid
    found: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "value"}:
            raise InputInvalid
        name, text = item["name"], item["value"]
        if not isinstance(name, str) or not isinstance(text, str):
            raise InputInvalid
        found.append((name, text))
    return tuple(found)


def is_record_type(value: object) -> TypeIs[str]:
    """Whether `value` is a record type key's text (`RECORD_TYPE_KEY`): a core key or an approved type's."""
    return isinstance(value, str) and RECORD_TYPE_KEY.fullmatch(value) is not None


def read_records(body: bytes) -> list[Record]:
    """The records run's JSON Lines, each `{id, page, type, name, fields: [{name, value}]}`: a UUID id (no id twice),
    a page within 1 to `MAX_PAGE`, a type that is a record type key (`RECORD_TYPE_KEY`), a non-empty name, and fields
    of name/value strings; `InputInvalid` otherwise."""
    found: list[Record] = []
    for item in _lines(body):
        page, record_type, name = item.get("page"), item.get("type"), item.get("name")
        if (
            type(page) is not int
            or not 1 <= page <= MAX_PAGE
            or not is_record_type(record_type)
            or not isinstance(name, str)
            or not name.strip()
        ):
            raise InputInvalid
        found.append(Record(_uuid(item.get("id")), page, record_type, name, _fields(item.get("fields"))))
    if len({record.record_id for record in found}) != len(found):
        raise InputInvalid
    return found


def record_text(record: Record) -> tuple[str, str]:
    """A record's embedded title and text: its name, then its fields as `name: value` lines (its name alone when it has
    none), the two cut together to `RECORD_TEXT_LIMIT` code points, name first."""
    title = record.name[:RECORD_TEXT_LIMIT]
    lines = "\n".join(f"{name}: {value}" for name, value in record.fields)
    text = lines[: RECORD_TEXT_LIMIT - len(title)] if lines else title[: RECORD_TEXT_LIMIT - len(title)]
    return title, text


def _passage_item(passage: Passage) -> _Item:
    return _Item(passage.passage_id, passage.page, passage.title, passage.text, passage.text)


def _record_item(record: Record) -> _Item:
    title, text = record_text(record)
    keyword_text = f"{title}\n{text}" if record.fields else title
    return _Item(record.record_id, record.page, title, text, keyword_text, record.record_type)


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
            log.info(
                "index.change_started",
                jobId=job_id,
                source=checked.source,
                recordsRun=checked.records,
                attempt=job.attempt,
            )
            body = self.files.read(job, checked.input_name, MAX_INPUT_BYTES)
            if checked.records:
                items = [_record_item(record) for record in read_records(body)]
            else:
                items = [_passage_item(passage) for passage in read_passages(body)]
            points, split = self._index(checked, items, cancel)
        except (InputInvalid, EmbeddingFailed, StoreFailed, httpx.HTTPError, FileTransferError, OSError) as error:
            return _failure(error, log, job_id, "index.change_failed")
        if points is None:
            log.warn("index.change_cancelled", jobId=job_id)
            return _failed(CANCELLED, retry=True)
        log.info(
            "index.change_done",
            jobId=job_id,
            source=checked.source,
            **{RECORDS if checked.records else PASSAGES: len(items)},
            points=points,
            splitPassages=split,
            seconds=round(self.clock() - started, 3),
        )
        return Outcome.done({"indexRecipe": INDEX_RECIPE, "points": points})

    def _index(self, checked: ChangeInput, items: list[_Item], cancel: threading.Event) -> tuple[int | None, int]:
        """Embeds and upserts the passages or records a batch at a time: the points written (None once cancelled) and
        how many were split for embedding."""
        points = split = 0
        with self.stores() as store:
            store.ensure_collection()
            for start in range(0, len(items), BATCH):
                if cancel.is_set():
                    return None, split
                batch = items[start : start + BATCH]
                embedded = self.embedder.embed([(item.title, item.text) for item in batch])
                store.upsert(
                    [
                        Point(
                            passage_id=item.point_id,
                            dense=vector.vector,
                            sparse=keyword_vector(item.keyword_text, checked.token_key),
                            owner=checked.owner,
                            source=checked.source,
                            kind=checked.kind,
                            page=item.page,
                            language=checked.language,
                            record_type=item.record_type,
                        )
                        for item, vector in zip(batch, embedded, strict=True)
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
