"""Finding the passages that answer a question (Story 2.3, AD-10): `find_passages`.

The question input is the backend's (`Scopes.questionInput`), never computed here: `{question, owner, tiers: [{tier,
sources, pending}], tokenKey}`, the tiers in order (campaign, then system), `pending` the ids of the tier's passages
whose index change has not ended (the backend lists at most the first 100 of each tier), and `tokenKey` Base64 of the
owner's 32-byte token key.

1. When no tier has a source, nothing is asked of Ollama, Qdrant or the backend: the list is empty.
2. The question is embedded with the indexing model under embeddinggemma's query prompt (`Embedder.embed_query`), and
   its keyword vector is `keyword_vector(question, tokenKey)`.
3. Per tier with sources, one Qdrant search (`PassageStore.search`): dense and sparse prefetches fused by RRF, `LIMIT`
   points, the owner and source filters always on. A tier without sources issues no query.
4. Pending: the first `MAX_PENDING` ids of the tier's `pending`, in input order, are read by id in one call; those that
   share at least one keyword token with the question are ranked by how many they share (input order breaks ties), and
   that ranking is fused with the tier's Qdrant results as a third RRF list (`fuse`), `LIMIT` ids. An id found both
   ways is one entry whose scores add up, as RRF does. The rest of `pending` is skipped and logged as a count. A
   question with no keyword token can match no pending passage: its pending passages are not read at all.
5. The text of every found id, in rank order, campaign tier first, is read in one call to `POST …/passages` (contract
   1.8.0): the handed-over list is `[Handed(id, tier, text, section_path, page)]`. Library text reaches the worker by
   no other path, and nothing of it reaches Qdrant.

Failures (`SearchFailed`): Ollama `retrieval.embedding_unavailable` and Qdrant `retrieval.store_unavailable`, both
retryable (nothing has been sent yet); a question the model refuses as too long `retrieval.question_too_long`; a refusal
of the backend (4xx) its own code (`scope.passage_refused`, ...), or `retrieval.passages_refused` when it names none,
with no retry; any other failure of the passages route
`retrieval.passages_unavailable`, retryable. An input that is not the backend's is `InputInvalid`. Lines carry ids,
counts and timings: never the question, a text, a token or the key.
"""

from __future__ import annotations

import base64
import binascii
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from gmshield_worker.backend.client import ClaimedJob, PassagesAnswer, ReadPassage
from gmshield_worker.log import TraceLog
from gmshield_worker.retrieval.embedding import Embedder, EmbeddingFailed
from gmshield_worker.retrieval.indexing import StoreFactory
from gmshield_worker.retrieval.keywords import KEY_BYTES, keyword_vector
from gmshield_worker.retrieval.store import RRF_K, Found, StoreFailed

TIERS = ("campaign", "system")
LIMIT = 8
MAX_PENDING = 100
MAX_QUESTION = 4000
INPUT_INVALID = "retrieval.input_invalid"
EMBEDDING_UNAVAILABLE = "retrieval.embedding_unavailable"
STORE_UNAVAILABLE = "retrieval.store_unavailable"
PASSAGES_UNAVAILABLE = "retrieval.passages_unavailable"
PASSAGES_REFUSED = "retrieval.passages_refused"
_UUID = re.compile(r"[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", re.ASCII)


class InputInvalid(ValueError):
    """The claim's input is not the backend's question input. Names nothing."""


class SearchFailed(Exception):
    """The search could not end: the code, and whether another attempt could change that."""

    def __init__(self, code: str, *, retry: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retry = retry


@dataclass(frozen=True)
class TierScope:
    """One tier of the question's scope: its allowed sources and its pending passages, in the backend's order."""

    tier: str
    sources: tuple[str, ...]
    pending: tuple[str, ...]


@dataclass(frozen=True)
class QuestionInput:
    owner: str
    tiers: tuple[TierScope, ...]
    question: str = field(repr=False)
    token_key: bytes = field(repr=False)


@dataclass(frozen=True)
class Handed:
    """A passage handed over for the question: its id, its tier, its page; its text and section path never printed."""

    passage_id: str
    tier: str
    page: int
    text: str = field(repr=False)
    section_path: list[str] = field(repr=False)


class PassageReader(Protocol):
    """Reading passages by id under the claim's grant (`backend.client.JobsApi`, or a stand-in in tests)."""

    def passages(self, job: ClaimedJob, ids: list[str]) -> PassagesAnswer: ...


def _uuid(value: object) -> str:
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        raise InputInvalid
    return value.lower()


def _uuids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise InputInvalid
    found = tuple(_uuid(item) for item in value)
    if len(set(found)) != len(found):
        raise InputInvalid
    return found


def _tiers(value: object) -> tuple[TierScope, ...]:
    """The tiers, each at most once, in the order of `TIERS`."""
    if not isinstance(value, list):
        raise InputInvalid
    found: list[TierScope] = []
    for item in value:
        if not isinstance(item, dict) or item.get("tier") not in TIERS:
            raise InputInvalid
        tier = str(item["tier"])
        if found and TIERS.index(tier) <= TIERS.index(found[-1].tier):
            raise InputInvalid
        found.append(TierScope(tier, _uuids(item.get("sources")), _uuids(item.get("pending"))))
    return tuple(found)


def question_input(raw: dict[str, Any]) -> QuestionInput:
    """The question input, checked; `InputInvalid` otherwise."""
    question = raw.get("question")
    if not isinstance(question, str) or not 1 <= len(question) <= MAX_QUESTION:
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
    return QuestionInput(
        owner=_uuid(raw.get("owner")), tiers=_tiers(raw.get("tiers")), question=question, token_key=key
    )


def rank_pending(passages: Sequence[ReadPassage], question_tokens: set[int], key: bytes) -> list[str]:
    """The ids of the passages that share at least one keyword token with the question, by how many they share, most
    first; input order breaks ties."""
    shared = [
        (len(question_tokens & set(keyword_vector(item.text, key).indices)), item.passage_id) for item in passages
    ]
    return [passage_id for count, passage_id in sorted(shared, key=lambda pair: -pair[0]) if count > 0]


def fuse(found: Sequence[Found], pending: Sequence[str], limit: int = LIMIT) -> list[str]:
    """Qdrant's fused results with the pending ranking as a third RRF list: a pending id at rank r (from 0) scores
    1 / (r + RRF_K), as Qdrant scores each of its prefetches. An id found both ways is one entry whose two scores add
    up, as RRF sums across lists; a tie keeps Qdrant's order, then the pending order."""
    scores: dict[str, float] = {item.passage_id: item.score for item in found}
    for rank, passage_id in enumerate(pending):
        scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (rank + RRF_K)
    return sorted(scores, key=lambda passage_id: -scores[passage_id])[:limit]


def _read(reader: PassageReader, job: ClaimedJob, ids: list[str], tiers: dict[str, str]) -> list[ReadPassage]:
    """The passages with these ids, in this order, each of the expected tier; `SearchFailed` otherwise."""
    try:
        answer = reader.passages(job, ids)
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise SearchFailed(PASSAGES_UNAVAILABLE, retry=True) from error
    if answer.status == 200 and answer.passages is not None:
        read = answer.passages
        if [item.passage_id for item in read] != ids or any(item.tier != tiers[item.passage_id] for item in read):
            raise SearchFailed(PASSAGES_UNAVAILABLE, retry=True)
        return read
    if 400 <= answer.status < 500:
        raise SearchFailed(answer.code or PASSAGES_REFUSED, retry=False)
    raise SearchFailed(PASSAGES_UNAVAILABLE, retry=True)


@dataclass
class _Timings:
    clock: Callable[[], float]
    started: float = 0.0
    spent: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def lap(self, name: str) -> None:
        now = self.clock()
        self.spent[name] = round(self.spent.get(name, 0.0) + now - self.started, 3)
        self.started = now


def find_passages(
    job: ClaimedJob,
    checked: QuestionInput,
    *,
    embedder: Embedder,
    stores: StoreFactory,
    reader: PassageReader,
    log: TraceLog,
    clock: Callable[[], float] = time.monotonic,
) -> list[Handed]:
    """The passages that answer the question within its scope, as the module says."""
    job_id = str(job.job_id)
    searched = [tier for tier in checked.tiers if tier.sources]
    if not searched:
        log.info("retrieval.search_done", jobId=job_id, tiers=0, handed=0)
        return []
    timings = _Timings(clock)
    try:
        dense = embedder.embed_query(checked.question)
    except EmbeddingFailed as error:
        raise SearchFailed(EMBEDDING_UNAVAILABLE if error.retry else error.code, retry=error.retry) from error
    sparse = keyword_vector(checked.question, checked.token_key)
    timings.lap("embedSeconds")
    ranked: list[tuple[str, list[str]]] = []
    fetched = skipped = 0
    try:
        with stores() as store:
            store.ensure_collection()
            for tier in searched:
                found = store.search(checked.owner, tier.sources, dense, sparse, LIMIT)
                timings.lap("searchSeconds")
                pending = list(tier.pending[:MAX_PENDING])
                extra = len(tier.pending) - len(pending)
                if extra > 0:
                    skipped += extra
                    log.info("retrieval.pending_skipped", jobId=job_id, tier=tier.tier, skipped=extra)
                matching: list[str] = []
                if pending and sparse.indices:
                    read = _read(reader, job, pending, dict.fromkeys(pending, tier.tier))
                    matching = rank_pending(read, set(sparse.indices), checked.token_key)
                    fetched += len(pending)
                    timings.lap("pendingSeconds")
                ranked.append((tier.tier, fuse(found, matching)))
    except StoreFailed as error:
        raise SearchFailed(STORE_UNAVAILABLE, retry=True) from error
    tiers = {passage_id: tier for tier, ids in ranked for passage_id in ids}
    ids = list(tiers)
    handed: list[Handed] = []
    if ids:
        handed = [
            Handed(item.passage_id, item.tier, item.page, item.text, list(item.section_path))
            for item in _read(reader, job, ids, tiers)
        ]
        timings.lap("readSeconds")
    log.info(
        "retrieval.search_done",
        jobId=job_id,
        tiers=len(searched),
        handed=len(handed),
        ids=[item.passage_id for item in handed],
        pendingFetched=fetched,
        pendingSkipped=skipped,
        **timings.spent,
    )
    return handed
