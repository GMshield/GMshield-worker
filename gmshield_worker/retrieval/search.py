"""Finding the passages that answer a question (Story 2.3, AD-10): `find_passages`.

The question input is the backend's (`Scopes.questionInput`), never computed here: `{question, owner, tiers: [{tier,
sources, pending}], tokenKey}`, the tiers in order (campaign, system, then other from internal contract 1.13.0, Story
3.5; the backend works out each tier, the worker never does), `pending` the ids of the tier's passages
that are not indexed yet (their index change has not ended, or it failed; the backend lists at most the first 5,000 of
each tier), and `tokenKey` Base64 of the owner's 32-byte token key.

1. When no tier has a source, nothing is asked of Ollama, Qdrant or the backend: the list is empty.
2. The question is embedded with the indexing model under embeddinggemma's query prompt (`Embedder.embed_query`), and
   its keyword vector is `query_vector(question, tokenKey)` (each distinct token valued 1).
3. Per tier with sources, one Qdrant search (`PassageStore.search`): dense and sparse prefetches fused by RRF, `LIMIT`
   points, the owner and source filters always on. A tier without sources issues no query.
4. Pending: once every tier is searched, the first `MAX_PENDING` ids of each tier's `pending`, in input order, are read
   by id in pages of `PAGE` (the backend's most ids per call, `JobPassageReads.MAX_IDS`), at most `PARALLEL` pages at a
   time, the tiers' pages in turns (campaign, system, other, campaign, ..., then the rest of the longer ones), under one
   deadline `PENDING_BUDGET_SECONDS` after the question's first pending read (`read_pending`), so the first words stay
   within NFR-14's 5 s: no page starts after it, and a page not back by it is dropped, never waited for. Each page is
   tokenized as it comes back, inside the deadline: each passage keeps only the question's distinct folded tokens of two
   characters or more (`question_terms`, from `keywords.tokens`, never an HMAC) that it contains, and its text is let
   go. After the deadline only the counting runs: each token weighs BM25's ln(1 + (N - df + 0.5) / (df + 0.5)), N the
   pending passages read for the question (every tier) and df how many of them contain it (`rarity`); passages are
   ordered by the weight of the rarest question token they contain, then by the sum of their tokens' weights, so the
   passage naming what the question is about (a monster, a spell, a person) comes first; those with no token are dropped
   and ties keep input order (`rank_pending`). That ranking is fused with the tier's Qdrant results as a third RRF list
   (`fuse`), `LIMIT` ids. An id found both ways is one entry whose scores add up, as RRF does. The ids not read (past
   `MAX_PENDING`, in pages the deadline left out, or in pages it dropped) are skipped and logged as a count per tier
   (`retrieval.pending_skipped`). A question with no token of two characters or more can match no pending passage: its
   pending passages are not read at all.
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
import functools
import itertools
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from gmshield_worker.backend.client import ClaimedJob, PassagesAnswer, ReadPassage
from gmshield_worker.log import TraceLog
from gmshield_worker.retrieval.embedding import Embedder, EmbeddingFailed
from gmshield_worker.retrieval.indexing import StoreFactory
from gmshield_worker.retrieval.keywords import KEY_BYTES, query_vector, tokens
from gmshield_worker.retrieval.store import RRF_K, Found, StoreFailed

TIERS = ("campaign", "system", "other")
LIMIT = 8
# The most pending ids of a tier the worker reads: as many as the backend lists (`Scopes.MAX_PENDING`).
MAX_PENDING = 5000
# The most ids of one passages call (the backend's `JobPassageReads.MAX_IDS`).
PAGE = 100
# The most passages calls of a question's pending reads at a time, every tier together. 8, not 4: on roberto-server
# while indexing loaded it (2026-11-19), 4 read about 2,100-2,900 of 3,900 pending passages in the 1.5 s and 8 read
# all 3,721 in about 1 s.
PARALLEL = 8
# The pending reads' deadline, from the question's first pending read, for every tier together.
PENDING_BUDGET_SECONDS = 1.5
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


@dataclass(frozen=True)
class PendingPassage:
    """A pending passage read for a question: its id, and the question's distinct folded tokens it contains (never
    printed)."""

    passage_id: str
    terms: frozenset[str] = field(repr=False)


def question_terms(question: str) -> frozenset[str]:
    """The question's distinct folded tokens the pending ranking compares, one-character ones (the "s" of
    "Tarrasque's") left out."""
    return frozenset(term for term in tokens(question) if len(term) > 1)


def rarity(read: Iterable[PendingPassage]) -> dict[str, float]:
    """Each token's weight over the pending passages read for the question, BM25's ln(1 + (N - df + 0.5) / (df + 0.5)),
    N the passages and df how many of them contain it: about 0 for a token in nearly every passage."""
    passages = list(read)
    total = len(passages)
    found = Counter(term for item in passages for term in item.terms)
    return {term: math.log(1 + (total - count + 0.5) / (count + 0.5)) for term, count in found.items()}


def rank_pending(passages: Sequence[PendingPassage], weights: dict[str, float]) -> list[str]:
    """The ids of the passages that contain at least one of the question's tokens, highest first by the weight of the
    rarest question token they contain (`rarity`, 0 for a token missing from `weights`), then by the sum of those
    tokens' weights; those scoring 0 are dropped and input order breaks ties. The rarest token comes first because a
    sum lets several mid-frequency words outvote the one name the question is about: on roberto-server (2026-11-19),
    "challenge" + "rating" outweighed "tarrasque" and buried its stat block. `math.fsum` makes the sum correctly
    rounded, whatever the order."""
    scored = [
        (
            max((weights.get(term, 0.0) for term in item.terms), default=0.0),
            math.fsum(weights.get(term, 0.0) for term in item.terms),
            item.passage_id,
        )
        for item in passages
    ]
    ranked = sorted(scored, key=lambda entry: (-entry[0], -entry[1]))
    return [passage_id for rarest, _, passage_id in ranked if rarest > 0]


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


@functools.lru_cache(maxsize=64)
def _terms_pattern(terms: frozenset[str]) -> re.Pattern[str] | None:
    """One regex for the ASCII terms, each a whole word: `\\b(?:term|…)\\b`; None when no term is ASCII."""
    ascii_terms = sorted((term for term in terms if term.isascii()), key=lambda term: (-len(term), term))
    if not ascii_terms:
        return None
    return re.compile(r"\b(?:" + "|".join(map(re.escape, ascii_terms)) + r")\b")


def terms_in(text: str, terms: frozenset[str]) -> frozenset[str]:
    """The `terms` the text's folded tokens contain: exactly `terms ∩ tokens(text)`. For ASCII text (the SRD), folding
    is lowercasing and a token is a run of `[A-Za-z0-9_]`, so one regex over the lowered text finds the same terms
    about three times faster (measured on roberto-server: 0.13 against 0.40 ms for a 1,500-character passage); any
    other text is folded and tokenized whole."""
    if text.isascii():
        pattern = _terms_pattern(terms)
        return frozenset() if pattern is None else frozenset(pattern.findall(text.lower()))
    return terms.intersection(tokens(text))


def _read_terms(
    reader: PassageReader, job: ClaimedJob, ids: list[str], tiers: dict[str, str], terms: frozenset[str]
) -> list[PendingPassage]:
    """A page of pending passages (`_read`), each kept as its id and the `terms` its folded text contains
    (`terms_in`)."""
    return [PendingPassage(item.passage_id, terms_in(item.text, terms)) for item in _read(reader, job, ids, tiers)]


@dataclass(frozen=True)
class PendingRead:
    """What `read_pending` read of one tier: the passages of its pages back by the deadline, in input order, and how
    many of its ids it left unread (pages not started, or dropped)."""

    passages: list[PendingPassage]
    unread: int


def _in_turns(pending: Sequence[tuple[str, Sequence[str]]], page: int) -> list[tuple[str, int, list[str]]]:
    """The tiers' pages as `(tier, index, ids)`, in turns: each tier's first page in tier order, then each one's
    second, and so on; once a tier has no page left, the others' go on alone."""
    paged = [
        [(tier, index, list(ids[start : start + page])) for index, start in enumerate(range(0, len(ids), page))]
        for tier, ids in pending
    ]
    return [item for turn in itertools.zip_longest(*paged) for item in turn if item is not None]


def read_pending(
    reader: PassageReader,
    job: ClaimedJob,
    pending: Sequence[tuple[str, Sequence[str]]],
    *,
    terms: frozenset[str],
    clock: Callable[[], float] = time.monotonic,
    budget: float = PENDING_BUDGET_SECONDS,
    page: int = PAGE,
    parallel: int = PARALLEL,
) -> dict[str, PendingRead]:
    """Reads the pending passages of each `(tier, ids)` in pages of `page` ids, at most `parallel` pages at a time,
    the tiers' pages in turns (`_in_turns`), under one deadline `budget` seconds after the first read: no page starts
    after it, and a page not back by it is dropped, never waited for (its thread ends on its own, its answer unread).
    A page that fails before the deadline fails the whole read (`SearchFailed`), as one read did before. Each page is
    tokenized in its own thread as it comes back (`_read_terms`: the `terms` each passage contains), so that work is
    inside the deadline too."""
    pages = _in_turns(pending, page)
    read: dict[tuple[str, int], list[PendingPassage]] = {}
    if pages:
        deadline = clock() + budget
        running: dict[Future[list[PendingPassage]], tuple[str, int]] = {}
        started = 0
        pool = ThreadPoolExecutor(max_workers=max(1, parallel), thread_name_prefix="pending-read")
        try:
            while True:
                while started < len(pages) and len(running) < parallel and clock() < deadline:
                    tier, index, ids = pages[started]
                    future = pool.submit(_read_terms, reader, job, ids, dict.fromkeys(ids, tier), terms)
                    running[future] = (tier, index)
                    started += 1
                left = deadline - clock()
                if not running or left <= 0:
                    # Past the deadline nothing more is read: a page that ends, or fails, from now on is dropped.
                    break
                done, _ = wait(running, timeout=left, return_when=FIRST_COMPLETED)
                for future in done:
                    read[running.pop(future)] = future.result()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    result: dict[str, PendingRead] = {}
    for name, asked in pending:
        passages = [item for key in sorted(key for key in read if key[0] == name) for item in read[key]]
        result[name] = PendingRead(passages=passages, unread=len(asked) - len(passages))
    return result


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
    sparse = query_vector(checked.question, checked.token_key)
    timings.lap("embedSeconds")
    found: list[tuple[TierScope, list[Found]]] = []
    try:
        with stores() as store:
            store.ensure_collection()
            for tier in searched:
                found.append((tier, store.search(checked.owner, tier.sources, dense, sparse, LIMIT)))
                timings.lap("searchSeconds")
    except StoreFailed as error:
        raise SearchFailed(STORE_UNAVAILABLE, retry=True) from error
    capped = [(tier.tier, list(tier.pending[:MAX_PENDING])) for tier, _ in found if tier.pending]
    terms = question_terms(checked.question)
    read: dict[str, PendingRead] = {}
    if capped and terms:
        read = read_pending(
            reader,
            job,
            capped,
            terms=terms,
            clock=clock,
            budget=PENDING_BUDGET_SECONDS,
            page=PAGE,
            parallel=PARALLEL,
        )
        timings.lap("pendingSeconds")
    weights = rarity(item for tier_read in read.values() for item in tier_read.passages)
    ranked: list[tuple[str, list[str]]] = []
    fetched = skipped = 0
    for tier, tier_found in found:
        unread = len(tier.pending) - min(len(tier.pending), MAX_PENDING)
        matching: list[str] = []
        tier_read = read.get(tier.tier)
        if tier_read is not None:
            matching = rank_pending(tier_read.passages, weights)
            fetched += len(tier_read.passages)
            unread += tier_read.unread
        if unread > 0:
            skipped += unread
            log.info("retrieval.pending_skipped", jobId=job_id, tier=tier.tier, skipped=unread)
        ranked.append((tier.tier, fuse(tier_found, matching)))
    if read:
        timings.lap("rankSeconds")
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
