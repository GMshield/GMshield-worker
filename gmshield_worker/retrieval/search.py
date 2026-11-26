"""Finding the passages that answer a question (Story 2.3, AD-10): `find_passages`.

The question input is the backend's (`Scopes.questionInput`), never computed here: `{question, owner, tiers: [{tier,
sources, pending}], tokenKey}`, the tiers in order (campaign, system, then other from internal contract 1.13.0, Story
3.5; the backend works out each tier, the worker never does), `pending` the ids of the tier's passages
that are not indexed yet (their index change has not ended, or it failed; the backend lists at most the first 5,000 of
each tier), and `tokenKey` Base64 of the owner's 32-byte token key. From internal contract 1.16.0 (Story 3.10) it may
carry `history: [{question, answer}]`, the conversation's earlier turns oldest first (at most `MAX_HISTORY`, each answer
at most `MAX_HISTORY_ANSWER` code points): checked here and handed to the model as data only (`lanes.answer`); the
search below never reads it.

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
6. Story 4.3 part C (contract 1.19.0), an answer only (`find_sources` with a record reader): per tier, beside its
   passages search and with the same vectors, one more Qdrant query for record points only (`store.RECORDS`, any type of
   the scope's `recordTypes`, `RECORD_LIMIT` points), so records never take a passage's slot and passages rank exactly
   as before. The found record ids, in rank order, campaign tier first, are read in one call to `POST …/records`, never
   `…/passages`: the handed-over records are `[HandedRecord(id, tier, type, name, fields, page, section_path)]`. A
   record id never reaches the pending ranking. `retrieval.search_done` keeps `ids` (the passages) and adds `records`
   (the record ids).

7. Story 4.6 (contract 1.23.0), an answer only (`find_sources` with a term resolver): alongside the question's
   embedding, the backend is asked to resolve the question's terms (`POST …/terms`, no body: it reads the turn's
   question and matches its words against the system's ontology and the scope's record names). Its answer is used once
   the embedding is back, waiting at most `TERMS_BUDGET_SECONDS` from the ask: the resolved names join the keyword
   query (`query_vector` over the question and the names); each tier's records search gets one more Qdrant query,
   records of the resolved types (those of the scope's `recordTypes`) only, fused with it by RRF (`merge_records`);
   and the resolved record ids come first among the records read, any tier the backend labels, then the records
   search's, at most `RESOLVED_LIMIT` in all (no more than that search alone hands). An empty answer, a 404 (a backend
   before 1.23.0), any failure or a timeout leaves the search exactly as before, with the line
   `answer.terms_unavailable` (its reason and status); a records read refused with named records in it is read again
   without them (the line then carries the read's code and retry flag). The worker derives no definition and no type
   from the names, and the names reach no prompt and no line: lines carry counts and `termsSeconds`, the term call's
   own duration (the budget waited when it is late).

The library search (Story 3.9, `lanes/search.py`) reuses steps 1 to 4 (`rank_tiers`) on the backend's search input
(`Scopes.searchInput`: the same shape with `query`, 1 to `MAX_QUERY` code points, in place of `question`;
`question_input(raw, field="query")`), with its own limit per tier, and keeps each id's fused score. From internal
contract 1.18.0 (Story 4.3 part B) it searches records too (`store.EVERYTHING`), and a search input may carry
`recordType`, a record type key (`indexing.RECORD_TYPE_KEY`; a core key before Story 4.4 part B), which leaves only
that type's records (`store.records_of`): then no pending passage is read, since a passage is never a record. An
answer still searches passages only (`store.PASSAGES`) beside its records search.

Story 4.4 part B (internal contract 1.21.0): a question or a search input may carry `recordTypes`, the record type keys
of the campaign's system (distinct, at most `MAX_RECORD_TYPES`). Every Qdrant query that may answer records then keeps
only record points of those keys (`store.limited`, a `MatchAny` on `record_type`), so a record of a type that system
never approved is never found; passages are untouched. An input without it (a backend before 1.21.0) limits nothing.

Failures (`SearchFailed`): Ollama `retrieval.embedding_unavailable` and Qdrant `retrieval.store_unavailable`, both
retryable (nothing has been sent yet); a question the model refuses as too long `retrieval.question_too_long`; a refusal
of the backend (4xx) its own code (`scope.passage_refused`, ...), or `retrieval.passages_refused` when it names none,
with no retry; any other failure of the passages route
`retrieval.passages_unavailable`, retryable. The records route fails the same way, with `retrieval.records_refused` and
`retrieval.records_unavailable`. An input that is not the backend's is `InputInvalid`. Lines carry ids,
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from gmshield_worker.backend.client import (
    ClaimedJob,
    PassagesAnswer,
    ReadPassage,
    ReadRecord,
    RecordsAnswer,
    TermsAnswer,
)
from gmshield_worker.log import TraceLog
from gmshield_worker.retrieval.embedding import Embedder, EmbeddingFailed
from gmshield_worker.retrieval.indexing import StoreFactory, is_record_type
from gmshield_worker.retrieval.keywords import KEY_BYTES, Sparse, query_vector, tokens
from gmshield_worker.retrieval.store import (
    PASSAGES,
    RECORDS,
    RRF_K,
    Found,
    Kinds,
    PassageStore,
    StoreFailed,
    limited,
)

TIERS = ("campaign", "system", "other")
LIMIT = 8
# Story 4.3 part C: the most record points an answer's records search finds per tier, beside its `LIMIT` passages.
RECORD_LIMIT = 4
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
# The longest search query, in code points (the backend's `Scopes.MAX_QUERY`, Story 3.9).
MAX_QUERY = 500
# Story 3.10: the most earlier turns a question input carries, and the longest earlier answer, in code points (the
# backend's `Conversations.HISTORY_TURNS` and `HISTORY_ANSWER`).
MAX_HISTORY = 6
MAX_HISTORY_ANSWER = 4000
# Story 4.4 part B: the most record type keys a question or a search input carries (a system's core and approved types).
MAX_RECORD_TYPES = 1000
# Story 4.6: how long an answer waits for its question's terms (`POST …/terms`, asked alongside the question's
# embedding), from the moment it asked; past it the search runs as before.
TERMS_BUDGET_SECONDS = 1.0
# Story 4.6: the most record ids a terms answer may carry (the backend's `Scopes.MAX_TERM_RECORDS`), and the most
# records an answer hands to the model, the named ones first, then its records search's: as many as that search alone
# finds at most, so the prompt holds no more record blocks than before.
MAX_TERM_RECORDS = 100
RESOLVED_LIMIT = RECORD_LIMIT * 3
# Story 4.6: the line of a term resolution that gave nothing to use (timeout, refusal, failure, an older backend).
TERMS_UNAVAILABLE = "answer.terms_unavailable"
INPUT_INVALID = "retrieval.input_invalid"
EMBEDDING_UNAVAILABLE = "retrieval.embedding_unavailable"
STORE_UNAVAILABLE = "retrieval.store_unavailable"
PASSAGES_UNAVAILABLE = "retrieval.passages_unavailable"
PASSAGES_REFUSED = "retrieval.passages_refused"
RECORDS_UNAVAILABLE = "retrieval.records_unavailable"
RECORDS_REFUSED = "retrieval.records_refused"
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
class EarlierTurn:
    """An earlier turn of the question's conversation (Story 3.10): data for the model only, never searched with and
    never a source. Its question and answer are never printed."""

    question: str = field(repr=False)
    answer: str = field(repr=False)


@dataclass(frozen=True)
class QuestionInput:
    """The checked question input. `model_knowledge` (Story 3.8, contract 1.15.0) is the turn's frozen switch: whether
    the model may add its own knowledge, each such part under `[[tier:model]]`; absent from the input, it is off.
    `history` (Story 3.10, contract 1.16.0) is the conversation's earlier turns, oldest first; absent, there is none.
    `record_type` (Story 4.3 part B, contract 1.18.0) is a search's record-type filter, a record type key; absent,
    None. `record_types` (Story 4.4 part B, contract 1.21.0) is the campaign's system's keys, to which every search
    that may answer records is limited; absent, None (nothing is limited). Retrieval reads `question` alone."""

    owner: str
    tiers: tuple[TierScope, ...]
    question: str = field(repr=False)
    token_key: bytes = field(repr=False)
    model_knowledge: bool = False
    history: tuple[EarlierTurn, ...] = field(default=(), repr=False)
    record_type: str | None = None
    record_types: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Handed:
    """A passage handed over for the question: its id, its tier, its page; its text and section path never printed."""

    passage_id: str
    tier: str
    page: int
    text: str = field(repr=False)
    section_path: list[str] = field(repr=False)


@dataclass(frozen=True)
class HandedField:
    """One field of a handed record: never printed."""

    name: str = field(repr=False)
    value: str = field(repr=False)


@dataclass(frozen=True)
class HandedRecord:
    """A record handed over for the question (Story 4.3 part C): its id, its tier, its type key and its first page;
    its name, fields and section path never printed."""

    record_id: str
    tier: str
    type: str
    page: int
    name: str = field(repr=False)
    fields: tuple[HandedField, ...] = field(repr=False)
    section_path: list[str] = field(repr=False)


@dataclass(frozen=True)
class Sources:
    """What `find_sources` hands over: the passages, then (an answer only) the records, each in rank order."""

    passages: list[Handed]
    records: list[HandedRecord] = field(default_factory=list)


class PassageReader(Protocol):
    """Reading passages by id under the claim's grant (`backend.client.JobsApi`, or a stand-in in tests)."""

    def passages(self, job: ClaimedJob, ids: list[str]) -> PassagesAnswer: ...


class RecordReader(Protocol):
    """Reading records by id under the claim's grant (`backend.client.JobsApi`, or a stand-in in tests; Story 4.3 part
    C)."""

    def records(self, job: ClaimedJob, ids: list[str]) -> RecordsAnswer: ...


class TermResolver(Protocol):
    """Resolving the question's terms under the claim's grant (`backend.client.JobsApi`, or a stand-in in tests; Story
    4.6)."""

    def terms(self, job: ClaimedJob) -> TermsAnswer: ...


@dataclass(frozen=True)
class ResolvedTerms:
    """What the backend resolved the question's terms to (Story 4.6): the record type keys it names, the names of its
    matched terms (with their aliases) and records, and the ids of the in-scope records it names, in the backend's
    order. The names are library data: they join the keyword query only, never a prompt, a line or a repr."""

    record_types: tuple[str, ...] = ()
    names: tuple[str, ...] = field(default=(), repr=False)
    records: tuple[str, ...] = ()


NO_TERMS_RESOLVED = ResolvedTerms()


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


def _history(value: object) -> tuple[EarlierTurn, ...]:
    """The earlier turns (Story 3.10): a list of at most `MAX_HISTORY` objects of exactly `question` (1 to
    `MAX_QUESTION` code points) and `answer` (0 to `MAX_HISTORY_ANSWER`), both strings."""
    if not isinstance(value, list) or len(value) > MAX_HISTORY:
        raise InputInvalid
    found: list[EarlierTurn] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"question", "answer"}:
            raise InputInvalid
        question, earlier_answer = item["question"], item["answer"]
        if not isinstance(question, str) or not 1 <= len(question) <= MAX_QUESTION:
            raise InputInvalid
        if not isinstance(earlier_answer, str) or len(earlier_answer) > MAX_HISTORY_ANSWER:
            raise InputInvalid
        found.append(EarlierTurn(question, earlier_answer))
    return tuple(found)


def question_input(raw: dict[str, Any], *, field: str = "question") -> QuestionInput:
    """The question input, checked; `InputInvalid` otherwise. With `field="query"` (Story 3.9) it is the backend's
    search input instead: the same shape, its text (1 to `MAX_QUERY` code points) under `query`, kept as `question`;
    a search input carries no `history`, and may carry `recordType` (Story 4.3 part B), a record type key, which a
    question input never does; either may carry `recordTypes` (Story 4.4 part B)."""
    if field not in ("question", "query"):
        raise ValueError(field)
    question = raw.get(field)
    longest = MAX_QUESTION if field == "question" else MAX_QUERY
    if not isinstance(question, str) or not 1 <= len(question) <= longest:
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
    model_knowledge = raw.get("modelKnowledge", False)
    if not isinstance(model_knowledge, bool):
        raise InputInvalid
    if field == "query" and "history" in raw:
        raise InputInvalid
    if field == "question" and "recordType" in raw:
        raise InputInvalid
    record_type = raw.get("recordType")
    if record_type is not None and not is_record_type(record_type):
        raise InputInvalid
    return QuestionInput(
        owner=_uuid(raw.get("owner")),
        tiers=_tiers(raw.get("tiers")),
        question=question,
        token_key=key,
        model_knowledge=model_knowledge,
        history=_history(raw["history"]) if "history" in raw else (),
        record_type=record_type,
        record_types=_record_types(raw["recordTypes"]) if "recordTypes" in raw else None,
    )


def _record_types(value: object) -> tuple[str, ...]:
    """The scope's record type keys (Story 4.4 part B): a list of at most `MAX_RECORD_TYPES` distinct keys."""
    if not isinstance(value, list) or len(value) > MAX_RECORD_TYPES:
        raise InputInvalid
    if not all(is_record_type(item) for item in value) or len(set(value)) != len(value):
        raise InputInvalid
    return tuple(value)


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


def fused(found: Sequence[Found], pending: Sequence[str], limit: int = LIMIT) -> list[tuple[str, float]]:
    """Qdrant's fused results with the pending ranking as a third RRF list, each id with its score: a pending id at
    rank r (from 0) scores 1 / (r + RRF_K), as Qdrant scores each of its prefetches. An id found both ways is one entry
    whose two scores add up, as RRF sums across lists; a tie keeps Qdrant's order, then the pending order."""
    scores: dict[str, float] = {item.passage_id: item.score for item in found}
    for rank, passage_id in enumerate(pending):
        scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (rank + RRF_K)
    ranked = sorted(scores, key=lambda passage_id: -scores[passage_id])[:limit]
    return [(passage_id, scores[passage_id]) for passage_id in ranked]


def fuse(found: Sequence[Found], pending: Sequence[str], limit: int = LIMIT) -> list[str]:
    """The ids of `fused`, best first."""
    return [passage_id for passage_id, _ in fused(found, pending, limit)]


def read_passages(reader: PassageReader, job: ClaimedJob, ids: list[str], tiers: dict[str, str]) -> list[ReadPassage]:
    """The passages with these ids (1 to `PAGE`), in this order, each of the expected tier; `SearchFailed`
    otherwise."""
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


def read_records(
    reader: RecordReader, job: ClaimedJob, ids: list[str], tiers: Mapping[str, str | None]
) -> list[ReadRecord]:
    """The records with these ids (1 to `PAGE`), in this order, each of the expected tier (any of `TIERS` for an id
    whose tier is None: a record the question's terms named, Story 4.6); `SearchFailed` otherwise."""
    try:
        answer = reader.records(job, ids)
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise SearchFailed(RECORDS_UNAVAILABLE, retry=True) from error
    if answer.status == 200 and answer.records is not None:
        read = answer.records
        if [item.record_id for item in read] != ids or any(
            item.tier not in TIERS or tiers[item.record_id] not in (None, item.tier) for item in read
        ):
            raise SearchFailed(RECORDS_UNAVAILABLE, retry=True)
        return read
    if 400 <= answer.status < 500:
        raise SearchFailed(answer.code or RECORDS_REFUSED, retry=False)
    raise SearchFailed(RECORDS_UNAVAILABLE, retry=True)


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
    """A page of pending passages (`read_passages`), each kept as its id and the `terms` its folded text contains
    (`terms_in`)."""
    read = read_passages(reader, job, ids, tiers)
    return [PendingPassage(item.passage_id, terms_in(item.text, terms)) for item in read]


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

    def record(self, name: str, seconds: float) -> None:
        """`name` took `seconds`, timed on its own (Story 4.6: the term call, which ran alongside the embedding); the
        next lap starts now, so the wait for it counts in no lap."""
        self.spent[name] = round(seconds, 3)
        self.started = self.clock()


@dataclass(frozen=True)
class TierRanking:
    """One searched tier's fused ranking (`fused`): its ids, best first, each with its RRF score, and (Story 4.3
    part B) which of them are records' points; (Story 4.3 part C) the ids of its separate records search, best
    first."""

    tier: str
    scored: list[tuple[str, float]]
    records: frozenset[str] = frozenset()
    found_records: tuple[str, ...] = ()

    @property
    def ids(self) -> list[str]:
        return [passage_id for passage_id, _ in self.scored]


@dataclass
class Ranked:
    """What `rank_tiers` found: each searched tier's ranking in tier order (empty when no tier has a source), how many
    pending passages it read and skipped, and its timings so far (None when no tier has a source: nothing was timed)."""

    tiers: list[TierRanking]
    searched: int
    fetched: int
    skipped: int
    timings: _Timings | None
    terms: ResolvedTerms = NO_TERMS_RESOLVED


@dataclass(frozen=True)
class _Resolution:
    """What one term call gave: the checked terms, or None with why (`reason`) and the status; and how long the call
    took (`seconds`, `time.monotonic` around it)."""

    terms: ResolvedTerms | None
    reason: str
    status: int
    seconds: float


def _resolve(resolver: TermResolver, job: ClaimedJob) -> _Resolution:
    """The question's terms as the backend resolved them, checked, timed around the call. Never raises: a term
    resolution only sharpens the search."""
    started = time.monotonic()
    try:
        answer = resolver.terms(job)
    except Exception as error:  # any failure of the call leaves the search as it was
        return _Resolution(None, f"transport:{type(error).__name__}", 0, time.monotonic() - started)
    seconds = time.monotonic() - started
    found, reason = _checked(answer)
    return _Resolution(found, reason, answer.status, seconds)


def _checked(answer: TermsAnswer) -> tuple[ResolvedTerms | None, str]:
    """A terms answer's terms, or None with why: a refusal's code, or `invalid` for one out of the contract."""
    if answer.status != 200:
        return None, answer.code or "status"
    record_types, names, records = answer.record_types, answer.names, answer.records
    if (
        record_types is None
        or names is None
        or records is None
        or not all(is_record_type(item) for item in record_types)
        or not all(isinstance(item, str) for item in names)
        or len(records) > MAX_TERM_RECORDS
        or not all(isinstance(item, str) and _UUID.fullmatch(item) for item in records)
    ):
        return None, "invalid"
    unique = tuple(dict.fromkeys(item.lower() for item in records))
    return ResolvedTerms(tuple(dict.fromkeys(record_types)), tuple(dict.fromkeys(names)), unique), ""


def _await_terms(
    future: Future[_Resolution], deadline: float, budget: float, job_id: str, log: TraceLog
) -> tuple[ResolvedTerms, float]:
    """The resolved terms once back, waiting at most until `deadline` (`time.monotonic`), and how long the term call
    took (the `budget` waited when it is late); nothing when it failed or is late, with a line saying so
    (`TERMS_UNAVAILABLE`)."""
    try:
        resolution = future.result(timeout=max(0.0, deadline - time.monotonic()))
    except FutureTimeout:
        resolution = _Resolution(None, "timeout", 0, budget)
    seconds = round(resolution.seconds, 3)
    resolved = resolution.terms
    if resolved is None:
        log.warn(
            TERMS_UNAVAILABLE, jobId=job_id, reason=resolution.reason, status=resolution.status, termsSeconds=seconds
        )
        return NO_TERMS_RESOLVED, resolution.seconds
    log.info(
        "retrieval.terms_resolved",
        jobId=job_id,
        recordTypes=len(resolved.record_types),
        names=len(resolved.names),
        records=len(resolved.records),
        termsSeconds=seconds,
    )
    return resolved, resolution.seconds


def merge_records(found: Sequence[Found], typed: Sequence[Found], limit: int) -> list[Found]:
    """A tier's records search and its resolved types' records search (Story 4.6) fused as RRF fuses its lists: an id
    found by both is one entry whose two scores add up; best first, a tie keeping the records search's order, then the
    typed one's; at most `limit`."""
    scores: dict[str, float] = {}
    for item in [*found, *typed]:
        scores[item.passage_id] = scores.get(item.passage_id, 0.0) + item.score
    ranked = sorted(scores, key=lambda passage_id: -scores[passage_id])[:limit]
    return [Found(passage_id, scores[passage_id], record=True) for passage_id in ranked]


def _embed_and_resolve(
    job: ClaimedJob,
    checked: QuestionInput,
    *,
    embedder: Embedder,
    resolver: TermResolver | None,
    terms_budget: float,
    timings: _Timings,
    log: TraceLog,
) -> tuple[list[float], ResolvedTerms]:
    """The question's embedding and (with a `resolver`, Story 4.6) its resolved terms, asked alongside it in a thread of
    its own and waited for at most `terms_budget` seconds from the ask; `embedSeconds` and `termsSeconds` timed."""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="terms") if resolver is not None else None
    future = pool.submit(_resolve, resolver, job) if pool is not None and resolver is not None else None
    deadline = time.monotonic() + terms_budget
    try:
        try:
            dense = embedder.embed_query(checked.question)
        except EmbeddingFailed as error:
            raise SearchFailed(EMBEDDING_UNAVAILABLE if error.retry else error.code, retry=error.retry) from error
        timings.lap("embedSeconds")
        resolved = NO_TERMS_RESOLVED
        if future is not None:
            resolved, seconds = _await_terms(future, deadline, terms_budget, str(job.job_id), log)
            timings.record("termsSeconds", seconds)
    finally:
        if pool is not None:
            # A late resolution is not waited for: its thread ends on its own, its answer unread.
            pool.shutdown(wait=False, cancel_futures=True)
    return dense, resolved


def _search_records(
    store: PassageStore,
    tier: TierScope,
    *,
    owner: str,
    vectors: tuple[list[float], Sparse],
    records: int,
    searches: tuple[Kinds, Kinds | None],
) -> list[Found]:
    """A tier's records search (step 6): at most `records` record points of the scope's keys; with resolved types
    (Story 4.6), one more query of those types only, fused with it (`merge_records`)."""
    dense, sparse = vectors
    any_kind, typed = searches
    hits = [item for item in store.search(owner, tier.sources, dense, sparse, records, kinds=any_kind) if item.record]
    if typed is None:
        return hits
    extra = [item for item in store.search(owner, tier.sources, dense, sparse, records, kinds=typed) if item.record]
    return merge_records(hits, extra, records)


def rank_tiers(
    job: ClaimedJob,
    checked: QuestionInput,
    *,
    embedder: Embedder,
    stores: StoreFactory,
    reader: PassageReader,
    log: TraceLog,
    clock: Callable[[], float] = time.monotonic,
    limit: int = LIMIT,
    kinds: Kinds = PASSAGES,
    records: int = 0,
    resolver: TermResolver | None = None,
    terms_budget: float = TERMS_BUDGET_SECONDS,
) -> Ranked:
    """Steps 1 to 4 of the module: each tier with sources searched in Qdrant (`limit` points of the `kinds` it may
    answer) and its pending passages ranked, fused into at most `limit` ids per tier with their scores. No passage text
    is read but the pending passages' (to rank them); with a record-type filter no pending passage is read at all.
    With `records` above 0 (step 6), each tier is also searched for at most that many record points. With a `resolver`
    (an answer, step 7), the question's terms are resolved alongside the embedding, within `terms_budget` seconds."""
    job_id = str(job.job_id)
    searched = [tier for tier in checked.tiers if tier.sources]
    if not searched:
        return Ranked([], 0, 0, 0, None)
    # Story 4.4 part B: the records of the campaign's system's keys only, whichever search may answer them.
    kinds = limited(kinds, checked.record_types)
    record_kinds = limited(RECORDS, checked.record_types)
    timings = _Timings(clock)
    dense, resolved = _embed_and_resolve(
        job, checked, embedder=embedder, resolver=resolver, terms_budget=terms_budget, timings=timings, log=log
    )
    # Story 4.6: the names the question's terms resolved to join its keyword query; with none, it is exactly as before.
    keywords = "\n".join((checked.question, *resolved.names)) if resolved.names else checked.question
    sparse = query_vector(keywords, checked.token_key)
    allowed = checked.record_types
    typed = [key for key in resolved.record_types if allowed is None or key in allowed]
    record_searches = (record_kinds, limited(RECORDS, typed) if typed else None) if records > 0 else None
    found: list[tuple[TierScope, list[Found]]] = []
    found_records: dict[str, tuple[str, ...]] = {}
    try:
        with stores() as store:
            store.ensure_collection()
            for tier in searched:
                found.append((tier, store.search(checked.owner, tier.sources, dense, sparse, limit, kinds=kinds)))
                if record_searches is not None:
                    hits = _search_records(
                        store,
                        tier,
                        owner=checked.owner,
                        vectors=(dense, sparse),
                        records=records,
                        searches=record_searches,
                    )
                    found_records[tier.tier] = tuple(item.passage_id for item in hits)
                timings.lap("searchSeconds")
    except StoreFailed as error:
        raise SearchFailed(STORE_UNAVAILABLE, retry=True) from error
    capped = [
        (tier.tier, list(tier.pending[:MAX_PENDING])) for tier, _ in found if tier.pending and kinds.record_type is None
    ]
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
    ranked: list[TierRanking] = []
    fetched = skipped = 0
    for tier, tier_found in found:
        unread = len(tier.pending) - min(len(tier.pending), MAX_PENDING) if kinds.record_type is None else 0
        matching: list[str] = []
        tier_read = read.get(tier.tier)
        if tier_read is not None:
            matching = rank_pending(tier_read.passages, weights)
            fetched += len(tier_read.passages)
            unread += tier_read.unread
        if unread > 0:
            skipped += unread
            log.info("retrieval.pending_skipped", jobId=job_id, tier=tier.tier, skipped=unread)
        found_ids = frozenset(item.passage_id for item in tier_found if item.record)
        ranked.append(
            TierRanking(tier.tier, fused(tier_found, matching, limit), found_ids, found_records.get(tier.tier, ()))
        )
    if read:
        timings.lap("rankSeconds")
    return Ranked(ranked, len(searched), fetched, skipped, timings, resolved)


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
    """The passages that answer the question within its scope, as the module says (steps 1 to 5)."""
    return find_sources(job, checked, embedder=embedder, stores=stores, reader=reader, log=log, clock=clock).passages


def find_sources(
    job: ClaimedJob,
    checked: QuestionInput,
    *,
    embedder: Embedder,
    stores: StoreFactory,
    reader: PassageReader,
    log: TraceLog,
    clock: Callable[[], float] = time.monotonic,
    record_reader: RecordReader | None = None,
    resolver: TermResolver | None = None,
    terms_budget: float = TERMS_BUDGET_SECONDS,
) -> Sources:
    """The passages and, with a `record_reader` (an answer, Story 4.3 part C), the records that answer the question
    within its scope, as the module says; with a `resolver` (an answer, Story 4.6), sharpened by its resolved terms."""
    job_id = str(job.job_id)
    ranked = rank_tiers(
        job,
        checked,
        embedder=embedder,
        stores=stores,
        reader=reader,
        log=log,
        clock=clock,
        limit=LIMIT,
        records=RECORD_LIMIT if record_reader is not None else 0,
        resolver=resolver,
        terms_budget=terms_budget,
    )
    timings = ranked.timings
    if timings is None:
        log.info("retrieval.search_done", jobId=job_id, tiers=0, handed=0, records=[])
        return Sources([])
    tiers = {passage_id: tier.tier for tier in ranked.tiers for passage_id in tier.ids}
    ids = list(tiers)
    handed: list[Handed] = []
    if ids:
        handed = [
            Handed(item.passage_id, item.tier, item.page, item.text, list(item.section_path))
            for item in read_passages(reader, job, ids, tiers)
        ]
        timings.lap("readSeconds")
    found_tiers = {record_id: tier.tier for tier in ranked.tiers for record_id in tier.found_records}
    records: list[HandedRecord] = []
    named = 0
    if record_reader is not None:
        # Story 4.6: the records the question's terms named come first, then those the records search found.
        resolved = list(ranked.terms.records)
        records, named = _read_handed(
            record_reader, job, resolved=resolved, found_tiers=found_tiers, job_id=job_id, log=log
        )
        if records or resolved or found_tiers:
            timings.lap("recordsSeconds")
    log.info(
        "retrieval.search_done",
        jobId=job_id,
        tiers=ranked.searched,
        handed=len(handed),
        ids=[item.passage_id for item in handed],
        records=[item.record_id for item in records],
        resolvedRecords=named,
        resolvedLeftOut=len(ranked.terms.records) - named,
        pendingFetched=ranked.fetched,
        pendingSkipped=ranked.skipped,
        **timings.spent,
    )
    return Sources(handed, records)


def _read_handed(
    record_reader: RecordReader,
    job: ClaimedJob,
    *,
    resolved: list[str],
    found_tiers: dict[str, str],
    job_id: str,
    log: TraceLog,
) -> tuple[list[HandedRecord], int]:
    """The records to hand over and how many of them the terms named: those the terms named (`resolved`, of any tier),
    then those the records search found (`found_tiers`, each of its tier), each once, at most `RESOLVED_LIMIT` in all
    (as many as the records search alone hands at most), in one read. When the read with named records fails, they are
    let go (`TERMS_UNAVAILABLE`, with the failure's code and retry flag) and the found ones alone are read, as before:
    term resolution never fails an answer."""
    wanted: dict[str, str | None] = dict.fromkeys(resolved)
    wanted.update(found_tiers)
    kept = list(wanted)[:RESOLVED_LIMIT]
    if not kept:
        return [], 0
    named = len(set(kept) & set(resolved))
    try:
        read = read_records(record_reader, job, kept, wanted)
    except SearchFailed as failed:
        if named == 0:
            raise
        log.warn(TERMS_UNAVAILABLE, jobId=job_id, reason="records", code=failed.code, retry=failed.retry)
        read = read_records(record_reader, job, list(found_tiers), found_tiers) if found_tiers else []
        named = 0
    handed = [
        HandedRecord(
            item.record_id,
            item.tier,
            item.type,
            item.page,
            item.name,
            tuple(HandedField(one.name, one.value) for one in item.fields),
            list(item.section_path),
        )
        for item in read
    ]
    return handed, named
