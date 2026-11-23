"""The `library.search` job kind (Story 3.9): the GM's search of the library, by meaning or exact words, over the
campaign's scope only. It calls no model.

Its claim input is the backend's search input (`Scopes.searchInput`): the question input's shape with `query` (1 to
`retrieval.search.MAX_QUERY` code points) in place of `question`, its tiers already narrowed by the search's filters,
and from internal contract 1.18.0 (Story 4.3 part B) an optional `recordType`, a core key. The backend works out every
tier; the worker never does.

1. Each tier with sources is searched as a question's is (`retrieval.search.rank_tiers`: Qdrant's dense and sparse
   prefetches fused by RRF, under the owner and source filters, and the tier's pending passages ranked by the query's
   tokens), up to `LIMIT` candidates per tier, each with its fused score. From Story 4.3 part B the search also answers
   records' points (`store.EVERYTHING`), mixed with passages by that score; with `recordType` it answers only that
   type's records (`store.records_of`) and reads no pending passage. Records still being indexed have no point yet, so
   they are not found (there is no keyword fallback for records).
2. The tiers are merged by that score, highest first; a tie keeps tier order (campaign, system, other), then each
   tier's own order (`merge`).
3. The passage candidates' text is read through the backend under the claim's grant (`POST …/passages`), in calls of
   at most `PAGE` ids, each checked to be the tier the backend labels it (`read_texts`). Library text reaches the worker
   by no other path, and a record's name or fields never do (points carry none, AD-10).
4. Every passage whose folded text (`keywords.fold`, runs of white space made one space) contains the folded query
   moves to the front, in its existing order (`exact_first`), so a passage that holds the query verbatim, whatever its
   case or accents, comes first. Records keep their fused rank among the rest: a record whose name equals the query is
   put first by the backend, which holds names.
5. The result is the first `LIMIT` ids, passages and records alike, `{"ids": [...]}`. The backend tells them apart and
   re-checks them against the scope.

The index's payload, its vectors, the keyword numbering and `INDEX_RECIPE` are unchanged. An input that is not the
backend's fails with `retrieval.input_invalid` and `retry: false`; a search failure with its code
(`retrieval.search.SearchFailed`), retried by the backend only when another attempt could change it. Lines carry ids,
counts and timings: never the query, a text, a token or the key.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gmshield_worker.backend.client import ClaimedJob, Outcome, ReadPassage
from gmshield_worker.log import TraceLog
from gmshield_worker.retrieval import search, store
from gmshield_worker.retrieval.embedding import Embedder
from gmshield_worker.retrieval.indexing import StoreFactory
from gmshield_worker.retrieval.keywords import fold

KIND = "library.search"
UNIT = "worker.retrieval"
# The most candidates per tier, and the most ids the result holds (the backend's `Searches.MAX_RESULTS`).
LIMIT = 20
# The most ids of one passages call (the backend's `JobPassageReads.MAX_IDS`).
PAGE = search.PAGE


def merge(tiers: Sequence[search.TierRanking]) -> list[tuple[str, str]]:
    """Every tier's candidates as `(id, tier)`, highest fused score first; a tie keeps tier order, then the tier's
    own order. An id found in two tiers (it cannot be: a passage has one source) keeps its first."""
    entries = [
        (-score, tier_index, rank, passage_id, ranking.tier)
        for tier_index, ranking in enumerate(tiers)
        for rank, (passage_id, score) in enumerate(ranking.scored)
    ]
    merged: dict[str, str] = {}
    for _, _, _, passage_id, tier in sorted(entries, key=lambda entry: entry[:3]):
        merged.setdefault(passage_id, tier)
    return list(merged.items())


def flat(text: str) -> str:
    """The folded text (`keywords.fold`), its runs of white space made one space and its ends trimmed."""
    return " ".join(fold(text).split())


def exact_first(read: Sequence[ReadPassage], query: str, order: Sequence[str] | None = None) -> list[str]:
    """The ids of `order` (by default the read passages' own), those of the read passages whose flattened text contains
    the flattened query first, each group in its existing order. An id of `order` with no text read (a record's, Story
    4.3 part B) keeps its place among the rest."""
    ids = [item.passage_id for item in read] if order is None else list(order)
    wanted = flat(query)
    if not wanted:
        return ids
    holding = [item.passage_id for item in read if wanted in flat(item.text)]
    held = set(holding)
    return holding + [item for item in ids if item not in held]


def read_texts(
    reader: search.PassageReader, job: ClaimedJob, candidates: Sequence[tuple[str, str]], page: int = PAGE
) -> list[ReadPassage]:
    """The candidates' passages, in their order, read in calls of at most `page` ids, each call checked as
    `search.read_passages` checks it. Only passage candidates are given: a record has no text to read."""
    tiers = dict(candidates)
    ids = [passage_id for passage_id, _ in candidates]
    read: list[ReadPassage] = []
    for start in range(0, len(ids), page):
        read.extend(search.read_passages(reader, job, ids[start : start + page], tiers))
    return read


def find_ids(
    job: ClaimedJob,
    checked: search.QuestionInput,
    *,
    embedder: Embedder,
    stores: StoreFactory,
    reader: search.PassageReader,
    log: TraceLog,
    clock: Callable[[], float] = time.monotonic,
) -> list[str]:
    """The ids the search finds within its scope, best first, as the module says."""
    job_id = str(job.job_id)
    kinds = store.EVERYTHING if checked.record_type is None else store.records_of(checked.record_type)
    ranked = search.rank_tiers(
        job, checked, embedder=embedder, stores=stores, reader=reader, log=log, clock=clock, limit=LIMIT, kinds=kinds
    )
    timings = ranked.timings
    if timings is None:
        log.info("retrieval.library_search_done", jobId=job_id, tiers=0, candidates=0, exact=0, records=0, ids=[])
        return []
    candidates = merge(ranked.tiers)
    records = frozenset(item for ranking in ranked.tiers for item in ranking.records)
    ordered: list[str] = []
    exact = 0
    if candidates:
        passages = [(item, tier) for item, tier in candidates if item not in records]
        read = read_texts(reader, job, passages) if passages else []
        timings.lap("readSeconds")
        ordered = exact_first(read, checked.question, [item for item, _ in candidates])
        wanted = flat(checked.question)
        exact = sum(1 for item in read if wanted and wanted in flat(item.text))
        timings.lap("rerankSeconds")
    ids = ordered[:LIMIT]
    log.info(
        "retrieval.library_search_done",
        jobId=job_id,
        tiers=ranked.searched,
        candidates=len(candidates),
        exact=exact,
        records=sum(1 for item in ids if item in records),
        ids=ids,
        pendingFetched=ranked.fetched,
        pendingSkipped=ranked.skipped,
        **timings.spent,
    )
    return ids


@dataclass
class SearchJob:
    """The `library.search` handler: `(job, cancel) -> Outcome`, as the lanes call it."""

    reader: search.PassageReader
    embedder: Embedder
    stores: StoreFactory

    def __call__(self, job: ClaimedJob, cancel: threading.Event, /) -> Outcome:
        log = TraceLog(UNIT, job.trace_id)
        job_id = str(job.job_id)
        try:
            checked = search.question_input(job.input, field="query")
            log.info("retrieval.library_search_started", jobId=job_id, attempt=job.attempt)
            ids = find_ids(job, checked, embedder=self.embedder, stores=self.stores, reader=self.reader, log=log)
        except search.InputInvalid:
            log.warn("retrieval.library_search_failed", jobId=job_id, code=search.INPUT_INVALID, retry=False)
            return Outcome.failed(search.INPUT_INVALID, retry=False)
        except search.SearchFailed as failed:
            error_type = type(failed.__cause__).__name__ if failed.__cause__ is not None else type(failed).__name__
            log.warn(
                "retrieval.library_search_failed",
                jobId=job_id,
                code=failed.code,
                retry=failed.retry,
                errorType=error_type,
            )
            return Outcome.failed(failed.code, retry=None if failed.retry else False)
        return Outcome.done({"ids": ids})
