"""Story 2.3: finding the passages that answer a question. Every row of the story's I/O matrix, asserted on the ids
handed over, with Qdrant's local mode (`QdrantClient(":memory:")`), the real embeddinggemma vectors recorded in
`fixtures/retrieval` (an unrecorded input fails the test) and the backend's stand-in serving passages by id. A spy on
the Qdrant client proves the owner and source filters are on every query. No model is called."""

from __future__ import annotations

import base64
import itertools
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import pytest
from qdrant_client import QdrantClient, models

from gmshield_worker.backend.client import ClaimedJob, JobsApi, Outcome, PassagesAnswer, ReadPassage
from gmshield_worker.lanes import question
from gmshield_worker.log import TraceLog, new_trace_id
from gmshield_worker.retrieval import embedding, keywords, search, store
from gmshield_worker.settings import IndexSettings
from tests import retrieval_fixtures
from tests.conftest import StandIn, StandInPassage, log_lines

CREDENTIAL = "w" * 64
KEY = bytes(range(32))
OWNER = "0195a3c0-0000-7000-8000-0000000000aa"
OTHER_OWNER = "0195a3c0-0000-7000-8000-0000000000ab"
SRD = "0195a3c0-0000-7000-8000-0000000000e1"  # the English SRD passages' source
PT = "0195a3c0-0000-7000-8000-0000000000e2"  # the Portuguese passages' source
OUTSIDE = "0195a3c0-0000-7000-8000-0000000000e3"  # the owner's source that no tier holds
NEW = "0195a3c0-0000-7000-8000-0000000000e4"  # a just-accepted version, none of its points indexed

PASSAGES = retrieval_fixtures.passages()
BY_KEY = {item.key: item for item in PASSAGES}
QUESTIONS = {item.key: item for item in retrieval_fixtures.questions()}


class Spy:
    """The local client, recording every `query_points` call's arguments."""

    def __init__(self, client: QdrantClient) -> None:
        self.client = client
        self.queries: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def query_points(self, collection: str, **arguments: Any) -> Any:
        self.queries.append(arguments)
        return self.client.query_points(collection, **arguments)


@dataclass
class Library:
    """Every fixture passage indexed for the owner (SRD and Portuguese sources), plus copies that are out of every
    scope: the same vectors under another owner, and under a source of the owner that no tier holds."""

    spy: Spy
    outside_ids: set[str] = field(default_factory=set)


def library() -> Library:
    client = QdrantClient(":memory:")
    passages = store.PassageStore(client)
    passages.ensure_collection()
    vectors = retrieval_fixtures.RecordedOllama().embedder().embed([(item.title, item.text) for item in PASSAGES])
    points: list[store.Point] = []
    outside: set[str] = set()
    for item, vector in zip(PASSAGES, vectors, strict=True):
        sparse = keywords.keyword_vector(item.text, KEY)
        source = SRD if item.srd else PT
        points.append(store.Point(item.passage_id, vector.vector, sparse, OWNER, source, "rulebook", item.page, None))
        for owner, copy_source in ((OTHER_OWNER, source), (OWNER, OUTSIDE)):
            copy_id = str(uuid.uuid4())
            outside.add(copy_id)
            points.append(store.Point(copy_id, vector.vector, sparse, owner, copy_source, "rulebook", item.page, None))
    passages.upsert(points)
    return Library(Spy(client), outside)


@pytest.fixture(scope="module")
def indexed() -> Library:
    return library()


@contextmanager
def local(spy: Spy) -> Iterator[store.PassageStore]:
    yield store.PassageStore(spy)  # type: ignore[arg-type]


def serve_fixtures(stand_in: StandIn) -> None:
    """The backend holds every fixture passage, and the out-of-scope copies are of sources no tier holds."""
    stand_in.credential = CREDENTIAL
    for item in PASSAGES:
        stand_in.passages[item.passage_id] = StandInPassage(
            SRD if item.srd else PT, item.text, item.section_path, item.page
        )


def tier(name: str, sources: list[str], pending: list[str] | None = None) -> dict[str, Any]:
    return {"tier": name, "sources": sources, "pending": pending or []}


def question_input(text: str, tiers: list[dict[str, Any]]) -> dict[str, Any]:
    return {"question": text, "owner": OWNER, "tiers": tiers, "tokenKey": base64.b64encode(KEY).decode()}


def claim(stand_in: StandIn, job_input: dict[str, Any]) -> ClaimedJob:
    job_id = stand_in.offer(question.KIND, job_input, new_trace_id())
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    return answer.job


@dataclass
class Run:
    outcome: Outcome
    job: ClaimedJob

    @property
    def ids(self) -> list[str]:
        assert self.outcome.outcome == "done", self.outcome
        assert self.outcome.result is not None
        return list(self.outcome.result["ids"])


def ask(
    stand_in: StandIn,
    lib: Library,
    text: str,
    tiers: list[dict[str, Any]],
    ollama: retrieval_fixtures.RecordedOllama | None = None,
) -> Run:
    """The `test.question` handler over the claim of a job with this question input."""
    job = claim(stand_in, question_input(text, tiers))
    handler = question.QuestionProbe(
        JobsApi(stand_in.url, CREDENTIAL),
        (ollama or retrieval_fixtures.RecordedOllama()).embedder(),
        lambda: local(lib.spy),
    )
    return Run(handler(job, threading.Event()), job)


def find(stand_in: StandIn, lib: Library, text: str, tiers: list[dict[str, Any]]) -> list[search.Handed]:
    """`find_passages` itself, for what the result's ids do not show (tiers, texts)."""
    job = claim(stand_in, question_input(text, tiers))
    return search.find_passages(
        job,
        search.question_input(job.input),
        embedder=retrieval_fixtures.RecordedOllama().embedder(),
        stores=lambda: local(lib.spy),
        reader=JobsApi(stand_in.url, CREDENTIAL),
        log=TraceLog("test", job.trace_id),
    )


def passages_requests(stand_in: StandIn) -> list[list[str]]:
    return [json.loads(item["body"])["ids"] for item in stand_in.requests if item["path"].endswith("/passages")]


def assert_filtered(query: dict[str, Any], sources: list[str]) -> None:
    """The owner and the tier's sources filter the query and each of its prefetches."""
    expected = models.Filter(
        must=[
            models.FieldCondition(key="owner", match=models.MatchValue(value=OWNER)),
            models.FieldCondition(key="source", match=models.MatchAny(any=sources)),
        ]
    )
    assert query["query_filter"] == expected
    assert query["prefetch"]
    assert all(prefetch.filter == expected for prefetch in query["prefetch"])
    assert [prefetch.using for prefetch in query["prefetch"]] == ["dense", "sparse"]
    assert all(prefetch.limit == 40 for prefetch in query["prefetch"])
    assert query["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    assert query["limit"] == 8


# --- the matrix


@pytest.mark.parametrize("key", [key for key, item in QUESTIONS.items() if item.case == "pt-en"])
def test_a_portuguese_question_finds_its_english_passage(stand_in: StandIn, indexed: Library, key: str) -> None:
    serve_fixtures(stand_in)
    asked = QUESTIONS[key]

    ids = ask(stand_in, indexed, asked.question, [tier("system", [SRD, PT])]).ids

    expected = BY_KEY[asked.expected].passage_id
    assert expected in ids
    assert ids.index(expected) < 3
    assert BY_KEY[asked.expected].language == "en"


@pytest.mark.parametrize("key", [key for key, item in QUESTIONS.items() if item.case == "en-pt"])
def test_an_english_question_finds_its_portuguese_passage(stand_in: StandIn, indexed: Library, key: str) -> None:
    serve_fixtures(stand_in)
    asked = QUESTIONS[key]

    ids = ask(stand_in, indexed, asked.question, [tier("system", [SRD, PT])]).ids

    expected = BY_KEY[asked.expected].passage_id
    assert expected in ids
    assert ids.index(expected) < 3
    assert BY_KEY[asked.expected].language == "pt-BR"


def test_an_exact_name_ranks_its_passage_in_the_top_three_among_passages_about_swords(
    stand_in: StandIn, indexed: Library
) -> None:
    serve_fixtures(stand_in)
    swords = [item for item in PASSAGES if "sword" in item.text.lower() or "Sword" in item.title]
    assert len(swords) >= 5

    ids = ask(stand_in, indexed, QUESTIONS["q-vorpal"].question, [tier("system", [SRD])]).ids

    assert ids.index(BY_KEY["srd-vorpal-sword"].passage_id) < 3


def test_another_owners_passages_and_a_source_out_of_the_tiers_are_never_handed_over(
    stand_in: StandIn, indexed: Library
) -> None:
    serve_fixtures(stand_in)
    before = len(indexed.spy.queries)

    for asked in QUESTIONS.values():
        ids = ask(stand_in, indexed, asked.question, [tier("campaign", [PT]), tier("system", [SRD])]).ids
        assert ids
        assert indexed.outside_ids.isdisjoint(ids)
        assert all(item in stand_in.passages for item in ids)

    # Every query filtered on the owner and on its tier's sources, two per question.
    queries = indexed.spy.queries[before:]
    assert len(queries) == 2 * len(QUESTIONS)
    for index, query in enumerate(queries):
        assert_filtered(query, [PT] if index % 2 == 0 else [SRD])


def test_with_matches_in_both_tiers_the_campaign_tiers_ids_come_first_each_labelled(
    stand_in: StandIn, indexed: Library
) -> None:
    serve_fixtures(stand_in)

    handed = find(
        stand_in, indexed, QUESTIONS["q-pt-exhaustion"].question, [tier("campaign", [PT]), tier("system", [SRD])]
    )

    tiers = [item.tier for item in handed]
    assert "campaign" in tiers
    assert "system" in tiers
    assert tiers == sorted(tiers)  # "campaign" < "system": every campaign id first
    by_id = {item.passage_id: item for item in PASSAGES}
    for item in handed:
        assert by_id[item.passage_id].srd is (item.tier == "system")
        assert item.text == by_id[item.passage_id].text
        assert item.section_path == by_id[item.passage_id].section_path
        assert item.page == by_id[item.passage_id].page
    assert BY_KEY["pt-coroa"].passage_id in [item.passage_id for item in handed if item.tier == "campaign"]
    assert BY_KEY["srd-exhaustion"].passage_id in [item.passage_id for item in handed if item.tier == "system"]
    # One call fetched the text of every found id, in rank order.
    assert passages_requests(stand_in) == [[item.passage_id for item in handed]]
    assert "Exhaustion" not in repr(handed)


def test_a_pending_passage_past_the_first_100_that_names_a_word_of_the_question_is_handed_over(
    stand_in: StandIn, indexed: Library, capsys: pytest.CaptureFixture[str]
) -> None:
    serve_fixtures(stand_in)
    pending = [str(uuid.uuid4()) for _ in range(150)]
    for index, passage_id in enumerate(pending):
        stand_in.passages[passage_id] = StandInPassage(NEW, f"Página de enchimento número {index}.", ["Notas"], 1)
    taverna = BY_KEY["pt-taverna"]
    stand_in.passages[pending[10]] = StandInPassage(NEW, taverna.text, taverna.section_path, taverna.page)
    # Matrix row "Big pending tier": a better match after the first 100, in the second page.
    stand_in.passages[pending[120]] = StandInPassage(NEW, "Osvaldo Brandão é quem é Osvaldo Brandão.", ["Notas"], 2)
    before = len(indexed.spy.queries)

    run = ask(stand_in, indexed, QUESTIONS["q-pending"].question, [tier("campaign", [SRD, NEW], pending)])

    assert pending[10] in run.ids
    assert pending[120] in run.ids
    fetched = passages_requests(stand_in)
    # Two pages of at most 100 ids (read at once, in either order), then the found ids' text.
    assert sorted(fetched[:2], key=len, reverse=True) == [pending[:100], pending[100:]]
    assert fetched[2] == run.ids
    assert len(fetched) == 3
    # The sources still went to Qdrant, the new one included.
    assert len(indexed.spy.queries) == before + 1
    assert_filtered(indexed.spy.queries[-1], [SRD, NEW])
    lines = log_lines(capsys.readouterr().out)
    assert not [line for line in lines if line["event"] == "retrieval.pending_skipped"]
    done = [line for line in lines if line["event"] == "retrieval.search_done"]
    assert (done[0]["pendingFetched"], done[0]["pendingSkipped"]) == (150, 0)


def test_the_pending_ids_past_the_cap_are_skipped_and_logged_as_a_count(
    stand_in: StandIn, indexed: Library, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_fixtures(stand_in)
    monkeypatch.setattr(search, "MAX_PENDING", 120)
    pending = [str(uuid.uuid4()) for _ in range(150)]
    for index, passage_id in enumerate(pending):
        stand_in.passages[passage_id] = StandInPassage(NEW, f"Página de enchimento número {index}.", ["Notas"], 1)
    # The best match, past the cap: never fetched, never handed over.
    stand_in.passages[pending[130]] = StandInPassage(NEW, "Osvaldo Brandão é quem é Osvaldo Brandão.", ["Notas"], 2)

    run = ask(stand_in, indexed, QUESTIONS["q-pending"].question, [tier("campaign", [SRD, NEW], pending)])

    assert pending[130] not in run.ids
    fetched = passages_requests(stand_in)
    assert sorted(fetched[:2], key=len, reverse=True) == [pending[:100], pending[100:120]]
    lines = log_lines(capsys.readouterr().out)
    skipped = [line for line in lines if line["event"] == "retrieval.pending_skipped"]
    assert [(line["tier"], line["skipped"]) for line in skipped] == [("campaign", 30)]
    done = [line for line in lines if line["event"] == "retrieval.search_done"]
    assert (done[0]["pendingFetched"], done[0]["pendingSkipped"]) == (120, 30)
    assert search.MAX_PENDING == 120
    monkeypatch.undo()
    assert (search.MAX_PENDING, search.PAGE, search.PARALLEL, search.PENDING_BUDGET_SECONDS) == (5000, 100, 4, 1.5)


def test_a_spent_budget_starts_no_page_and_every_unread_id_is_logged_as_skipped(
    stand_in: StandIn, indexed: Library, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_fixtures(stand_in)
    monkeypatch.setattr(search, "PENDING_BUDGET_SECONDS", 0.0)
    pending = [str(uuid.uuid4()) for _ in range(150)]
    for index, passage_id in enumerate(pending):
        stand_in.passages[passage_id] = StandInPassage(NEW, f"Página de enchimento número {index}.", ["Notas"], 1)

    run = ask(stand_in, indexed, QUESTIONS["q-vorpal"].question, [tier("system", [SRD], pending)])

    # The search still answers from Qdrant; no pending page was read.
    assert run.ids
    assert passages_requests(stand_in) == [run.ids]
    lines = log_lines(capsys.readouterr().out)
    skipped = [line for line in lines if line["event"] == "retrieval.pending_skipped"]
    assert [(line["tier"], line["skipped"]) for line in skipped] == [("system", 150)]
    done = [line for line in lines if line["event"] == "retrieval.search_done"]
    assert (done[0]["pendingFetched"], done[0]["pendingSkipped"]) == (0, 150)


def test_both_tiers_pending_pages_are_read_in_turns_and_each_tier_keeps_its_matches(
    stand_in: StandIn, indexed: Library, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_fixtures(stand_in)
    # One page at a time, so the order the stand-in sees is the order the pages started.
    monkeypatch.setattr(search, "PARALLEL", 1)
    campaign = [str(uuid.uuid4()) for _ in range(150)]
    system = [str(uuid.uuid4()) for _ in range(250)]
    for index, passage_id in enumerate(campaign):
        stand_in.passages[passage_id] = StandInPassage(NEW, f"Página de enchimento número {index}.", ["Notas"], 1)
    for index, passage_id in enumerate(system):
        stand_in.passages[passage_id] = StandInPassage(SRD, f"Página de enchimento número {index}.", ["Notas"], 1)
    stand_in.passages[campaign[120]] = StandInPassage(NEW, "Osvaldo Brandão é quem é Osvaldo Brandão.", ["Notas"], 2)
    stand_in.passages[system[220]] = StandInPassage(SRD, "Osvaldo Brandão, de novo Osvaldo Brandão.", ["Notas"], 3)

    handed = find(
        stand_in,
        indexed,
        QUESTIONS["q-pending"].question,
        [tier("campaign", [NEW], campaign), tier("system", [SRD], system)],
    )

    # Matrix row "Both tiers pending": campaign, system, campaign, system, then the rest of the longer one.
    fetched = passages_requests(stand_in)
    assert fetched[:5] == [campaign[:100], system[:100], campaign[100:], system[100:200], system[200:]]
    by_id = {item.passage_id: item.tier for item in handed}
    assert (by_id[campaign[120]], by_id[system[220]]) == ("campaign", "system")
    lines = log_lines(capsys.readouterr().out)
    assert not [line for line in lines if line["event"] == "retrieval.pending_skipped"]
    done = [line for line in lines if line["event"] == "retrieval.search_done"]
    assert (done[0]["pendingFetched"], done[0]["pendingSkipped"]) == (400, 0)


class PageReader:
    """Serves any id as a passage of its tier (`tiers`, else `default`), recording each call's ids in the order they
    started. The first calls, as many as `gate` (a barrier) has parties, wait at it, and a call whose page is in `hold`
    waits for `release` before it answers."""

    def __init__(
        self,
        default: str = "system",
        tiers: dict[str, str] | None = None,
        refuse_page: int | None = None,
        gate: threading.Barrier | None = None,
        hold: tuple[list[str], ...] = (),
    ) -> None:
        self.default = default
        self.tiers = tiers or {}
        self.refuse_page = refuse_page
        self.gate = gate
        self.hold = [list(page) for page in hold]
        self.release = threading.Event()
        self.calls: list[list[str]] = []
        self.running = 0
        self.most = 0
        self.lock = threading.Lock()

    def passages(self, job: ClaimedJob, ids: list[str]) -> PassagesAnswer:
        with self.lock:
            self.calls.append(list(ids))
            page = len(self.calls) - 1
            self.running += 1
            self.most = max(self.most, self.running)
        try:
            if self.gate is not None and page < self.gate.parties:
                self.gate.wait(timeout=5)
            if ids in self.hold:
                self.release.wait(timeout=5)
            if self.refuse_page is not None and page == self.refuse_page:
                return PassagesAnswer(403, code="scope.passage_refused")
            return PassagesAnswer(
                200, passages=[ReadPassage(item, self.tiers.get(item, self.default), 1, item, []) for item in ids]
            )
        finally:
            with self.lock:
                self.running -= 1


def fake_job() -> ClaimedJob:
    return ClaimedJob(
        job_id=uuid.uuid4(),
        claim_id=uuid.uuid4(),
        attempt=1,
        renew_at=datetime.now(UTC),
        lapses_at=datetime.now(UTC),
        kind=question.KIND,
        input={},
        trace_id=new_trace_id(),
        grant="g" * 43,
    )


def stepping_clock(step: float) -> Callable[[], float]:
    """A clock for one page at a time (`parallel=1`): the first reading (the deadline's start) is 0, then each pair of
    readings, a page's start check and the wait after it, reads `step` more. The deadline is only ever passed at a
    start check, while no page runs."""
    readings = itertools.count()

    def now() -> float:
        return ((next(readings) + 1) // 2) * step

    return now


def test_pending_passages_are_read_in_pages_of_100_exactly_4_at_a_time_in_input_order() -> None:
    ids = [str(uuid.uuid4()) for _ in range(1050)]
    # The first four calls only go on once all four run at once: fewer in parallel breaks the barrier.
    gate = threading.Barrier(4)
    reader = PageReader(gate=gate)

    read = search.read_pending(reader, fake_job(), [("system", ids)], clock=lambda: 0.0)

    assert [item.passage_id for item in read["system"].passages] == ids
    assert read["system"].unread == 0
    assert sorted(reader.calls) == sorted(ids[start : start + 100] for start in range(0, 1050, 100))
    assert reader.most == 4


def test_the_tiers_pages_are_started_in_turns_then_the_rest_of_the_longer_tier() -> None:
    campaign = [str(uuid.uuid4()) for _ in range(250)]
    system = [str(uuid.uuid4()) for _ in range(450)]
    reader = PageReader(tiers=dict.fromkeys(campaign, "campaign"))

    read = search.read_pending(
        reader, fake_job(), [("campaign", campaign), ("system", system)], clock=lambda: 0.0, parallel=1
    )

    assert reader.calls == [
        campaign[:100],
        system[:100],
        campaign[100:200],
        system[100:200],
        campaign[200:],
        system[200:300],
        system[300:400],
        system[400:],
    ]
    assert [item.passage_id for item in read["campaign"].passages] == campaign
    assert [item.passage_id for item in read["system"].passages] == system
    assert (read["campaign"].unread, read["system"].unread) == (0, 0)


def test_one_deadline_for_both_tiers_and_no_page_starts_after_it() -> None:
    campaign = [str(uuid.uuid4()) for _ in range(1000)]
    system = [str(uuid.uuid4()) for _ in range(3142)]
    reader = PageReader(tiers=dict.fromkeys(campaign, "campaign"))

    # Pages start at 0.5, 1.0 and 1.5 s past the deadline's start: with 2 s, three start, the fourth check is at 2 s.
    read = search.read_pending(
        reader,
        fake_job(),
        [("campaign", campaign), ("system", system)],
        clock=stepping_clock(0.5),
        budget=2.0,
        parallel=1,
    )

    # Matrix row "Both tiers pending": three pages in all, not three per tier.
    assert reader.calls == [campaign[:100], system[:100], campaign[100:200]]
    assert [item.passage_id for item in read["campaign"].passages] == campaign[:200]
    assert [item.passage_id for item in read["system"].passages] == system[:100]
    assert (read["campaign"].unread, read["system"].unread) == (800, 3042)


def test_a_page_not_back_by_the_deadline_is_dropped_and_not_waited_for() -> None:
    campaign = [str(uuid.uuid4()) for _ in range(200)]
    system = [str(uuid.uuid4()) for _ in range(300)]
    # The system tier's second page answers only once released, long after the deadline.
    reader = PageReader(tiers=dict.fromkeys(campaign, "campaign"), hold=(system[100:200],))
    started = time.monotonic()
    try:
        read = search.read_pending(reader, fake_job(), [("campaign", campaign), ("system", system)], budget=0.5)
        spent = time.monotonic() - started
    finally:
        reader.release.set()

    assert spent < 3.0  # the held page waits up to 5 s: it was not waited for
    assert system[100:200] in reader.calls
    assert [item.passage_id for item in read["campaign"].passages] == campaign
    assert [item.passage_id for item in read["system"].passages] == system[:100] + system[200:]
    assert (read["campaign"].unread, read["system"].unread) == (0, 100)


def test_a_page_that_fails_after_the_deadline_is_dropped_and_its_ids_are_skipped() -> None:
    ids = [str(uuid.uuid4()) for _ in range(200)]
    # One page at a time: the second page is refused, and the clock passes the deadline only once it has answered.
    reader = PageReader(refuse_page=1)
    readings = itertools.count()

    def clock() -> float:
        if next(readings) < 4:  # the deadline's start, then the first page's start check and wait, the second's start
            return 0.0
        time.sleep(0.2)  # the refusal is back well before this reading
        return 5.0

    read = search.read_pending(reader, fake_job(), [("system", ids)], clock=clock, budget=1.5, parallel=1)

    assert reader.calls == [ids[:100], ids[100:]]
    assert [item.passage_id for item in read["system"].passages] == ids[:100]
    assert read["system"].unread == 100


def test_a_refused_page_fails_the_whole_read_with_the_backends_code() -> None:
    ids = [str(uuid.uuid4()) for _ in range(500)]

    with pytest.raises(search.SearchFailed) as failed:
        search.read_pending(PageReader(refuse_page=2), fake_job(), [("system", ids)], clock=lambda: 0.0)

    assert (failed.value.code, failed.value.retry) == ("scope.passage_refused", False)
    with pytest.raises(search.SearchFailed) as other_tier:
        search.read_pending(PageReader(default="campaign"), fake_job(), [("system", ids)], clock=lambda: 0.0)
    assert (other_tier.value.code, other_tier.value.retry) == ("retrieval.passages_unavailable", True)


def test_an_id_found_both_in_qdrant_and_pending_ranks_above_where_either_list_alone_puts_it(
    stand_in: StandIn, indexed: Library
) -> None:
    serve_fixtures(stand_in)
    asked = QUESTIONS["q-pt-exhaustion"].question
    exhaustion, vorpal = BY_KEY["srd-exhaustion"].passage_id, BY_KEY["srd-vorpal-sword"].passage_id
    alone = ask(stand_in, indexed, asked, [tier("system", [SRD])]).ids
    # Both share one keyword token with the question: the pending ranking keeps their input order.
    tokens = set(keywords.keyword_vector(asked, KEY).indices)
    read = [stand_in.passages[passage_id] for passage_id in (vorpal, exhaustion)]
    for item in read:
        assert len(tokens & set(keywords.keyword_vector(item.text, KEY).indices)) == 1

    # The source is being indexed again: its passages are pending and already in Qdrant.
    fused = ask(stand_in, indexed, asked, [tier("system", [SRD], [vorpal, exhaustion])]).ids

    assert alone.index(exhaustion) == 1  # second in Qdrant alone, second in the pending ranking
    assert fused[0] == exhaustion
    assert fused.count(exhaustion) == 1


def test_a_question_with_no_keyword_token_reads_no_pending_passage(stand_in: StandIn, indexed: Library) -> None:
    serve_fixtures(stand_in)
    recorded = retrieval_fixtures.recorded()
    # "—?" has no word token; it borrows a recorded question's vector, so only its keyword vector is empty.
    vectors = {**recorded, embedding.query_input("—?"): recorded[embedding.query_input(QUESTIONS["q-vorpal"].question)]}
    pending = [BY_KEY["srd-vorpal-sword"].passage_id, BY_KEY["srd-exhaustion"].passage_id]

    run = ask(stand_in, indexed, "—?", [tier("system", [SRD], pending)], retrieval_fixtures.RecordedOllama(vectors))

    assert run.ids
    assert passages_requests(stand_in) == [run.ids]  # only the found ids' text, no pending read
    assert [prefetch.using for prefetch in indexed.spy.queries[-1]["prefetch"]] == ["dense"]


def test_an_empty_scope_answers_an_empty_list_with_no_query_and_no_passages_call(
    stand_in: StandIn, indexed: Library
) -> None:
    serve_fixtures(stand_in)
    ollama = retrieval_fixtures.RecordedOllama()
    before = len(indexed.spy.queries)

    run = ask(stand_in, indexed, "Qualquer coisa?", [tier("campaign", []), tier("system", [])], ollama)

    assert run.outcome == Outcome.done({"ids": []})
    assert len(indexed.spy.queries) == before
    assert passages_requests(stand_in) == []
    assert ollama.calls == []  # not even the question is embedded


def test_a_tier_with_no_source_issues_no_query(stand_in: StandIn, indexed: Library) -> None:
    serve_fixtures(stand_in)
    before = len(indexed.spy.queries)

    run = ask(stand_in, indexed, QUESTIONS["q-vorpal"].question, [tier("campaign", []), tier("system", [SRD])])

    assert run.ids
    assert len(indexed.spy.queries) == before + 1
    assert_filtered(indexed.spy.queries[-1], [SRD])


def test_a_refusal_of_the_backend_fails_the_job_with_its_code_and_no_retry(stand_in: StandIn, indexed: Library) -> None:
    serve_fixtures(stand_in)
    stand_in.passages_status = 403

    run = ask(stand_in, indexed, QUESTIONS["q-vorpal"].question, [tier("system", [SRD])])

    assert run.outcome == Outcome.failed("scope.passage_refused", retry=False)


def test_an_id_the_backend_does_not_hold_in_the_scope_is_refused(stand_in: StandIn, indexed: Library) -> None:
    serve_fixtures(stand_in)
    del stand_in.passages[BY_KEY["srd-vorpal-sword"].passage_id]

    run = ask(stand_in, indexed, QUESTIONS["q-vorpal"].question, [tier("system", [SRD])])

    assert run.outcome == Outcome.failed("scope.passage_refused", retry=False)


def test_a_store_down_or_ollama_down_is_retryable_and_reads_no_passage(stand_in: StandIn, indexed: Library) -> None:
    serve_fixtures(stand_in)
    asked = QUESTIONS["q-vorpal"].question
    down = IndexSettings(qdrant_url="http://127.0.0.1:9")
    job = claim(stand_in, question_input(asked, [tier("system", [SRD])]))
    store_down = question.QuestionProbe(
        JobsApi(stand_in.url, CREDENTIAL),
        retrieval_fixtures.RecordedOllama().embedder(),
        lambda: store.passage_store(down),
    )(job, threading.Event())

    ollama_down = ask(
        stand_in, indexed, asked, [tier("system", [SRD])], retrieval_fixtures.RecordedOllama(status=503)
    ).outcome

    assert store_down == Outcome.failed("retrieval.store_unavailable")
    assert ollama_down == Outcome.failed("retrieval.embedding_unavailable")
    assert passages_requests(stand_in) == []


def test_a_passages_route_down_is_retryable(stand_in: StandIn, indexed: Library) -> None:
    serve_fixtures(stand_in)
    stand_in.passages_status, stand_in.passages_code = 503, "server.unavailable"

    run = ask(stand_in, indexed, QUESTIONS["q-vorpal"].question, [tier("system", [SRD])])

    assert run.outcome == Outcome.failed("retrieval.passages_unavailable")


@dataclass
class OddReader:
    """A passages route that fails in transit, or answers what the contract does not."""

    answer: str

    def passages(self, job: ClaimedJob, ids: list[str]) -> PassagesAnswer:
        if self.answer == "unreachable":
            raise httpx.ConnectError("refused")
        if self.answer == "missing":
            return PassagesAnswer(
                200, passages=[ReadPassage(passage_id, "system", 1, "t", []) for passage_id in ids[1:]]
            )
        return PassagesAnswer(200, passages=[ReadPassage(passage_id, "campaign", 1, "t", []) for passage_id in ids])


@pytest.mark.parametrize("answer", ["unreachable", "missing", "another tier"])
def test_a_passages_answer_out_of_the_contract_is_retryable(stand_in: StandIn, indexed: Library, answer: str) -> None:
    serve_fixtures(stand_in)
    job = claim(stand_in, question_input(QUESTIONS["q-vorpal"].question, [tier("system", [SRD])]))

    outcome = question.QuestionProbe(
        OddReader(answer), retrieval_fixtures.RecordedOllama().embedder(), lambda: local(indexed.spy)
    )(job, threading.Event())

    assert outcome == Outcome.failed("retrieval.passages_unavailable")


def test_a_question_the_model_refuses_as_too_long_fails_without_a_retry(stand_in: StandIn, indexed: Library) -> None:
    serve_fixtures(stand_in)

    class TooLong(retrieval_fixtures.RecordedOllama):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "the input length exceeds the context length"})

    run = ask(stand_in, indexed, "x" * 4000, [tier("system", [SRD])], TooLong())

    assert run.outcome == Outcome.failed("retrieval.question_too_long", retry=False)


def test_an_unrecorded_question_fails_the_test(stand_in: StandIn, indexed: Library) -> None:
    serve_fixtures(stand_in)

    with pytest.raises(AssertionError, match="unrecorded"):
        ask(stand_in, indexed, "A question nobody recorded?", [tier("system", [SRD])])


@pytest.mark.parametrize(
    "change",
    [
        {"question": ""},
        {"question": "x" * 4001},
        {"question": None},
        {"owner": "not-a-uuid"},
        {"tokenKey": base64.b64encode(b"short").decode()},
        {"tokenKey": "not base64!"},
        {"tiers": [tier("system", [SRD]), tier("campaign", [PT])]},
        {"tiers": [tier("system", [SRD]), tier("system", [PT])]},
        {"tiers": [tier("other", [SRD])]},
        {"tiers": [tier("system", ["not-a-uuid"])]},
        {"tiers": [tier("system", [SRD, SRD])]},
        {"tiers": [{"tier": "system", "sources": [SRD]}]},
        {"tiers": None},
    ],
)
def test_an_input_that_is_not_the_backends_fails_without_a_retry(
    stand_in: StandIn, indexed: Library, change: dict[str, Any]
) -> None:
    serve_fixtures(stand_in)
    job = claim(stand_in, {**question_input("Vorpal Sword", [tier("system", [SRD])]), **change})

    outcome = question.QuestionProbe(
        JobsApi(stand_in.url, CREDENTIAL), retrieval_fixtures.RecordedOllama().embedder(), lambda: local(indexed.spy)
    )(job, threading.Event())

    assert outcome == Outcome.failed("retrieval.input_invalid", retry=False)


def test_the_handed_over_ids_are_logged_never_the_question_a_text_or_the_key(
    stand_in: StandIn, indexed: Library, capsys: pytest.CaptureFixture[str]
) -> None:
    serve_fixtures(stand_in)
    asked = QUESTIONS["q-en-tavern"].question

    run = ask(stand_in, indexed, asked, [tier("campaign", [PT]), tier("system", [SRD])])

    captured = capsys.readouterr()
    logged = captured.out + captured.err
    done = [line for line in log_lines(captured.out) if line["event"] == "retrieval.search_done"]
    assert done[0]["ids"] == run.ids
    assert done[0]["tiers"] == 2
    assert {"embedSeconds", "searchSeconds", "readSeconds"} <= set(done[0])
    for secret in (asked, "smugglers", "Osvaldo", "contrabandistas", base64.b64encode(KEY).decode(), KEY.hex()):
        assert secret not in logged
    checked = search.question_input(run.job.input)
    assert asked not in repr(checked)
    assert base64.b64encode(KEY).decode() not in repr(checked)


# --- the parts


def test_the_question_is_embedded_under_the_query_prompt_with_no_truncation() -> None:
    ollama = retrieval_fixtures.RecordedOllama()
    asked = QUESTIONS["q-vorpal"].question

    vector = ollama.embedder().embed_query(asked)

    assert ollama.calls == [[f"task: search result | query: {asked}"]]
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-9


def test_the_pending_ranking_counts_shared_keyword_tokens_and_fusion_sums_an_ids_scores_in_one_entry() -> None:
    def read(passage_id: str, text: str) -> ReadPassage:
        return ReadPassage(passage_id, "campaign", 1, text, [])

    tokens = set(keywords.keyword_vector("espada vorpal de prata", KEY).indices)
    ranked = search.rank_pending(
        [read("a", "uma espada"), read("b", "nada aqui"), read("c", "espada vorpal"), read("d", "a prata")], tokens, KEY
    )
    assert ranked == ["c", "a", "d"]

    found = [store.Found("x", 1.0), store.Found("y", 0.6), store.Found("c", 0.4)]
    fused = search.fuse(found, ranked, limit=4)
    # "c" is one entry whose scores add up (0.4 + 1/2 as the first pending): above "y", where neither list alone puts
    # it; "a" (1/3) follows; "d" (1/4) is cut.
    assert fused == ["x", "c", "y", "a"]
    assert search.fuse([], [], limit=8) == []


def test_a_search_without_a_source_asks_nothing_and_one_without_keywords_has_no_sparse_prefetch(
    indexed: Library,
) -> None:
    passages = store.PassageStore(indexed.spy)  # type: ignore[arg-type]
    dense = retrieval_fixtures.RecordedOllama().embedder().embed_query(QUESTIONS["q-vorpal"].question)
    before = len(indexed.spy.queries)

    assert passages.search(OWNER, [], dense, keywords.Sparse([], []), 8) == []
    assert len(indexed.spy.queries) == before

    found = passages.search(OWNER, [SRD], dense, keywords.Sparse([], []), 8)

    assert BY_KEY["srd-vorpal-sword"].passage_id in [item.passage_id for item in found]
    assert [prefetch.using for prefetch in indexed.spy.queries[-1]["prefetch"]] == ["dense"]
    assert indexed.spy.queries[-1]["query_filter"].must[0].match == models.MatchValue(value=OWNER)
