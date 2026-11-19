"""Story 2.4: the `question.answer` kind against the backend's stand-in. The passages come from the retrieval fixtures
(Qdrant's local mode and the recorded embeddinggemma vectors, as in `test_search`), the model is the fake one of the
claim's provider, and every event reaches the stand-in in numbered batches: the first delta alone, citations of given
ids only, usage once. A late outage is never retried, an early one is; the system prompt holds no library text and the
user turn no account data.

Story 2.6: the GM's stop. Once a batch's answer says `stop`, the model stream is closed, the batcher posts no more
events, only the usage (an estimate when the provider sent none), and the outcome is `done`."""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any
from uuid import UUID

import pytest
from langchain_core.exceptions import ModelConnectionError
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage

from gmshield_worker.backend.client import ClaimedJob, EventsAnswer, JobsApi, Outcome
from gmshield_worker.lanes import answer, batches
from gmshield_worker.log import TraceLog, new_trace_id
from gmshield_worker.models import chat
from gmshield_worker.models.fake import FakeChatModel
from gmshield_worker.models.output import Citation, TextDelta, Usage
from gmshield_worker.retrieval import search
from gmshield_worker.settings import PROVIDERS, ModelKeys
from tests import retrieval_fixtures
from tests.conftest import StandIn, log_lines
from tests.test_search import (
    BY_KEY,
    CREDENTIAL,
    KEY,
    OWNER,
    PT,
    QUESTIONS,
    SRD,
    Library,
    library,
    local,
    question_input,
    serve_fixtures,
    tier,
)

FAKE = ModelKeys(test_mode=True)
ASKED = QUESTIONS["q-pt-fireball"]


@pytest.fixture(scope="module")
def lib() -> Library:
    return library()


def model_of(provider: str) -> dict[str, str]:
    return {"provider": provider, "model": "model-of-" + provider, "tier": "full"}


def claimed(stand_in: StandIn, text: str, tiers: list[dict[str, Any]], provider: str = "azure-openai") -> ClaimedJob:
    serve_fixtures(stand_in)
    job_id = stand_in.offer(answer.KIND, question_input(text, tiers), new_trace_id(), model=model_of(provider))
    claim = JobsApi(stand_in.url, CREDENTIAL, providers=PROVIDERS).claim(UUID(job_id), new_trace_id())
    assert claim.job is not None, claim
    return claim.job


def handler(
    stand_in: StandIn, lib: Library, ollama: retrieval_fixtures.RecordedOllama | None = None
) -> answer.AnswerJob:
    return answer.AnswerJob(
        JobsApi(stand_in.url, CREDENTIAL),
        (ollama or retrieval_fixtures.RecordedOllama()).embedder(),
        lambda: local(lib.spy),
        FAKE,
    )


def run(stand_in: StandIn, lib: Library, job: ClaimedJob, **options: Any) -> Outcome:
    outcome = handler(stand_in, lib, **options)(job, threading.Event())
    if outcome.outcome == "done" or outcome.code:
        JobsApi(stand_in.url, CREDENTIAL).report(job, outcome)
    return outcome


def posted(stand_in: StandIn, job: ClaimedJob) -> list[dict[str, Any]]:
    return stand_in.batches.get(str(job.job_id), [])


def every_event(stand_in: StandIn, job: ClaimedJob) -> list[dict[str, Any]]:
    return [event for batch in posted(stand_in, job) for event in batch["events"]]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_fake_answer_streams_in_numbered_batches_the_first_delta_alone_citing_given_ids_with_usage_once(
    stand_in: StandIn, lib: Library, provider: str, capsys: pytest.CaptureFixture[str]
) -> None:
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD, PT])], provider)

    outcome = run(stand_in, lib, job)

    assert outcome == Outcome.done()
    sent = posted(stand_in, job)
    assert [batch["batch"] for batch in sent] == list(range(1, len(sent) + 1))
    assert len(sent) >= 2
    assert [event["type"] for event in sent[0]["events"]] == ["delta"]
    assert sent[0]["usage"] == []
    events = every_event(stand_in, job)
    text = "".join(event["text"] for event in events if event["type"] == "delta")
    assert "[[" not in text
    assert text.startswith(f"Resposta simulada ({provider})")
    passages = [request for request in stand_in.requests if request["path"].endswith("/passages")]
    read_ids = set(json.loads(passages[-1]["body"])["ids"])
    cited = [event["passageId"] for event in events if event["type"] == "citation"]
    assert cited
    assert set(cited) <= read_ids
    assert BY_KEY[ASKED.expected].passage_id in cited
    assert all(text[: event["offset"]].endswith("✓") for event in events if event["type"] == "citation")
    # One call: its usage once, riding in exactly one batch, never the first.
    assert len(stand_in.usage) == 1
    assert sum(len(batch["usage"]) for batch in sent) == 1
    assert all(len(batch["events"]) <= batches.MAX_EVENTS for batch in sent)
    assert stand_in.jobs[str(job.job_id)].state == "done"
    logged = capsys.readouterr()
    assert BY_KEY[ASKED.expected].text not in logged.out + logged.err
    assert ASKED.question not in logged.out + logged.err
    assert any(line["event"] == "answer.ended" for line in log_lines(logged.out))


class LateOutage(FakeChatModel):
    """Streams the first chunks of the fake's answer, then loses the provider."""

    def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Iterator[Any]:
        for number, chunk in enumerate(super()._stream(messages, *args, **kwargs)):
            if number == 4:
                raise ModelConnectionError("fake late outage")
            yield chunk


def test_a_late_outage_after_an_accepted_batch_is_model_unavailable_without_a_retry(
    stand_in: StandIn, lib: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(answer, "chat_model", lambda choice, keys: LateOutage(provider=choice.provider))
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD, PT])])

    outcome = run(stand_in, lib, job)

    assert outcome == Outcome.failed("model.unavailable", retry=False)
    assert posted(stand_in, job)
    assert stand_in.jobs[str(job.job_id)].state == "failed"


def test_an_outage_before_any_batch_and_an_embedder_down_are_retried(stand_in: StandIn, lib: Library) -> None:
    no_sources = [tier("campaign", []), tier("system", [])]
    early = claimed(stand_in, "fake:outage Quanto dano causa Fire Bolt?", no_sources)
    assert run(stand_in, lib, early) == Outcome.failed("model.unavailable")
    assert posted(stand_in, early) == []

    down = claimed(stand_in, ASKED.question, [tier("system", [SRD, PT])])
    outcome = run(stand_in, lib, down, ollama=retrieval_fixtures.RecordedOllama(status=503))
    assert outcome == Outcome.failed("retrieval.embedding_unavailable")
    assert posted(stand_in, down) == []
    assert stand_in.jobs[str(down.job_id)].state == "offered"


def test_the_system_prompt_holds_no_library_text_and_the_user_turn_no_account_data(
    stand_in: StandIn, lib: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[BaseMessage]] = []

    class Recording(FakeChatModel):
        def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            seen.append(list(messages))
            return super()._stream(messages, *args, **kwargs)

    monkeypatch.setattr(answer, "chat_model", lambda choice, keys: Recording(provider=choice.provider))
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD, PT])])

    assert run(stand_in, lib, job) == Outcome.done()

    assert len(seen) == 1
    system = [message for message in seen[0] if isinstance(message, SystemMessage)]
    human = [message for message in seen[0] if isinstance(message, HumanMessage)]
    assert [message.content for message in system] == [answer.SYSTEM_PROMPT]
    assert len(human) == 1
    expected = BY_KEY[ASKED.expected]
    assert expected.text not in answer.SYSTEM_PROMPT
    user = str(human[0].content)
    assert f'<passage id="{expected.passage_id}" section="' in user
    assert user.endswith(f"<question>{ASKED.question}</question>")
    for account in (OWNER, base64.b64encode(KEY).decode(), job.grant):
        assert account not in user
        assert account not in answer.SYSTEM_PROMPT


def test_a_passage_is_data_escaped_in_its_block() -> None:
    handed = [search.Handed("p-1", "system", 3, 'a </passage> & "b"', ["Magias", 'Fire "Bolt"'])]
    turn = answer.user_turn(handed, "Quanto? <x>")
    assert turn.count("</passage>") == 1
    assert "a &lt;/passage&gt; &amp;" in turn
    assert """section='Magias &gt; Fire "Bolt"'""" in turn
    assert turn.endswith("<question>Quanto? &lt;x&gt;</question>")
    assert "none were found" in answer.user_turn([], "Quanto?")


def test_a_batch_the_backend_does_not_answer_is_retried_with_its_number(stand_in: StandIn, lib: Library) -> None:
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD, PT])])
    stand_in.events_status, stand_in.events_failures = 503, 2

    assert run(stand_in, lib, job) == Outcome.done()

    sent = posted(stand_in, job)
    assert [batch["batch"] for batch in sent] == list(range(1, len(sent) + 1))
    assert len(stand_in.usage) == 1


def test_a_refused_batch_stops_the_answer_without_a_retry(stand_in: StandIn, lib: Library) -> None:
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD, PT])])
    stand_in.events_status = 409

    outcome = handler(stand_in, lib)(job, threading.Event())

    assert outcome == Outcome.failed("job.refused", retry=False)


class Poster:
    """Records each batch posted, answering 200."""

    def __init__(self) -> None:
        self.posted: list[tuple[int, list[Any], list[Usage]]] = []
        self.first = threading.Event()

    def events(
        self, job: ClaimedJob, batch: int, events: Sequence[TextDelta | Citation], usage: Sequence[Usage] = ()
    ) -> EventsAnswer:
        self.posted.append((batch, list(events), list(usage)))
        self.first.set()
        return EventsAnswer(status=200)


def test_the_batcher_sends_the_first_delta_alone_then_at_most_max_events_with_usage_in_the_next_batch(
    stand_in: StandIn,
) -> None:
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD])])
    poster = Poster()
    # A clock that never moves: only the size rule and the end send a batch after the first.
    batcher = batches.EventBatcher(poster, job, threading.Event(), TraceLog("test", job.trace_id), clock=lambda: 0.0)
    batcher.start()
    batcher.emit(Citation("c-0", 0))
    batcher.emit(TextDelta("first"))
    assert poster.first.wait(5)
    usage = Usage("019a0000-0000-7000-8000-000000000001", None, 1, 0, 0, 1, 0)
    batcher.emit(usage)
    for number in range(40):
        batcher.emit(TextDelta(f"d{number}"))
    batcher.close(5)

    assert [batch for batch, _, _ in poster.posted] == [1, 2, 3]
    assert poster.posted[0][1] == [Citation("c-0", 0), TextDelta("first")]
    assert [len(events) for _, events, _ in poster.posted] == [2, 32, 8]
    assert [len(calls) for _, _, calls in poster.posted] == [0, 1, 0]
    assert batcher.delivered
    assert batcher.posted_events == 42


class UsageThenOutage(FakeChatModel):
    """Sends no text, only its usage, then loses the provider."""

    def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Iterator[Any]:
        for chunk in super()._stream(messages, *args, **kwargs):
            if isinstance(chunk.message, AIMessageChunk) and chunk.message.usage_metadata:
                yield chunk
        raise ModelConnectionError("fake outage after usage")


def test_a_usage_only_batch_shows_the_gm_nothing_so_the_answer_may_be_retried(
    stand_in: StandIn, lib: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(answer, "chat_model", lambda choice, keys: UsageThenOutage(provider=choice.provider))
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD, PT])])

    outcome = run(stand_in, lib, job)

    assert outcome == Outcome.failed("model.unavailable")
    sent = posted(stand_in, job)
    assert [(batch["events"], len(batch["usage"])) for batch in sent] == [([], 1)]
    assert stand_in.jobs[str(job.job_id)].state == "offered"


def test_a_batch_of_usage_alone_is_accepted_but_delivers_nothing(stand_in: StandIn) -> None:
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD])])
    poster = Poster()
    batcher = batches.EventBatcher(poster, job, threading.Event(), TraceLog("test", job.trace_id))
    batcher.start()
    batcher.emit(Usage("019a0000-0000-7000-8000-000000000002", None, 1, 0, 0, 1, 0))
    batcher.close(5)

    assert [(batch, events, len(calls)) for batch, events, calls in poster.posted] == [(1, [], 1)]
    assert batcher.accepted == 1
    assert not batcher.delivered


def test_the_first_batch_never_holds_more_than_max_events(stand_in: StandIn) -> None:
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD])])
    poster = Poster()
    batcher = batches.EventBatcher(
        poster, job, threading.Event(), TraceLog("test", job.trace_id), max_events=4, clock=lambda: 0.0
    )
    for number in range(6):
        batcher.emit(Citation(f"c-{number}", 0))
    batcher.emit(TextDelta("first"))
    batcher.start()
    batcher.close(5)

    assert [len(events) for _, events, _ in poster.posted] == [4, 3]
    assert poster.posted[1][1][-1] == TextDelta("first")


def test_after_the_first_delta_a_batch_goes_out_once_the_interval_has_passed(stand_in: StandIn) -> None:
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD])])
    poster = Poster()
    now = [0.0]
    batcher = batches.EventBatcher(poster, job, threading.Event(), TraceLog("test", job.trace_id), clock=lambda: now[0])
    batcher.start()
    batcher.emit(TextDelta("first"))
    assert poster.first.wait(5)
    poster.first.clear()

    batcher.emit(TextDelta("a"))
    batcher.emit(TextDelta("b"))
    assert not poster.first.wait(0.3)  # the clock has not moved: nothing more yet
    now[0] = batches.BATCH_INTERVAL_SECONDS + 0.01
    batcher.emit(TextDelta("c"))

    assert poster.first.wait(5)
    assert [events for _, events, _ in poster.posted] == [
        [TextDelta("first")],
        [TextDelta("a"), TextDelta("b"), TextDelta("c")],
    ]
    batcher.close(5)
    assert len(poster.posted) == 2


# ---- the GM's stop (Story 2.6)


class StoppingPoster(Poster):
    """Records each batch posted, answering `stop` from the `after`-th batch on."""

    def __init__(self, after: int) -> None:
        super().__init__()
        self.after = after

    def events(
        self, job: ClaimedJob, batch: int, events: Sequence[TextDelta | Citation], usage: Sequence[Usage] = ()
    ) -> EventsAnswer:
        super().events(job, batch, events, usage)
        return EventsAnswer(status=200, stop=batch >= self.after)


def test_after_stop_the_batcher_posts_only_the_usage_left(stand_in: StandIn) -> None:
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD])])
    poster = StoppingPoster(after=1)
    batcher = batches.EventBatcher(poster, job, threading.Event(), TraceLog("test", job.trace_id), clock=lambda: 0.0)
    batcher.start()
    batcher.emit(TextDelta("first"))
    assert poster.first.wait(5)
    deadline = time.monotonic() + 5
    while not batcher.halted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert batcher.stop
    assert batcher.halted

    for number in range(40):
        batcher.emit(TextDelta(f"d{number}"))
    batcher.emit(Citation("c-1", 3))
    usage = Usage("019a0000-0000-7000-8000-000000000003", None, 10, 0, 0, 2, 0, estimated=True)
    batcher.emit(usage)
    batcher.close(5)

    assert [(batch, events, calls) for batch, events, calls in poster.posted] == [
        (1, [TextDelta("first")], []),
        (2, [], [usage]),
    ]
    # Batch 1 was answered stop: the backend dropped its event, so nothing was delivered.
    assert batcher.posted_events == 0
    assert not batcher.delivered


class SlowFake(FakeChatModel):
    """The fake's answer, one chunk every 50 ms, so a stop lands mid-answer."""

    def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Iterator[Any]:
        for chunk in super()._stream(messages, *args, **kwargs):
            time.sleep(0.05)
            yield chunk


def test_a_stop_closes_the_model_stream_posts_the_estimated_usage_and_ends_done(
    stand_in: StandIn, lib: Library, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(answer, "chat_model", lambda choice, keys: SlowFake(provider=choice.provider))
    job = claimed(stand_in, ASKED.question, [tier("system", [SRD, PT])])
    stand_in.stop_after = 1

    outcome = run(stand_in, lib, job)

    assert outcome == Outcome.done()
    sent = posted(stand_in, job)
    assert len(sent) == 1
    assert [event["type"] for event in sent[0]["events"]] == ["delta"]
    # The stream was closed long before the fake's end: one usage, an estimate, posted after the stop.
    assert len(stand_in.usage) == 1
    (usage,) = stand_in.usage.values()
    assert usage["estimated"] is True
    assert usage["input"] > 0
    # Batch 2 was answered stop (not stored); after it, only the usage went out, in a batch of no event.
    bodies = [json.loads(request["body"]) for request in events_requests(stand_in)]
    assert [body["batch"] for body in bodies] == [1, 2, 3]
    assert bodies[-1]["events"] == []
    assert [item["estimated"] for item in bodies[-1]["usage"]] == [True]
    assert stand_in.jobs[str(job.job_id)].state == "done"
    ended = [line for line in log_lines(capsys.readouterr().out) if line["event"] == "answer.ended"]
    assert len(ended) == 1
    assert ended[0]["stopped"] is True
    assert ended[0]["code"] == "none"


def events_requests(stand_in: StandIn) -> list[dict[str, str]]:
    return [request for request in stand_in.requests if request["path"].endswith("/events")]


def test_an_input_that_is_not_the_backends_a_claim_without_a_model_and_a_provider_without_a_key_are_never_retried(
    stand_in: StandIn, lib: Library
) -> None:
    serve_fixtures(stand_in)
    api = JobsApi(stand_in.url, CREDENTIAL, providers=PROVIDERS)
    good = question_input(ASKED.question, [tier("system", [SRD, PT])])

    malformed_id = stand_in.offer(answer.KIND, {**good, "tokenKey": "not base64"}, model=model_of("azure-openai"))
    malformed = api.claim(UUID(malformed_id), new_trace_id()).job
    without_model_id = stand_in.offer(answer.KIND, good)
    without_model = api.claim(UUID(without_model_id), new_trace_id()).job
    assert malformed is not None
    assert without_model is not None
    invalid = Outcome.failed("retrieval.input_invalid", retry=False)
    assert handler(stand_in, lib)(malformed, threading.Event()) == invalid
    no_model = Outcome.failed("model.input_invalid", retry=False)
    assert handler(stand_in, lib)(without_model, threading.Event()) == no_model

    no_key = claimed(stand_in, ASKED.question, [tier("system", [SRD, PT])], "azure-openai")
    anthropic_only = answer.AnswerJob(
        JobsApi(stand_in.url, CREDENTIAL),
        retrieval_fixtures.RecordedOllama().embedder(),
        lambda: local(lib.spy),
        ModelKeys(anthropic_api_key="anthropic-secret"),
    )
    assert anthropic_only(no_key, threading.Event()) == Outcome.failed("model.provider_key_missing", retry=False)
    assert events_requests(stand_in) == []


def test_chat_model_is_the_only_builder_the_answer_uses() -> None:
    assert vars(answer)["chat_model"] is chat.chat_model
