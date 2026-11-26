"""The work lanes (Story 1.12) on an in-memory stream and the stand-in internal API: claims and
acknowledgements, lane priority and capacity, renewal and its refusal, reporting, take-over at start,
and the job's trace id on every line."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest
import redis

from gmshield_worker.backend.client import ClaimedJob, JobsApi, Outcome
from gmshield_worker.ingestion.step import IngestionStep
from gmshield_worker.lanes import Lanes, build_lanes, consumer_name, handlers, probe
from gmshield_worker.lanes.consumer import Lane
from gmshield_worker.lanes.handlers import Handler
from gmshield_worker.lanes.streams import Message, RedisWorkStream, connect
from gmshield_worker.log import new_trace_id
from gmshield_worker.settings import RedisSettings, check_settings
from tests.conftest import StandIn, log_lines

CREDENTIAL = "w" * 64


@dataclass
class FakeStream:
    """A stream of one lane, in memory: new messages, pending ones, acknowledged ones."""

    name: str
    new: queue.Queue[Message] = field(default_factory=queue.Queue)
    pending: dict[str, Message] = field(default_factory=dict)
    acked: list[str] = field(default_factory=list)
    stale: list[Message] = field(default_factory=list)
    failing_reads: int = 0
    reads: int = 0
    _sequence: int = 0

    def add(self, job_id: str) -> str:
        self._sequence += 1
        message = Message(f"{self._sequence}-0", job_id)
        self.new.put(message)
        return message.message_id

    def read(self, consumer: str, block_ms: int) -> list[Message]:
        self.reads += 1
        if self.failing_reads > 0:
            self.failing_reads -= 1
            raise ConnectionError("redis down")
        try:
            message = self.new.get(timeout=block_ms / 1000)
        except queue.Empty:
            return []
        self.pending[message.message_id] = message
        return [message]

    def ack(self, message_id: str) -> None:
        self.acked.append(message_id)
        self.pending.pop(message_id, None)

    def take_over(self, consumer: str, min_idle_ms: int) -> list[Message]:
        taken, self.stale = self.stale, []
        for message in taken:
            self.pending[message.message_id] = message
        return taken


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


@contextmanager
def lane(
    stand_in: StandIn,
    name: str = "interactive",
    capacity: int = 2,
    handlers_by_kind: Mapping[str, Handler] | None = None,
    stream: FakeStream | None = None,
    *,
    report_give_up_seconds: float = 120.0,
) -> Iterator[tuple[Lane, FakeStream]]:
    work = stream or FakeStream(f"gmshield:work:{name}")
    running = Lane(
        name,
        work,
        capacity=capacity,
        api=JobsApi(stand_in.url, CREDENTIAL),
        handlers=handlers_by_kind if handlers_by_kind is not None else {probe.KIND: probe.run},
        consumer="test-consumer",
        trace_id=new_trace_id(),
        stop=threading.Event(),
        read_block_ms=50,
        backoff_initial=0.01,
        renewal_retry_seconds=0.05,
        report_give_up_seconds=report_give_up_seconds,
    )
    running.start()
    try:
        yield running, work
    finally:
        running.stop()


def job_state(stand_in: StandIn, job_id: str) -> str:
    return stand_in.jobs[job_id].state


# ---- claim and acknowledgement


def test_a_claimed_job_runs_its_outcome_is_reported_and_every_line_carries_the_jobs_trace(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    job_trace = new_trace_id()
    job_id = stand_in.offer("probe", {"workMs": 10, "failTimes": 0}, job_trace)
    with lane(stand_in) as (_lane, stream):
        message_id = stream.add(job_id)
        wait_for(lambda: job_state(stand_in, job_id) == "done")
        wait_for(lambda: message_id in stream.acked)
    assert stand_in.jobs[job_id].outcomes == [{"outcome": "done", "result": {"workedMs": 10}}]
    assert stream.pending == {}
    lines = [line for line in log_lines(capsys.readouterr().out) if line.get("jobId") == job_id]
    assert [line["event"] for line in lines if line["event"] != "job.renewed"] == ["job.claimed", "job.reported"]
    assert {line["traceId"] for line in lines} == {job_trace}
    # The renewal and the outcome carry the job's trace to the backend, with the grant.
    later = [request for request in stand_in.job_requests(job_id) if "/claims/" in request["path"]]
    assert later
    assert all(request["traceparent"].split("-")[1] == job_trace for request in later)
    assert all(request["x-job-grant"] for request in later)


@pytest.mark.parametrize("state", ["running", "done"])
def test_a_definite_refusal_acknowledges_the_message_and_runs_nothing(
    stand_in: StandIn, state: str, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = stand_in.offer("probe", {"workMs": 0})
    stand_in.jobs[job_id].state = state
    if state == "running":
        stand_in.jobs[job_id].lapses_at = stand_in.times()[2]
    with lane(stand_in) as (_lane, stream):
        message_id = stream.add(job_id)
        wait_for(lambda: message_id in stream.acked)
    assert stand_in.jobs[job_id].outcomes == []
    refused = [line for line in log_lines(capsys.readouterr().out) if line["event"] == "claim.refused"]
    assert refused[0]["status"] == 409
    assert refused[0]["code"] == ("job.claimed" if state == "running" else "job.not_offered")


def test_no_answer_or_a_5xx_leaves_the_message_pending(stand_in: StandIn) -> None:
    job_id = stand_in.offer("probe", {"workMs": 0})
    stand_in.claim_status = 503
    with lane(stand_in) as (_lane, stream):
        message_id = stream.add(job_id)
        wait_for(lambda: any("/claims" in request["path"] for request in stand_in.requests))
        time.sleep(0.2)
        assert stream.acked == []
        assert message_id in stream.pending
    unreachable = StandIn(url="http://127.0.0.1:9")
    with lane(unreachable) as (_lane, stream):
        message_id = stream.add(job_id)
        wait_for(lambda: stream.reads >= 3)
        assert stream.acked == []
        assert message_id in stream.pending


def test_a_message_without_a_job_id_is_left_alone(stand_in: StandIn, capsys: pytest.CaptureFixture[str]) -> None:
    with lane(stand_in) as (_lane, stream):
        stream.new.put(Message("9-0", None))
        stream.new.put(Message("10-0", "not-a-uuid"))
        wait_for(lambda: stream.reads >= 3)
    assert stream.acked == []
    assert [line["event"] for line in log_lines(capsys.readouterr().out)].count("message.invalid") == 2
    assert stand_in.requests == []


# ---- lanes


def test_an_interactive_job_is_claimed_at_once_while_the_ingestion_lane_is_busy(stand_in: StandIn) -> None:
    long_job = stand_in.offer("probe", {"workMs": 3000})
    queued_ingestion = stand_in.offer("probe", {"workMs": 0})
    interactive_job = stand_in.offer("probe", {"workMs": 0})
    with (
        lane(stand_in, "ingestion", capacity=1) as (_ingestion, ingestion_stream),
        lane(stand_in, "interactive") as (_interactive, interactive_stream),
    ):
        ingestion_stream.add(long_job)
        wait_for(lambda: job_state(stand_in, long_job) == "running")
        ingestion_stream.add(queued_ingestion)
        started = time.monotonic()
        interactive_stream.add(interactive_job)
        wait_for(lambda: job_state(stand_in, interactive_job) == "done")
        assert time.monotonic() - started < 1.5
        assert job_state(stand_in, long_job) == "running"
        # The busy ingestion lane leaves its next message in the stream for another worker.
        assert job_state(stand_in, queued_ingestion) == "offered"
        assert ingestion_stream.new.qsize() == 1


def test_a_lane_runs_at_most_its_capacity_at_once(stand_in: StandIn) -> None:
    running = 0
    most = 0
    lock = threading.Lock()

    def counted(job: ClaimedJob, cancel: threading.Event, /) -> Outcome:
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
        cancel.wait(0.3)
        with lock:
            running -= 1
        return Outcome.done()

    jobs = [stand_in.offer("count") for _ in range(5)]
    with lane(stand_in, capacity=2, handlers_by_kind={"count": counted}) as (_lane, stream):
        for job_id in jobs:
            stream.add(job_id)
        wait_for(lambda: all(job_state(stand_in, job_id) == "done" for job_id in jobs))
    assert most == 2


def test_at_start_a_lane_takes_over_messages_left_pending_by_a_dead_worker(stand_in: StandIn) -> None:
    job_id = stand_in.offer("probe", {"workMs": 0})
    stream = FakeStream("gmshield:work:interactive", stale=[Message("1-0", job_id)])
    with lane(stand_in, stream=stream):
        wait_for(lambda: job_state(stand_in, job_id) == "done")
        wait_for(lambda: "1-0" in stream.acked)


def test_a_stream_that_fails_is_retried_with_backoff(stand_in: StandIn, capsys: pytest.CaptureFixture[str]) -> None:
    job_id = stand_in.offer("probe", {"workMs": 0})
    stream = FakeStream("gmshield:work:interactive", failing_reads=3)
    stream.add(job_id)
    with lane(stand_in, stream=stream):
        wait_for(lambda: job_state(stand_in, job_id) == "done")
    failures = [line for line in log_lines(capsys.readouterr().out) if line["event"] == "stream.unavailable"]
    assert [line["attempt"] for line in failures] == [1, 2, 3]
    assert failures[0]["errorType"] == "ConnectionError"


def test_a_job_of_an_unknown_kind_is_reported_failed(stand_in: StandIn) -> None:
    job_id = stand_in.offer("nothing-handles-this")
    with lane(stand_in) as (_lane, stream):
        stream.add(job_id)
        wait_for(lambda: bool(stand_in.jobs[job_id].outcomes))
    assert stand_in.jobs[job_id].outcomes[0] == {"outcome": "failed", "code": "worker.unknown_kind"}


def test_a_handler_that_raises_is_a_failed_attempt_logged_by_type_only(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(job: ClaimedJob, cancel: threading.Event, /) -> Outcome:
        raise RuntimeError("content that must not be logged")

    job_id = stand_in.offer("broken")
    with lane(stand_in, handlers_by_kind={"broken": broken}) as (_lane, stream):
        stream.add(job_id)
        wait_for(lambda: bool(stand_in.jobs[job_id].outcomes))
    assert stand_in.jobs[job_id].outcomes[0] == {"outcome": "failed", "code": "worker.handler_failed"}
    output = capsys.readouterr()
    assert "content that must not be logged" not in output.out + output.err


# ---- renewal and outcome


def test_a_long_job_renews_its_claim_and_finishes(stand_in: StandIn) -> None:
    stand_in.renew_seconds, stand_in.lapse_seconds = 0.1, 0.6
    job_id = stand_in.offer("probe", {"workMs": 900})
    with lane(stand_in) as (_lane, stream):
        stream.add(job_id)
        wait_for(lambda: job_state(stand_in, job_id) == "done")
    assert stand_in.jobs[job_id].renewals >= 3


def test_a_refused_renewal_stops_the_work_and_nothing_is_reported(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.renew_seconds, stand_in.lapse_seconds = 0.1, 5.0
    stand_in.renew_status = 403
    job_id = stand_in.offer("probe", {"workMs": 5000})
    with lane(stand_in) as (_lane, stream):
        stream.add(job_id)
        started = time.monotonic()
        output = Output(capsys)
        wait_for(lambda: output.reason() == "renewal_refused")
        assert time.monotonic() - started < 2
    assert stand_in.jobs[job_id].outcomes == []
    assert not any("/outcome" in request["path"] for request in stand_in.requests)


@dataclass
class Output:
    """The log lines written so far, read as they come."""

    capsys: pytest.CaptureFixture[str]
    seen: list[dict[str, object]] = field(default_factory=list)

    def reason(self) -> object:
        """The reason of the last `job.abandoned` line, if any."""
        self.seen.extend(log_lines(self.capsys.readouterr().out))
        reasons = [line.get("reason") for line in self.seen if line["event"] == "job.abandoned"]
        return reasons[-1] if reasons else None


def test_renewals_without_an_answer_stop_the_work_once_the_claim_lapsed(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.renew_seconds, stand_in.lapse_seconds = 0.1, 0.5
    stand_in.renew_status = 503
    job_id = stand_in.offer("probe", {"workMs": 5000})
    with lane(stand_in) as (_lane, stream):
        stream.add(job_id)
        output = Output(capsys)
        wait_for(lambda: output.reason() == "claim_lapsed")
    assert stand_in.jobs[job_id].outcomes == []


def test_an_outcome_without_an_answer_is_retried(stand_in: StandIn) -> None:
    stand_in.outcome_status = 503
    job_id = stand_in.offer("probe", {"workMs": 0})
    with lane(stand_in) as (_lane, stream):
        stream.add(job_id)
        wait_for(lambda: sum("/outcome" in request["path"] for request in stand_in.requests) >= 2)
        stand_in.outcome_status = None
        wait_for(lambda: job_state(stand_in, job_id) == "done")


def test_an_outcome_that_never_gets_an_answer_is_given_up_and_frees_the_slot(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.outcome_status = 503
    first = stand_in.offer("probe", {"workMs": 0})
    second = stand_in.offer("probe", {"workMs": 0})
    with lane(stand_in, capacity=1, report_give_up_seconds=0.5) as (_lane, stream):
        stream.add(first)
        output = Output(capsys)
        wait_for(lambda: output.reason() == "outcome_unanswered")
        stand_in.outcome_status = None
        stream.add(second)
        wait_for(lambda: job_state(stand_in, second) == "done")
    # The first job is left to the backend: its claim lapses, nothing more is sent about it.
    assert stand_in.jobs[first].state == "running"
    assert stand_in.jobs[first].outcomes == []


def test_stopping_the_lane_abandons_a_running_job_unreported(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    job_id = stand_in.offer("probe", {"workMs": 30000})
    with lane(stand_in) as (running, stream):
        stream.add(job_id)
        wait_for(lambda: job_state(stand_in, job_id) == "running")
        started = time.monotonic()
        running.stop()
        assert time.monotonic() - started < 5
    assert stand_in.jobs[job_id].outcomes == []
    abandoned = [line for line in log_lines(capsys.readouterr().out) if line["event"] == "job.abandoned"]
    assert abandoned[0]["reason"] == "worker_stopping"


# ---- the probe and the registry


def claimed(job_input: Mapping[str, object], attempt: int = 1) -> ClaimedJob:
    from datetime import UTC, datetime
    from uuid import uuid4

    now = datetime.now(UTC)
    return ClaimedJob(uuid4(), uuid4(), attempt, now, now, "probe", dict(job_input), new_trace_id(), "g" * 43)


def test_the_probe_fails_its_first_attempts_then_is_done() -> None:
    never = threading.Event()
    job_input = {"workMs": 1, "failTimes": 2}
    assert probe.run(claimed(job_input, 1), never) == Outcome.failed("probe.failed")
    assert probe.run(claimed(job_input, 2), never) == Outcome.failed("probe.failed")
    assert probe.run(claimed(job_input, 3), never) == Outcome.done({"workedMs": 1})


def test_the_probe_stops_when_cancelled_and_refuses_bad_input() -> None:
    cancel = threading.Event()
    cancel.set()
    assert probe.run(claimed({"workMs": 60000}), cancel) == Outcome.failed("probe.cancelled")
    for bad in ({"workMs": -1}, {"workMs": "1"}, {"failTimes": True}, {"workMs": 10**9}):
        with pytest.raises(ValueError, match="probe input"):
            probe.run(claimed(bad), threading.Event())


def test_ingestion_runs_in_every_mode_and_the_probes_in_test_mode_only() -> None:
    always = {"index.change", "index.removal", "index.record_removal", "question.answer", "library.search"}
    test_mode = check_settings({"GMSHIELD_MODE": "test"})
    assert set(handlers.handlers_for(test_mode)) == {
        "ingestion",
        "probe",
        "test.question",
        "test.model",
        *always,
    }
    production = check_settings(
        {
            "GMSHIELD_BACKEND_URL": "http://b:8080",
            "GMSHIELD_ANTHROPIC_API_KEY": "sk",
            "GMSHIELD_WORKER_CREDENTIAL": CREDENTIAL,
            "GMSHIELD_REDIS_HOST": "redis",
            "GMSHIELD_REDIS_USERNAME": "worker",
            "GMSHIELD_REDIS_PASSWORD": "p",
            "GMSHIELD_OLLAMA_URL": "http://172.30.5.1:11434",
            "GMSHIELD_QDRANT_URL": "https://qdrant:6333",
            "GMSHIELD_QDRANT_API_KEY": "q",
        }
    )
    assert set(handlers.handlers_for(production)) == {"ingestion", *always}
    # Story 4.2: the ingestion handler holds the providers' keys, which its records steps' model calls need.
    for settings in (test_mode, production):
        ingestion = handlers.handlers_for(settings)["ingestion"]
        assert isinstance(ingestion, IngestionStep)
        assert ingestion.keys is settings.models


def test_both_lanes_start_interactive_first_and_stop_together() -> None:
    order: list[str] = []

    class Recorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def start(self) -> None:
            order.append(f"start {self.name}")

        def stop(self) -> None:
            order.append(f"stop {self.name}")

    lanes = Lanes(lanes=[Recorder("interactive"), Recorder("ingestion")])  # type: ignore[list-item]
    lanes.start()
    lanes.stop()
    assert order == ["start interactive", "start ingestion", "stop interactive", "stop ingestion"]
    assert lanes.stop_event.is_set()
    first, second = consumer_name(), consumer_name()
    assert first != second
    assert len(first) <= 60


def test_redis_is_reached_over_verified_tls_unless_switched_off() -> None:
    settings = RedisSettings(host="redis", username="worker", password="p", tls=True)  # noqa: S106 - a test value
    secure = connect(settings).connection_pool
    assert secure.connection_class is redis.SSLConnection
    options = secure.connection_kwargs
    assert options["ssl_cert_reqs"] == "required"
    assert options["ssl_check_hostname"] is True
    assert options["ssl_ca_certs"] == "/run/gmshield/certs/ca.crt"
    assert options["protocol"] == 2
    plain = connect(RedisSettings(host="redis", tls=False)).connection_pool
    assert plain.connection_class is redis.Connection
    assert "ssl_ca_certs" not in plain.connection_kwargs


def test_build_lanes_puts_the_interactive_lane_first_with_its_capacity() -> None:
    settings = check_settings({"GMSHIELD_MODE": "test", "GMSHIELD_INTERACTIVE_CAPACITY": "4"})
    built = build_lanes(settings, new_trace_id())
    first, second = built.lanes
    assert (first.name, first.capacity) == ("interactive", 4)
    assert (second.name, second.capacity) == ("ingestion", 1)
    assert isinstance(first.stream, RedisWorkStream)
    assert isinstance(second.stream, RedisWorkStream)
    assert first.stream.name == "gmshield:work:interactive"
    assert second.stream.name == "gmshield:work:ingestion"
    assert first.consumer == second.consumer
    assert first.stop_event is second.stop_event is built.stop_event
    assert set(first.handlers) == {
        "ingestion",
        "probe",
        "test.question",
        "test.model",
        "index.change",
        "index.removal",
        "index.record_removal",
        "question.answer",
        "library.search",
    }
    # Story 2.10: every claim states the providers; test mode states all three.
    assert first.api is second.api
    assert isinstance(first.api, JobsApi)
    assert first.api.providers == ("azure-openai", "anthropic", "openai")


def test_build_lanes_states_only_the_providers_this_worker_holds_a_key_for() -> None:
    settings = check_settings(
        {
            "GMSHIELD_BACKEND_URL": "http://b:8080",
            "GMSHIELD_ANTHROPIC_API_KEY": "anthropic-key",
            "GMSHIELD_WORKER_CREDENTIAL": CREDENTIAL,
            "GMSHIELD_REDIS_HOST": "redis",
            "GMSHIELD_REDIS_USERNAME": "worker",
            "GMSHIELD_REDIS_PASSWORD": "p",
            "GMSHIELD_REDIS_TLS": "false",
            "GMSHIELD_OLLAMA_URL": "http://172.30.5.1:11434",
            "GMSHIELD_QDRANT_URL": "https://qdrant:6333",
            "GMSHIELD_QDRANT_API_KEY": "q",
        }
    )
    first, second = build_lanes(settings, new_trace_id()).lanes
    assert isinstance(first.api, JobsApi)
    assert first.api is second.api
    assert first.api.providers == ("anthropic",)
