"""Story 1.12 against a real Redis (test mode: `redis:6379`, or `GMSHIELD_REDIS_HOST` and
`GMSHIELD_REDIS_PORT`), with the stand-in internal API in process. The worker connects as an
account with exactly the production ACL (GMshield-pipeline `deploy/data-credentials.sh`), so any
other command it sent would fail here. The test itself plays the backend's part in Redis: it
creates the group, adds the messages and flushes Redis.

Run by the Dockerfile's `integration` stage beside a throwaway Redis (`.gmshield/pipeline.toml`
`[test]`), never by the unit test run.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
import redis

from gmshield_worker.backend.client import JobsApi
from gmshield_worker.lanes import ingestion, interactive, probe
from gmshield_worker.lanes.consumer import Lane
from gmshield_worker.lanes.streams import GROUP, RedisWorkStream, connect
from gmshield_worker.log import new_trace_id
from gmshield_worker.settings import RedisSettings
from tests.conftest import StandIn, log_lines

HOST = os.environ.get("GMSHIELD_REDIS_HOST") or "redis"
PORT = int(os.environ.get("GMSHIELD_REDIS_PORT") or "6379")
WORKER = "gmshield-it-worker"
STREAMS = (interactive.STREAM, ingestion.STREAM)
# The worker's account in production, word for word (deploy/data-credentials.sh), with a password.
WORKER_ACL = [
    "reset",
    "on",
    "resetkeys",
    *(f"~{stream}" for stream in STREAMS),
    "resetchannels",
    "-@all",
    *("+auth", "+hello", "+ping", "+xreadgroup", "+xack", "+xautoclaim", "+xclaim", "+xpending"),
]


@pytest.fixture
def admin() -> Iterator[redis.Redis]:
    client = redis.Redis(host=HOST, port=PORT, decode_responses=True, protocol=2)
    client.flushall()
    yield client
    client.flushall()
    client.close()


@pytest.fixture
def worker_redis(admin: redis.Redis) -> Iterator[redis.Redis]:
    password = secrets.token_hex(16)
    admin.execute_command("ACL", "SETUSER", WORKER, *WORKER_ACL, f">{password}")  # type: ignore[no-untyped-call]
    client = connect(RedisSettings(host=HOST, port=PORT, username=WORKER, password=password, tls=False))
    yield client
    client.close()
    admin.execute_command("ACL", "DELUSER", WORKER)  # type: ignore[no-untyped-call]


def create_groups(admin: redis.Redis) -> None:
    for stream in STREAMS:
        admin.xgroup_create(stream, GROUP, id="0", mkstream=True)


def wait_for(condition: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.02)


@contextmanager
def lane(
    client: redis.Redis, stand_in: StandIn, stream: str = interactive.STREAM, take_over_after_ms: int = 120_000
) -> Iterator[Lane]:
    running = Lane(
        "interactive" if stream == interactive.STREAM else "ingestion",
        RedisWorkStream(client, stream),
        capacity=2,
        api=JobsApi(stand_in.url, "w" * 64),
        handlers={probe.KIND: probe.run},
        consumer=f"it-{secrets.token_hex(4)}",
        trace_id=new_trace_id(),
        stop=threading.Event(),
        read_block_ms=200,
        take_over_after_ms=take_over_after_ms,
        backoff_initial=0.05,
    )
    running.start()
    try:
        yield running
    finally:
        running.stop()


def pending(admin: redis.Redis, stream: str = interactive.STREAM) -> int:
    summary = admin.xpending(stream, GROUP)
    return int(summary["pending"])


def test_a_job_offered_on_the_stream_is_claimed_run_reported_and_acknowledged(
    admin: redis.Redis, worker_redis: redis.Redis, stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    create_groups(admin)
    job_trace = new_trace_id()
    job_id = stand_in.offer("probe", {"workMs": 50, "failTimes": 0}, job_trace)
    with lane(worker_redis, stand_in):
        admin.xadd(interactive.STREAM, {"job": job_id})
        wait_for(lambda: stand_in.jobs[job_id].state == "done")
        wait_for(lambda: pending(admin) == 0)
    lines = [line for line in log_lines(capsys.readouterr().out) if line.get("jobId") == job_id]
    assert {line["traceId"] for line in lines} == {job_trace}
    # The worker only read: the stream still holds the one message the "backend" added.
    assert admin.xlen(interactive.STREAM) == 1


def test_a_claim_without_a_definite_answer_leaves_the_message_pending(
    admin: redis.Redis, worker_redis: redis.Redis, stand_in: StandIn
) -> None:
    create_groups(admin)
    job_id = stand_in.offer("probe", {"workMs": 0})
    stand_in.claim_status = 503
    with lane(worker_redis, stand_in):
        admin.xadd(interactive.STREAM, {"job": job_id})
        wait_for(lambda: any("/claims" in request["path"] for request in stand_in.requests))
        time.sleep(0.3)
        assert pending(admin) == 1


def test_at_start_a_lane_takes_over_what_a_dead_worker_left_pending(
    admin: redis.Redis, worker_redis: redis.Redis, stand_in: StandIn
) -> None:
    create_groups(admin)
    job_id = stand_in.offer("probe", {"workMs": 0})
    admin.xadd(interactive.STREAM, {"job": job_id})
    # A worker read it and died before claiming.
    assert admin.xreadgroup(GROUP, "dead-worker", {interactive.STREAM: ">"}, count=1)
    time.sleep(0.3)
    with lane(worker_redis, stand_in, take_over_after_ms=200):
        wait_for(lambda: stand_in.jobs[job_id].state == "done")
        wait_for(lambda: pending(admin) == 0)


def test_a_lane_waits_for_the_group_and_resumes_after_redis_is_flushed(
    admin: redis.Redis, worker_redis: redis.Redis, stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    first = stand_in.offer("probe", {"workMs": 0})
    second = stand_in.offer("probe", {"workMs": 0})
    with lane(worker_redis, stand_in, stream=ingestion.STREAM):
        time.sleep(0.5)  # no group yet (NOGROUP): the lane retries
        # The backend's sweep creates the group from the start of the stream, so an offer made
        # before it existed is still read.
        admin.xadd(ingestion.STREAM, {"job": first})
        create_groups(admin)
        wait_for(lambda: stand_in.jobs[first].state == "done")

        admin.flushall()  # Redis restarted empty
        time.sleep(0.5)
        create_groups(admin)
        admin.xadd(ingestion.STREAM, {"job": second})
        wait_for(lambda: stand_in.jobs[second].state == "done")
    failures = [line for line in log_lines(capsys.readouterr().out) if line["event"] == "stream.unavailable"]
    assert failures
    assert {line["errorType"] for line in failures} <= {"ResponseError", "ConnectionError"}


def test_the_workers_account_cannot_add_trim_or_create_groups(admin: redis.Redis, worker_redis: redis.Redis) -> None:
    create_groups(admin)
    for command in (
        lambda: worker_redis.xadd(interactive.STREAM, {"job": "x"}),
        lambda: worker_redis.xtrim(interactive.STREAM, minid="0"),
        lambda: worker_redis.xgroup_create(interactive.STREAM, "other", id="0"),
        lambda: worker_redis.get("gmshield:session:x"),
    ):
        with pytest.raises(redis.exceptions.NoPermissionError):
            command()
    assert worker_redis.ping()
