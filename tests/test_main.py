"""The start-up rules, one test per row of the story's matrix, plus the port and connection guards."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from gmshield_worker import health, main
from gmshield_worker.backend import contract
from gmshield_worker.backend.client import RecipeAnswer, VersionAnswer, fetch_contract_version
from gmshield_worker.retrieval.recipe import INDEX_RECIPE
from tests.conftest import StandIn, log_lines

ROOT = Path(__file__).resolve().parent.parent


class FakeLanes:
    """Stands in for the lanes: records when the supervisor starts and stops them."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")


@contextmanager
def running(
    environment: dict[str, str], health_file: Path, lanes: FakeLanes | None = None, **options: Any
) -> Iterator[dict[str, object]]:
    """Runs the worker in a thread until the block ends; yields a dict that receives the status."""
    stop = threading.Event()
    result: dict[str, object] = {}
    fake = lanes or FakeLanes()

    def target() -> None:
        result["status"] = main.run(
            environment,
            stop,
            health_file=health_file,
            backoff_initial=0.01,
            recheck_seconds=0.05,
            lanes_factory=lambda _settings, _trace: fake,
            **options,
        )

    thread = threading.Thread(target=target)
    thread.start()
    try:
        yield result
    finally:
        stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()


def no_lanes(_settings: object, _trace_id: str) -> FakeLanes:
    return FakeLanes()


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def mode_env(stand_in: StandIn, tmp_path: Path) -> dict[str, str]:
    return {"GMSHIELD_MODE": "test", "GMSHIELD_BACKEND_URL": stand_in.url, "GMSHIELD_WORK_DIR": str(tmp_path / "work")}


def test_healthy_start(stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file) as result:
        wait_for(lambda: health.is_healthy(health_file))
        wait_for(lambda: len(stand_in.requests) >= 3)  # re-checked
    assert result["status"] == 0
    assert not health_file.exists()
    events = [line["event"] for line in log_lines(capsys.readouterr().out)]
    assert events.count("worker.ready") == 1
    assert events[-1] == "worker.stopped"


def test_backend_late(stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls = 0

    def late(url: str, trace_id: str) -> VersionAnswer:
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise httpx.ConnectError("[Errno 111] Connection refused to http://secret-host")
        return fetch_contract_version(url, trace_id)

    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file, fetch=late):
        wait_for(lambda: health.is_healthy(health_file))
    output = capsys.readouterr().out
    lines = log_lines(output)
    failures = [line for line in lines if line["event"] == "backend.unreachable"]
    assert [(f["attempt"], f["errorType"]) for f in failures] == [(n, "ConnectError") for n in (1, 2, 3)]
    assert set(failures[0]) == {"timestamp", "level", "traceId", "unit", "event", "attempt", "errorType"}
    assert "secret-host" not in output
    assert any(line["event"] == "worker.ready" for line in lines)


def test_backoff_grows_from_1_to_30_seconds_with_jitter() -> None:
    low = [main.backoff_delay(n, lambda: 0.0) for n in range(1, 10)]
    high = [main.backoff_delay(n, lambda: 1.0) for n in range(1, 10)]
    middle = [main.backoff_delay(n, lambda: 0.5) for n in range(1, 10)]
    assert middle == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0, 30.0]
    assert all(1.0 <= delay <= 30.0 for delay in low + high)
    assert high[1] == pytest.approx(2.4)
    assert low[2] == pytest.approx(3.2)
    assert main.backoff_delay(10_000, lambda: 1.0) == 30.0


def test_old_backend(stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stand_in.version = "0.9.0"
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file):
        wait_for(lambda: len(stand_in.requests) >= 4)
        assert not health.is_healthy(health_file)
    lines = log_lines(capsys.readouterr().out)
    incompatible = [line for line in lines if line["event"] == "backend.incompatible"]
    assert len(incompatible) == 1
    assert incompatible[0]["backendContractVersion"] == "0.9.0"
    assert incompatible[0]["minimumContractVersion"] == "1.18.0"
    assert not any(line["event"] == "worker.ready" for line in lines)


def test_an_old_backend_that_is_upgraded_becomes_healthy(stand_in: StandIn, tmp_path: Path) -> None:
    stand_in.version = "0.9.0"
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file):
        wait_for(lambda: len(stand_in.requests) >= 2)
        stand_in.version = "1.18.0"
        wait_for(lambda: health.is_healthy(health_file))
        stand_in.version = "0.9.0"
        wait_for(lambda: not health_file.exists())


def test_a_body_that_is_not_a_version_is_never_echoed(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.version = "<script>leak</script>"
    with running(mode_env(stand_in, tmp_path), tmp_path / "healthy"):
        wait_for(lambda: len(stand_in.requests) >= 2)
    output = capsys.readouterr().out
    assert "leak" not in output
    incompatible = [line for line in log_lines(output) if line["event"] == "backend.incompatible"]
    assert incompatible[0]["backendContractVersion"] == "invalid"


def test_a_body_that_is_not_the_contract_is_retried(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.status = 500
    stand_in.body = b"internal details that must not be logged"
    with running(mode_env(stand_in, tmp_path), tmp_path / "healthy"):
        wait_for(lambda: len(stand_in.requests) >= 2)
    output = capsys.readouterr().out
    assert "internal details" not in output
    failures = [line for line in log_lines(output) if line["event"] == "backend.unreachable"]
    assert failures
    assert failures[0]["errorType"] == "JSONDecodeError"


def test_a_problem_answer_is_retried_until_the_backend_answers(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.status = 503
    stand_in.body = (
        b'{"type":"urn:gmshield:problem:service.unavailable","title":"t","status":503,'
        b'"category":"service","code":"service.unavailable","params":{},"traceId":"x"}'
    )
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file):
        wait_for(lambda: len(stand_in.requests) >= 3)
        assert not health_file.exists()
        stand_in.status, stand_in.body = 200, None
        wait_for(lambda: health.is_healthy(health_file))
    lines = log_lines(capsys.readouterr().out)
    unexpected = [line for line in lines if line["event"] == "backend.unexpected_answer"]
    assert len(unexpected) >= 3
    assert all(line["status"] == 503 for line in unexpected)
    attempts = [line["attempt"] for line in unexpected]
    assert attempts == list(range(1, len(attempts) + 1))
    assert "service.unavailable" not in str(lines)
    assert any(line["event"] == "worker.ready" for line in lines)


def test_missing_secret(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    environment = {"GMSHIELD_BACKEND_URL": "http://marker-host-123:8080"}
    status = main.run(environment, threading.Event(), health_file=tmp_path / "healthy", lanes_factory=no_lanes)
    assert status == 78
    captured = capsys.readouterr()
    lines = log_lines(captured.err)
    assert lines[-1]["event"] == "startup.refused"
    assert lines[-1]["settings"] == [
        "GMSHIELD_REDIS_HOST",
        "GMSHIELD_REDIS_USERNAME",
        "GMSHIELD_OLLAMA_URL",
        "GMSHIELD_QDRANT_URL",
        "GMSHIELD_WORKER_CREDENTIAL",
        "GMSHIELD_REDIS_PASSWORD",
        "GMSHIELD_QDRANT_API_KEY",
        "GMSHIELD_AZURE_OPENAI_API_KEY",
        "GMSHIELD_ANTHROPIC_API_KEY",
        "GMSHIELD_OPENAI_API_KEY",
    ]
    assert "marker-host-123" not in captured.out + captured.err


def test_test_mode_needs_no_secret_and_uses_the_fake_model(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environment = mode_env(stand_in, tmp_path)
    assert not any("ANTHROPIC" in name for name in environment)
    health_file = tmp_path / "healthy"
    with running(environment, health_file):
        wait_for(lambda: health.is_healthy(health_file))
    checked = next(line for line in log_lines(capsys.readouterr().out) if line["event"] == "startup.checked")
    assert checked["model"] == "fake"
    assert checked["providers"] == ["azure-openai", "anthropic", "openai"]
    assert checked["testMode"] is True


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("LANGSMITH_TRACING", "true"),
        ("LANGCHAIN_TRACING_V2", "1"),
        ("LANGSMITH_OTEL_ENABLED", "TRUE"),
        ("LANGSMITH_GATEWAY", "https://gateway.example"),
        ("OTEL_TRACES_EXPORTER", "otlp"),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318"),
        ("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "content"),
    ],
)
def test_any_trace_export_refuses_the_start_even_in_test_mode(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str], variable: str, value: str
) -> None:
    environment = {**mode_env(stand_in, tmp_path), variable: value}
    status = main.run(environment, threading.Event(), health_file=tmp_path / "healthy", lanes_factory=no_lanes)
    assert status == 78
    captured = capsys.readouterr()
    refused = log_lines(captured.err)[-1]
    assert refused["event"] == "startup.refused"
    assert refused["reason"] == "tracing.on"
    assert refused["settings"] == [variable]
    if "://" in value:
        assert value not in captured.out + captured.err


def test_dirty_work_folder(stand_in: StandIn, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    (work / "sub" / "piece").write_text("x", encoding="utf-8")
    (work / "file").write_text("x", encoding="utf-8")
    (work / "out").symlink_to(outside, target_is_directory=True)
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file):
        wait_for(lambda: health.is_healthy(health_file))
        assert work.is_dir()
        assert list(work.iterdir()) == []
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_start_up_never_listens_and_connects_only_to_the_backend(
    stand_in: StandIn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listened: list[object] = []
    connected: list[object] = []
    real_connect = socket.socket.connect

    def listen(self: socket.socket, *args: object) -> None:
        listened.append(args)
        raise OSError("listening is not allowed")

    def connect(self: socket.socket, address: object) -> None:
        connected.append(address)
        real_connect(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "listen", listen)
    monkeypatch.setattr(socket.socket, "connect", connect)
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file):
        wait_for(lambda: health.is_healthy(health_file))
    assert listened == []
    port = int(stand_in.url.rsplit(":", 1)[1])
    assert connected
    assert all(address == ("127.0.0.1", port) for address in connected)


def run_process(environment: dict[str, str], health_file: Path) -> subprocess.Popen[str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT),
        "GMSHIELD_HEALTH_FILE": str(health_file),
        # The lanes run for real here: a Redis port where nothing listens, so they only retry.
        "GMSHIELD_REDIS_HOST": "127.0.0.1",
        "GMSHIELD_REDIS_PORT": "9",
        **environment,
    }
    return subprocess.Popen(
        [sys.executable, "-m", "gmshield_worker.main"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for_event(process: subprocess.Popen[str], event: str) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        if f'"event":"{event}"' in line:
            return
    raise AssertionError(f"{event} never logged")


@pytest.mark.parametrize("state", ["retrying", "ready"])
def test_sigterm_ends_with_status_0(stand_in: StandIn, tmp_path: Path, state: str) -> None:
    url = stand_in.url if state == "ready" else "http://127.0.0.1:9"
    work_dir = str(tmp_path / "w")
    process = run_process(
        {"GMSHIELD_MODE": "test", "GMSHIELD_BACKEND_URL": url, "GMSHIELD_WORK_DIR": work_dir}, tmp_path / "healthy"
    )
    try:
        wait_for_event(process, "worker.ready" if state == "ready" else "backend.unreachable")
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0
    finally:
        process.kill()
        process.communicate()


def test_the_process_refuses_without_secrets(tmp_path: Path) -> None:
    process = run_process({}, tmp_path / "healthy")
    _out, err = process.communicate(timeout=30)
    assert process.returncode == 78
    refused = log_lines(err)[-1]
    assert refused["event"] == "startup.refused"
    # run_process gives a Redis host: everything else is missing.
    assert refused["settings"] == [
        "GMSHIELD_BACKEND_URL",
        "GMSHIELD_REDIS_USERNAME",
        "GMSHIELD_OLLAMA_URL",
        "GMSHIELD_QDRANT_URL",
        "GMSHIELD_WORKER_CREDENTIAL",
        "GMSHIELD_REDIS_PASSWORD",
        "GMSHIELD_QDRANT_API_KEY",
        "GMSHIELD_AZURE_OPENAI_API_KEY",
        "GMSHIELD_ANTHROPIC_API_KEY",
        "GMSHIELD_OPENAI_API_KEY",
    ]


def test_the_signal_stop_wakes_a_waiter() -> None:
    stop = main.SignalStop()
    assert not stop.wait(0.01)
    started = time.monotonic()
    threading.Timer(0.05, stop.set).start()
    assert stop.wait(5)
    assert time.monotonic() - started < 2
    stop.set()  # twice: no error
    assert stop.is_set()


def test_a_link_in_place_of_the_work_folder_refuses_the_start(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "work").symlink_to(outside, target_is_directory=True)
    status = main.run(
        mode_env(stand_in, tmp_path), threading.Event(), health_file=tmp_path / "healthy", lanes_factory=no_lanes
    )
    assert status == 78
    refused = log_lines(capsys.readouterr().err)[-1]
    assert refused["event"] == "startup.refused"
    assert refused["reason"] == "workdir.unusable"
    assert refused["settings"] == ["GMSHIELD_WORK_DIR"]
    assert stand_in.requests == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unwritable_parent_refuses_the_start(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = tmp_path / "locked"
    parent.mkdir(mode=0o500)
    environment = {**mode_env(stand_in, tmp_path), "GMSHIELD_WORK_DIR": str(parent / "work")}
    try:
        status = main.run(environment, threading.Event(), health_file=tmp_path / "healthy", lanes_factory=no_lanes)
    finally:
        parent.chmod(0o700)
    assert status == 78
    lines = log_lines(capsys.readouterr().err)
    assert [line["event"] for line in lines[-2:]] == ["workdir.unusable", "startup.refused"]
    assert lines[-1]["reason"] == "workdir.unusable"


def test_the_lanes_start_after_the_first_compatible_answer_and_stop_at_the_end(
    stand_in: StandIn, tmp_path: Path
) -> None:
    stand_in.version = "1.0.0"
    lanes = FakeLanes()
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file, lanes=lanes):
        wait_for(lambda: len(stand_in.requests) >= 3)
        assert lanes.events == []  # an older backend: nothing is claimed from it
        stand_in.version = "1.18.0"
        wait_for(lambda: health.is_healthy(health_file))
        wait_for(lambda: len(stand_in.requests) >= 6)  # re-checked, never started twice
        assert lanes.events == ["start"]
    assert lanes.events == ["start", "stop"]


def test_every_check_carries_the_workers_credential(stand_in: StandIn, tmp_path: Path) -> None:
    stand_in.credential = "gmshield-test-mode-worker-credential-not-a-secret"
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file):
        wait_for(lambda: health.is_healthy(health_file))
    assert all(
        request["authorization"] == "Bearer gmshield-test-mode-worker-credential-not-a-secret"
        for request in stand_in.requests
    )


def test_a_wrong_credential_is_an_unexpected_answer_and_never_logged(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.credential = "another-credential-of-at-least-32-characters"
    with running(mode_env(stand_in, tmp_path), tmp_path / "healthy"):
        wait_for(lambda: len(stand_in.requests) >= 2)
    output = capsys.readouterr().out
    unexpected = [line for line in log_lines(output) if line["event"] == "backend.unexpected_answer"]
    assert unexpected
    assert unexpected[0]["status"] == 401
    assert "not-a-secret" not in output


# ---- Indexing at worker start: the recipe stated at every check (contract 1.12.0)


def recipe_requests(stand_in: StandIn) -> list[dict[str, str]]:
    return [request for request in stand_in.requests if request["path"] == "/internal/v1/index-recipe"]


def test_the_recipe_is_stated_at_every_compatible_check_of_a_backend_at_1_14_0_with_no_job_claimed(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.version = "1.18.0"
    stand_in.credential = "gmshield-test-mode-worker-credential-not-a-secret"
    lanes = FakeLanes()
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file, lanes=lanes):
        wait_for(lambda: len(stand_in.recipe_statements) >= 3)
        assert health.is_healthy(health_file)
    # Matrix rows "Deploy" and "Same recipe again": the first check states it, and every check after it.
    assert set(stand_in.recipe_statements) == {INDEX_RECIPE}
    assert all(json.loads(request["body"]) == {"indexRecipe": INDEX_RECIPE} for request in recipe_requests(stand_in))
    assert all(request["method"] == "PUT" for request in recipe_requests(stand_in))
    assert all(
        request["authorization"] == "Bearer gmshield-test-mode-worker-credential-not-a-secret"
        and "x-job-grant" not in request
        for request in recipe_requests(stand_in)
    )
    # Each statement follows a compatible version answer; no job was claimed, and the lanes started once.
    paths = [request["path"] for request in stand_in.requests]
    assert paths[:2] == ["/internal/v1/version", "/internal/v1/index-recipe"]
    assert not any("/jobs/" in path for path in paths)
    assert lanes.events == ["start", "stop"]
    lines = log_lines(capsys.readouterr().out)
    stated = [line for line in lines if line["event"] == "index.recipe_stated"]
    assert len(stated) == 1
    assert stated[0]["recipe"] == INDEX_RECIPE
    assert not any(line["event"] == "index.recipe_unstated" for line in lines)


@pytest.mark.parametrize("version", ["1.12.0", "1.15.0", "1.16.0", "1.17.0", "1.17.9"])
def test_a_backend_before_1_18_0_is_incompatible_and_never_told_the_recipe(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str], version: str
) -> None:
    # Story 4.3 part B: the minimum is 1.18.0 (records in a library search; 1.17.0 since Story 4.2), so an older
    # backend leaves the worker unhealthy, with no lane started and the recipe route never called.
    stand_in.version = version
    lanes = FakeLanes()
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file, lanes=lanes):
        wait_for(lambda: len(stand_in.requests) >= 3)
        assert not health.is_healthy(health_file)
        assert lanes.events == []  # nothing is claimed from it
    assert recipe_requests(stand_in) == []
    assert contract.MINIMUM_BACKEND_CONTRACT_VERSION == "1.18.0"
    events = [line["event"] for line in log_lines(capsys.readouterr().out)]
    assert "backend.incompatible" in events
    assert "index.recipe_stated" not in events


def test_a_failed_statement_is_logged_changes_neither_health_nor_lanes_and_is_stated_again_at_the_next_check(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.version = "1.18.0"
    stand_in.recipe_status = 503
    lanes = FakeLanes()
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file, lanes=lanes):
        wait_for(lambda: len(recipe_requests(stand_in)) >= 2)
        assert health.is_healthy(health_file)
        assert lanes.events == ["start"]
        stand_in.recipe_status = None
        wait_for(lambda: len(stand_in.recipe_statements) >= 2)
        assert health.is_healthy(health_file)
    lines = log_lines(capsys.readouterr().out)
    unstated = [line for line in lines if line["event"] == "index.recipe_unstated"]
    assert unstated
    assert (unstated[0]["level"], unstated[0]["status"], unstated[0]["errorType"]) == ("WARN", 503, "unexpected_answer")
    assert [line["event"] for line in lines].count("index.recipe_stated") == 1
    assert not any(line["event"] == "worker.unhealthy" for line in lines)


def test_a_statement_that_raises_is_logged_by_error_type_only(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in.version = "1.18.0"
    calls = 0

    def unreachable(_url: str, _trace_id: str) -> RecipeAnswer:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("[Errno 111] Connection refused to http://secret-host")

    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file, state_recipe=unreachable):
        wait_for(lambda: calls >= 2)
        assert health.is_healthy(health_file)
    output = capsys.readouterr().out
    unstated = [line for line in log_lines(output) if line["event"] == "index.recipe_unstated"]
    assert unstated[0]["errorType"] == "ConnectError"
    assert set(unstated[0]) == {"timestamp", "level", "traceId", "unit", "event", "errorType"}
    assert "secret-host" not in output


# ---- Story 1.15: non-dumpable, and the work folder's sweep


class FakeSweep:
    def __init__(self, events: list[str], work_dir: str) -> None:
        self.events = events
        events.append(f"sweep:{work_dir}")

    def start(self) -> None:
        self.events.append("sweep.start")

    def stop(self) -> None:
        self.events.append("sweep.stop")


def test_the_worker_makes_itself_non_dumpable_before_anything_else_and_runs_the_sweep(
    stand_in: StandIn, tmp_path: Path
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    events: list[str] = []
    health_file = tmp_path / "healthy"
    with running(
        mode_env(stand_in, tmp_path),
        health_file,
        sweep_factory=lambda work_dir, _trace: FakeSweep(events, work_dir),
    ):
        wait_for(lambda: health.is_healthy(health_file))
        assert libc.prctl(3, 0, 0, 0, 0) == 0  # PR_GET_DUMPABLE: no longer dumpable
    assert events == [f"sweep:{tmp_path / 'work'}", "sweep.start", "sweep.stop"]


def test_a_worker_that_cannot_become_non_dumpable_refuses_to_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def refused() -> None:
        raise OSError(1, "refused")

    status = main.run(
        {"GMSHIELD_MODE": "test"},
        threading.Event(),
        health_file=tmp_path / "healthy",
        lanes_factory=no_lanes,
        non_dumpable=refused,
    )
    assert status == 78
    assert log_lines(capsys.readouterr().err)[-1]["reason"] == "process.dumpable"
