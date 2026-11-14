"""The start-up rules, one test per row of the story's matrix, plus the port and connection guards."""

from __future__ import annotations

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
from gmshield_worker.backend.client import VersionAnswer, fetch_contract_version
from tests.conftest import StandIn, log_lines

ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def running(environment: dict[str, str], health_file: Path, **options: Any) -> Iterator[dict[str, object]]:
    """Runs the worker in a thread until the block ends; yields a dict that receives the status."""
    stop = threading.Event()
    result: dict[str, object] = {}

    def target() -> None:
        result["status"] = main.run(
            environment, stop, health_file=health_file, backoff_initial=0.01, recheck_seconds=0.05, **options
        )

    thread = threading.Thread(target=target)
    thread.start()
    try:
        yield result
    finally:
        stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()


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
    assert incompatible[0]["minimumContractVersion"] == "1.0.0"
    assert not any(line["event"] == "worker.ready" for line in lines)


def test_an_old_backend_that_is_upgraded_becomes_healthy(stand_in: StandIn, tmp_path: Path) -> None:
    stand_in.version = "0.9.0"
    health_file = tmp_path / "healthy"
    with running(mode_env(stand_in, tmp_path), health_file):
        wait_for(lambda: len(stand_in.requests) >= 2)
        stand_in.version = "1.0.0"
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


def test_an_error_status_is_retried(stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stand_in.status = 500
    stand_in.body = b"internal details that must not be logged"
    with running(mode_env(stand_in, tmp_path), tmp_path / "healthy"):
        wait_for(lambda: len(stand_in.requests) >= 2)
    output = capsys.readouterr().out
    assert "internal details" not in output
    assert any(line["event"] in ("backend.unexpected_answer", "backend.unreachable") for line in log_lines(output))


def test_missing_secret(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    environment = {"GMSHIELD_BACKEND_URL": "http://marker-host-123:8080"}
    status = main.run(environment, threading.Event(), health_file=tmp_path / "healthy")
    assert status == 78
    captured = capsys.readouterr()
    lines = log_lines(captured.err)
    assert lines[-1]["event"] == "startup.refused"
    assert lines[-1]["settings"] == ["GMSHIELD_ANTHROPIC_API_KEY"]
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
    assert checked["testMode"] is True


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


def run_process(environment: dict[str, str]) -> subprocess.Popen[str]:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT), **environment}
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
    process = run_process({"GMSHIELD_MODE": "test", "GMSHIELD_BACKEND_URL": url, "GMSHIELD_WORK_DIR": work_dir})
    try:
        wait_for_event(process, "worker.ready" if state == "ready" else "backend.unreachable")
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 0
    finally:
        process.kill()
        process.communicate()


def test_the_process_refuses_without_secrets() -> None:
    process = run_process({})
    _out, err = process.communicate(timeout=30)
    assert process.returncode == 78
    refused = log_lines(err)[-1]
    assert refused["event"] == "startup.refused"
    assert refused["settings"] == ["GMSHIELD_BACKEND_URL", "GMSHIELD_ANTHROPIC_API_KEY"]


def test_the_signal_stop_wakes_a_waiter() -> None:
    stop = main.SignalStop()
    assert not stop.wait(0.01)
    started = time.monotonic()
    threading.Timer(0.05, stop.set).start()
    assert stop.wait(5)
    assert time.monotonic() - started < 2
    stop.set()  # twice: no error
    assert stop.is_set()
