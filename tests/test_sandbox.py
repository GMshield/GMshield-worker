"""The sandbox of a parsing child (Story 1.15): its environment and folder, the seccomp filter (no network socket, no
io_uring), the non-dumpable worker whose environment the child cannot read, the deadline and the cancel."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import socket
import struct
import threading
import time
from pathlib import Path

import pytest

from gmshield_worker.ingestion import sandbox

PR_GET_DUMPABLE = 3

PROBE = """
import ctypes, json, os, socket
found = {}
for family in ("AF_INET", "AF_INET6", "AF_UNIX", "AF_NETLINK", "AF_PACKET"):
    kind = socket.SOCK_RAW if family == "AF_PACKET" else socket.SOCK_DGRAM
    try:
        socket.socket(getattr(socket, family), kind).close()
        found[family] = "allowed"
    except OSError as error:
        found[family] = error.errno
libc = ctypes.CDLL(None, use_errno=True)
found["io_uring_setup"] = ctypes.get_errno() if libc.syscall(425, 1, None) < 0 else "allowed"
# socket(AF_INET, SOCK_DGRAM) through the x32 ABI (x86-64 only): the filter's other-ABI branch.
found["x32_socket"] = ctypes.get_errno() if libc.syscall(41 | 0x40000000, 2, 2, 0) < 0 else "allowed"
try:
    with open(f"/proc/{os.getppid()}/environ", "rb") as environ:
        found["parent_environ"] = "read" if environ.read() is not None else "empty"
except OSError as error:
    found["parent_environ"] = error.errno
found["environment"] = sorted(os.environ)
found["home"] = os.environ.get("HOME")
found["cwd"] = os.getcwd()
targets = []
for fd in os.listdir("/proc/self/fd"):
    try:
        targets.append(os.readlink(f"/proc/self/fd/{fd}"))
    except OSError:
        pass
found["fds"] = targets
with open("found.json", "w") as out:
    json.dump(found, out)
"""


def dumpable() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    return int(libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0))


def probe(folder: Path) -> dict[str, object]:
    (folder / "probe.py").write_text(PROBE, encoding="utf-8")
    result = sandbox.run_child(folder / "probe.py", [], folder=folder, cancel=threading.Event())
    assert result.returncode == 0
    found: dict[str, object] = json.loads((folder / "found.json").read_text(encoding="utf-8"))
    return found


def test_a_child_cannot_open_a_network_socket_or_io_uring_and_gets_eacces(tmp_path: Path) -> None:
    found = probe(tmp_path)
    for family in ("AF_INET", "AF_INET6", "AF_NETLINK", "AF_PACKET"):
        assert found[family] == 13, family  # EACCES
    assert found["AF_UNIX"] == "allowed"
    assert found["io_uring_setup"] == 13


@pytest.mark.skipif(platform.machine() != "x86_64", reason="the x32 ABI exists on x86-64 only")
def test_a_child_cannot_open_a_socket_through_the_x32_abi(tmp_path: Path) -> None:
    assert probe(tmp_path)["x32_socket"] == 13  # EACCES from the filter, not ENOSYS from the kernel


def test_a_child_gets_only_path_lang_and_home_its_folder_and_no_other_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GMSHIELD_WORKER_CREDENTIAL", "w" * 64)
    monkeypatch.setenv("LANG", "C.UTF-8")
    leaked = os.open(tmp_path / "parent-only", os.O_CREAT | os.O_WRONLY, 0o600)
    os.set_inheritable(leaked, True)
    try:
        found = probe(tmp_path)
    finally:
        os.close(leaked)
    assert found["environment"] == ["HOME", "LANG", "PATH"]
    assert found["home"] == str(tmp_path)
    assert found["cwd"] == str(tmp_path)
    # Standard input, output and error on /dev/null; nothing the parent had open, inheritable or not.
    targets = found["fds"]
    assert isinstance(targets, list)
    assert targets.count("/dev/null") >= 3
    assert not any("parent-only" in str(target) for target in targets)


def test_a_child_cannot_read_the_environment_of_a_non_dumpable_worker(tmp_path: Path) -> None:
    assert dumpable() == 1
    assert probe(tmp_path)["parent_environ"] == "read"  # the check is real: a dumpable parent's is readable

    sandbox.make_non_dumpable()
    assert dumpable() == 0
    assert probe(tmp_path)["parent_environ"] == 13


def test_the_deadline_kills_the_child_and_what_it_started(tmp_path: Path) -> None:
    marker = tmp_path / "late"
    (tmp_path / "slow.py").write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3); open(\"late\", \"w\").close()'])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    started = time.monotonic()

    result = sandbox.run_child(tmp_path / "slow.py", [], folder=tmp_path, cancel=threading.Event(), deadline_seconds=1)

    assert result.timed_out
    assert not result.cancelled
    assert result.returncode < 0
    assert time.monotonic() - started < 10
    time.sleep(3.5)
    assert not marker.exists()


def test_a_cancel_kills_the_child_at_once(tmp_path: Path) -> None:
    (tmp_path / "slow.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()

    result = sandbox.run_child(tmp_path / "slow.py", [], folder=tmp_path, cancel=cancel)

    assert result.cancelled
    assert not result.timed_out
    assert result.seconds < 5


def test_the_filter_is_known_for_x86_64_and_aarch64_only() -> None:
    for machine in ("x86_64", "aarch64"):
        program = sandbox.filter_program(machine)
        assert len(program) % 8 == 0
        assert struct.unpack_from("=HBBI", program, 0) == (sandbox.BPF_LD_W_ABS, 0, 0, sandbox.OFFSET_ARCH)
    with pytest.raises(sandbox.SandboxUnavailable):
        sandbox.filter_program("riscv64")


def test_the_launcher_never_runs_the_program_without_its_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(_machine: str | None = None) -> None:
        raise sandbox.SandboxUnavailable("riscv64")

    monkeypatch.setattr(sandbox, "install_filter", refused)
    monkeypatch.setattr(os, "execv", lambda *_args: pytest.fail("the program ran unsandboxed"))
    assert sandbox.launch(["sandbox.py", "program.py"]) == sandbox.EXIT_UNSANDBOXED
    assert sandbox.launch(["sandbox.py"]) == sandbox.EXIT_UNSANDBOXED


def test_the_environment_falls_back_to_a_plain_path() -> None:
    assert sandbox.child_environment(Path("/work/j"), {"SECRET": "x"}) == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/work/j",
    }


def test_a_socket_in_this_process_is_still_allowed() -> None:
    # The filter is only ever installed in the child: the worker itself keeps its network.
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close()
