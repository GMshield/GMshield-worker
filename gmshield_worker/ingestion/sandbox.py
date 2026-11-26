"""The sandbox every parsing child runs in (Story 1.15, AD-21).

The parent (`run_child`) starts this file as a script, in isolated mode, with:

- an empty environment but `PATH`, `LANG` and `HOME` (the step folder): no credential, no setting;
- the step folder as its working directory, every other descriptor closed, standard input, output and error on
  `/dev/null`, and a session of its own, so the deadline kills the child and anything it started;
- a deadline: the child is killed at `DEADLINE_SECONDS` (9 minutes 30 seconds), or as soon as the step is cancelled;
  whatever it left running in its session is killed when it ends.

As `__main__`, this file is the launcher: it installs a seccomp filter (no new privileges first, which needs no
capability) and then `exec`s the step program in its place, so the filter holds before any of that program runs and
for everything it starts. The filter refuses, with `EACCES`, `socket()` for every address family but `AF_UNIX`, and
`io_uring_setup` (whose rings would bypass the filter); a system call of another architecture's ABI (i386 or x32 on
x86-64) is refused too. It is written as classic BPF through ctypes, for x86-64 and AArch64; on any other machine
the launcher refuses to start the child at all.

A network namespace is not possible here: the container drops every capability and runs with no-new-privileges, so
`unshare(CLONE_NEWNET)` is refused. The filter needs no privilege and holds whatever networks the container joins.

The launcher imports the standard library only: run isolated, it has no path to `gmshield_worker`.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import os
import platform
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

DEADLINE_SECONDS = 9 * 60 + 30
POLL_SECONDS = 0.2
# The launcher's own exit status when it could not sandbox the child: the step program never ran.
EXIT_UNSANDBOXED = 125

# prctl(2) and seccomp(2)
PR_SET_DUMPABLE = 4
PR_SET_SECCOMP = 22
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_KILL_PROCESS = 0x80000000
AF_UNIX = 1

# Classic BPF (linux/bpf_common.h): instruction classes, sizes, modes and sources
BPF_LD = 0x00
BPF_JMP = 0x05
BPF_RET = 0x06
BPF_W = 0x00
BPF_ABS = 0x20
BPF_JEQ = 0x10
BPF_JGE = 0x30
BPF_K = 0x00
BPF_LD_W_ABS = BPF_LD | BPF_W | BPF_ABS
BPF_JMP_JEQ_K = BPF_JMP | BPF_JEQ | BPF_K
BPF_JMP_JGE_K = BPF_JMP | BPF_JGE | BPF_K
BPF_RET_K = BPF_RET | BPF_K
# struct seccomp_data: int nr; __u32 arch; __u64 instruction_pointer; __u64 args[6] (little-endian low words).
OFFSET_NR = 0
OFFSET_ARCH = 4
OFFSET_ARG0 = 16
X32_SYSCALL_BIT = 0x40000000


@dataclass(frozen=True)
class Architecture:
    audit: int
    socket: int
    io_uring_setup: int
    # x86-64 also runs x32 system calls under its own audit architecture: refused.
    x32_bit: bool


ARCHITECTURES = {
    "x86_64": Architecture(audit=0xC000003E, socket=41, io_uring_setup=425, x32_bit=True),
    "aarch64": Architecture(audit=0xC00000B7, socket=198, io_uring_setup=425, x32_bit=False),
}


class SandboxUnavailable(Exception):
    """The filter cannot be installed here (an unknown machine, or the kernel refused): the child must not run."""


def _instruction(code: int, jt: int, jf: int, k: int) -> bytes:
    return struct.pack("=HBBI", code, jt, jf, k)


def filter_program(machine: str) -> bytes:
    """The filter for `machine` (`platform.machine()`), as the bytes of its `struct sock_filter` array."""
    arch = ARCHITECTURES.get(machine)
    if arch is None:
        raise SandboxUnavailable(machine)
    refused = SECCOMP_RET_ERRNO | errno.EACCES
    program = [
        # Another ABI than the machine's own: refused.
        _instruction(BPF_LD_W_ABS, 0, 0, OFFSET_ARCH),
        _instruction(BPF_JMP_JEQ_K, 1, 0, arch.audit),
        _instruction(BPF_RET_K, 0, 0, refused),
        _instruction(BPF_LD_W_ABS, 0, 0, OFFSET_NR),
    ]
    if arch.x32_bit:
        program += [_instruction(BPF_JMP_JGE_K, 0, 1, X32_SYSCALL_BIT), _instruction(BPF_RET_K, 0, 0, refused)]
    program += [
        _instruction(BPF_JMP_JEQ_K, 0, 1, arch.io_uring_setup),
        _instruction(BPF_RET_K, 0, 0, refused),
        _instruction(BPF_JMP_JEQ_K, 1, 0, arch.socket),
        _instruction(BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW),
        # socket(): only AF_UNIX.
        _instruction(BPF_LD_W_ABS, 0, 0, OFFSET_ARG0),
        _instruction(BPF_JMP_JEQ_K, 0, 1, AF_UNIX),
        _instruction(BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW),
        _instruction(BPF_RET_K, 0, 0, refused),
    ]
    return b"".join(program)


class _SockFprog(ctypes.Structure):
    _fields_ = (("len", ctypes.c_ushort), ("filter", ctypes.c_void_p))


def _prctl(option: int, argument: int, extra: int = 0) -> None:
    """`prctl(option, argument, extra, 0, 0)`; raises with the error's name when the kernel refuses."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(option, argument, extra, 0, 0) != 0:
        raise SandboxUnavailable(errno.errorcode.get(ctypes.get_errno(), "error"))


def install_filter(machine: str | None = None) -> None:
    """No new privileges, then the filter, on the calling process and everything it starts afterwards."""
    program = filter_program(machine or platform.machine())
    buffer = ctypes.create_string_buffer(program, len(program))
    fprog = _SockFprog(len(program) // 8, ctypes.addressof(buffer))
    _prctl(PR_SET_NO_NEW_PRIVS, 1)
    _prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.addressof(fprog))


def make_non_dumpable() -> None:
    """`PR_SET_DUMPABLE 0` on this process: no child (same user, no ptrace capability) can read its memory or
    `/proc/<pid>/environ`, which holds the worker's credential."""
    _prctl(PR_SET_DUMPABLE, 0)


def child_environment(folder: Path, environment: dict[str, str] | None = None) -> dict[str, str]:
    """`PATH` and `LANG` from the worker's environment when set, and `HOME` the step folder: nothing else."""
    source = os.environ if environment is None else environment
    kept = {name: source[name] for name in ("PATH", "LANG") if name in source}
    kept.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    kept["HOME"] = str(folder)
    return kept


@dataclass(frozen=True)
class ChildResult:
    """How a child ended: its exit status (negative: the signal), whether the deadline or a cancel killed it, its peak
    resident memory in KiB, and how long it ran. The peak starts from this process's resident size when it forked the
    child (Linux keeps the high-water mark across exec), so it is the child's own only when it is above that."""

    returncode: int
    timed_out: bool
    cancelled: bool
    max_rss_kib: int
    seconds: float


def run_child(
    script: Path,
    arguments: list[str],
    *,
    folder: Path,
    cancel: threading.Event,
    deadline_seconds: float = DEADLINE_SECONDS,
) -> ChildResult:
    """Runs `script` with `arguments` in the sandbox, in `folder`, until it ends, the deadline, or `cancel`."""
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), str(script), *arguments]
    started = time.monotonic()
    child = subprocess.Popen(  # noqa: S603 - a fixed interpreter and our own scripts, never a shell or a user's value
        command,
        cwd=folder,
        env=child_environment(folder),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    timed_out = cancelled = False
    while True:
        pid, status, usage = os.wait4(child.pid, os.WNOHANG)
        if pid == child.pid:
            break
        elapsed = time.monotonic() - started
        if not timed_out and not cancelled and (elapsed >= deadline_seconds or cancel.is_set()):
            timed_out, cancelled = elapsed >= deadline_seconds, not elapsed >= deadline_seconds
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
        time.sleep(POLL_SECONDS)
    # Whatever the child left running in its session goes with it.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, signal.SIGKILL)
    child.returncode = os.waitstatus_to_exitcode(status)
    return ChildResult(
        returncode=child.returncode,
        timed_out=timed_out,
        cancelled=cancelled,
        max_rss_kib=int(usage.ru_maxrss),
        seconds=time.monotonic() - started,
    )


def launch(argv: list[str]) -> int:
    """The launcher: the filter, then the step program in this process's place. Returns only when it could not."""
    if len(argv) < 2:
        return EXIT_UNSANDBOXED
    try:
        install_filter()
    except SandboxUnavailable:
        return EXIT_UNSANDBOXED
    os.execv(sys.executable, [sys.executable, "-I", "-B", *argv[1:]])  # noqa: S606 - the step program, isolated
    return EXIT_UNSANDBOXED  # pragma: no cover - execv does not return


if __name__ == "__main__":
    sys.exit(launch(sys.argv))
