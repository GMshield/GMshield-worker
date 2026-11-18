from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from gmshield_worker import workdir
from tests.conftest import log_lines

TRACE = "a" * 32


def make_dirty(work: Path, outside: Path) -> None:
    (outside / "keep.txt").write_text("outside", encoding="utf-8")
    (outside / "keep-dir").mkdir()
    (outside / "keep-dir" / "inner.txt").write_text("outside", encoding="utf-8")
    work.mkdir()
    (work / "page-001.bin").write_bytes(b"\x00" * 16)
    (work / ".hidden").write_text("x", encoding="utf-8")
    nested = work / "job" / "step" / "deep"
    nested.mkdir(parents=True)
    (nested / "piece.txt").write_text("x", encoding="utf-8")
    (work / "link-to-file").symlink_to(outside / "keep.txt")
    (work / "link-to-dir").symlink_to(outside / "keep-dir", target_is_directory=True)
    (work / "job" / "link-in-nested").symlink_to(outside / "keep-dir", target_is_directory=True)
    (work / "dangling").symlink_to(outside / "missing")


def test_empties_the_folder_but_never_follows_a_link(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    work, outside = tmp_path / "work", tmp_path / "outside"
    outside.mkdir()
    make_dirty(work, outside)

    assert workdir.empty_work_dir(str(work), TRACE) == 0

    assert work.is_dir()
    assert list(work.iterdir()) == []
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "outside"
    assert (outside / "keep-dir" / "inner.txt").read_text(encoding="utf-8") == "outside"
    lines = log_lines(capsys.readouterr().out)
    assert lines[-1]["event"] == "workdir.emptied"
    assert lines[-1]["failures"] == 0


def test_creates_a_missing_folder(tmp_path: Path) -> None:
    work = tmp_path / "a" / "work"
    assert workdir.empty_work_dir(str(work), TRACE) == 0
    assert work.is_dir()


def test_a_link_in_place_of_the_folder_is_not_followed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("outside", encoding="utf-8")
    work = tmp_path / "work"
    work.symlink_to(outside, target_is_directory=True)

    with pytest.raises(workdir.WorkDirUnusable):
        workdir.empty_work_dir(str(work), TRACE)

    assert (outside / "keep.txt").exists()
    assert log_lines(capsys.readouterr().err)[-1]["event"] == "workdir.not_a_directory"


def test_a_file_in_place_of_the_folder_is_unusable(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.write_text("x", encoding="utf-8")
    with pytest.raises(workdir.WorkDirUnusable):
        workdir.empty_work_dir(str(work), TRACE)
    assert work.read_text(encoding="utf-8") == "x"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_folder_that_cannot_be_created_is_unusable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parent = tmp_path / "locked"
    parent.mkdir(mode=0o500)
    try:
        with pytest.raises(workdir.WorkDirUnusable):
            workdir.empty_work_dir(str(parent / "work"), TRACE)
    finally:
        parent.chmod(0o700)
    line = log_lines(capsys.readouterr().err)[-1]
    assert line["event"] == "workdir.unusable"
    assert line["errorType"] == "PermissionError"
    assert "locked" not in str(line)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_folder_that_cannot_be_read_is_unusable(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir(mode=0o300)
    try:
        with pytest.raises(workdir.WorkDirUnusable):
            workdir.empty_work_dir(str(work), TRACE)
    finally:
        work.chmod(0o700)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_folder_entry_that_is_not_fully_removed_is_not_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    stuck = work / "job" / "stuck"
    stuck.mkdir(parents=True)
    (stuck / "piece").write_text("x", encoding="utf-8")
    (work / "file").write_text("x", encoding="utf-8")
    stuck.chmod(0o500)
    try:
        assert workdir.empty_work_dir(str(work), TRACE) >= 1
    finally:
        stuck.chmod(0o700)
    emptied = log_lines(capsys.readouterr().out)[-1]
    assert emptied["event"] == "workdir.emptied"
    assert emptied["removed"] == 1


def test_a_removal_error_is_logged_and_the_rest_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    for name in ("a.txt", "stuck.txt", "z.txt"):
        (work / name).write_text("secret content", encoding="utf-8")
    real_unlink = os.unlink

    def unlink(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        if str(path).endswith("stuck.txt"):
            raise PermissionError(13, "denied", str(path))
        real_unlink(path)

    monkeypatch.setattr(os, "unlink", unlink)

    assert workdir.empty_work_dir(str(work), TRACE) == 1

    assert sorted(p.name for p in work.iterdir()) == ["stuck.txt"]
    output = capsys.readouterr().out
    failure = next(line for line in log_lines(output) if line["event"] == "workdir.removal_failed")
    assert failure["errorType"] == "PermissionError"
    assert "stuck.txt" not in output
    assert "secret content" not in output


# ---- Story 1.15: a step's own folder and the sweep


def test_a_step_folder_is_private_and_erased_when_the_step_ends(tmp_path: Path) -> None:
    with workdir.step_folder(str(tmp_path), "job-1", TRACE) as folder:
        assert folder == tmp_path / "job-1"
        assert folder.stat().st_mode & 0o777 == 0o700
        (folder / "out").mkdir()
        (folder / "out" / "batch.pdf").write_bytes(b"%PDF")
    assert list(tmp_path.iterdir()) == []


def test_a_step_folder_is_erased_when_the_step_fails_and_what_an_earlier_attempt_left_goes_first(
    tmp_path: Path,
) -> None:
    (tmp_path / "job-1").mkdir()
    (tmp_path / "job-1" / "stale.pdf").write_bytes(b"left by a crash")
    with pytest.raises(RuntimeError, match="boom"), workdir.step_folder(str(tmp_path), "job-1", TRACE) as folder:
        assert list(folder.iterdir()) == []
        (folder / "source.pdf").write_bytes(b"%PDF")
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


def test_a_step_folder_is_one_plain_name_used_once_at_a_time(tmp_path: Path) -> None:
    for bad in ("", "a/b", "..", "."):
        with pytest.raises(ValueError, match="plain name"), workdir.step_folder(str(tmp_path), bad, TRACE):
            pass
    with workdir.step_folder(str(tmp_path), "job-1", TRACE):
        with pytest.raises(RuntimeError, match="in use"), workdir.step_folder(str(tmp_path), "job-1", TRACE):
            pass
        assert (tmp_path / "job-1").is_dir()
    assert not (tmp_path / "job-1").exists()


def test_the_sweep_removes_old_entries_but_never_a_running_step_or_a_recent_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = time.time() - 601
    for name in ("old-folder", "old-file", "running", "recent"):
        path = tmp_path / name
        if name.endswith("file"):
            path.write_bytes(b"x")
        else:
            path.mkdir()
            (path / "inner.pdf").write_bytes(b"x")
        if name != "recent":
            os.utime(path, (old, old))
    (tmp_path / "old-link").symlink_to(tmp_path / "recent")
    os.utime(tmp_path / "old-link", (old, old), follow_symlinks=False)

    with workdir.step_folder(str(tmp_path), "running", TRACE) as running:
        os.utime(running, (old, old))
        assert workdir.sweep_once(str(tmp_path), TRACE) == 3
        assert sorted(path.name for path in tmp_path.iterdir()) == ["recent", "running"]
    assert (tmp_path / "recent" / "inner.pdf").exists()
    assert log_lines(capsys.readouterr().out)[-1]["event"] == "workdir.swept"
    assert workdir.sweep_once(str(tmp_path / "missing"), TRACE) == 0


def test_the_sweep_thread_runs_on_its_interval_and_stops(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    stale.mkdir()
    sweep = workdir.WorkDirSweep(str(tmp_path), TRACE, every=0.05, older_than=0)
    sweep.start()
    deadline = time.monotonic() + 5
    while stale.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    sweep.stop()
    assert not stale.exists()
    assert workdir.SWEEP_EVERY_SECONDS == 120
    assert workdir.SWEEP_OLDER_THAN_SECONDS == 600
