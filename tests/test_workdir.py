from __future__ import annotations

import os
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

    assert workdir.empty_work_dir(str(work), TRACE) == 1

    assert (outside / "keep.txt").exists()
    assert log_lines(capsys.readouterr().err)[-1]["event"] == "workdir.not_a_directory"


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
