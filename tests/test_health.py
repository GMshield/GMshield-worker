from __future__ import annotations

import os
from pathlib import Path

import pytest

from gmshield_worker import health


def test_marking(tmp_path: Path) -> None:
    path = tmp_path / "healthy"
    assert not health.is_healthy(path)
    health.mark_healthy(path)
    assert health.is_healthy(path)
    health.mark_unhealthy(path)
    assert not health.is_healthy(path)
    health.mark_unhealthy(path)  # already gone: no error


def test_a_stale_file_is_unhealthy(tmp_path: Path) -> None:
    path = tmp_path / "healthy"
    health.mark_healthy(path)
    old = path.stat().st_mtime - health.MAX_AGE_SECONDS - 1
    os.utime(path, (old, old))
    assert not health.is_healthy(path)


def test_the_command_reads_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "HEALTH_FILE", tmp_path / "healthy")
    assert health.main() == 1
    health.mark_healthy(tmp_path / "healthy")
    assert health.main() == 0


def test_a_file_from_the_future_is_unhealthy(tmp_path: Path) -> None:
    path = tmp_path / "healthy"
    health.mark_healthy(path)
    assert not health.is_healthy(path, now=path.stat().st_mtime - 60)


def test_the_default_file_is_outside_world_writable_folders() -> None:
    # SonarQube python:S5443: the image keeps it in /run/gmshield-worker, owned by the worker, mode 0700.
    if "GMSHIELD_HEALTH_FILE" not in os.environ:
        expected = Path("/run/gmshield-worker/healthy")
        assert expected == health.HEALTH_FILE
    assert not str(health.HEALTH_FILE).startswith(("/tmp/", "/var/tmp/", "/dev/shm/"))
