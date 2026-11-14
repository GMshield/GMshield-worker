"""The worker's settings, read from the environment and checked before anything else starts.

- `GMSHIELD_MODE=test` (set only by the pipeline): no secret is needed, the model is the fake
  model, and the backend is `http://stand-in-backend:8080` unless `GMSHIELD_BACKEND_URL` is set.
- No mode: `GMSHIELD_BACKEND_URL` and every secret in `REQUIRED_SECRETS` are required.
- `GMSHIELD_WORK_DIR` is optional (default `/tmp/gmshield-work`; Story 1.5 moves it onto the
  encrypted volume).

A refusal names settings and never holds a value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

MODE = "GMSHIELD_MODE"
TEST_MODE = "test"
BACKEND_URL = "GMSHIELD_BACKEND_URL"
WORK_DIR = "GMSHIELD_WORK_DIR"
ANTHROPIC_API_KEY = "GMSHIELD_ANTHROPIC_API_KEY"

TEST_MODE_BACKEND_URL = "http://stand-in-backend:8080"
DEFAULT_WORK_DIR = "/tmp/gmshield-work"  # noqa: S108 - the only writable place; Story 1.5 moves it

# Required outside test mode (Dalton, 2026-11-13). The worker's service credential and its Redis
# credentials join this list in Story 1.12, which first uses them.
REQUIRED_SECRETS: tuple[str, ...] = (ANTHROPIC_API_KEY,)

type RefusalReason = Literal["mode.unknown", "setting.missing", "setting.invalid"]


class SettingsRefused(Exception):
    """The settings do not allow a start. Holds setting names only, never a value."""

    def __init__(self, reason: RefusalReason, settings: list[str]) -> None:
        super().__init__(f"{reason}: {', '.join(settings)}")
        self.reason: RefusalReason = reason
        self.settings = settings


@dataclass(frozen=True)
class Settings:
    test_mode: bool
    backend_url: str
    work_dir: str
    # Never printed: excluded from repr.
    anthropic_api_key: str | None = field(default=None, repr=False)


def _present(value: str | None) -> bool:
    return value is not None and value.strip() != ""


def _is_http_url(value: str) -> bool:
    try:
        url = urlsplit(value)
        port = url.port  # raises on an invalid port
    except ValueError:
        return False
    del port
    return (
        url.scheme in ("http", "https")
        and bool(url.hostname)
        and url.username is None
        and url.password is None
        and not url.query
        and not url.fragment
    )


def check_settings(environment: Mapping[str, str]) -> Settings:
    """The settings, or a `SettingsRefused` that names what is wrong."""
    mode = environment.get(MODE)
    test_mode = _present(mode)
    if test_mode and mode != TEST_MODE:
        raise SettingsRefused("mode.unknown", [MODE])

    configured_url = environment.get(BACKEND_URL)
    if not test_mode:
        missing = [name for name in (BACKEND_URL, *REQUIRED_SECRETS) if not _present(environment.get(name))]
        if missing:
            raise SettingsRefused("setting.missing", missing)

    backend_url = configured_url.strip() if configured_url and _present(configured_url) else TEST_MODE_BACKEND_URL
    if not _is_http_url(backend_url):
        raise SettingsRefused("setting.invalid", [BACKEND_URL])

    configured_dir = environment.get(WORK_DIR)
    work_dir = configured_dir.strip() if configured_dir and _present(configured_dir) else DEFAULT_WORK_DIR
    path = PurePosixPath(work_dir)
    if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
        raise SettingsRefused("setting.invalid", [WORK_DIR])

    return Settings(
        test_mode=test_mode,
        backend_url=backend_url.rstrip("/"),
        work_dir=str(path),
        anthropic_api_key=None if test_mode else environment[ANTHROPIC_API_KEY].strip(),
    )
