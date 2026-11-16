"""The worker's settings, read from the environment and checked before anything else starts.

- `GMSHIELD_MODE=test` (set only by the pipeline): no secret is needed, the model is the fake
  model, the backend is `http://stand-in-backend:8080` unless `GMSHIELD_BACKEND_URL` is set, Redis
  is `redis:6379` without an account, a password or TLS (`GMSHIELD_REDIS_HOST` and
  `GMSHIELD_REDIS_PORT` may move it), and the worker's credential is the fixed, public test-mode
  value that the backend's test mode knows.
- No mode: every setting in `REQUIRED_SETTINGS` and every secret in `REQUIRED_SECRETS` are
  required. Redis is reached over TLS verified against the private authority's certificate
  (`/run/gmshield/certs/ca.crt`) unless `GMSHIELD_REDIS_TLS=false`, which only the development
  setup sets.
- `GMSHIELD_WORK_DIR` is optional (default `/var/lib/gmshield-worker/work`, a folder of the image
  owned by the worker; production mounts the named volume `gmshield-worker-work` there, on the
  encrypted volume).
- `GMSHIELD_INTERACTIVE_CAPACITY` is optional: how many interactive jobs run at once (default 2,
  1 to 16). The ingestion lane runs one job at a time.
- `GMSHIELD_HEALTH_FILE` (read in `health.py`) is for tests that run the worker outside the image.

A refusal names settings and never holds a value.
"""

from __future__ import annotations

import re
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
WORKER_CREDENTIAL = "GMSHIELD_WORKER_CREDENTIAL"
REDIS_HOST = "GMSHIELD_REDIS_HOST"
REDIS_PORT = "GMSHIELD_REDIS_PORT"
REDIS_USERNAME = "GMSHIELD_REDIS_USERNAME"
REDIS_PASSWORD = "GMSHIELD_REDIS_PASSWORD"  # noqa: S105 - the setting's name, not a value
REDIS_TLS = "GMSHIELD_REDIS_TLS"
INTERACTIVE_CAPACITY = "GMSHIELD_INTERACTIVE_CAPACITY"

TEST_MODE_BACKEND_URL = "http://stand-in-backend:8080"
TEST_MODE_REDIS_HOST = "redis"
# The fixed, public credential of test mode, the same as the backend's test mode. Never a secret.
TEST_MODE_WORKER_CREDENTIAL = "gmshield-test-mode-worker-credential-not-a-secret"
DEFAULT_WORK_DIR = "/var/lib/gmshield-worker/work"
DEFAULT_REDIS_PORT = 6379
DEFAULT_INTERACTIVE_CAPACITY = 2
MAX_INTERACTIVE_CAPACITY = 16
# The private authority's certificate, mounted read-only by the pipeline (`certs = true`).
REDIS_CA_FILE = "/run/gmshield/certs/ca.crt"

# Required outside test mode, not secret.
REQUIRED_SETTINGS: tuple[str, ...] = (BACKEND_URL, REDIS_HOST, REDIS_USERNAME)
# Required outside test mode (Dalton, 2026-11-13). Story 1.12 adds the worker's credential and its
# Redis password, the first time the worker uses them.
REQUIRED_SECRETS: tuple[str, ...] = (ANTHROPIC_API_KEY, WORKER_CREDENTIAL, REDIS_PASSWORD)

# The backend accepts 32 to 512 header-safe characters (the pipeline writes 64 hex characters).
_CREDENTIAL = re.compile(r"[A-Za-z0-9._~+/=-]{32,512}", re.ASCII)
_HOST = re.compile(r"[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", re.ASCII)

type RefusalReason = Literal["mode.unknown", "setting.missing", "setting.invalid"]


class SettingsRefused(Exception):
    """The settings do not allow a start. Holds setting names only, never a value."""

    def __init__(self, reason: RefusalReason, settings: list[str]) -> None:
        super().__init__(f"{reason}: {', '.join(settings)}")
        self.reason: RefusalReason = reason
        self.settings = settings


@dataclass(frozen=True)
class RedisSettings:
    """Where the work streams are. Test mode: no account, no password, no TLS."""

    host: str = TEST_MODE_REDIS_HOST
    port: int = DEFAULT_REDIS_PORT
    username: str | None = None
    tls: bool = False
    # Never printed: excluded from repr.
    password: str | None = field(default=None, repr=False)
    ca_file: str = REDIS_CA_FILE


@dataclass(frozen=True)
class Settings:
    test_mode: bool
    backend_url: str
    work_dir: str
    redis: RedisSettings = field(default_factory=RedisSettings)
    interactive_capacity: int = DEFAULT_INTERACTIVE_CAPACITY
    # Never printed: excluded from repr.
    anthropic_api_key: str | None = field(default=None, repr=False)
    worker_credential: str = field(default=TEST_MODE_WORKER_CREDENTIAL, repr=False)


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


def _whole_number(value: str | None, default: int, low: int, high: int) -> int | None:
    """The number, the default when unset, or `None` when it is not a whole number in [low, high]."""
    if value is None or not _present(value):
        return default
    text = value.strip()
    if not text.isascii() or not text.isdigit():
        return None
    number = int(text)
    return number if low <= number <= high else None


def _redis_settings(environment: Mapping[str, str], test_mode: bool) -> RedisSettings:
    configured_host = environment.get(REDIS_HOST)
    host = configured_host.strip() if configured_host and _present(configured_host) else TEST_MODE_REDIS_HOST
    if not _HOST.fullmatch(host):
        raise SettingsRefused("setting.invalid", [REDIS_HOST])
    port = _whole_number(environment.get(REDIS_PORT), DEFAULT_REDIS_PORT, 1, 65535)
    if port is None:
        raise SettingsRefused("setting.invalid", [REDIS_PORT])
    if test_mode:
        return RedisSettings(host=host, port=port)
    tls = (environment.get(REDIS_TLS) or "").strip()
    if tls not in ("", "true", "false"):
        raise SettingsRefused("setting.invalid", [REDIS_TLS])
    return RedisSettings(
        host=host,
        port=port,
        username=environment[REDIS_USERNAME].strip(),
        password=environment[REDIS_PASSWORD].strip(),
        tls=tls != "false",
    )


def check_settings(environment: Mapping[str, str]) -> Settings:
    """The settings, or a `SettingsRefused` that names what is wrong."""
    mode = environment.get(MODE)
    test_mode = _present(mode)
    if test_mode and mode != TEST_MODE:
        raise SettingsRefused("mode.unknown", [MODE])

    configured_url = environment.get(BACKEND_URL)
    if not test_mode:
        missing = [name for name in (*REQUIRED_SETTINGS, *REQUIRED_SECRETS) if not _present(environment.get(name))]
        if missing:
            raise SettingsRefused("setting.missing", missing)

    backend_url = configured_url.strip() if configured_url and _present(configured_url) else TEST_MODE_BACKEND_URL
    if not _is_http_url(backend_url):
        raise SettingsRefused("setting.invalid", [BACKEND_URL])

    configured_dir = environment.get(WORK_DIR)
    work_dir = configured_dir.strip() if configured_dir and _present(configured_dir) else DEFAULT_WORK_DIR
    # Repeated slashes collapse ("//" is not "/" to PurePosixPath, but it is the root).
    parts = [part for part in work_dir.split("/") if part not in ("", ".")]
    path = PurePosixPath("/" + "/".join(parts))
    if not work_dir.startswith("/") or not parts or ".." in parts:
        raise SettingsRefused("setting.invalid", [WORK_DIR])

    credential = TEST_MODE_WORKER_CREDENTIAL
    if not test_mode:
        credential = environment[WORKER_CREDENTIAL].strip()
        # The public test-mode value would let anyone act as the worker: never a real credential.
        if not _CREDENTIAL.fullmatch(credential) or credential == TEST_MODE_WORKER_CREDENTIAL:
            raise SettingsRefused("setting.invalid", [WORKER_CREDENTIAL])

    capacity = _whole_number(
        environment.get(INTERACTIVE_CAPACITY), DEFAULT_INTERACTIVE_CAPACITY, 1, MAX_INTERACTIVE_CAPACITY
    )
    if capacity is None:
        raise SettingsRefused("setting.invalid", [INTERACTIVE_CAPACITY])

    return Settings(
        test_mode=test_mode,
        backend_url=backend_url.rstrip("/"),
        work_dir=str(path),
        redis=_redis_settings(environment, test_mode),
        interactive_capacity=capacity,
        anthropic_api_key=None if test_mode else environment[ANTHROPIC_API_KEY].strip(),
        worker_credential=credential,
    )
