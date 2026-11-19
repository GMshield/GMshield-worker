"""The worker's settings, read from the environment and checked before anything else starts.

- `GMSHIELD_MODE=test` (set only by the pipeline): no secret is needed, the model is the fake
  model whatever the provider (the worker states all three), the backend is
  `http://stand-in-backend:8080` unless `GMSHIELD_BACKEND_URL` is set, Redis is `redis:6379`
  without an account, a password or TLS (`GMSHIELD_REDIS_HOST` and `GMSHIELD_REDIS_PORT` may move
  it), and the worker's credential is the fixed, public test-mode
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
- Story 2.1, the index: `GMSHIELD_OLLAMA_URL` (embeddinggemma through Ollama, http or https without
  credentials), `GMSHIELD_QDRANT_URL` and the secret `GMSHIELD_QDRANT_API_KEY` are required outside
  test mode. An https Qdrant is verified against the private authority's certificate, as Redis is.
  Test mode's stand-ins: `http://stand-in-ollama:11434` and `http://stand-in-qdrant:6333` with no key,
  unless the variables are set.
- Story 2.10, the model providers' keys, all secrets and each optional: `GMSHIELD_AZURE_OPENAI_API_KEY` with
  `GMSHIELD_AZURE_OPENAI_BASE_URL` (the resource's `https://<host>/openai/v1/`; an Azure key without it is refused),
  `GMSHIELD_OPENAI_API_KEY` and `GMSHIELD_ANTHROPIC_API_KEY`. Outside test mode at least one key is required. The
  worker states at each claim the providers it holds a key for; the backend chooses the provider. Test mode reads none
  of them.

Before any of these, `check_tracing` refuses a start (`tracing.on`) when any trace export is on (Story 2.10): LangSmith
tracing, its OpenTelemetry export or its gateway, any OpenTelemetry exporter, or Anthropic's content capture. Prompts
and answers carry library text, which must never leave for a tracing service.

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
AZURE_OPENAI_API_KEY = "GMSHIELD_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_BASE_URL = "GMSHIELD_AZURE_OPENAI_BASE_URL"
OPENAI_API_KEY = "GMSHIELD_OPENAI_API_KEY"
WORKER_CREDENTIAL = "GMSHIELD_WORKER_CREDENTIAL"
REDIS_HOST = "GMSHIELD_REDIS_HOST"
REDIS_PORT = "GMSHIELD_REDIS_PORT"
REDIS_USERNAME = "GMSHIELD_REDIS_USERNAME"
REDIS_PASSWORD = "GMSHIELD_REDIS_PASSWORD"  # noqa: S105 - the setting's name, not a value
REDIS_TLS = "GMSHIELD_REDIS_TLS"
INTERACTIVE_CAPACITY = "GMSHIELD_INTERACTIVE_CAPACITY"
OLLAMA_URL = "GMSHIELD_OLLAMA_URL"
QDRANT_URL = "GMSHIELD_QDRANT_URL"
QDRANT_API_KEY = "GMSHIELD_QDRANT_API_KEY"

TEST_MODE_BACKEND_URL = "http://stand-in-backend:8080"
TEST_MODE_REDIS_HOST = "redis"
TEST_MODE_OLLAMA_URL = "http://stand-in-ollama:11434"
TEST_MODE_QDRANT_URL = "http://stand-in-qdrant:6333"
# The fixed, public credential of test mode, the same as the backend's test mode. Never a secret.
TEST_MODE_WORKER_CREDENTIAL = "gmshield-test-mode-worker-credential-not-a-secret"
DEFAULT_WORK_DIR = "/var/lib/gmshield-worker/work"
DEFAULT_REDIS_PORT = 6379
DEFAULT_INTERACTIVE_CAPACITY = 2
MAX_INTERACTIVE_CAPACITY = 16
# The private authority's certificate, mounted read-only by the pipeline (`certs = true`): Redis and Qdrant.
REDIS_CA_FILE = "/run/gmshield/certs/ca.crt"
CA_FILE = REDIS_CA_FILE

# Required outside test mode, not secret.
REQUIRED_SETTINGS: tuple[str, ...] = (BACKEND_URL, REDIS_HOST, REDIS_USERNAME, OLLAMA_URL, QDRANT_URL)
# Required outside test mode (Dalton, 2026-11-13). Story 1.12 adds the worker's credential and its
# Redis password, the first time the worker uses them; Story 2.1 the Qdrant API key. Story 2.10 drops the
# Anthropic key: each provider's key is optional, and at least one of `MODEL_KEYS` is required.
REQUIRED_SECRETS: tuple[str, ...] = (WORKER_CREDENTIAL, REDIS_PASSWORD, QDRANT_API_KEY)
# The model providers' keys (Story 2.10), in the order a refusal names them.
MODEL_KEYS: tuple[str, ...] = (AZURE_OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)
# The providers, as the backend names them; test mode states all three.
AZURE_OPENAI = "azure-openai"
ANTHROPIC = "anthropic"
OPENAI = "openai"
PROVIDERS: tuple[str, ...] = (AZURE_OPENAI, ANTHROPIC, OPENAI)

# The backend accepts 32 to 512 header-safe characters (the pipeline writes 64 hex characters).
_CREDENTIAL = re.compile(r"[A-Za-z0-9._~+/=-]{32,512}", re.ASCII)
_HOST = re.compile(r"[A-Za-z0-9]([A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", re.ASCII)
# A provider key goes into a header: visible ASCII only, without spaces.
_PROVIDER_KEY = re.compile(r"[\x21-\x7e]{1,512}", re.ASCII)
# The Azure OpenAI resource's v1 API: https://<host>/openai/v1/ (Story 2.10). No api-version is ever sent.
AZURE_OPENAI_PATH = "/openai/v1/"

type RefusalReason = Literal["mode.unknown", "setting.missing", "setting.invalid", "tracing.on"]

# Trace exports (Story 2.10): on when truthy, when set, or when not `none`.
# LangSmith reads each switch under both prefixes (langsmith 0.14.16: `TRACING_V2`, else `TRACING`; `OTEL_ENABLED`;
# `OTEL_ONLY`).
TRACING_FLAGS: tuple[str, ...] = tuple(
    f"{prefix}_{name}"
    for prefix in ("LANGSMITH", "LANGCHAIN")
    for name in ("TRACING", "TRACING_V2", "OTEL_ENABLED", "OTEL_ONLY")
)
TRACING_GATEWAY = "LANGSMITH_GATEWAY"
OTEL_EXPORTERS: tuple[str, ...] = ("OTEL_TRACES_EXPORTER", "OTEL_METRICS_EXPORTER", "OTEL_LOGS_EXPORTER")
OTLP_PREFIX = "OTEL_EXPORTER_OTLP_"
ANTHROPIC_CONTENT_MODE = "ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE"
_FALSE = frozenset({"", "false", "0", "no", "off"})


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
class IndexSettings:
    """Where the index's two services are (Story 2.1). Test mode: the stand-ins, no key."""

    ollama_url: str = TEST_MODE_OLLAMA_URL
    qdrant_url: str = TEST_MODE_QDRANT_URL
    # Never printed: excluded from repr.
    qdrant_api_key: str | None = field(default=None, repr=False)
    ca_file: str = CA_FILE


@dataclass(frozen=True)
class ModelKeys:
    """The model providers' keys (Story 2.10). Each is optional; the Azure base URL comes with the Azure key. Test mode
    holds none and states every provider, because the fake model answers for each."""

    test_mode: bool = False
    # Never printed: excluded from repr.
    azure_openai_api_key: str | None = field(default=None, repr=False)
    azure_openai_base_url: str | None = None
    openai_api_key: str | None = field(default=None, repr=False)
    anthropic_api_key: str | None = field(default=None, repr=False)

    @property
    def providers(self) -> tuple[str, ...]:
        """The providers this worker holds a key for, as it states them at each claim."""
        if self.test_mode:
            return PROVIDERS
        held = {
            AZURE_OPENAI: self.azure_openai_api_key is not None and self.azure_openai_base_url is not None,
            ANTHROPIC: self.anthropic_api_key is not None,
            OPENAI: self.openai_api_key is not None,
        }
        return tuple(name for name in PROVIDERS if held[name])


@dataclass(frozen=True)
class Settings:
    test_mode: bool
    backend_url: str
    work_dir: str
    redis: RedisSettings = field(default_factory=RedisSettings)
    index: IndexSettings = field(default_factory=IndexSettings)
    interactive_capacity: int = DEFAULT_INTERACTIVE_CAPACITY
    models: ModelKeys = field(default_factory=ModelKeys)
    # Never printed: excluded from repr.
    worker_credential: str = field(default=TEST_MODE_WORKER_CREDENTIAL, repr=False)


def _present(value: str | None) -> bool:
    return value is not None and value.strip() != ""


def check_tracing(environment: Mapping[str, str]) -> None:
    """Refuses (`tracing.on`, naming the variables) when any trace export is on, in every mode:

    - any of `LANGSMITH_` or `LANGCHAIN_` followed by `TRACING`, `TRACING_V2`, `OTEL_ENABLED` or `OTEL_ONLY` (eight
      variables) is truthy (anything but empty, `false`, `0`, `no` or `off`, in any case);
    - `LANGSMITH_GATEWAY` is set (not blank);
    - `OTEL_TRACES_EXPORTER`, `OTEL_METRICS_EXPORTER` or `OTEL_LOGS_EXPORTER` is anything other than empty or `none`;
    - any `OTEL_EXPORTER_OTLP_*` variable is set (not blank);
    - `ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE` is `content`.
    """
    on = [name for name in TRACING_FLAGS if (environment.get(name) or "").strip().lower() not in _FALSE]
    if _present(environment.get(TRACING_GATEWAY)):
        on.append(TRACING_GATEWAY)
    on += [name for name in OTEL_EXPORTERS if (environment.get(name) or "").strip().lower() not in ("", "none")]
    on += sorted(name for name, value in environment.items() if name.startswith(OTLP_PREFIX) and _present(value))
    if (environment.get(ANTHROPIC_CONTENT_MODE) or "").strip().lower() == "content":
        on.append(ANTHROPIC_CONTENT_MODE)
    if on:
        raise SettingsRefused("tracing.on", on)


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


def _url(environment: Mapping[str, str], name: str, default: str) -> str:
    """The setting's URL (http or https, without credentials), or the default when unset; refused otherwise."""
    configured = environment.get(name)
    url = configured.strip() if configured and _present(configured) else default
    if not _is_http_url(url):
        raise SettingsRefused("setting.invalid", [name])
    return url.rstrip("/")


def _index_settings(environment: Mapping[str, str], test_mode: bool) -> IndexSettings:
    ollama = _url(environment, OLLAMA_URL, TEST_MODE_OLLAMA_URL)
    qdrant = _url(environment, QDRANT_URL, TEST_MODE_QDRANT_URL)
    if test_mode:
        return IndexSettings(ollama_url=ollama, qdrant_url=qdrant)
    return IndexSettings(ollama_url=ollama, qdrant_url=qdrant, qdrant_api_key=environment[QDRANT_API_KEY].strip())


def _azure_base_url(value: str) -> str | None:
    """The Azure OpenAI v1 URL as `https://<host>[:port]/openai/v1/`, or `None` when it is not one."""
    if not _is_http_url(value):
        return None
    url = urlsplit(value)
    if url.scheme != "https" or url.path.rstrip("/") + "/" != AZURE_OPENAI_PATH:
        return None
    return f"https://{url.netloc}{AZURE_OPENAI_PATH}"


def _missing_model_keys(environment: Mapping[str, str]) -> list[str]:
    """Outside test mode: every key when none is set, the Azure base URL when the Azure key comes without it."""
    if not any(_present(environment.get(name)) for name in MODEL_KEYS):
        return list(MODEL_KEYS)
    if _present(environment.get(AZURE_OPENAI_API_KEY)) and not _present(environment.get(AZURE_OPENAI_BASE_URL)):
        return [AZURE_OPENAI_BASE_URL]
    return []


def _model_keys(environment: Mapping[str, str], test_mode: bool) -> ModelKeys:
    """The keys that are set, each checked; the Azure base URL whenever it is set (it is refused when invalid)."""
    if test_mode:
        return ModelKeys(test_mode=True)
    keys: dict[str, str | None] = {}
    for name in MODEL_KEYS:
        value = environment.get(name)
        key = value.strip() if value is not None and _present(value) else None
        if key is not None and not _PROVIDER_KEY.fullmatch(key):
            raise SettingsRefused("setting.invalid", [name])
        keys[name] = key
    configured_url = environment.get(AZURE_OPENAI_BASE_URL)
    base_url = None
    if configured_url is not None and _present(configured_url):
        base_url = _azure_base_url(configured_url.strip())
        if base_url is None:
            raise SettingsRefused("setting.invalid", [AZURE_OPENAI_BASE_URL])
    return ModelKeys(
        azure_openai_api_key=keys[AZURE_OPENAI_API_KEY],
        azure_openai_base_url=base_url if keys[AZURE_OPENAI_API_KEY] is not None else None,
        openai_api_key=keys[OPENAI_API_KEY],
        anthropic_api_key=keys[ANTHROPIC_API_KEY],
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
        missing += _missing_model_keys(environment)
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
        index=_index_settings(environment, test_mode),
        interactive_capacity=capacity,
        models=_model_keys(environment, test_mode),
        worker_credential=credential,
    )
