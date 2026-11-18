from __future__ import annotations

import pytest

from gmshield_worker.settings import (
    ANTHROPIC_API_KEY,
    BACKEND_URL,
    DEFAULT_WORK_DIR,
    INTERACTIVE_CAPACITY,
    MODE,
    OLLAMA_URL,
    QDRANT_API_KEY,
    QDRANT_URL,
    REDIS_CA_FILE,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_TLS,
    REDIS_USERNAME,
    TEST_MODE_BACKEND_URL,
    TEST_MODE_OLLAMA_URL,
    TEST_MODE_QDRANT_URL,
    TEST_MODE_WORKER_CREDENTIAL,
    WORK_DIR,
    WORKER_CREDENTIAL,
    SettingsRefused,
    check_settings,
)

CREDENTIAL = "c0ffee" * 10 + "beef"
COMPLETE = {
    BACKEND_URL: "http://backend:8080/",
    ANTHROPIC_API_KEY: "sk-secret-value",
    WORKER_CREDENTIAL: CREDENTIAL,
    REDIS_HOST: "redis",
    REDIS_USERNAME: "worker",
    REDIS_PASSWORD: "redis-secret-value",
    OLLAMA_URL: "http://172.30.5.1:11434/",
    QDRANT_URL: "https://qdrant:6333",
    QDRANT_API_KEY: "qdrant-secret-value",
}
SECRETS = ("sk-secret-value", CREDENTIAL, "redis-secret-value", "qdrant-secret-value")


def test_complete_settings_start() -> None:
    settings = check_settings(COMPLETE)
    assert not settings.test_mode
    assert settings.backend_url == "http://backend:8080"
    assert settings.work_dir == DEFAULT_WORK_DIR
    assert settings.anthropic_api_key == "sk-secret-value"
    assert settings.worker_credential == CREDENTIAL
    assert settings.interactive_capacity == 2
    redis = settings.redis
    assert (redis.host, redis.port, redis.username, redis.password) == ("redis", 6379, "worker", "redis-secret-value")
    assert redis.tls
    assert redis.ca_file == REDIS_CA_FILE == "/run/gmshield/certs/ca.crt"
    index = settings.index
    assert (index.ollama_url, index.qdrant_url, index.qdrant_api_key) == (
        "http://172.30.5.1:11434",
        "https://qdrant:6333",
        "qdrant-secret-value",
    )
    assert index.ca_file == REDIS_CA_FILE
    for secret in SECRETS:
        assert secret not in repr(settings)


@pytest.mark.parametrize(
    ("environment", "missing"),
    [
        (
            {},
            [
                BACKEND_URL,
                REDIS_HOST,
                REDIS_USERNAME,
                OLLAMA_URL,
                QDRANT_URL,
                ANTHROPIC_API_KEY,
                WORKER_CREDENTIAL,
                REDIS_PASSWORD,
                QDRANT_API_KEY,
            ],
        ),
        ({**COMPLETE, QDRANT_API_KEY: " "}, [QDRANT_API_KEY]),
        ({k: v for k, v in COMPLETE.items() if k not in (OLLAMA_URL, QDRANT_URL)}, [OLLAMA_URL, QDRANT_URL]),
        ({**COMPLETE, ANTHROPIC_API_KEY: ""}, [ANTHROPIC_API_KEY]),
        ({k: v for k, v in COMPLETE.items() if k != BACKEND_URL}, [BACKEND_URL]),
        ({**COMPLETE, ANTHROPIC_API_KEY: "   "}, [ANTHROPIC_API_KEY]),
        ({**COMPLETE, WORKER_CREDENTIAL: " "}, [WORKER_CREDENTIAL]),
        ({k: v for k, v in COMPLETE.items() if k not in (REDIS_HOST, REDIS_PASSWORD)}, [REDIS_HOST, REDIS_PASSWORD]),
    ],
)
def test_missing_settings_are_refused_by_name(environment: dict[str, str], missing: list[str]) -> None:
    with pytest.raises(SettingsRefused) as refused:
        check_settings(environment)
    assert refused.value.reason == "setting.missing"
    assert refused.value.settings == missing
    for secret in SECRETS:
        assert secret not in str(refused.value)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://backend",
        "backend:8080",
        "http://user:pass@backend:8080",
        "http://",
        "http://backend:99999",
        "http://b/?q=1",
    ],
)
def test_invalid_backend_url_is_refused_without_its_value(url: str) -> None:
    with pytest.raises(SettingsRefused) as refused:
        check_settings({**COMPLETE, BACKEND_URL: url})
    assert refused.value.reason == "setting.invalid"
    assert refused.value.settings == [BACKEND_URL]
    assert str(refused.value) == f"setting.invalid: {BACKEND_URL}"


@pytest.mark.parametrize("work_dir", ["relative/dir", "/", "//", "///", "///x/..", "/./", "/tmp/../etc"])
def test_invalid_work_dir_is_refused(work_dir: str) -> None:
    with pytest.raises(SettingsRefused) as refused:
        check_settings({**COMPLETE, WORK_DIR: work_dir})
    assert refused.value.settings == [WORK_DIR]


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(SettingsRefused) as refused:
        check_settings({**COMPLETE, MODE: "production"})
    assert refused.value.reason == "mode.unknown"


def test_test_mode_needs_no_secret_and_uses_the_stand_in() -> None:
    settings = check_settings({MODE: "test"})
    assert settings.test_mode
    assert settings.backend_url == TEST_MODE_BACKEND_URL
    assert settings.anthropic_api_key is None
    assert settings.worker_credential == TEST_MODE_WORKER_CREDENTIAL
    redis = settings.redis
    assert (redis.host, redis.port, redis.username, redis.password, redis.tls) == ("redis", 6379, None, None, False)
    index = settings.index
    assert (index.ollama_url, index.qdrant_url, index.qdrant_api_key) == (
        TEST_MODE_OLLAMA_URL,
        TEST_MODE_QDRANT_URL,
        None,
    )


def test_test_mode_ignores_redis_credentials_and_tls_but_may_move_redis() -> None:
    settings = check_settings(
        {
            MODE: "test",
            REDIS_HOST: "127.0.0.1",
            REDIS_PORT: "16379",
            REDIS_USERNAME: "u",
            REDIS_PASSWORD: "p",
            REDIS_TLS: "true",
            WORKER_CREDENTIAL: CREDENTIAL,
        }
    )
    redis = settings.redis
    assert (redis.host, redis.port, redis.username, redis.password, redis.tls) == (
        "127.0.0.1",
        16379,
        None,
        None,
        False,
    )
    assert settings.worker_credential == TEST_MODE_WORKER_CREDENTIAL


def test_redis_tls_can_be_switched_off_for_development_only_by_false() -> None:
    assert not check_settings({**COMPLETE, REDIS_TLS: "false"}).redis.tls
    assert check_settings({**COMPLETE, REDIS_TLS: "true"}).redis.tls
    with pytest.raises(SettingsRefused) as refused:
        check_settings({**COMPLETE, REDIS_TLS: "no"})
    assert refused.value.settings == [REDIS_TLS]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (WORKER_CREDENTIAL, "short"),
        (WORKER_CREDENTIAL, "has space " + "a" * 32),
        (WORKER_CREDENTIAL, TEST_MODE_WORKER_CREDENTIAL),
        (REDIS_HOST, "bad host"),
        (REDIS_HOST, "-leading"),
        (REDIS_PORT, "0"),
        (REDIS_PORT, "65536"),
        (REDIS_PORT, "six"),
        (INTERACTIVE_CAPACITY, "0"),
        (INTERACTIVE_CAPACITY, "17"),
        (INTERACTIVE_CAPACITY, "two"),
    ],
)
def test_invalid_values_are_refused_by_name_never_by_value(name: str, value: str) -> None:
    with pytest.raises(SettingsRefused) as refused:
        check_settings({**COMPLETE, name: value})
    assert refused.value.reason == "setting.invalid"
    assert refused.value.settings == [name]
    assert value not in str(refused.value)


def test_the_interactive_capacity_is_configurable() -> None:
    assert check_settings({**COMPLETE, INTERACTIVE_CAPACITY: "4"}).interactive_capacity == 4
    assert check_settings({MODE: "test", INTERACTIVE_CAPACITY: "1"}).interactive_capacity == 1


def test_test_mode_ignores_a_given_key_and_honours_the_backend_url() -> None:
    settings = check_settings({MODE: "test", BACKEND_URL: "http://127.0.0.1:9", ANTHROPIC_API_KEY: "sk-x"})
    assert settings.backend_url == "http://127.0.0.1:9"
    assert settings.anthropic_api_key is None


@pytest.mark.parametrize("name", [OLLAMA_URL, QDRANT_URL])
@pytest.mark.parametrize("url", ["ftp://ollama", "ollama:11434", "http://user:pass@ollama:11434", "http://o/?q=1"])
def test_an_invalid_index_url_is_refused_by_name_never_by_value(name: str, url: str) -> None:
    for environment in (COMPLETE, {MODE: "test"}):
        with pytest.raises(SettingsRefused) as refused:
            check_settings({**environment, name: url})
        assert refused.value.settings == [name]
        assert url not in str(refused.value)


def test_test_mode_may_move_the_index_services_but_ignores_a_qdrant_key() -> None:
    settings = check_settings(
        {MODE: "test", OLLAMA_URL: "http://127.0.0.1:1", QDRANT_URL: "http://127.0.0.1:2", QDRANT_API_KEY: "k"}
    )
    assert (settings.index.ollama_url, settings.index.qdrant_url, settings.index.qdrant_api_key) == (
        "http://127.0.0.1:1",
        "http://127.0.0.1:2",
        None,
    )


def test_work_dir_is_configurable() -> None:
    assert check_settings({**COMPLETE, WORK_DIR: "/data/work/"}).work_dir == "/data/work"
    assert check_settings({**COMPLETE, WORK_DIR: "//data//work"}).work_dir == "/data/work"
