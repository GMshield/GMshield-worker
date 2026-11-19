from __future__ import annotations

import pytest

from gmshield_worker.settings import (
    ANTHROPIC_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_BASE_URL,
    BACKEND_URL,
    DEFAULT_WORK_DIR,
    INTERACTIVE_CAPACITY,
    MODE,
    OLLAMA_URL,
    OPENAI_API_KEY,
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
    ModelKeys,
    SettingsRefused,
    check_settings,
    check_tracing,
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
    assert settings.models.anthropic_api_key == "sk-secret-value"
    assert settings.models.providers == ("anthropic",)
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
                WORKER_CREDENTIAL,
                REDIS_PASSWORD,
                QDRANT_API_KEY,
                AZURE_OPENAI_API_KEY,
                ANTHROPIC_API_KEY,
                OPENAI_API_KEY,
            ],
        ),
        ({**COMPLETE, QDRANT_API_KEY: " "}, [QDRANT_API_KEY]),
        ({k: v for k, v in COMPLETE.items() if k not in (OLLAMA_URL, QDRANT_URL)}, [OLLAMA_URL, QDRANT_URL]),
        ({**COMPLETE, ANTHROPIC_API_KEY: ""}, [AZURE_OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY]),
        ({**COMPLETE, ANTHROPIC_API_KEY: "", AZURE_OPENAI_API_KEY: "azure-secret-value"}, [AZURE_OPENAI_BASE_URL]),
        ({**COMPLETE, AZURE_OPENAI_API_KEY: "k", AZURE_OPENAI_BASE_URL: " "}, [AZURE_OPENAI_BASE_URL]),
        ({k: v for k, v in COMPLETE.items() if k != BACKEND_URL}, [BACKEND_URL]),
        ({**COMPLETE, ANTHROPIC_API_KEY: "   "}, [AZURE_OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY]),
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
    assert settings.models == ModelKeys(test_mode=True)
    assert settings.models.providers == ("azure-openai", "anthropic", "openai")
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
    settings = check_settings(
        {MODE: "test", BACKEND_URL: "http://127.0.0.1:9", ANTHROPIC_API_KEY: "sk-x", AZURE_OPENAI_BASE_URL: "bad"}
    )
    assert settings.backend_url == "http://127.0.0.1:9"
    assert settings.models.anthropic_api_key is None
    assert settings.models.azure_openai_api_key is None


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


# ---- Story 2.10: the model providers' keys and the tracing refusal

AZURE = {
    **{k: v for k, v in COMPLETE.items() if k != ANTHROPIC_API_KEY},
    AZURE_OPENAI_API_KEY: "azure-secret-value",
    AZURE_OPENAI_BASE_URL: "https://gmshield-test.openai.azure.com/openai/v1/",
}


def test_an_azure_key_with_its_base_url_is_the_only_provider_and_never_printed() -> None:
    settings = check_settings(AZURE)
    models = settings.models
    assert models.providers == ("azure-openai",)
    assert models.azure_openai_base_url == "https://gmshield-test.openai.azure.com/openai/v1/"
    assert (models.anthropic_api_key, models.openai_api_key) == (None, None)
    assert "azure-secret-value" not in repr(settings)
    # Without the trailing slash the URL is the same one.
    trimmed = check_settings({**AZURE, AZURE_OPENAI_BASE_URL: "https://gmshield-test.openai.azure.com/openai/v1"})
    assert trimmed.models.azure_openai_base_url == "https://gmshield-test.openai.azure.com/openai/v1/"


def test_every_key_set_states_every_provider() -> None:
    settings = check_settings({**AZURE, ANTHROPIC_API_KEY: "sk-ant", OPENAI_API_KEY: "sk-openai"})
    assert settings.models.providers == ("azure-openai", "anthropic", "openai")
    for secret in ("azure-secret-value", "sk-ant", "sk-openai"):
        assert secret not in repr(settings)


@pytest.mark.parametrize(
    "url",
    [
        "http://gmshield-test.openai.azure.com/openai/v1/",
        "https://gmshield-test.openai.azure.com/",
        "https://gmshield-test.openai.azure.com/openai/",
        "https://gmshield-test.openai.azure.com/openai/deployments/x/",
        "https://gmshield-test.openai.azure.com/openai/v1/?api-version=2024-10-21",
        "https://user:pass@gmshield-test.openai.azure.com/openai/v1/",
        "gmshield-test.openai.azure.com/openai/v1/",
    ],
)
def test_an_azure_base_url_that_is_not_the_v1_api_is_refused_by_name(url: str) -> None:
    with pytest.raises(SettingsRefused) as refused:
        check_settings({**AZURE, AZURE_OPENAI_BASE_URL: url})
    assert (refused.value.reason, refused.value.settings) == ("setting.invalid", [AZURE_OPENAI_BASE_URL])
    assert url not in str(refused.value)


@pytest.mark.parametrize("name", [AZURE_OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY])
def test_a_key_that_cannot_go_in_a_header_is_refused_by_name(name: str) -> None:
    with pytest.raises(SettingsRefused) as refused:
        check_settings({**AZURE, name: "two words"})
    assert (refused.value.reason, refused.value.settings) == ("setting.invalid", [name])


@pytest.mark.parametrize(
    ("environment", "named"),
    [
        ({"LANGSMITH_TRACING": "true"}, ["LANGSMITH_TRACING"]),
        ({"LANGSMITH_TRACING": "Yes", "LANGCHAIN_TRACING_V2": "1"}, ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"]),
        ({"LANGSMITH_OTEL_ENABLED": "on"}, ["LANGSMITH_OTEL_ENABLED"]),
        ({"LANGSMITH_TRACING_V2": "true"}, ["LANGSMITH_TRACING_V2"]),
        ({"LANGSMITH_OTEL_ONLY": "true"}, ["LANGSMITH_OTEL_ONLY"]),
        ({"LANGCHAIN_TRACING": "true"}, ["LANGCHAIN_TRACING"]),
        ({"LANGCHAIN_OTEL_ENABLED": "1"}, ["LANGCHAIN_OTEL_ENABLED"]),
        ({"LANGCHAIN_OTEL_ONLY": "yes"}, ["LANGCHAIN_OTEL_ONLY"]),
        ({"LANGSMITH_GATEWAY": "https://gateway"}, ["LANGSMITH_GATEWAY"]),
        ({"OTEL_METRICS_EXPORTER": "console"}, ["OTEL_METRICS_EXPORTER"]),
        ({"OTEL_LOGS_EXPORTER": "otlp", "OTEL_TRACES_EXPORTER": "NONE"}, ["OTEL_LOGS_EXPORTER"]),
        (
            {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://c", "OTEL_EXPORTER_OTLP_HEADERS": "a=b"},
            ["OTEL_EXPORTER_OTLP_HEADERS", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"],
        ),
        ({"ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE": "Content"}, ["ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE"]),
    ],
)
def test_any_trace_export_is_refused_naming_the_variables(environment: dict[str, str], named: list[str]) -> None:
    with pytest.raises(SettingsRefused) as refused:
        check_tracing({**COMPLETE, **environment})
    assert (refused.value.reason, refused.value.settings) == ("tracing.on", named)


def test_trace_settings_that_export_nothing_are_accepted() -> None:
    check_tracing(
        {
            **COMPLETE,
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "",
            "LANGSMITH_OTEL_ENABLED": "0",
            "LANGSMITH_OTEL_ONLY": "false",
            "LANGCHAIN_TRACING": "off",
            "LANGSMITH_GATEWAY": " ",
            "OTEL_TRACES_EXPORTER": "none",
            "OTEL_METRICS_EXPORTER": "",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "",
            "ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE": "metadata",
            "OTEL_SERVICE_NAME": "gmshield-worker",
        }
    )
