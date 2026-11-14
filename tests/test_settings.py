from __future__ import annotations

import pytest

from gmshield_worker.settings import (
    ANTHROPIC_API_KEY,
    BACKEND_URL,
    DEFAULT_WORK_DIR,
    MODE,
    TEST_MODE_BACKEND_URL,
    WORK_DIR,
    SettingsRefused,
    check_settings,
)

COMPLETE = {BACKEND_URL: "http://backend:8080/", ANTHROPIC_API_KEY: "sk-secret-value"}


def test_complete_settings_start() -> None:
    settings = check_settings(COMPLETE)
    assert not settings.test_mode
    assert settings.backend_url == "http://backend:8080"
    assert settings.work_dir == DEFAULT_WORK_DIR
    assert settings.anthropic_api_key == "sk-secret-value"
    assert "sk-secret-value" not in repr(settings)


@pytest.mark.parametrize(
    ("environment", "missing"),
    [
        ({}, [BACKEND_URL, ANTHROPIC_API_KEY]),
        ({BACKEND_URL: "http://backend:8080"}, [ANTHROPIC_API_KEY]),
        ({ANTHROPIC_API_KEY: "sk-secret-value"}, [BACKEND_URL]),
        ({BACKEND_URL: "http://backend:8080", ANTHROPIC_API_KEY: "   "}, [ANTHROPIC_API_KEY]),
    ],
)
def test_missing_settings_are_refused_by_name(environment: dict[str, str], missing: list[str]) -> None:
    with pytest.raises(SettingsRefused) as refused:
        check_settings(environment)
    assert refused.value.reason == "setting.missing"
    assert refused.value.settings == missing
    assert "sk-secret-value" not in str(refused.value)


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


def test_test_mode_ignores_a_given_key_and_honours_the_backend_url() -> None:
    settings = check_settings({MODE: "test", BACKEND_URL: "http://127.0.0.1:9", ANTHROPIC_API_KEY: "sk-x"})
    assert settings.backend_url == "http://127.0.0.1:9"
    assert settings.anthropic_api_key is None


def test_work_dir_is_configurable() -> None:
    assert check_settings({**COMPLETE, WORK_DIR: "/data/work/"}).work_dir == "/data/work"
    assert check_settings({**COMPLETE, WORK_DIR: "//data//work"}).work_dir == "/data/work"
