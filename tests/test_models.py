from __future__ import annotations

import socket

import pytest

from gmshield_worker.models import select_model_client
from gmshield_worker.models.claude import DEFAULT_MODEL, ClaudeModelClient
from gmshield_worker.models.fake import FakeModelClient
from gmshield_worker.settings import Settings, check_settings


def test_the_fake_model_is_deterministic() -> None:
    fake = FakeModelClient()
    assert fake.complete("system", "prompt") == fake.complete("system", "prompt")
    assert fake.complete("system", "prompt") != fake.complete("system", "other")
    assert fake.kind == "fake"


def test_test_mode_always_selects_the_fake_model() -> None:
    settings = check_settings({"GMSHIELD_MODE": "test", "GMSHIELD_ANTHROPIC_API_KEY": "sk-real-looking"})
    assert isinstance(select_model_client(settings), FakeModelClient)


def test_outside_test_mode_the_claude_client_is_built_without_calling_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_connection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("building the Claude client must not connect")

    monkeypatch.setattr(socket.socket, "connect", no_connection)
    monkeypatch.setattr(socket, "getaddrinfo", no_connection)
    settings = check_settings({"GMSHIELD_BACKEND_URL": "http://b:8080", "GMSHIELD_ANTHROPIC_API_KEY": "sk-abc"})
    client = select_model_client(settings)
    assert isinstance(client, ClaudeModelClient)
    assert client.kind == "claude"
    assert client.model == DEFAULT_MODEL
    assert "sk-abc" not in repr(client)


def test_the_claude_client_sends_one_message_through_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    import httpx2 as httpx

    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": DEFAULT_MODEL,
                "content": [{"type": "text", "text": "an answer"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )

    client = ClaudeModelClient(api_key="sk-abc")
    sdk = anthropic.Anthropic(api_key="sk-abc", http_client=httpx.Client(transport=httpx.MockTransport(answer)))
    monkeypatch.setattr(client, "_client", sdk)
    assert client.complete("system", "prompt", max_tokens=16) == "an answer"
    assert seen[0].url.path == "/v1/messages"


def test_outside_test_mode_a_missing_key_is_an_error_not_the_fake_model() -> None:
    settings = Settings(test_mode=False, backend_url="http://b:8080", work_dir="/tmp/w", anthropic_api_key=None)
    with pytest.raises(ValueError, match="API key"):
        select_model_client(settings)
