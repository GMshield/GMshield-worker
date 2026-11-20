"""Each real chat model against its provider's recorded HTTP stream (Story 2.10): the same event shapes and usage as
the fake, and the recorded refusal, 503, length stop and stream error mapped to the same codes. The streams are
replayed by a local server (`tests/model_fixtures.py`), so the request each model sends is checked too: the explicit
path, no `api-version`, storage off, usage streamed, no tool, one request only (no retry)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gmshield_worker.models.chat import build_chat_model
from gmshield_worker.models.output import (
    QUESTION_FLAGGED,
    REFUSED,
    SOURCE_FLAGGED,
    TRUNCATED,
    UNAVAILABLE,
    Citation,
    CitationMarkers,
    ModelEvent,
    ModelFailed,
    TextDelta,
    Usage,
    stream_call,
)
from gmshield_worker.models.providers import Endpoint, ModelChoice
from tests import model_fixtures
from tests.model_fixtures import PASSAGE_ID, SYSTEM_PROMPT, USER_PROMPT, Replay

KEY = "test-key-not-a-secret"


def model_for(provider: str, server: Replay) -> Any:
    base = server.url + "/openai/v1/" if provider == "azure-openai" else server.url
    choice = ModelChoice(provider, str(server.fixture["model"]), "full")
    return build_chat_model(choice, Endpoint(provider, base, KEY), max_tokens=512)


def call(model: Any) -> list[ModelEvent]:
    from langchain_core.messages import HumanMessage, SystemMessage

    events: list[ModelEvent] = []
    stream_call(
        model,
        [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)],
        markers=CitationMarkers([PASSAGE_ID]),
        emit=events.append,
    )
    return events


def expected_usage(provider: str, case: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """(input, cache read, cache write, output, reasoning) as the provider reported them in the stream."""
    if provider == "anthropic":
        usage = next(data["usage"] for data in model_fixtures.stream_data(case) if data["type"] == "message_delta")
        read, write = usage.get("cache_read_input_tokens") or 0, usage.get("cache_creation_input_tokens") or 0
        details = usage.get("output_tokens_details") or {}
        return (
            usage["input_tokens"] + read + write,
            read,
            write,
            usage["output_tokens"],
            details.get("thinking_tokens", 0),
        )
    usage = next(data["usage"] for data in model_fixtures.stream_data(case) if data.get("usage"))
    prompt, completion = usage.get("prompt_tokens_details") or {}, usage.get("completion_tokens_details") or {}
    return (
        usage["prompt_tokens"],
        prompt.get("cached_tokens") or 0,
        prompt.get("cache_write_tokens") or 0,
        usage["completion_tokens"],
        completion.get("reasoning_tokens") or 0,
    )


def assert_request(provider: str, server: Replay) -> None:
    assert len(server.requests) == 1, "a model call is never retried inside the call"
    request = server.requests[0]
    body, headers = request["body"], request["headers"]
    assert body["stream"] is True
    assert "tools" not in body
    assert any(message.get("content") == USER_PROMPT for message in body["messages"])
    if provider == "anthropic":
        assert request["path"] == "/v1/messages"
        # The Anthropic SDK 2.x sends the API key as a bearer token.
        assert headers["authorization"] == f"Bearer {KEY}"
        assert "anthropic-version" in headers
        assert body["system"] == SYSTEM_PROMPT or SYSTEM_PROMPT in json.dumps(body["system"])
        return
    assert request["path"] == "/openai/v1/chat/completions", "no api-version, no deployments path"
    assert headers["authorization"] == f"Bearer {KEY}"
    assert body["store"] is False
    assert body["stream_options"] == {"include_usage": True}
    assert body["model"] == server.fixture["model"]


@pytest.mark.parametrize("provider", list(model_fixtures.PROVIDER_FILES))
def test_a_recorded_answer_gives_display_text_its_citation_and_its_usage(provider: str) -> None:
    with model_fixtures.replay(provider) as server:
        events = call(model_for(provider, server))
        assert_request(provider, server)
    case = server.fixture["cases"]["answer"]
    display, citations = model_fixtures.display_and_citations(model_fixtures.streamed_text(provider, case))

    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == display
    assert "[[" not in display
    assert [event for event in events if isinstance(event, Citation)] == [Citation(i, o) for i, o in citations]
    assert (PASSAGE_ID, citations[0][1]) in citations
    usage = [event for event in events if isinstance(event, Usage)]
    assert len(usage) == 1
    assert events[-1] == usage[0]
    found = usage[0]
    assert (found.input, found.cache_read, found.cache_write, found.output, found.reasoning) == expected_usage(
        provider, case
    )
    recorded_id = model_fixtures.ANTHROPIC_RESPONSE_ID if provider == "anthropic" else model_fixtures.OPENAI_RESPONSE_ID
    assert found.response_id == recorded_id
    assert not any(isinstance(event, ModelFailed) for event in events)


CASES = [
    ("azure-openai", "refusal", REFUSED),
    ("azure-openai", "refusal_delta", REFUSED),
    ("azure-openai", "outage", UNAVAILABLE),
    ("azure-openai", "length", TRUNCATED),
    ("azure-openai", "prompt_filter_jailbreak", QUESTION_FLAGGED),
    ("azure-openai", "prompt_filter_indirect_attack", SOURCE_FLAGGED),
    ("anthropic", "refusal", REFUSED),
    ("anthropic", "outage", UNAVAILABLE),
    ("anthropic", "length", TRUNCATED),
    ("anthropic", "stream_error", UNAVAILABLE),
]


@pytest.mark.parametrize(("provider", "case", "code"), CASES)
def test_a_recorded_failure_is_one_typed_failure(provider: str, case: str, code: str) -> None:
    with model_fixtures.replay(provider) as server:
        server.case = case
        events = call(model_for(provider, server))
        assert len(server.requests) == 1
    assert [event for event in events if isinstance(event, ModelFailed)] == [ModelFailed(code)]
    assert events[-1] == ModelFailed(code)
    assert KEY not in repr(events)
    if code in {REFUSED, QUESTION_FLAGGED, SOURCE_FLAGGED}:
        assert not any(isinstance(event, TextDelta) for event in events)


@pytest.mark.parametrize("provider", list(model_fixtures.PROVIDER_FILES))
def test_the_fixtures_hold_no_header_key_or_response_id(provider: str) -> None:
    fixture = model_fixtures.load(provider)
    text = json.dumps(fixture)
    assert fixture["provider"] == provider
    for case in fixture["cases"].values():
        assert case["source"] in ("recorded", "hand-built")
        assert set(case) <= {"source", "recorded", "status", "contentType", "body"}
    for leak in ("api-key", "authorization", "x-request-id", "apim-request-id", "set-cookie"):
        assert leak not in text.lower()
    for case in fixture["cases"].values():
        if isinstance(case["body"], list):
            for data in model_fixtures.stream_data(case):
                ids = {data.get("id"), (data.get("message") or {}).get("id")} - {None, ""}
                assert ids <= {model_fixtures.OPENAI_RESPONSE_ID, model_fixtures.ANTHROPIC_RESPONSE_ID}


def recording_tool() -> Any:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "tools" / "record_model_responses.py"
    spec = importlib.util.spec_from_file_location("record_model_responses", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_recording_tool_sends_what_the_worker_sends() -> None:
    tool = recording_tool()
    with model_fixtures.replay("azure-openai") as server:
        call(model_for("azure-openai", server))
    sent = server.requests[0]["body"]
    recorded = tool.request_body(server.fixture["model"], 512)
    assert set(recorded) <= set(sent)
    for name in ("model", "stream", "stream_options", "store", "max_completion_tokens"):
        assert recorded[name] == sent[name], name
    assert [message["content"] for message in recorded["messages"]] == [
        message["content"] for message in sent["messages"]
    ]


def test_the_recording_tool_drops_ids_and_comments_and_refuses_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = recording_tool()
    lines = [
        ": keep-alive",
        'data: {"id":"chatcmpl-AbC123","choices":[]}',
        "",
        'data: {"id":"","choices":[]}',
        "data: [DONE]",
    ]
    assert tool.clean(lines, ["secret-key", "res.openai.azure.com"]) == [
        'data: {"id":"chatcmpl-recorded","choices":[]}',
        "",
        'data: {"id":"","choices":[]}',
        "data: [DONE]",
    ]
    with pytest.raises(tool.RecordingFailed):
        tool.clean(['data: {"id":"x","note":"res.openai.azure.com"}'], ["secret-key", "res.openai.azure.com"])
    with pytest.raises(tool.RecordingFailed, match="not JSON"):
        tool.clean(["data: {not json"], [])
    monkeypatch.setenv(tool.KEY, "secret-key")
    monkeypatch.setenv(tool.BASE_URL, "https://res.openai.azure.com/openai/v1")
    assert tool.settings() == ("secret-key", "https://res.openai.azure.com/openai/v1/")
    monkeypatch.setenv(tool.BASE_URL, "https://res.openai.azure.com/openai/deployments/x")
    with pytest.raises(tool.RecordingFailed) as refused:
        tool.settings()
    assert "secret-key" not in str(refused.value)
    assert tool.shape(model_fixtures.load("azure-openai")["cases"]["answer"])["finishes"] == ["stop"]
