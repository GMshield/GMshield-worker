"""The models module (Story 2.10): the providers and their keys, the chat models each one builds, the fake model for
every provider and directive, the citation markers, and the typed failures."""

from __future__ import annotations

import socket

import anthropic
import httpx2
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai.chat_models.base import OpenAIAPIContextOverflowError, OpenAIContextOverflowError

from gmshield_worker.models.chat import chat_model
from gmshield_worker.models.fake import FakeChatModel
from gmshield_worker.models.output import (
    REFUSED,
    TRUNCATED,
    UNAVAILABLE,
    Citation,
    CitationMarkers,
    ModelEvent,
    ModelFailed,
    TextDelta,
    Usage,
    failure_code,
    stream_call,
)
from gmshield_worker.models.providers import (
    ANTHROPIC_BASE_URL,
    OPENAI_BASE_URL,
    PROVIDER_KEY_MISSING,
    PROVIDER_UNKNOWN,
    PROVIDERS,
    ModelChoice,
    ProviderUnavailable,
    endpoint,
)
from gmshield_worker.settings import ModelKeys

A = "019a0000-0000-7000-8000-00000000000a"
B = "019a0000-0000-7000-8000-00000000000b"
UNKNOWN = "019a0000-0000-7000-8000-0000000000ff"
AZURE_URL = "https://gmshield-test.openai.azure.com/openai/v1/"
KEYS = ModelKeys(
    azure_openai_api_key="azure-secret",
    azure_openai_base_url=AZURE_URL,
    openai_api_key="openai-secret",
    anthropic_api_key="anthropic-secret",
)
FAKE = ModelKeys(test_mode=True)


def choice(provider: str) -> ModelChoice:
    return ModelChoice(provider, "model-of-" + provider, "full")


def run(model: object, prompt: str, given: tuple[str, ...] = (A, B)) -> list[ModelEvent]:
    events: list[ModelEvent] = []
    stream_call(
        model,  # type: ignore[arg-type]
        [SystemMessage("system"), HumanMessage(prompt)],
        markers=CitationMarkers(given),
        emit=events.append,
    )
    return events


def display(events: list[ModelEvent] | list[TextDelta | Citation]) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


# ---- providers and chat models


def test_each_provider_has_its_explicit_base_url_and_key() -> None:
    assert endpoint(KEYS, "azure-openai").base_url == AZURE_URL
    assert endpoint(KEYS, "openai").base_url == OPENAI_BASE_URL == "https://api.openai.com/v1"
    assert endpoint(KEYS, "anthropic").base_url == ANTHROPIC_BASE_URL == "https://api.anthropic.com"
    assert endpoint(KEYS, "anthropic").api_key == "anthropic-secret"
    assert "anthropic-secret" not in repr(endpoint(KEYS, "anthropic"))


@pytest.mark.parametrize(
    ("keys", "provider", "code"),
    [
        (ModelKeys(anthropic_api_key="k"), "azure-openai", PROVIDER_KEY_MISSING),
        (ModelKeys(azure_openai_api_key="k", azure_openai_base_url=AZURE_URL), "openai", PROVIDER_KEY_MISSING),
        (ModelKeys(openai_api_key="k"), "anthropic", PROVIDER_KEY_MISSING),
        (KEYS, "bedrock", PROVIDER_UNKNOWN),
        (FAKE, "bedrock", PROVIDER_UNKNOWN),
    ],
)
def test_a_provider_without_a_key_or_unknown_is_unavailable_never_another(
    keys: ModelKeys, provider: str, code: str
) -> None:
    with pytest.raises(ProviderUnavailable) as unavailable:
        chat_model(choice(provider), keys)
    assert unavailable.value.code == code


def test_the_chat_models_are_built_explicitly_without_calling_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_connection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("building a chat model must not connect")

    monkeypatch.setattr(socket.socket, "connect", no_connection)
    monkeypatch.setattr(socket, "getaddrinfo", no_connection)
    for provider, base_url in (("azure-openai", AZURE_URL), ("openai", OPENAI_BASE_URL)):
        model = chat_model(choice(provider), KEYS)
        assert isinstance(model, ChatOpenAI)
        assert not isinstance(model, AzureChatOpenAI)
        assert model.openai_api_base == base_url
        assert model.model_name == "model-of-" + provider
        assert model.use_responses_api is False
        assert model.store is False
        assert model.stream_usage is True
        assert model.max_retries == 0
        assert not model.model_kwargs
        assert "secret" not in repr(model)
    claude = chat_model(choice("anthropic"), KEYS)
    assert isinstance(claude, ChatAnthropic)
    assert claude.anthropic_api_url == ANTHROPIC_BASE_URL
    assert claude.model == "model-of-anthropic"
    assert claude.max_retries == 0
    assert claude.stream_usage is True
    assert "anthropic-secret" not in repr(claude)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_test_mode_answers_every_provider_with_the_fake_model(provider: str) -> None:
    model = chat_model(choice(provider), FAKE)
    assert isinstance(model, FakeChatModel)
    assert model.provider == provider


# ---- the fake model through the uniform output


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_fake_streams_deltas_citations_and_one_usage_for_each_provider(provider: str) -> None:
    events = run(chat_model(choice(provider), FAKE), f"Uma pergunta sobre {A} e {B} e {UNKNOWN}.")

    deltas = [event for event in events if isinstance(event, TextDelta)]
    citations = [event for event in events if isinstance(event, Citation)]
    usage = [event for event in events if isinstance(event, Usage)]
    text = display(events)
    assert len(deltas) > 3
    assert "[[" not in text
    assert provider in text
    assert [citation.passage_id for citation in citations] == [A, B]
    for citation in citations:
        assert text[: citation.offset].endswith("✓")
    assert len(usage) == 1
    assert events[-1] == usage[0]
    assert usage[0].input > 0
    assert usage[0].output >= len(deltas)
    assert usage[0].response_id is not None
    assert usage[0].response_id.startswith("msg_fake" if provider == "anthropic" else "chatcmpl-fake")
    assert not any(isinstance(event, ModelFailed) for event in events)
    # Deterministic: the same prompt streams the same text.
    assert display(run(chat_model(choice(provider), FAKE), f"Uma pergunta sobre {A} e {B} e {UNKNOWN}.")) == text


def test_without_passages_the_fake_says_the_sources_are_silent() -> None:
    events = run(chat_model(choice("azure-openai"), FAKE), "Pergunta sem trechos.")
    assert "as fontes não dizem nada" in display(events)
    assert not any(isinstance(event, Citation) for event in events)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    ("directive", "code"), [("fake:refusal", REFUSED), ("fake:outage", UNAVAILABLE), ("fake:truncated", TRUNCATED)]
)
def test_each_directive_ends_in_one_typed_failure(provider: str, directive: str, code: str) -> None:
    events = run(chat_model(choice(provider), FAKE), f"{directive} {A}")

    assert [event for event in events if isinstance(event, ModelFailed)] == [ModelFailed(code)]
    assert events[-1] == ModelFailed(code)
    if directive == "fake:outage":
        assert events == [ModelFailed(UNAVAILABLE)]
    if directive == "fake:refusal":
        assert display(events) == ""
    if directive == "fake:truncated":
        assert display(events)
        assert any(isinstance(event, Usage) for event in events)


# ---- citation markers


def test_a_marker_split_across_chunks_is_removed_and_cited_where_it_stood() -> None:
    markers = CitationMarkers([A])
    events = markers.feed("Fire bolt deals 1d10[[019a0000-0000") + markers.feed("-7000-8000-00000000000a]] fire.")
    events += markers.flush()

    assert display(events) == "Fire bolt deals 1d10 fire."
    assert [event for event in events if isinstance(event, Citation)] == [Citation(A, 20)]


def test_every_split_of_the_stream_gives_the_same_display_and_citations() -> None:
    text = f"Um [[{A}]]dois [colchete] [[{UNKNOWN}]] três[[{B}]]. [[não é marca]] fim [[{A.upper()}]]"
    whole = CitationMarkers([A, B])
    expected = whole.feed(text) + whole.flush()
    expected_citations = [event for event in expected if isinstance(event, Citation)]
    assert display(expected) == "Um dois [colchete]  três. [[não é marca]] fim "
    assert expected_citations == [Citation(A, 3), Citation(B, 24), Citation(A, 46)]
    for first in range(len(text) + 1):
        for second in (first, min(first + 3, len(text))):
            markers = CitationMarkers([A, B])
            events = markers.feed(text[:first]) + markers.feed(text[first:second]) + markers.feed(text[second:])
            events += markers.flush()
            assert display(events) == display(expected), (first, second)
            assert [event for event in events if isinstance(event, Citation)] == expected_citations, (first, second)


def test_an_unknown_id_is_removed_and_cites_nothing() -> None:
    markers = CitationMarkers([A])
    events = markers.feed(f"Texto[[{UNKNOWN}]] fim.") + markers.flush()
    assert display(events) == "Texto fim."
    assert not any(isinstance(event, Citation) for event in events)


def test_the_offset_counts_code_points_of_multi_byte_text() -> None:
    markers = CitationMarkers([A])
    prefix = "Ação ✓ 🐉 dano"
    events = markers.feed(prefix + "[[" + A[:10]) + markers.feed(A[10:] + "]]!") + markers.flush()
    citation = next(event for event in events if isinstance(event, Citation))
    assert citation.offset == len(prefix) == 13
    assert len(prefix.encode()) != 13
    assert len(prefix.encode("utf-16-le")) // 2 != 13
    assert display(events) == prefix + "!"


def test_brackets_that_never_become_a_marker_are_text() -> None:
    markers = CitationMarkers([A])
    held = markers.feed("um [")
    assert display(held) == "um "
    events = held + markers.feed("[[" + A + "]] e [[sem fim") + markers.flush()
    assert display(events) == "um [ e [[sem fim"
    assert [event for event in events if isinstance(event, Citation)] == [Citation(A, 4)]
    assert markers.offset == len("um [ e [[sem fim")


def test_the_display_offset_runs_across_calls() -> None:
    markers = CitationMarkers()
    markers.feed("abc")
    markers.give([A])
    events = markers.feed(f"de[[{A}]]")
    assert events[-1] == Citation(A, 5)


# ---- typed failures


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", "https://provider.test/v1"))


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (openai.InternalServerError("x", response=_response(503), body=None), UNAVAILABLE),
        (openai.RateLimitError("x", response=_response(429), body=None), UNAVAILABLE),
        (openai.APITimeoutError(request=httpx2.Request("POST", "https://provider.test")), UNAVAILABLE),
        (openai.APIConnectionError(request=httpx2.Request("POST", "https://provider.test")), UNAVAILABLE),
        (openai.APIError("in the stream", request=httpx2.Request("POST", "https://p.test"), body=None), UNAVAILABLE),
        (
            openai.BadRequestError("x", response=_response(400), body={"code": "content_filter", "message": "m"}),
            REFUSED,
        ),
        (anthropic.InternalServerError("x", response=_response(500), body=None), UNAVAILABLE),
        (
            anthropic.APIStatusError(
                "x", response=_response(200), body={"type": "error", "error": {"type": "overloaded_error"}}
            ),
            UNAVAILABLE,
        ),
        (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://provider.test")), UNAVAILABLE),
        (OpenAIContextOverflowError("context_length_exceeded", response=_response(400), body=None), TRUNCATED),
        (
            OpenAIAPIContextOverflowError(
                message="exceeds the context window", request=httpx2.Request("POST", "https://p.test"), body=None
            ),
            TRUNCATED,
        ),
        (openai.AuthenticationError("x", response=_response(401), body=None), None),
        (anthropic.BadRequestError("x", response=_response(400), body=None), None),
        (ValueError("not a provider's"), None),
    ],
)
def test_provider_errors_map_to_the_typed_failures_and_others_propagate(error: BaseException, code: str | None) -> None:
    assert failure_code(error) == code


def test_an_error_that_is_not_a_typed_failure_propagates_from_the_call() -> None:
    class Broken(FakeChatModel):
        def _stream(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            raise ValueError("broken")
            yield

    with pytest.raises(ValueError, match="broken"):
        run(Broken(provider="openai"), "x")
