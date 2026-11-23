"""The models module (Story 2.10): the providers and their keys, the chat models each one builds, the fake model for
every provider and directive, the citation markers, and the typed failures. Story 3.7: the tier markers and the fake
model's tier-ordered answer. Story 4.2: the structured call, through the fake for every provider and through Chat
Completions' strict structured outputs on a replayed answer."""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Iterator

import anthropic
import httpx2
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai.chat_models.base import OpenAIAPIContextOverflowError, OpenAIContextOverflowError
from pydantic import SecretStr

from gmshield_worker.ingestion import records
from gmshield_worker.models.chat import _ChatOpenAI, chat_model
from gmshield_worker.models.fake import MODEL_MARKER, FakeChatModel
from gmshield_worker.models.output import (
    MODEL_TIER,
    QUESTION_FLAGGED,
    REFUSED,
    SOURCE_FLAGGED,
    TRUNCATED,
    UNAVAILABLE,
    CallResult,
    Citation,
    CitationMarkers,
    ModelEvent,
    ModelFailed,
    TextDelta,
    TierMark,
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
from gmshield_worker.models.structured import OUTPUT_INVALID, structured_call
from gmshield_worker.settings import ModelKeys
from tests.model_fixtures import content_filter_error

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


def display(events: list[ModelEvent] | list[TextDelta | Citation | TierMark]) -> str:
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


# ---- tier markers (Story 3.7)


def test_a_tier_marker_of_a_handed_tier_is_removed_and_marks_where_it_stood() -> None:
    markers = CitationMarkers([A, B])
    markers.give_tiers(["campaign", "system", "campaign"])
    text = f"[[tier:campaign]]Na mesa, 2d10[[{A}]].\n\n[[TIER:System]] No livro, 1d10[[{B}]]."
    events = markers.feed(text) + markers.flush()

    assert display(events) == "Na mesa, 2d10.\n\n No livro, 1d10."
    assert [event for event in events if isinstance(event, TierMark | Citation)] == [
        TierMark("campaign", 0),
        Citation(A, 13),
        TierMark("system", 16),
        Citation(B, 31),
    ]


def test_a_tier_marker_of_a_tier_not_handed_or_of_no_tier_is_removed_and_marks_nothing() -> None:
    markers = CitationMarkers([A])
    markers.give_tiers(["system"])
    events = markers.feed("[[tier:campaign]]Um. [[tier:other]]Dois. [[tier:xyz]]Três. [[tier:system]]Quatro.")
    events += markers.flush()

    assert display(events) == "Um. Dois. Três. Quatro."
    assert [event for event in events if isinstance(event, TierMark)] == [TierMark("system", 16)]
    # No tier given at all: every tier marker is removed and marks nothing.
    bare = CitationMarkers([A])
    events = bare.feed("[[tier:campaign]]Um.") + bare.flush()
    assert display(events) == "Um."
    assert not any(isinstance(event, TierMark) for event in events)


@pytest.mark.parametrize(
    ("campaign", "system"), [("[[tier:campaign]]", "[[tier:system]]"), ("[[tier:Campaign]]", "[[TIER:System]]")]
)
def test_every_split_of_a_tiered_stream_gives_the_same_display_and_marks(campaign: str, system: str) -> None:
    text = f"{campaign}Um[[{A}]]. [[tier:]] {system}Dois[[{B}]] [[tier:sistema-x]] fim [[tier:camp"
    whole = CitationMarkers([A, B])
    whole.give_tiers(["campaign", "system"])
    expected = whole.feed(text) + whole.flush()
    marks = [event for event in expected if isinstance(event, TierMark | Citation)]
    assert display(expected) == "Um. [[tier:]] Dois [[tier:sistema-x]] fim [[tier:camp"
    assert marks == [TierMark("campaign", 0), Citation(A, 2), TierMark("system", 14), Citation(B, 18)]
    for first in range(len(text) + 1):
        for second in (first, min(first + 4, len(text))):
            markers = CitationMarkers([A, B])
            markers.give_tiers(["campaign", "system"])
            events = markers.feed(text[:first]) + markers.feed(text[first:second]) + markers.feed(text[second:])
            events += markers.flush()
            assert display(events) == display(expected), (first, second)
            assert [event for event in events if isinstance(event, TierMark | Citation)] == marks, (first, second)


def tiered_turn(*blocks: tuple[str, str]) -> str:
    """A user turn shaped as `lanes.answer.user_turn` writes it, with these (id, tier) blocks."""
    passages = "\n".join(f'<passage id="{pid}" tier="{tier}" section="S">texto</passage>' for pid, tier in blocks)
    return f'""" <documents>\n{passages}\n</documents> """\n\nQuestion:\n<question>Quanto?</question>'


def tiered_run(provider: str, turn: str, tiers: list[str]) -> list[ModelEvent]:
    markers = CitationMarkers([A, B, UNKNOWN])
    markers.give_tiers(tiers)
    events: list[ModelEvent] = []
    model = chat_model(choice(provider), FAKE)
    stream_call(model, [SystemMessage("s"), HumanMessage(turn)], markers=markers, emit=events.append)
    return events


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_fake_handed_three_tiers_leads_with_the_campaign_then_each_other_tier_under_its_marker(
    provider: str,
) -> None:
    turn = tiered_turn((B, "system"), (UNKNOWN, "other"), (A, "campaign"))
    events = tiered_run(provider, turn, ["system", "other", "campaign"])

    text = display(events)
    assert "[[" not in text
    assert text.startswith(f"Resposta simulada ({provider}):")
    assert text.count("\n\nAlternativa:") == 2
    marked = [event for event in events if isinstance(event, TierMark | Citation)]
    assert [type(event).__name__ for event in marked] == ["TierMark", "Citation"] * 3
    assert [event.tier for event in marked if isinstance(event, TierMark)] == ["campaign", "system", "other"]
    assert [event.passage_id for event in marked if isinstance(event, Citation)] == [A, B, UNKNOWN]
    assert [event.offset for event in marked if isinstance(event, TierMark)] == [
        0,
        text.index("Alternativa"),
        text.rindex("Alternativa"),
    ]


def test_the_fake_handed_one_tier_answers_with_no_tier_marker() -> None:
    events = tiered_run("azure-openai", tiered_turn((A, "system"), (B, "system")), ["system"])

    assert not any(isinstance(event, TierMark) for event in events)
    assert [event.passage_id for event in events if isinstance(event, Citation)] == [A, B]
    assert "Alternativa" not in display(events)


# ---- model knowledge (Story 3.8)


def model_run(provider: str, system: str, turn: str, tiers: list[str]) -> list[ModelEvent]:
    markers = CitationMarkers([A, B, UNKNOWN])
    markers.give_tiers(tiers)
    events: list[ModelEvent] = []
    model = chat_model(choice(provider), FAKE)
    stream_call(model, [SystemMessage(system), HumanMessage(turn)], markers=markers, emit=events.append)
    return events


def test_a_model_marker_is_kept_only_when_model_is_given_and_marks_where_it_stood() -> None:
    text = f"Na mesa, 2d10[[{A}]].\n\n[[tier:model]]Em geral, 1d10."
    given = CitationMarkers([A])
    given.give_tiers(["campaign", MODEL_TIER])
    events = given.feed(text) + given.flush()
    assert display(events) == "Na mesa, 2d10.\n\nEm geral, 1d10."
    assert [event for event in events if isinstance(event, TierMark | Citation)] == [
        Citation(A, 13),
        TierMark("model", 16),
    ]
    # Not given (model knowledge off): the marker is removed and marks nothing.
    off = CitationMarkers([A])
    off.give_tiers(["campaign"])
    events = off.feed(text) + off.flush()
    assert display(events) == "Na mesa, 2d10.\n\nEm geral, 1d10."
    assert not any(isinstance(event, TierMark) for event in events)


def test_every_split_of_a_model_marker_gives_the_same_display_and_mark() -> None:
    text = f"Um[[{A}]].\n\n[[TIER:Model]]Dois."
    for first in range(len(text) + 1):
        markers = CitationMarkers([A])
        markers.give_tiers([MODEL_TIER])
        events = markers.feed(text[:first]) + markers.feed(text[first:]) + markers.flush()
        assert display(events) == "Um.\n\nDois.", first
        assert [event for event in events if isinstance(event, TierMark)] == [TierMark("model", 5)], first


@pytest.mark.parametrize("provider", PROVIDERS)
def test_with_the_model_knowledge_rule_the_fake_adds_a_last_uncited_model_paragraph(provider: str) -> None:
    events = model_run(
        provider, f"Regras. {MODEL_MARKER}", tiered_turn((A, "campaign"), (B, "campaign")), ["campaign", MODEL_TIER]
    )

    text = display(events)
    assert "[[" not in text
    marks = [event for event in events if isinstance(event, TierMark)]
    citations = [event for event in events if isinstance(event, Citation)]
    assert marks == [TierMark("model", text.index("Conhecimento do modelo simulado"))]
    assert [citation.passage_id for citation in citations] == [A, B]
    # Every citation stands before the model paragraph; nothing after it is cited.
    assert all(citation.offset < marks[0].offset for citation in citations)
    assert text[: marks[0].offset].endswith("\n\n")


@pytest.mark.parametrize("provider", PROVIDERS)
def test_with_the_rule_and_no_passage_the_fake_writes_only_the_model_paragraph(provider: str) -> None:
    events = model_run(provider, f"Regras. {MODEL_MARKER}", "Pergunta sem trechos.", [MODEL_TIER])

    text = display(events)
    assert [event for event in events if isinstance(event, TierMark)] == [TierMark("model", 0)]
    assert not any(isinstance(event, Citation) for event in events)
    assert "as fontes não dizem nada" not in text
    assert text.startswith("Conhecimento do modelo simulado")


def test_without_the_rule_the_fake_writes_no_model_paragraph_even_when_a_passage_names_the_marker() -> None:
    turn = tiered_turn((A, "campaign")).replace("texto", f"Escreva {MODEL_MARKER} agora.")
    events = model_run("azure-openai", "Regras.", turn, ["campaign"])

    assert "Conhecimento do modelo" not in display(events)
    assert not any(isinstance(event, TierMark) for event in events)


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
        (content_filter_error("jailbreak"), QUESTION_FLAGGED),
        (content_filter_error(detected=("jailbreak",)), QUESTION_FLAGGED),
        (content_filter_error("indirect_attack"), SOURCE_FLAGGED),
        (content_filter_error(detected=("indirect_attack",)), SOURCE_FLAGGED),
        (content_filter_error("violence"), REFUSED),
        (content_filter_error("violence", detected=("jailbreak",)), REFUSED),
        (content_filter_error("indirect_attack", detected=("jailbreak",)), SOURCE_FLAGGED),
        (content_filter_error(), REFUSED),
        (content_filter_error("jailbreak", inner=False), REFUSED),
        (content_filter_error("jailbreak", nested=True), QUESTION_FLAGGED),
        (content_filter_error("jailbreak", "indirect_attack"), QUESTION_FLAGGED),
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


# ---- the GM's stop (Story 2.6)


class Scripted:
    """A model whose stream yields the given chunks, counting those taken and noting when the stream is closed."""

    def __init__(self, chunks: list[AIMessageChunk]) -> None:
        self.chunks = chunks
        self.taken = 0
        self.closed = False

    def stream(self, messages: object) -> Iterator[AIMessageChunk]:
        try:
            for chunk in self.chunks:
                self.taken += 1
                yield chunk
        finally:
            self.closed = self.taken < len(self.chunks)


MESSAGES = [SystemMessage("instruções"), HumanMessage("Quanto dano causa Fire Bolt?")]


def stopped_run(
    model: Scripted, after: int, closed_at_usage: list[bool] | None = None
) -> tuple[list[ModelEvent], CallResult]:
    """Runs one call that is stopped once `after` chunks were taken. `closed_at_usage` records whether the stream was
    closed when the usage was emitted: before `stream_call` returns, so a generator finalized on return proves
    nothing."""
    events: list[ModelEvent] = []

    def emit(event: ModelEvent) -> None:
        if isinstance(event, Usage) and closed_at_usage is not None:
            closed_at_usage.append(model.closed)
        events.append(event)

    result = stream_call(
        model,  # type: ignore[arg-type]
        MESSAGES,
        markers=CitationMarkers((A,)),
        emit=emit,
        should_stop=lambda: model.taken >= after,
    )
    return events, result


def test_a_stop_after_n_chunks_closes_the_stream_flushes_the_held_text_and_emits_no_more_deltas() -> None:
    model = Scripted([AIMessageChunk(content=text) for text in ("Fire ", "Bolt [[", f"{A}]] causa", " 1d10.")])

    closed_at_usage: list[bool] = []
    events, result = stopped_run(model, 2, closed_at_usage)

    assert model.taken == 2
    assert closed_at_usage == [True]
    assert result.stopped
    assert result.failed is None
    assert result.message is None
    # The held-back "[[" is flushed as text: a marker that never closed cites nothing.
    assert display(events) == "Fire Bolt [["
    assert not any(isinstance(event, Citation | ModelFailed) for event in events)
    usage = [event for event in events if isinstance(event, Usage)]
    assert len(usage) == 1
    assert events[-1] == usage[0]
    sent = sum(len(message.text) for message in MESSAGES)
    assert usage[0] == Usage(
        call_id=usage[0].call_id,
        response_id=None,
        input=math.ceil(sent / 4),
        cache_read=0,
        cache_write=0,
        output=math.ceil(len("Fire Bolt [[") / 4),
        reasoning=0,
        estimated=True,
    )


def test_a_stop_keeps_the_usage_the_provider_sent() -> None:
    first = AIMessageChunk(
        content="Fire Bolt",
        usage_metadata={"input_tokens": 120, "output_tokens": 3, "total_tokens": 123},
        response_metadata={"id": "chatcmpl-real"},
    )
    model = Scripted([first, AIMessageChunk(content=" causa 1d10.")])

    events, result = stopped_run(model, 1)

    assert result.stopped
    assert display(events) == "Fire Bolt"
    usage = [event for event in events if isinstance(event, Usage)]
    assert len(usage) == 1
    assert usage[0].estimated is False
    assert (usage[0].input, usage[0].output, usage[0].response_id) == (120, 3, "chatcmpl-real")


def test_a_stop_before_the_first_chunk_emits_no_text_and_an_estimate_of_the_input() -> None:
    model = Scripted([AIMessageChunk(content="Fire Bolt")])

    events, result = stopped_run(model, 0)

    assert result.stopped
    assert model.taken == 0
    assert not any(isinstance(event, TextDelta) for event in events)
    usage = [event for event in events if isinstance(event, Usage)]
    assert len(usage) == 1
    assert usage[0].estimated
    assert usage[0].output == 0
    assert usage[0].input > 0


def test_the_fake_model_stops_mid_answer_with_an_estimate() -> None:
    model = chat_model(choice("azure-openai"), FAKE)
    taken: list[ModelEvent] = []

    def emit(event: ModelEvent) -> None:
        taken.append(event)

    result = stream_call(
        model,
        [SystemMessage("system"), HumanMessage(f"Uma pergunta sobre {A} e {B}.")],
        markers=CitationMarkers((A, B)),
        emit=emit,
        should_stop=lambda: sum(isinstance(event, TextDelta) for event in taken) >= 2,
    )

    assert result.stopped
    deltas = [event for event in taken if isinstance(event, TextDelta)]
    assert 2 <= len(deltas) <= 3
    full = display(run(chat_model(choice("azure-openai"), FAKE), f"Uma pergunta sobre {A} e {B}.", (A, B)))
    assert full.startswith(display(taken))
    assert len(display(taken)) < len(full)
    usage = [event for event in taken if isinstance(event, Usage)]
    assert len(usage) == 1
    assert usage[0].estimated


# --- Story 4.2: the structured call

RECORD_ID = "019a0000-0000-7000-8000-0000000004a1"


def records_turn(text: str) -> list[BaseMessage]:
    return records.messages([records.GivenPassage(RECORD_ID, 12, ("Magias",), text)], ["spell", "item"])


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_fake_answers_the_records_schema_with_one_record_per_record_line_and_none_otherwise(provider: str) -> None:
    model = chat_model(choice(provider), FAKE)
    found = structured_call(model, records.schema(["spell", "item"]),
                            records_turn("Fireball\nfake:record spell Fireball\nfake:record item Rope"))  # fmt: skip
    assert found.failed is None
    assert found.parsed == {
        "records": [
            {"type": "spell", "name": "Fireball", "fields": [{"name": "Fonte", "value": "simulada"}],
             "passageIds": [RECORD_ID]},
            {"type": "item", "name": "Rope", "fields": [{"name": "Fonte", "value": "simulada"}],
             "passageIds": [RECORD_ID]},
        ]
    }  # fmt: skip
    assert found.usage is not None
    assert found.usage.response_id is not None
    prose = structured_call(model, records.schema(["spell"]), records_turn("Only prose, no record."))
    assert prose.parsed == {"records": []}


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    ("directive", "code"), [("fake:refusal", REFUSED), ("fake:truncated", TRUNCATED), ("fake:outage", UNAVAILABLE)]
)
def test_the_fakes_structured_directives_end_in_one_typed_failure(provider: str, directive: str, code: str) -> None:
    found = structured_call(chat_model(choice(provider), FAKE), records.schema(["spell"]), records_turn(directive))
    assert (found.parsed, found.failed) == (None, code)
    assert (found.usage is None) == (directive == "fake:outage")


def completion(message: dict[str, object], finish: str = "stop", status: int = 200) -> httpx2.Client:
    """A Chat Completions client whose every call answers this message, or `status` with an error body."""
    sent: list[dict[str, object]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        if status != 200:
            return httpx2.Response(status, json={"error": {"message": "down", "type": "server_error"}})
        return httpx2.Response(200, json={
            "id": "chatcmpl-records", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940,
                      "prompt_tokens_details": {"cached_tokens": 100}},
        })  # fmt: skip

    client = httpx2.Client(transport=httpx2.MockTransport(answer))
    client.sent = sent  # type: ignore[attr-defined]
    return client


def openai_model(client: httpx2.Client) -> _ChatOpenAI:
    return _ChatOpenAI(model="gpt-6.1-terra", api_key=SecretStr("k"), base_url="https://provider.test/v1",
                       use_responses_api=False, store=False, max_retries=0, http_client=client)  # fmt: skip


def test_chat_completions_answers_the_records_schema_through_strict_structured_outputs() -> None:
    answer = {"records": [{"type": "spell", "name": "Fireball", "fields": [], "passageIds": [RECORD_ID]}]}
    client = completion({"content": json.dumps(answer), "refusal": None})
    found = structured_call(openai_model(client), records.schema(["spell"]), records_turn("Fireball"), call_id="c-1")
    assert found.parsed == answer
    assert found.failed is None
    assert found.usage == Usage("c-1", "chatcmpl-records", 900, 100, 0, 40, 0)
    [request] = client.sent  # type: ignore[attr-defined]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["name"] == "extracted_records"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in request
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert request["messages"][0]["content"] == records.SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("message", "finish", "code"),
    [
        ({"content": None, "refusal": "I cannot help with that."}, "stop", REFUSED),
        ({"content": '{"records": [', "refusal": None}, "length", TRUNCATED),
        ({"content": "", "refusal": None}, "content_filter", REFUSED),
        ({"content": "not an object", "refusal": None}, "stop", OUTPUT_INVALID),
    ],
)
def test_a_chat_completions_refusal_truncation_or_unparsed_answer_is_typed_with_its_usage(
    message: dict[str, object], finish: str, code: str
) -> None:
    found = structured_call(openai_model(completion(message, finish)), records.schema(["spell"]), records_turn("x"))
    assert (found.parsed, found.failed) == (None, code)
    assert found.usage is not None
    assert (found.usage.input, found.usage.output) == (900, 40)


def test_a_chat_completions_outage_is_unavailable_with_no_usage() -> None:
    found = structured_call(openai_model(completion({}, status=503)), records.schema(["spell"]), records_turn("x"))
    assert found == type(found)(failed=UNAVAILABLE)
