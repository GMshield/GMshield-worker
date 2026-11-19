"""The `test.model` kind against the backend's stand-in (Story 2.10): the proof of the whole model path. The claim
states the providers and answers the model; the one-node graph calls the fake model of that provider, which calls
`read_passages` under the job's grant; the answer cites what it read; the result holds the events and the usage."""

from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from gmshield_worker.agents import graphs
from gmshield_worker.agents.tools import read_passages_tool
from gmshield_worker.backend.client import ClaimedJob, JobsApi, Outcome
from gmshield_worker.lanes import model_probe
from gmshield_worker.log import new_trace_id
from gmshield_worker.models import chat
from gmshield_worker.models.fake import FakeChatModel
from gmshield_worker.models.output import CitationMarkers, ModelFailed, Usage
from gmshield_worker.models.providers import ModelChoice
from gmshield_worker.settings import PROVIDERS, ModelKeys
from tests.conftest import StandIn, StandInPassage, log_lines

CREDENTIAL = "w" * 64
SOURCE = "019a0000-0000-7000-8000-0000000005c0"
A = "019a0000-0000-7000-8000-00000000000a"
B = "019a0000-0000-7000-8000-00000000000b"
TEXT_A = "Texto da biblioteca A: a magia fere com fogo."
TEXT_B = "Texto da biblioteca B: o escudo dura um minuto."
FAKE = ModelKeys(test_mode=True)


def model_of(provider: str) -> dict[str, str]:
    return {"provider": provider, "model": "model-of-" + provider, "tier": "full"}


def claimed(stand_in: StandIn, prompt: str, model: dict[str, str] | None, providers: tuple[str, ...]) -> ClaimedJob:
    stand_in.passages[A] = StandInPassage(SOURCE, TEXT_A, ["Magias", "Bola de fogo"], 3)
    stand_in.passages[B] = StandInPassage(SOURCE, TEXT_B, ["Magias", "Escudo"], 4)
    job_input = {"prompt": prompt, "tiers": [{"tier": "campaign", "sources": [SOURCE]}]}
    job_id = stand_in.offer(model_probe.KIND, job_input, new_trace_id(), model=model)
    answer = JobsApi(stand_in.url, CREDENTIAL, providers=providers).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None, answer
    return answer.job


def probe(stand_in: StandIn, job: ClaimedJob, keys: ModelKeys = FAKE) -> Outcome:
    return model_probe.ModelProbe(JobsApi(stand_in.url, CREDENTIAL), keys)(job, threading.Event())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_test_model_reads_passages_under_the_grant_and_cites_them_for_every_provider(
    stand_in: StandIn, provider: str, capsys: pytest.CaptureFixture[str]
) -> None:
    job = claimed(stand_in, f"fake:tool O que diz {A} e {B}?", model_of(provider), PROVIDERS)
    assert job.model == ModelChoice(provider, "model-of-" + provider, "full")

    outcome = probe(stand_in, job)

    assert outcome.outcome == "done", outcome
    assert outcome.result is not None
    result: dict[str, Any] = outcome.result
    assert result["model"] == model_of(provider)
    events = result["events"]
    text = "".join(event["text"] for event in events if event["type"] == "delta")
    assert "[[" not in text
    citations = [(event["passageId"], event["offset"]) for event in events if event["type"] == "citation"]
    assert [passage_id for passage_id, _ in citations] == [A, B]
    assert all(text[:offset].endswith("✓") for _, offset in citations)
    # Two calls: the tool call, then the answer; each reports its usage once under its own call id.
    usage = result["usage"]
    assert len(usage) == 2
    assert len({item["callId"] for item in usage}) == 2
    assert [event for event in events if event["type"] == "usage"] == [{"type": "usage", **item} for item in usage]
    passages = [request for request in stand_in.requests if request["path"].endswith("/passages")]
    assert len(passages) == 1
    assert passages[0]["x-job-grant"] == job.grant
    logged = capsys.readouterr()
    assert TEXT_A not in logged.out + logged.err
    assert any(line["event"] == "model.call_ended" for line in log_lines(logged.out))


def test_library_text_reaches_the_model_only_as_a_tool_result(
    stand_in: StandIn, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[BaseMessage]] = []

    class Recording(FakeChatModel):
        def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            seen.append(list(messages))
            return super()._stream(messages, *args, **kwargs)

    monkeypatch.setattr(
        model_probe, "chat_model", lambda choice, keys: Recording(provider=choice.provider, model_id=choice.model)
    )
    job = claimed(stand_in, f"fake:tool {A}", model_of("azure-openai"), PROVIDERS)

    assert probe(stand_in, job).outcome == "done"

    assert len(seen) == 2
    system = [message for message in seen[-1] if isinstance(message, SystemMessage)]
    assert [message.content for message in system] == [model_probe.SYSTEM_PROMPT]
    tools = [message for message in seen[-1] if isinstance(message, ToolMessage)]
    assert len(tools) == 1
    assert TEXT_A in str(tools[0].content)
    assert all(TEXT_A not in str(message.content) for message in seen[-1] if not isinstance(message, ToolMessage))


@pytest.mark.parametrize(
    ("directive", "code", "retry"),
    [
        ("fake:refusal", "model.refused", False),
        ("fake:truncated", "model.truncated", False),
        ("fake:outage", "model.unavailable", None),
    ],
)
def test_a_typed_failure_fails_the_attempt_with_its_code(
    stand_in: StandIn, directive: str, code: str, retry: bool | None
) -> None:
    job = claimed(stand_in, f"{directive} {A}", model_of("anthropic"), PROVIDERS)
    assert probe(stand_in, job) == Outcome.failed(code, retry=retry)


def test_a_provider_this_worker_holds_no_key_for_or_does_not_know_fails_without_a_retry(stand_in: StandIn) -> None:
    anthropic_only = ModelKeys(anthropic_api_key="anthropic-secret")
    job = claimed(stand_in, "pergunta", model_of("azure-openai"), PROVIDERS)
    assert probe(stand_in, job, anthropic_only) == Outcome.failed("model.provider_key_missing", retry=False)
    unknown = claimed(stand_in, "pergunta", model_of("bedrock"), ("bedrock",))
    assert probe(stand_in, unknown) == Outcome.failed("model.provider_unknown", retry=False)


def test_a_claim_without_a_model_or_a_prompt_is_invalid_input(stand_in: StandIn) -> None:
    without_model = claimed(stand_in, "pergunta", None, PROVIDERS)
    assert probe(stand_in, without_model) == Outcome.failed("model.input_invalid", retry=False)
    empty = claimed(stand_in, "", model_of("openai"), PROVIDERS)
    assert probe(stand_in, empty) == Outcome.failed("model.input_invalid", retry=False)


def test_a_worker_that_did_not_state_the_provider_never_gets_the_job(stand_in: StandIn) -> None:
    job_id = stand_in.offer(model_probe.KIND, {"prompt": "p"}, model=model_of("azure-openai"))
    answer = JobsApi(stand_in.url, CREDENTIAL, providers=("anthropic",)).claim(UUID(job_id), new_trace_id())
    assert (answer.status, answer.code) == (409, "job.not_offered")
    assert stand_in.jobs[job_id].ended_code == "job.provider_unavailable"


def test_a_model_that_keeps_calling_tools_is_cut_after_the_last_allowed_call(stand_in: StandIn) -> None:
    class Looping(FakeChatModel):
        def _stream(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            only_user = [message for message in messages if not isinstance(message, ToolMessage)]
            return super()._stream(only_user, *args, **kwargs)

    job = claimed(stand_in, f"fake:tool {A}", model_of("openai"), PROVIDERS)
    events: list[Any] = []
    markers = CitationMarkers()
    node = graphs.AnswerWithTools(
        model=Looping(provider="openai"),
        tools=[read_passages_tool(JobsApi(stand_in.url, CREDENTIAL), job, markers)],
        markers=markers,
        emit=events.append,
        max_calls=3,
    )
    state = graphs.run_graph(graphs.answer_graph(node), {"messages": [HumanMessage(f"fake:tool {A}")], "calls": 0})
    assert state["failed"] == "model.truncated"
    assert state["calls"] == 3
    assert events[-1] == ModelFailed("model.truncated")


def test_a_stop_after_the_tool_call_ends_the_answer_without_a_failure(stand_in: StandIn) -> None:
    job = claimed(stand_in, f"fake:tool {A}", model_of("openai"), PROVIDERS)
    events: list[Any] = []
    markers = CitationMarkers()
    node = graphs.AnswerWithTools(
        model=FakeChatModel(provider="openai"),
        tools=[read_passages_tool(JobsApi(stand_in.url, CREDENTIAL), job, markers)],
        markers=markers,
        emit=events.append,
        should_stop=lambda: any(isinstance(event, Usage) for event in events),
    )
    state = graphs.run_graph(graphs.answer_graph(node), {"messages": [HumanMessage(f"fake:tool {A}")], "calls": 0})
    assert state["calls"] == 1
    assert state["failed"] is None
    assert len([event for event in events if isinstance(event, Usage)]) == 1
    assert not any(isinstance(event, ModelFailed) for event in events)


def test_a_malformed_tool_argument_is_an_error_result_for_the_model(stand_in: StandIn) -> None:
    job = claimed(stand_in, "pergunta", model_of("openai"), PROVIDERS)
    tool = read_passages_tool(JobsApi(stand_in.url, CREDENTIAL), job, CitationMarkers())
    result = tool.invoke({"name": "read_passages", "args": {"ids": 7}, "id": "call_1", "type": "tool_call"})
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call_1"
    assert result.status == "error"
    assert not any(request["path"].endswith("/passages") for request in stand_in.requests)


def test_chat_model_is_the_only_builder_the_probe_uses() -> None:
    assert vars(model_probe)["chat_model"] is chat.chat_model
