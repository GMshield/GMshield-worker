"""One output shape for every provider (Story 2.10): text deltas, citations, usage and typed failures.

Each model call gets a worker-minted call id (a UUIDv7). `stream_call` streams one call of a LangChain chat model and
hands its events, in order, to `emit`:

- `TextDelta(text)`: display text, with the citation markers removed;
- `Citation(passage_id, offset)`: the model wrote `[[<passage id>]]` after a statement; the offset is where the marker
  stood, in code points of the display text. Only the ids that were given to the model (`CitationMarkers.give`) are
  kept: a marker with any other id is removed and cites nothing;
- `Usage(call_id, response_id, input, cache_read, cache_write, output, reasoning, estimated)`, once per call, from the
  model's `usage_metadata`. Input includes the cache reads and writes, output includes the reasoning, as LangChain
  counts them. A call stopped before the provider sent its usage reports an estimate instead (`estimated=True`, Story
  2.6): input is the code points of the messages' text ÷ 4, output the code points of the text emitted ÷ 4, both
  rounded up;
- `ModelFailed(code)`: `model.refused` (a refusal or a content filter), `model.question_flagged` (a prompt filter's
  jailbreak check flagged the user turn outside the documents: the question), `model.source_flagged` (its
  indirect-attack check flagged a document: a passage), `model.unavailable` (connection, timeout, 5xx, 429, an error
  inside the stream) or `model.truncated` (the length or max-tokens stop, or a prompt past the context window). The
  two prompt-filter codes come from Azure's 400 `content_filter` answer, `innererror.content_filter_result`; a filter
  block that flags neither is `model.refused`.

The markers are parsed across chunks: a chunk that ends inside a possible marker holds that part back until the next
chunk decides it. A provider's own citation feature may be used inside `models/` only; outside, a citation is always
this event. Nothing inside a call is retried: a job retry is the only retry.

A stop reason arrives at the end of the stream, so text deltas may already have been emitted when a `content_filter`,
refusal or length stop turns into `ModelFailed`: a consumer must treat a trailing `ModelFailed` as ending the answer,
and the deltas already passed on are not withdrawn.

The GM's stop (Story 2.6): `stream_call` asks `should_stop` before it takes each chunk. Once it answers true, the
provider's stream is closed (no more chunks are read), the held-back marker text is flushed, the usage (or its estimate)
is emitted, and the call ends `stopped`, with no failure.
"""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anthropic
import openai
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.runnables import Runnable

REFUSED = "model.refused"
QUESTION_FLAGGED = "model.question_flagged"
SOURCE_FLAGGED = "model.source_flagged"
UNAVAILABLE = "model.unavailable"
TRUNCATED = "model.truncated"
FAILURE_CODES: tuple[str, ...] = (REFUSED, QUESTION_FLAGGED, SOURCE_FLAGGED, UNAVAILABLE, TRUNCATED)

OPEN = "[["
CLOSE = "]]"
MAX_MARKER_ID = 64
_MARKER_ID = re.compile(r"[A-Za-z0-9-]{1,64}", re.ASCII)
# What may still become a marker once more text arrives: the id so far, and perhaps the first closing bracket.
_PARTIAL_MARKER = re.compile(r"[A-Za-z0-9-]{0,64}\]?", re.ASCII)

# The stop reasons of each provider's stream (Chat Completions `finish_reason`, Anthropic `stop_reason`).
_TRUNCATED_STOPS = frozenset({"length", "max_tokens", "model_context_window_exceeded"})
_REFUSED_STOPS = frozenset({"content_filter", "refusal"})
_CONTENT_FILTER = "content_filter"
# The prompt filter's categories with a code of their own, in the order they are checked (Azure's Prompt Shields).
_FLAGGED_CATEGORIES = (("jailbreak", QUESTION_FLAGGED), ("indirect_attack", SOURCE_FLAGGED))
# Anthropic's error events inside a stream that are the provider's own failure.
_UNAVAILABLE_ERRORS = frozenset({"overloaded_error", "api_error", "rate_limit_error", "timeout_error"})
_TOO_MANY_REQUESTS = 429
_SERVER_ERROR = 500
_FIRST_ERROR_STATUS = 400


@dataclass(frozen=True)
class TextDelta:
    """Display text, citation markers removed. Library-derived: never logged."""

    text: str


@dataclass(frozen=True)
class Citation:
    """A given passage id, cited at this code-point offset of the display text."""

    passage_id: str
    offset: int


@dataclass(frozen=True)
class Usage:
    """One call's tokens, reported once under the worker's call id, with the provider's response id when there is
    one."""

    call_id: str
    response_id: str | None
    input: int
    cache_read: int
    cache_write: int
    output: int
    reasoning: int
    estimated: bool = False


@dataclass(frozen=True)
class ModelFailed:
    """The call ended without a usable answer: `model.refused`, `model.question_flagged`, `model.source_flagged`,
    `model.unavailable` or `model.truncated`."""

    code: str


type ModelEvent = TextDelta | Citation | Usage | ModelFailed
type Emit = Callable[[ModelEvent], None]


def new_call_id() -> str:
    """A worker-minted call id: a UUIDv7, so ids sort by time."""
    return str(uuid.uuid7())


class CitationMarkers:
    """Removes `[[<passage id>]]` markers from streamed text and turns the given ids into citations.

    The display offset runs across every call of one answer, so a graph that calls the model again after a tool keeps
    counting where the last call stopped.
    """

    def __init__(self, given: Iterable[str] = ()) -> None:
        self._given: dict[str, str] = {}
        self._held = ""
        self._offset = 0
        self.give(given)

    @property
    def offset(self) -> int:
        """The length of the display text so far, in code points."""
        return self._offset

    def give(self, ids: Iterable[str]) -> None:
        """Ids handed to the model (tool results, user-turn data): only these may be cited."""
        for passage_id in ids:
            self._given.setdefault(passage_id.lower(), passage_id)

    def feed(self, text: str) -> list[TextDelta | Citation]:
        """The events of one more piece of text; a possible marker at its end is held back."""
        buffer, self._held = self._held + text, ""
        events: list[TextDelta | Citation] = []
        plain: list[str] = []
        position = 0
        while position < len(buffer):
            start = buffer.find(OPEN, position)
            if start < 0:
                end = len(buffer) - 1 if buffer.endswith("[") else len(buffer)
                plain.append(buffer[position:end])
                self._held = buffer[end:]
                break
            plain.append(buffer[position:start])
            close = buffer.find(CLOSE, start + len(OPEN))
            inside = buffer[start + len(OPEN) : close] if close >= 0 else buffer[start + len(OPEN) :]
            if close < 0 and _PARTIAL_MARKER.fullmatch(inside):
                self._held = buffer[start:]
                break
            if close >= 0 and _MARKER_ID.fullmatch(inside):
                self._text(plain, events)
                cited = self._given.get(inside.lower())
                if cited is not None:
                    events.append(Citation(cited, self._offset))
                position = close + len(CLOSE)
                continue
            # Not a marker: the first bracket is text, and the scan goes on from the second.
            plain.append("[")
            position = start + 1
        self._text(plain, events)
        return events

    def flush(self) -> list[TextDelta | Citation]:
        """What was held back at the end of the stream is text: a marker that never closed cites nothing."""
        held, self._held = self._held, ""
        events: list[TextDelta | Citation] = []
        self._text([held], events)
        return events

    def _text(self, plain: list[str], events: list[TextDelta | Citation]) -> None:
        text = "".join(plain)
        plain.clear()
        if text:
            events.append(TextDelta(text))
            self._offset += len(text)


def _count(details: Mapping[str, Any] | None, *names: str) -> int:
    if not details:
        return 0
    return sum(int(details.get(name) or 0) for name in names)


def usage_of(call_id: str, response_id: str | None, message: AIMessageChunk | None) -> Usage | None:
    """The call's usage from LangChain's `usage_metadata`, or `None` when the provider sent none."""
    metadata = message.usage_metadata if message is not None else None
    if not metadata:
        return None
    inputs = metadata.get("input_token_details")
    outputs = metadata.get("output_token_details")
    return Usage(
        call_id=call_id,
        response_id=response_id,
        input=int(metadata.get("input_tokens") or 0),
        cache_read=_count(inputs, "cache_read"),
        cache_write=_count(inputs, "cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"),
        output=int(metadata.get("output_tokens") or 0),
        reasoning=_count(outputs, "reasoning"),
    )


def stop_failure(metadata: Mapping[str, Any]) -> str | None:
    """`model.truncated` or `model.refused` from a chunk's stop reason or refusal; `None` for a normal stop."""
    stop = metadata.get("finish_reason") or metadata.get("stop_reason")
    if stop in _TRUNCATED_STOPS:
        return TRUNCATED
    if stop in _REFUSED_STOPS or metadata.get("refusal"):
        return REFUSED
    return None


def _error_code(body: object) -> str | None:
    """The `code` or `type` of a provider's error body, at its top or under `error`."""
    if not isinstance(body, Mapping):
        return None
    inner = body.get("error")
    for found in (body, inner if isinstance(inner, Mapping) else {}):
        for name in ("code", "type"):
            value = found.get(name)
            if isinstance(value, str) and value != "error":
                return value
    return None


def failure_code(error: BaseException) -> str | None:
    """The typed failure of an exception from a model call, or `None` for one that is not the provider's to report
    (it propagates: the lane reports the attempt failed by the exception's type only). A prompt past the context
    window is `model.truncated`, whatever status it came with: another attempt would overflow again."""
    if isinstance(error, ContextOverflowError):
        return TRUNCATED
    if isinstance(error, openai.APIStatusError | anthropic.APIStatusError):
        return _status_failure(error.status_code, error.body)
    unreachable = (
        ModelRateLimitError,
        ModelAPIError,
        ModelConnectionError,
        ModelTimeoutError,
        openai.APIConnectionError,
        anthropic.APIConnectionError,
        # An error event inside a Chat Completions stream: the provider failed after it answered 200.
        openai.APIError,
    )
    return UNAVAILABLE if isinstance(error, unreachable) else None


def _filter_failure(body: object) -> str:
    """The code of a content-filter block, from `innererror.content_filter_result` (the OpenAI SDK's `body` is the
    error object itself; under `error` too). The categories that are `filtered` decide: jailbreak first
    (`model.question_flagged`), then indirect attack (`model.source_flagged`), any other `model.refused`. Only when none
    is filtered does a category that is only `detected` (annotated) name the cause, in the same order."""
    if not isinstance(body, Mapping):
        return REFUSED
    nested = body.get("error")
    for found in (body, nested if isinstance(nested, Mapping) else {}):
        inner = found.get("innererror")
        result = inner.get("content_filter_result") if isinstance(inner, Mapping) else None
        if not isinstance(result, Mapping):
            continue
        filtered = {name for name, verdict in result.items() if _verdict(verdict, "filtered")}
        named = filtered or {name for name, verdict in result.items() if _verdict(verdict, "detected")}
        for category, code in _FLAGGED_CATEGORIES:
            if category in named:
                return code
        return REFUSED
    return REFUSED


def _verdict(verdict: object, key: str) -> bool:
    return isinstance(verdict, Mapping) and verdict.get(key) is True


def _status_failure(status: int, body: object) -> str | None:
    """A provider's error answer: 429 and 5xx are outages, a content filter a refusal (or a flagged question or
    source), and an error event inside an Anthropic stream (status 200) an outage when it is the provider's own
    failure."""
    if status == _TOO_MANY_REQUESTS or status >= _SERVER_ERROR:
        return UNAVAILABLE
    code = _error_code(body)
    if code == _CONTENT_FILTER:
        return _filter_failure(body)
    if status < _FIRST_ERROR_STATUS and code in _UNAVAILABLE_ERRORS:
        return UNAVAILABLE
    return None


def _response_id(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("id")
    return value if isinstance(value, str) and value and not value.startswith("lc_") else None


@dataclass
class CallResult:
    """How one call ended: the whole message (for its tool calls) when it answered, else the failure's code, or
    `stopped` (no message, no failure) when `should_stop` ended it."""

    message: AIMessageChunk | None
    failed: str | None = None
    stopped: bool = False


CODE_POINTS_PER_TOKEN = 4


def estimated_usage(call_id: str, messages: Sequence[BaseMessage], emitted: int) -> Usage:
    """An estimate for a call the provider sent no usage for: the code points of the messages' text, and of the
    `emitted` display text, each ÷ 4 and rounded up."""
    sent = sum(len(message.text) for message in messages)
    return Usage(
        call_id=call_id,
        response_id=None,
        input=math.ceil(sent / CODE_POINTS_PER_TOKEN),
        cache_read=0,
        cache_write=0,
        output=math.ceil(emitted / CODE_POINTS_PER_TOKEN),
        reasoning=0,
        estimated=True,
    )


def _close(stream: object) -> None:
    """Closes a provider's stream that was left before its end, so its connection is let go."""
    close = getattr(stream, "close", None)
    if callable(close):
        close()


@dataclass
class _Streamed:
    """What one call's stream gave so far: kept when the stream fails or is stopped."""

    whole: AIMessageChunk | None = None
    response_id: str | None = None
    stop_reason: str | None = None
    stopped: bool = False
    emitted: int = 0


def _read(
    stream: Any,
    markers: CitationMarkers,
    send: Callable[[TextDelta | Citation], None],
    should_stop: Callable[[], bool] | None,
    into: _Streamed,
) -> None:
    """Takes the stream's chunks into `into`, asking `should_stop` before each one."""
    chunks = iter(stream)
    while should_stop is None or not should_stop():
        chunk = next(chunks, None)
        if chunk is None:
            return
        if not isinstance(chunk, AIMessageChunk):
            continue
        into.whole = chunk if into.whole is None else into.whole + chunk
        into.response_id = into.response_id or _response_id(chunk.response_metadata)
        into.stop_reason = stop_failure(chunk.response_metadata) or into.stop_reason
        text = chunk.text
        if text:
            for event in markers.feed(text):
                send(event)
    into.stopped = True


def stream_call(
    model: BaseChatModel | Runnable[Any, Any],
    messages: Sequence[BaseMessage],
    *,
    markers: CitationMarkers,
    emit: Emit,
    call_id: str | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> CallResult:
    """Streams one model call, emitting its events as they come, then its usage, then a failure if it ended in one.

    `should_stop` is asked before each chunk is taken: once true, the call ends stopped, as the module says. An
    exception that is not a typed failure propagates; nothing is retried.
    """
    call = call_id or new_call_id()
    streamed = _Streamed()
    failed: str | None = None

    def send(event: TextDelta | Citation) -> None:
        if isinstance(event, TextDelta):
            streamed.emitted += len(event.text)
        emit(event)

    stream: Any = None
    try:
        stream = model.stream(list(messages))
        _read(stream, markers, send, should_stop, streamed)
    except Exception as error:
        failed = failure_code(error)
        if failed is None:
            raise
    finally:
        if streamed.stopped:
            _close(stream)
    for event in markers.flush():
        send(event)
    usage = usage_of(call, streamed.response_id, streamed.whole)
    if usage is None and streamed.stopped:
        usage = estimated_usage(call, messages, streamed.emitted)
    if usage is not None:
        emit(usage)
    if streamed.stopped:
        return CallResult(message=None, stopped=True)
    failed = failed or streamed.stop_reason
    if failed is not None:
        emit(ModelFailed(failed))
        return CallResult(message=None, failed=failed)
    return CallResult(message=streamed.whole)
