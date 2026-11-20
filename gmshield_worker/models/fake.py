"""The fake chat model (test mode, Story 2.10): it answers for any provider, and nothing leaves the process.

It is a LangChain chat model, so graphs, tools and `output.stream_call` treat it exactly as a real one, and it shapes
its chunks as its provider's integration does: Chat Completions chunks (`finish_reason`, a `chatcmpl-` id in every
chunk, usage in a last empty chunk) for `azure-openai` and `openai`, Anthropic chunks (`msg_` id at the start,
`stop_reason` and usage at the end) for `anthropic`.

Its answer is deterministic. It cites, with `[[<id>]]` markers, the passage ids it was given (in tool results, else in
the user turn), streamed in small chunks so that markers split across them. Directives in a user turn change it:

- `fake:refusal`: no text, the provider's refusal stop (`content_filter`, `refusal`);
- `fake:outage`: the call fails before any chunk, as an unreachable provider does;
- `fake:truncated`: half the answer, then the provider's length stop (`length`, `max_tokens`);
- `fake:tool`: when `read_passages` is bound and no tool result is in the conversation yet, a call of `read_passages`
  with the passage ids of the user turn; once the results are in, the answer cites them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import ModelConnectionError
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from gmshield_worker.models.providers import ANTHROPIC

REFUSAL = "fake:refusal"
OUTAGE = "fake:outage"
TRUNCATED = "fake:truncated"
TOOL = "fake:tool"
READ_PASSAGES = "read_passages"
_PASSAGE_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.ASCII)
_CHUNK = 7


def _text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.text)


def _digest(messages: Sequence[BaseMessage]) -> str:
    joined = "\x00".join(f"{message.type}:{_text(message)}" for message in messages)
    return hashlib.sha256(joined.encode()).hexdigest()[:24]


def answer_text(provider: str, ids: Sequence[str]) -> str:
    """The fake's answer: one cited statement per id, or the sources-are-silent sentence without any."""
    if not ids:
        return f"Resposta simulada ({provider}): as fontes não dizem nada sobre isso."
    statements = [f"Trecho {number} confirma a regra ✓[[{passage_id}]]" for number, passage_id in enumerate(ids, 1)]
    return f"Resposta simulada ({provider}): " + " ".join(statements) + " Fim."


class FakeChatModel(BaseChatModel):
    """A deterministic chat model shaped like `provider`'s integration."""

    provider: str
    model_id: str = "fake"

    @property
    def _llm_type(self) -> str:
        return "gmshield-fake"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        del tool_choice
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        whole: AIMessageChunk | None = None
        for generation in self._stream(messages, stop, run_manager, **kwargs):
            chunk = generation.message
            if isinstance(chunk, AIMessageChunk):
                whole = chunk if whole is None else whole + chunk
        message = AIMessage(content="") if whole is None else AIMessage(**whole.model_dump(exclude={"type"}))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del stop, run_manager
        asked = " ".join(_text(message) for message in messages if isinstance(message, HumanMessage))
        if OUTAGE in asked:
            raise ModelConnectionError("fake outage")
        results = [message for message in messages if isinstance(message, ToolMessage)]
        tools = {tool.get("function", {}).get("name") for tool in kwargs.get("tools") or []}
        digest = _digest(messages)
        anthropic = self.provider == ANTHROPIC
        response_id = f"msg_fake{digest}" if anthropic else f"chatcmpl-fake{digest}"
        words = sum(len(_text(message).split()) for message in messages)

        if TOOL in asked and READ_PASSAGES in tools and not results:
            ids = list(dict.fromkeys(_PASSAGE_ID.findall(asked)))
            call = tool_call_chunk(
                name=READ_PASSAGES, args=json.dumps({"ids": ids}), id=f"call_fake{digest[:12]}", index=0
            )
            yield self._chunk(AIMessageChunk(content="", tool_call_chunks=[call]), response_id, first=True)
            yield self._last("tool_use" if anthropic else "tool_calls", response_id, words, 1)
            return

        given = " ".join(_text(message) for message in results) if results else asked
        text = answer_text(self.provider, list(dict.fromkeys(_PASSAGE_ID.findall(given))))
        if REFUSAL in asked:
            text, stop_reason = "", "refusal" if anthropic else "content_filter"
        elif TRUNCATED in asked:
            text, stop_reason = text[: len(text) // 2], "max_tokens" if anthropic else "length"
        else:
            stop_reason = "end_turn" if anthropic else "stop"
        pieces = [text[start : start + _CHUNK] for start in range(0, len(text), _CHUNK)]
        yield self._chunk(AIMessageChunk(content=""), response_id, first=True)
        for piece in pieces:
            yield self._chunk(AIMessageChunk(content=piece), response_id, first=False)
        yield self._last(stop_reason, response_id, words, len(pieces))

    def _chunk(self, message: AIMessageChunk, response_id: str, *, first: bool) -> ChatGenerationChunk:
        if self.provider != ANTHROPIC:
            message.response_metadata.update({"id": response_id, "model_provider": "openai"})
        elif first:
            message.response_metadata.update({"id": response_id, "model_name": self.model_id})
        return ChatGenerationChunk(message=message)

    def _last(self, stop_reason: str, response_id: str, words: int, pieces: int) -> ChatGenerationChunk:
        if self.provider == ANTHROPIC:
            message = AIMessageChunk(
                content="",
                response_metadata={"stop_reason": stop_reason, "stop_sequence": None},
                usage_metadata={
                    "input_tokens": words,
                    "output_tokens": pieces,
                    "total_tokens": words + pieces,
                    "input_token_details": {"cache_read": 0, "cache_creation": 0},
                },
            )
            return ChatGenerationChunk(message=message)
        message = AIMessageChunk(
            content="",
            response_metadata={"finish_reason": stop_reason, "model_name": self.model_id},
            usage_metadata={
                "input_tokens": words,
                "output_tokens": pieces,
                "total_tokens": words + pieces,
                "input_token_details": {"cache_read": 0},
                "output_token_details": {"reasoning": 0},
            },
        )
        return self._chunk(message, response_id, first=False)
