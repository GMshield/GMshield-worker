"""The fake chat model (test mode, Story 2.10): it answers for any provider, and nothing leaves the process.

It is a LangChain chat model, so graphs, tools and `output.stream_call` treat it exactly as a real one, and it shapes
its chunks as its provider's integration does: Chat Completions chunks (`finish_reason`, a `chatcmpl-` id in every
chunk, usage in a last empty chunk) for `azure-openai` and `openai`, Anthropic chunks (`msg_` id at the start,
`stop_reason` and usage at the end) for `anthropic`.

Its answer is deterministic. It cites, with `[[<id>]]` markers, the passage ids it was given (in tool results, else in
the user turn), streamed in small chunks so that markers split across them. Story 3.7: when the passages it was given
carry more than one tier (the `tier="…"` of each block, the backend's label), it answers as the tiers disagreeing: one
paragraph per tier in tier order (campaign, system, other), each opened by `[[tier:<tier>]]` and citing that tier's
passages. Story 3.8: when its system message allows model knowledge (it holds `MODEL_MARKER`, which only
`lanes.answer.MODEL_KNOWLEDGE_RULE` writes there), it adds a final paragraph opened by `[[tier:model]]`, with no
citation; with no passage that paragraph is the whole answer, with no sources-are-silent sentence. Directives in a user
turn change it:

- `fake:refusal`: no text, the provider's refusal stop (`content_filter`, `refusal`);
- `fake:outage`: the call fails before any chunk, as an unreachable provider does;
- `fake:truncated`: half the answer, then the provider's length stop (`length`, `max_tokens`);
- `fake:tool`: when `read_passages` is bound and no tool result is in the conversation yet, a call of `read_passages`
  with the passage ids of the user turn; once the results are in, the answer cites them.

Story 4.2: when the records schema (`RECORDS_TOOL`, as `ingestion.records` names it) is bound as structured output,
the answer is one call of it: a record for each `fake:record <type> <name>` line of a passage block (the type a record
type key, Story 4.4 part B: `fake:record talento Ataque Poderoso`) (`<passage
id="…" …>`), with one field and that passage's id, and no record otherwise, so the pipeline's end-to-end run reads
prose into no record. `fake:refusal` and `fake:truncated` give no call and their stop, `fake:outage` fails as above.
From Story 4.4 the call also holds a proposal for each `fake:proposal <record_type|term> <name>` line of a passage
block, which quotes that line as its example (and, for a term, as its definition), and none otherwise.

Story 4.3 part C: a record block of the user turn (`<record id="…" tier="…" …>`, as `lanes.answer.user_turn` writes
it after the passages) is cited by its id as a passage is, in the order the blocks come, with its tier like a
passage's.

Story 5.5 part A: an artifact's record block that names the record it overrides (`overrides="…"`, which
`lanes.answer.record_block` writes only when that record is handed too) always meets that record's block of another
tier (campaign and system), so the answer is one paragraph per tier, the artifact's campaign paragraph first: the fake
reads an override as the tiers disagreeing, as `lanes.answer.SYSTEM_PROMPT` tells a real model to.

Story 5.3 part A: when its system message is a draft's (it holds `DRAFT_MARKER`, which only `lanes.draft.DRAFT_PROMPT`
writes there), it writes a draft instead (`draft_text`): "Rascunho simulado (provider)" under a heading, one field
citing each id it was given, in their order and with no tier marker, or one sentence when it was given none; the model
paragraph and the directives work as for an answer.

Story 5.3 part B: when the draft's proposal schema (`PROPOSAL_TOOL`, as `lanes.draft.PROPOSAL_TITLE` names it) is
bound as structured output, the answer is one call of it: `{name, recordType}` from a `fake:draft <type> <name>` line of
the request (the `<question>` element only, never the draft or the record types; the type a record type key, or `none`
for null), else "Rascunho simulado" with the first record type key of the user turn (`<record-type key="…">`), or null
when it lists none. `fake:refusal` and `fake:truncated` give no call and their stop, as for the records schema.

Story 3.10: the earlier turns of a conversation (`<earlier-turn>` elements, as `lanes.answer.user_turn` writes them) are
left out before anything is read from a user turn, so a directive or an id in an earlier question or answer neither
changes the answer nor is cited: directives and cited ids come from the question and the passages only.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import ModelConnectionError
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
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
# Story 4.2: the records step's structured output, and the line that asks the fake for one record.
RECORDS_TOOL = "extracted_records"
RECORD = "fake:record"
_RECORD_BLOCK = re.compile(r'<passage id="([0-9a-f-]{36})"[^>]*>(.*?)</passage>', re.ASCII | re.DOTALL)
# The name starts with a character that is no blank, so no run of blanks is ever matched two ways (linear).
_RECORD_LINE = re.compile(r"fake:record[ \t]+([a-z][a-z0-9_]{0,31})[ \t]+([^\s<][^\n<]{0,199})")
# Story 5.3 part B: the draft's proposal schema, the line of the request that sets its answer (the same linear shape
# as a record's line), the request element and a record type block of the proposal's user turn.
PROPOSAL_TOOL = "draft_proposal"
DRAFT_PROPOSAL = "fake:draft"
DRAFT_DEFAULT_NAME = "Rascunho simulado"
_DRAFT_LINE = re.compile(r"fake:draft[ \t]+([a-z][a-z0-9_]{0,31})[ \t]+([^\s<][^\n<]{0,199})")
_QUESTION = re.compile(r"<question>(.*?)</question>", re.DOTALL)
_RECORD_TYPE = re.compile(r'<record-type key="([a-z][a-z0-9_]{0,31})">', re.ASCII)
# Story 4.4: the line that asks the fake for one proposal, which quotes the line itself (the same linear shape).
PROPOSAL = "fake:proposal"
_PROPOSAL_LINE = re.compile(r"fake:proposal[ \t]+(record_type|term)[ \t]+([^\s<][^\n<]{0,119})")
_PASSAGE_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.ASCII)
# A data block's id and tier, as `lanes.answer.user_turn` writes them: a passage's or (Story 4.3 part C) a record's.
_TIERED = re.compile(r'<(?:passage|record) id="([0-9a-f-]{36})" tier="([a-z]+)"', re.ASCII)
TIER_ORDER = ("campaign", "system", "other")
# Story 3.8: the marker the model-knowledge rule names in the system message, and that opens the fake's model paragraph.
MODEL_MARKER = "[[tier:model]]"
# Story 5.3 part A: the words that open a draft's system message (`lanes.draft.DRAFT_PROMPT`), which tell the fake to
# write a draft instead of an answer.
DRAFT_MARKER = "You write a draft of game content"
_CHUNK = 7
# Story 3.10: an earlier turn of the conversation, as `lanes.answer.user_turn` writes it (its text escaped inside).
_EARLIER_OPEN = "<earlier-turn>"
_EARLIER_CLOSE = "</earlier-turn>"


def _text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.text)


def _without_history(text: str) -> str:
    """The text without its earlier turns (Story 3.10): each `<earlier-turn>` element is cut out whole."""
    kept: list[str] = []
    start = 0
    while (opened := text.find(_EARLIER_OPEN, start)) != -1:
        closed = text.find(_EARLIER_CLOSE, opened)
        if closed == -1:
            break
        kept.append(text[start:opened])
        start = closed + len(_EARLIER_CLOSE)
    kept.append(text[start:])
    return "".join(kept)


def _digest(messages: Sequence[BaseMessage]) -> str:
    joined = "\x00".join(f"{message.type}:{_text(message)}" for message in messages)
    return hashlib.sha256(joined.encode()).hexdigest()[:24]


def fake_records(text: str) -> list[dict[str, Any]]:
    """The fake's records (Story 4.2): one per `fake:record <type> <name>` line of each passage block, in order."""
    found: list[dict[str, Any]] = []
    for passage_id, body in _RECORD_BLOCK.findall(text):
        for kind, name in _RECORD_LINE.findall(body):
            found.append(
                {
                    "type": kind,
                    "name": name.strip(),
                    "fields": [{"name": "Fonte", "value": "simulada"}],
                    "passageIds": [passage_id],
                }
            )
    return found


def fake_proposals(text: str) -> list[dict[str, Any]]:
    """The fake's proposals (Story 4.4): one per `fake:proposal <record_type|term> <name>` line of each passage block,
    in order, quoting the line itself as its example (and a term's definition)."""
    found: list[dict[str, Any]] = []
    for passage_id, body in _RECORD_BLOCK.findall(text):
        for match in _PROPOSAL_LINE.finditer(body):
            kind, quote = match.group(1), match.group(0).strip()
            found.append(
                {
                    "kind": kind,
                    "name": match.group(2).strip(),
                    "definition": quote if kind == "term" else "",
                    "examples": [{"passageId": passage_id, "quote": quote}],
                }
            )
    return found


def fake_draft_proposal(text: str) -> dict[str, str | None]:
    """The fake's draft proposal (Story 5.3 part B), as the module says: from the request's `fake:draft` line, else
    `DRAFT_DEFAULT_NAME` with the first record type key listed."""
    request = _QUESTION.search(text)
    line = _DRAFT_LINE.search(request.group(1)) if request is not None else None
    if line is not None:
        kind = line.group(1)
        return {"name": html.unescape(line.group(2).strip()), "recordType": None if kind == "none" else kind}
    first = _RECORD_TYPE.search(text)
    return {"name": DRAFT_DEFAULT_NAME, "recordType": first.group(1) if first is not None else None}


def answer_text(
    provider: str, ids: Sequence[str], tiers: Mapping[str, str] | None = None, *, model_knowledge: bool = False
) -> str:
    """The fake's answer: one cited statement per id, or the sources-are-silent sentence without any. With the ids of
    more than one tier (`tiers`: id to tier), one paragraph per tier in tier order, each opened by its tier marker.
    With `model_knowledge`, a last paragraph opened by `MODEL_MARKER`, uncited; with no id, only that paragraph."""
    model = f"{MODEL_MARKER}Conhecimento do modelo simulado ({provider}): em geral, a regra vale assim."
    if not ids:
        return model if model_knowledge else f"Resposta simulada ({provider}): as fontes não dizem nada sobre isso."
    sourced = _sourced_text(provider, ids, tiers)
    return f"{sourced}\n\n{model}" if model_knowledge else sourced


def draft_text(provider: str, ids: Sequence[str], *, model_knowledge: bool = False) -> str:
    """The fake's draft (Story 5.3 part A): a heading naming "Rascunho simulado (provider)", then one cited field per
    id in the order given (the backend's tier order, campaign first), with no tier marker; without any id, one sentence
    saying nothing fits. With `model_knowledge`, a last paragraph opened by `MODEL_MARKER`, uncited, as `answer_text`
    adds it; with no id, only that paragraph."""
    model = f"{MODEL_MARKER}Conhecimento do modelo simulado ({provider}): em geral, assim se faz."
    if not ids:
        return model if model_knowledge else f"Rascunho simulado ({provider}): nada nas fontes serve para este pedido."
    fields = "\n".join(f"- Elemento {number}: da fonte ✓[[{source}]]" for number, source in enumerate(ids, 1))
    drafted = f"## Rascunho simulado ({provider})\n\n{fields}\n\nFim."
    return f"{drafted}\n\n{model}" if model_knowledge else drafted


def _sourced_text(provider: str, ids: Sequence[str], tiers: Mapping[str, str] | None) -> str:
    """The cited part of `answer_text`, as it says."""
    by_tier: dict[str, list[str]] = {}
    for passage_id in ids:
        by_tier.setdefault((tiers or {}).get(passage_id, ""), []).append(passage_id)
    if len(by_tier) < 2 or "" in by_tier:
        statements = [f"Trecho {number} confirma a regra ✓[[{passage_id}]]" for number, passage_id in enumerate(ids, 1)]
        return f"Resposta simulada ({provider}): " + " ".join(statements) + " Fim."
    paragraphs: list[str] = []
    number = 0
    for tier in sorted(by_tier, key=lambda name: TIER_ORDER.index(name) if name in TIER_ORDER else len(TIER_ORDER)):
        statements = []
        for passage_id in by_tier[tier]:
            number += 1
            statements.append(f"Trecho {number} confirma a regra ✓[[{passage_id}]]")
        lead = f"Resposta simulada ({provider}):" if not paragraphs else "Alternativa:"
        paragraphs.append(f"[[tier:{tier}]]{lead} " + " ".join(statements))
    return "\n\n".join(paragraphs) + " Fim."


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
        message = (
            AIMessage(content="")
            if whole is None
            else AIMessage(**whole.model_dump(exclude={"type", "tool_call_chunks", "chunk_position"}))
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del stop, run_manager
        asked = " ".join(_without_history(_text(message)) for message in messages if isinstance(message, HumanMessage))
        if OUTAGE in asked:
            raise ModelConnectionError("fake outage")
        results = [message for message in messages if isinstance(message, ToolMessage)]
        tools = {tool.get("function", {}).get("name") for tool in kwargs.get("tools") or []}
        digest = _digest(messages)
        anthropic = self.provider == ANTHROPIC
        response_id = f"msg_fake{digest}" if anthropic else f"chatcmpl-fake{digest}"
        words = sum(len(_text(message).split()) for message in messages)

        if RECORDS_TOOL in tools:
            yield from self._records(asked, response_id, words, anthropic=anthropic, digest=digest)
            return

        if PROPOSAL_TOOL in tools:
            yield from self._structured(
                PROPOSAL_TOOL, fake_draft_proposal(asked), asked, response_id, words, anthropic=anthropic, digest=digest
            )
            return

        if TOOL in asked and READ_PASSAGES in tools and not results:
            ids = list(dict.fromkeys(_PASSAGE_ID.findall(asked)))
            call = tool_call_chunk(
                name=READ_PASSAGES, args=json.dumps({"ids": ids}), id=f"call_fake{digest[:12]}", index=0
            )
            yield self._chunk(AIMessageChunk(content="", tool_call_chunks=[call]), response_id, first=True)
            yield self._last("tool_use" if anthropic else "tool_calls", response_id, words, 1)
            return

        given = " ".join(_text(message) for message in results) if results else asked
        tiers = {passage_id: tier for passage_id, tier in _TIERED.findall(given)}
        system = " ".join(_text(message) for message in messages if isinstance(message, SystemMessage))
        model_knowledge = MODEL_MARKER in system
        cited = list(dict.fromkeys(_PASSAGE_ID.findall(given)))
        if DRAFT_MARKER in system:
            text = draft_text(self.provider, cited, model_knowledge=model_knowledge)
        else:
            text = answer_text(self.provider, cited, tiers, model_knowledge=model_knowledge)
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

    def _records(
        self, asked: str, response_id: str, words: int, *, anthropic: bool, digest: str
    ) -> Iterator[ChatGenerationChunk]:
        """The structured answer of a records step (Story 4.2): one call of `RECORDS_TOOL` (from Story 4.4 with its
        proposals), or a refusal's or a truncation's stop with no call."""
        found = {"records": fake_records(asked), "proposals": fake_proposals(asked)}
        yield from self._structured(RECORDS_TOOL, found, asked, response_id, words, anthropic=anthropic, digest=digest)

    def _structured(
        self,
        tool: str,
        answer: Mapping[str, Any],
        asked: str,
        response_id: str,
        words: int,
        *,
        anthropic: bool,
        digest: str,
    ) -> Iterator[ChatGenerationChunk]:
        """A structured answer: one call of `tool` with `answer` as its arguments, or a refusal's or a truncation's
        stop with no call."""
        if REFUSAL in asked:
            yield self._chunk(AIMessageChunk(content=""), response_id, first=True)
            yield self._last("refusal" if anthropic else "content_filter", response_id, words, 0)
            return
        if TRUNCATED in asked:
            yield self._chunk(AIMessageChunk(content=""), response_id, first=True)
            yield self._last("max_tokens" if anthropic else "length", response_id, words, 1)
            return
        args = json.dumps(answer, ensure_ascii=False)
        call = tool_call_chunk(name=tool, args=args, id=f"call_fake{digest[:12]}", index=0)
        yield self._chunk(AIMessageChunk(content="", tool_call_chunks=[call]), response_id, first=True)
        yield self._last("tool_use" if anthropic else "tool_calls", response_id, words, 1)

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
