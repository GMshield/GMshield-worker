"""One structured call (Story 4.2): a model answers with a JSON object of a given schema, never with free text.

`structured_call` binds the schema with LangChain's `with_structured_output(include_raw=True)`: for Chat Completions
(Azure OpenAI and OpenAI, `ChatOpenAI`) through the provider's structured outputs (`method="json_schema"`, strict),
for any other model through a forced tool call, the model's own default. Nothing is streamed: the answer is one whole
object. The call's usage comes from the raw message's `usage_metadata`, under the worker-minted call id, with the
provider's response id when it sent one (`output.usage_of`); a call that raised has none to report.

How it ends, beside the object (`StructuredResult.failed`):

- `model.refused`: a refusal (Chat Completions' `refusal` field), a content filter or a refusal stop (the OpenAI SDK's
  parse raises `ContentFilterFinishReasonError` for it);
- `model.truncated`: the length or max-tokens stop (the SDK's parse raises `LengthFinishReasonError`, which still
  carries the call's usage), or a prompt past the context window;
- `model.unavailable`: connection, timeout, 5xx, 429 (`output.failure_code`); an exception that is none of these
  propagates, as `output.stream_call` lets it;
- `model.output_invalid`: the answer is no object of the schema's shape (it did not parse).

Nothing is retried here: a job retry is the only retry. The object is checked further by its caller, which trusts no
field of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from gmshield_worker.models.output import (
    REFUSED,
    TRUNCATED,
    Usage,
    failure_code,
    new_call_id,
    response_id_of,
    stop_failure,
    usage_of,
)

OUTPUT_INVALID = "model.output_invalid"


@dataclass(frozen=True)
class StructuredResult:
    """How one structured call ended: the object when it answered one, else the failure's code; and its usage when the
    provider sent it. The object is library-derived: never logged."""

    parsed: dict[str, Any] | None = field(default=None, repr=False)
    usage: Usage | None = None
    failed: str | None = None


def _completion_usage(call_id: str, completion: object) -> Usage | None:
    """The usage of a Chat Completion the OpenAI SDK refused to parse, or `None` when it carries none."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens_details", None)
    output = getattr(usage, "completion_tokens_details", None)
    response_id = getattr(completion, "id", None)
    return Usage(
        call_id=call_id,
        response_id=response_id if isinstance(response_id, str) and response_id else None,
        input=int(getattr(usage, "prompt_tokens", 0) or 0),
        cache_read=int(getattr(prompt, "cached_tokens", 0) or 0),
        cache_write=0,
        output=int(getattr(usage, "completion_tokens", 0) or 0),
        reasoning=int(getattr(output, "reasoning_tokens", 0) or 0),
    )


def _method(model: BaseChatModel) -> dict[str, Any]:
    """Chat Completions' strict structured outputs; any other model's own default (a forced tool call)."""
    if isinstance(model, ChatOpenAI):
        return {"method": "json_schema", "strict": True}
    return {}


def structured_call(
    model: BaseChatModel,
    schema: Mapping[str, Any],
    messages: Sequence[BaseMessage],
    *,
    call_id: str | None = None,
) -> StructuredResult:
    """One call of `model` for an object of `schema` (a JSON schema with a `title`), as the module says."""
    call = call_id or new_call_id()
    runnable = model.with_structured_output(dict(schema), include_raw=True, **_method(model))
    try:
        answer = runnable.invoke(list(messages), config={"callbacks": []})
    except openai.LengthFinishReasonError as error:
        return StructuredResult(usage=_completion_usage(call, error.completion), failed=TRUNCATED)
    except openai.ContentFilterFinishReasonError as error:
        return StructuredResult(usage=_completion_usage(call, error.completion), failed=REFUSED)
    except Exception as error:
        failed = failure_code(error)
        if failed is None:
            raise
        return StructuredResult(failed=failed)
    if not isinstance(answer, Mapping):
        answer = {}
    raw = answer.get("raw")
    message = raw if isinstance(raw, AIMessage) else None
    metadata = message.response_metadata if message is not None else {}
    usage = usage_of(call, response_id_of(metadata), message)
    failed = stop_failure(metadata)
    if message is not None and message.additional_kwargs.get("refusal"):
        failed = REFUSED
    parsed = answer.get("parsed")
    if failed is None and (answer.get("parsing_error") is not None or not isinstance(parsed, dict)):
        failed = OUTPUT_INVALID
    if failed is not None:
        return StructuredResult(usage=usage, failed=failed)
    return StructuredResult(parsed=parsed, usage=usage)
