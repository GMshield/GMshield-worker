"""The one way to a chat model (Story 2.10): `chat_model` builds the LangChain chat model of an attempt's choice.

- Azure OpenAI and OpenAI: `ChatOpenAI` on Chat Completions, chosen explicitly (`use_responses_api=False`), at the
  explicit base URL (Azure: the resource's `https://<host>/openai/v1/`, with the deployment name as the model; no
  `api-version` is sent and `AzureChatOpenAI` is not used). Storage is off (`store=False`), usage is streamed
  (`stream_usage=True`), and no provider-hosted tool is ever bound.
- Anthropic: `ChatAnthropic` at `https://api.anthropic.com`, usage streamed.
- Test mode: the fake chat model of `fake.py`, for whichever provider the claim names.

Every model runs with `max_retries=0`: a job retry is the only retry. Building one calls nothing. The key lives in the
model's secret field only and never appears in its repr. The provider's response id is kept in each chunk's
`response_metadata["id"]` (Chat Completions sends it in every chunk; Anthropic in `message_start`), which LangChain
drops otherwise.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from gmshield_worker.models.fake import FakeChatModel
from gmshield_worker.models.providers import (
    ANTHROPIC,
    PROVIDER_UNKNOWN,
    PROVIDERS,
    Endpoint,
    ModelChoice,
    ProviderUnavailable,
    endpoint,
)
from gmshield_worker.settings import ModelKeys

DEFAULT_MAX_TOKENS = 4096
# Connect, read and write limits of one call; a stream that stalls longer is `model.unavailable`.
REQUEST_TIMEOUT_SECONDS = 60.0


class _ChatOpenAI(ChatOpenAI):
    """`ChatOpenAI` that keeps the response id of each Chat Completions chunk."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if generation is None or not isinstance(generation.message, AIMessageChunk):
            return generation
        response_id = chunk.get("id")
        if isinstance(response_id, str) and response_id:
            generation.message.response_metadata["id"] = response_id
        choices = chunk.get("choices") or []
        delta = choices[0].get("delta") if choices and isinstance(choices[0], dict) else None
        if isinstance(delta, dict) and delta.get("refusal"):
            # A Chat Completions refusal arrives as its own delta field, which LangChain does not carry.
            generation.message.response_metadata["refusal"] = True
        return generation


class _ChatAnthropic(ChatAnthropic):
    """`ChatAnthropic` that keeps the message id of `message_start`."""

    def _make_message_chunk_from_anthropic_event(self, event: Any, **kwargs: Any) -> tuple[Any, Any]:
        message, block_start = super()._make_message_chunk_from_anthropic_event(event, **kwargs)
        if isinstance(message, AIMessageChunk) and getattr(event, "type", None) == "message_start":
            message_id = getattr(getattr(event, "message", None), "id", None)
            if isinstance(message_id, str) and message_id:
                message.response_metadata["id"] = message_id
        return message, block_start


def build_chat_model(choice: ModelChoice, where: Endpoint, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> BaseChatModel:
    """The provider's chat model at this endpoint. Calls nothing."""
    key = SecretStr(where.api_key)
    if where.provider == ANTHROPIC:
        return _ChatAnthropic(
            model_name=choice.model,
            api_key=key,
            base_url=where.base_url,
            max_tokens_to_sample=max_tokens,
            stop=None,
            max_retries=0,
            stream_usage=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    return _ChatOpenAI(
        model=choice.model,
        api_key=key,
        base_url=where.base_url,
        use_responses_api=False,
        store=False,
        stream_usage=True,
        max_retries=0,
        max_completion_tokens=max_tokens,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def chat_model(choice: ModelChoice, keys: ModelKeys, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> BaseChatModel:
    """The chat model the claim names: the fake one in test mode, else the provider's at its explicit base URL.

    Raises `ProviderUnavailable` (`model.provider_unknown`, `model.provider_key_missing`) when this worker cannot call
    that provider. Never another provider instead.
    """
    if keys.test_mode:
        if choice.provider not in PROVIDERS:
            raise ProviderUnavailable(PROVIDER_UNKNOWN)
        return FakeChatModel(provider=choice.provider, model_id=choice.model)
    return build_chat_model(choice, endpoint(keys, choice.provider), max_tokens=max_tokens)
