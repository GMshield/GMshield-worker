"""The Claude client, on the Anthropic SDK.

Building it only keeps the key: the SDK client is created on the first call, and nothing is
called at start. The key never appears in a repr or a log line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

# The architecture's full tier; the lightweight tier is claude-haiku-5-5. The backend will choose
# the model per job (AD-18).
DEFAULT_MODEL = "claude-sonnet-5-5"


class ClaudeModelClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._api_key = api_key
        self._model = model
        self._client: anthropic.Anthropic | None = None

    def __repr__(self) -> str:
        return f"ClaudeModelClient(model={self._model!r})"

    @property
    def kind(self) -> str:
        return "claude"

    @property
    def model(self) -> str:
        return self._model

    def _sdk(self) -> anthropic.Anthropic:
        if self._client is None:
            import anthropic  # noqa: PLC0415 - imported on first use: start-up stays light

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        message = self._sdk().messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
