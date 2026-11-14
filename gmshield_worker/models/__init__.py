"""Model clients: one interface, two implementations.

- `FakeModelClient` (`fake.py`): deterministic, no network. Test mode always selects it.
- `ClaudeModelClient` (`claude.py`): the Anthropic SDK. Building it calls nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from gmshield_worker.settings import Settings


class ModelClient(Protocol):
    """What the worker asks of a language model."""

    @property
    def kind(self) -> str:
        """`fake` or `claude`: logged at start, never a secret."""
        ...

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> str:
        """The model's text answer to one prompt."""
        ...


def select_model_client(settings: Settings) -> ModelClient:
    """The fake model in test mode; the Claude client otherwise. Neither calls anything here.

    Raises `ValueError` outside test mode without a key: the fake model never stands in silently.
    """
    from gmshield_worker.models.claude import ClaudeModelClient  # noqa: PLC0415
    from gmshield_worker.models.fake import FakeModelClient  # noqa: PLC0415

    if settings.test_mode:
        return FakeModelClient()
    if settings.anthropic_api_key is None:
        raise ValueError("The Claude client needs an API key outside test mode")
    return ClaudeModelClient(api_key=settings.anthropic_api_key)
