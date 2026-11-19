"""The providers a job's model may come from (Story 2.10, AD-18), and where each one is reached.

The backend chooses the provider, model and tier of every attempt and sends them with the claim (`ModelChoice`); the
worker never chooses one and never falls back to another. `endpoint` gives the explicit base URL and the key for a
provider: Azure OpenAI at the resource's `https://<host>/openai/v1/`, OpenAI at `https://api.openai.com/v1`,
Anthropic at `https://api.anthropic.com`. A provider this worker holds no key for fails the attempt with
`model.provider_key_missing`, one it does not know with `model.provider_unknown`; neither is retried, because no
attempt can change it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gmshield_worker.settings import ANTHROPIC, AZURE_OPENAI, OPENAI, PROVIDERS, ModelKeys

OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
TIERS: tuple[str, ...] = ("full", "lightweight")

PROVIDER_KEY_MISSING = "model.provider_key_missing"
PROVIDER_UNKNOWN = "model.provider_unknown"

__all__ = [
    "ANTHROPIC",
    "ANTHROPIC_BASE_URL",
    "AZURE_OPENAI",
    "OPENAI",
    "OPENAI_BASE_URL",
    "PROVIDERS",
    "PROVIDER_KEY_MISSING",
    "PROVIDER_UNKNOWN",
    "TIERS",
    "Endpoint",
    "ModelChoice",
    "ProviderUnavailable",
    "endpoint",
]


@dataclass(frozen=True)
class ModelChoice:
    """The model of one attempt, as the claim names it: the provider, its model id (for Azure OpenAI, the deployment
    name) and the tier."""

    provider: str
    model: str
    tier: str


@dataclass(frozen=True)
class Endpoint:
    """Where a provider is reached, and its key (never printed)."""

    provider: str
    base_url: str
    api_key: str = field(repr=False)


class ProviderUnavailable(Exception):
    """The claim names a provider this worker cannot call: `model.provider_key_missing` or `model.provider_unknown`.
    Never retried."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def endpoint(keys: ModelKeys, provider: str) -> Endpoint:
    """The provider's base URL and key; raises `ProviderUnavailable` for an unknown provider or a missing key."""
    if provider not in PROVIDERS:
        raise ProviderUnavailable(PROVIDER_UNKNOWN)
    found: dict[str, tuple[str | None, str | None]] = {
        AZURE_OPENAI: (keys.azure_openai_base_url, keys.azure_openai_api_key),
        OPENAI: (OPENAI_BASE_URL, keys.openai_api_key),
        ANTHROPIC: (ANTHROPIC_BASE_URL, keys.anthropic_api_key),
    }
    base_url, api_key = found[provider]
    if base_url is None or api_key is None:
        raise ProviderUnavailable(PROVIDER_KEY_MISSING)
    return Endpoint(provider=provider, base_url=base_url, api_key=api_key)
