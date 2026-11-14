"""Calls to the backend's internal API, through the generated client only.

A new client is built for every request: it sends a W3C `traceparent` with the caller's trace id
and a new span id, gives up after 2 seconds, follows no redirect, and ignores proxy settings from
the environment, so the worker connects to the backend's address and nowhere else.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from gmshield_worker.backend.generated import Client
from gmshield_worker.backend.generated.api.contract import get_internal_contract_version
from gmshield_worker.backend.generated.models import ContractVersion
from gmshield_worker.log import traceparent

REQUEST_TIMEOUT_SECONDS = 2.0


def backend_client(base_url: str, trace_id: str, *, httpx_args: Mapping[str, Any] | None = None) -> Client:
    """A client for one request of this trace."""
    return Client(
        base_url=base_url,
        headers={"traceparent": traceparent(trace_id), "accept": "application/json"},
        timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS),
        follow_redirects=False,
        raise_on_unexpected_status=False,
        httpx_args={"trust_env": False, **(httpx_args or {})},
    )


@dataclass(frozen=True)
class VersionAnswer:
    """What `GET /internal/v1/version` answered: the status, and the version when it is a 200."""

    status: int
    version: str | None


def fetch_contract_version(
    base_url: str, trace_id: str, *, httpx_args: Mapping[str, Any] | None = None
) -> VersionAnswer:
    """Asks the backend for its internal contract version.

    Raises the transport's exception (connection refused, timeout, a body that is not the
    contract's) for the caller to log by type and retry.
    """
    with backend_client(base_url, trace_id, httpx_args=httpx_args) as client:
        response = get_internal_contract_version.sync_detailed(client=client)
    parsed = response.parsed
    if response.status_code == 200 and isinstance(parsed, ContractVersion):
        return VersionAnswer(status=200, version=parsed.version)
    return VersionAnswer(status=int(response.status_code), version=None)
