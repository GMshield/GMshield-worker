"""Calls to the backend's internal API, through the generated client only.

A new client is built for every request: it sends the worker's credential
(`Authorization: Bearer`), a W3C `traceparent` with the given trace id and a new span id, gives up
after 2 seconds, follows no redirect, and ignores proxy settings from the environment, so the worker
connects to the backend's address and nowhere else. A renewal and an outcome also send the claim's
grant in `X-Job-Grant`, and carry the job's own trace id.

Every call returns what the backend answered (the status, and the problem's code when it refused).
A transport failure, or an answer that is not the contract's, raises for the caller, who treats it
as no answer at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import httpx

from gmshield_worker.backend.generated import Client
from gmshield_worker.backend.generated.api.contract import get_internal_contract_version
from gmshield_worker.backend.generated.api.jobs import claim_job, renew_job_claim, report_job_outcome
from gmshield_worker.backend.generated.models import (
    ContractVersion,
    JobClaim,
    JobOutcome,
    JobOutcomeResult,
    JobRenewal,
    Problem,
)
from gmshield_worker.log import parse_traceparent, traceparent

REQUEST_TIMEOUT_SECONDS = 2.0


def backend_client(
    base_url: str, trace_id: str, *, credential: str | None = None, httpx_args: Mapping[str, Any] | None = None
) -> Client:
    """A client for one request of this trace, with the worker's credential when given."""
    headers = {"traceparent": traceparent(trace_id), "accept": "application/json"}
    if credential is not None:
        headers["authorization"] = f"Bearer {credential}"
    return Client(
        base_url=base_url,
        headers=headers,
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
    base_url: str,
    trace_id: str,
    *,
    credential: str | None = None,
    httpx_args: Mapping[str, Any] | None = None,
) -> VersionAnswer:
    """Asks the backend for its internal contract version.

    Raises the transport's exception (connection refused, timeout, a body that is not the
    contract's) for the caller to log by type and retry.
    """
    with backend_client(base_url, trace_id, credential=credential, httpx_args=httpx_args) as client:
        response = get_internal_contract_version.sync_detailed(client=client)
    parsed = response.parsed
    if response.status_code == 200 and isinstance(parsed, ContractVersion):
        return VersionAnswer(status=200, version=parsed.version)
    return VersionAnswer(status=int(response.status_code), version=None)


@dataclass(frozen=True)
class ClaimedJob:
    """A live claim, as the backend gave it. The grant is never printed or logged."""

    job_id: UUID
    claim_id: UUID
    attempt: int
    renew_at: datetime
    lapses_at: datetime
    kind: str
    input: dict[str, Any]
    # The job's trace: every line about the job, and every later call, carries it.
    trace_id: str
    grant: str = field(repr=False)


@dataclass(frozen=True)
class ClaimAnswer:
    status: int
    job: ClaimedJob | None = None
    code: str | None = None

    @property
    def definite(self) -> bool:
        """The backend decided (2xx or 409): Postgres holds the truth, the message can go."""
        return 200 <= self.status < 300 or self.status == 409


@dataclass(frozen=True)
class RenewalAnswer:
    status: int
    renew_at: datetime | None = None
    lapses_at: datetime | None = None
    code: str | None = None


@dataclass(frozen=True)
class Answer:
    status: int
    code: str | None = None


@dataclass(frozen=True)
class Outcome:
    """How an attempt ended: `done` with an optional result, or `failed` with an optional code and, since contract
    1.2.0, `retry=False` for a failure no other attempt can change (the backend then ends the job at once)."""

    outcome: Literal["done", "failed"]
    code: str | None = None
    result: dict[str, Any] | None = None
    retry: bool | None = None

    @staticmethod
    def done(result: dict[str, Any] | None = None) -> Outcome:
        return Outcome("done", None, result)

    @staticmethod
    def failed(code: str, *, retry: bool | None = None) -> Outcome:
        return Outcome("failed", code, None, retry)


def _code(parsed: object) -> str | None:
    return parsed.code if isinstance(parsed, Problem) else None


@dataclass(frozen=True)
class JobsApi:
    """The job routes of the internal API, as the lanes use them."""

    base_url: str
    credential: str = field(repr=False)
    httpx_args: Mapping[str, Any] | None = None

    def _client(self, trace_id: str) -> Client:
        return backend_client(self.base_url, trace_id, credential=self.credential, httpx_args=self.httpx_args)

    def claim(self, job_id: UUID, trace_id: str) -> ClaimAnswer:
        """`POST /internal/v1/jobs/{id}/claims` under the lane's trace (the job's is not known yet)."""
        with self._client(trace_id) as client:
            response = claim_job.sync_detailed(job_id, client=client)
        parsed = response.parsed
        if response.status_code == 201 and isinstance(parsed, JobClaim):
            job_trace = parse_traceparent(parsed.traceparent) or trace_id
            return ClaimAnswer(
                status=201,
                job=ClaimedJob(
                    job_id=job_id,
                    claim_id=parsed.claim_id,
                    attempt=parsed.attempt,
                    renew_at=parsed.renew_at,
                    lapses_at=parsed.lapses_at,
                    kind=parsed.kind,
                    input=parsed.input_.to_dict(),
                    trace_id=job_trace,
                    grant=parsed.grant,
                ),
            )
        return ClaimAnswer(status=int(response.status_code), code=_code(parsed))

    def renew(self, job: ClaimedJob) -> RenewalAnswer:
        with self._client(job.trace_id) as client:
            response = renew_job_claim.sync_detailed(job.job_id, job.claim_id, client=client, x_job_grant=job.grant)
        parsed = response.parsed
        if response.status_code == 200 and isinstance(parsed, JobRenewal):
            return RenewalAnswer(status=200, renew_at=parsed.renew_at, lapses_at=parsed.lapses_at)
        return RenewalAnswer(status=int(response.status_code), code=_code(parsed))

    def report(self, job: ClaimedJob, outcome: Outcome) -> Answer:
        body = JobOutcome(outcome=outcome.outcome)
        if outcome.code is not None:
            body.code = outcome.code
        if outcome.result is not None:
            body.result = JobOutcomeResult.from_dict(outcome.result)
        if outcome.retry is not None:
            body.retry = outcome.retry
        with self._client(job.trace_id) as client:
            response = report_job_outcome.sync_detailed(
                job.job_id, job.claim_id, client=client, body=body, x_job_grant=job.grant
            )
        return Answer(status=int(response.status_code), code=_code(response.parsed))
