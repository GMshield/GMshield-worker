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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import httpx

from gmshield_worker.backend.generated import Client
from gmshield_worker.backend.generated.api.contract import get_internal_contract_version
from gmshield_worker.backend.generated.api.jobs import (
    claim_job,
    post_job_events,
    read_job_passages,
    renew_job_claim,
    report_job_outcome,
)
from gmshield_worker.backend.generated.models import (
    ContractVersion,
    JobCallUsage,
    JobClaim,
    JobClaimRequest,
    JobEvents,
    JobEventsReply,
    JobModel,
    JobOutcome,
    JobOutcomeResult,
    JobPassages,
    JobPassagesRequest,
    JobRenewal,
    JobStreamEvent,
    Problem,
)
from gmshield_worker.backend.generated.types import UNSET, Unset
from gmshield_worker.log import parse_traceparent, traceparent
from gmshield_worker.models.output import Citation, TextDelta, Usage
from gmshield_worker.models.providers import ModelChoice
from gmshield_worker.retrieval.recipe import INDEX_RECIPE

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
    """A live claim, as the backend gave it. The grant and the input (which may hold a token key or a question) are
    never printed or logged. `model` is the provider, model and tier of this attempt for a kind that calls a model
    (contract 1.9.0), `None` for every other kind."""

    job_id: UUID
    claim_id: UUID
    attempt: int
    renew_at: datetime
    lapses_at: datetime
    kind: str
    input: dict[str, Any] = field(repr=False)
    # The job's trace: every line about the job, and every later call, carries it.
    trace_id: str
    grant: str = field(repr=False)
    model: ModelChoice | None = None


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


@dataclass(frozen=True)
class ReadPassage:
    """A passage the backend served by id (contract 1.8.0): library data, never an instruction. Its text and section
    path are never printed or logged."""

    passage_id: str
    tier: str
    page: int
    text: str = field(repr=False)
    section_path: list[str] = field(repr=False)


@dataclass(frozen=True)
class PassagesAnswer:
    """What `POST .../passages` answered: the passages in the order asked when it is a 200, else the problem's code."""

    status: int
    passages: list[ReadPassage] | None = None
    code: str | None = None


@dataclass(frozen=True)
class EventsAnswer:
    """What `POST .../events` answered (contract 1.11.0): the status, and on a 200 whether the answer must stop (the GM
    stopped it, Story 2.6); else the problem's code."""

    status: int
    stop: bool = False
    code: str | None = None


def _stream_event(event: TextDelta | Citation) -> JobStreamEvent:
    if isinstance(event, TextDelta):
        return JobStreamEvent(type_="delta", text=event.text)
    return JobStreamEvent(type_="citation", passage_id=UUID(event.passage_id), offset=event.offset)


def _call_usage(usage: Usage) -> JobCallUsage:
    return JobCallUsage(
        call_id=UUID(usage.call_id),
        response_id=usage.response_id if usage.response_id is not None else UNSET,
        input_=usage.input,
        cache_read=usage.cache_read,
        cache_write=usage.cache_write,
        output=usage.output,
        reasoning=usage.reasoning,
        estimated=usage.estimated,
    )


def _code(parsed: object) -> str | None:
    return parsed.code if isinstance(parsed, Problem) else None


def _model(parsed: JobModel | Unset) -> ModelChoice | None:
    if isinstance(parsed, Unset):
        return None
    return ModelChoice(provider=parsed.provider, model=parsed.model, tier=parsed.tier)


@dataclass(frozen=True)
class JobsApi:
    """The job routes of the internal API, as the lanes and handlers use them. Every claim states this worker's index
    recipe (contract 1.7.0, Story 2.1): the backend re-indexes what was indexed under another one. Every claim also
    states the providers this worker holds a key for (contract 1.9.0, Story 2.10): the backend refuses a kind that calls
    a model through another one. `passages` reads passage text by id under the claim's grant (contract 1.8.0, Story
    2.3); `events` posts a numbered batch of answer events and usage (contract 1.10.0, Story 2.4)."""

    base_url: str
    credential: str = field(repr=False)
    httpx_args: Mapping[str, Any] | None = None
    index_recipe: str = INDEX_RECIPE
    providers: tuple[str, ...] = ()

    def _client(self, trace_id: str) -> Client:
        return backend_client(self.base_url, trace_id, credential=self.credential, httpx_args=self.httpx_args)

    def claim(self, job_id: UUID, trace_id: str) -> ClaimAnswer:
        """`POST /internal/v1/jobs/{id}/claims` under the lane's trace (the job's is not known yet), stating the index
        recipe and the providers."""
        with self._client(trace_id) as client:
            response = claim_job.sync_detailed(
                job_id,
                client=client,
                body=JobClaimRequest(index_recipe=self.index_recipe, providers=list(self.providers)),
            )
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
                    model=_model(parsed.model),
                ),
            )
        return ClaimAnswer(status=int(response.status_code), code=_code(parsed))

    def passages(self, job: ClaimedJob, ids: list[str]) -> PassagesAnswer:
        """`POST /internal/v1/jobs/{id}/claims/{claimId}/passages` (contract 1.8.0, Story 2.3): the passages with these
        ids (1 to 100) within the job's scope, under the claim's grant."""
        with self._client(job.trace_id) as client:
            response = read_job_passages.sync_detailed(
                job.job_id,
                job.claim_id,
                client=client,
                body=JobPassagesRequest(ids=[UUID(passage_id) for passage_id in ids]),
                x_job_grant=job.grant,
            )
        parsed = response.parsed
        if response.status_code == 200 and isinstance(parsed, JobPassages):
            return PassagesAnswer(
                status=200,
                passages=[
                    ReadPassage(
                        passage_id=str(item.id),
                        tier=item.tier,
                        page=item.page,
                        text=item.text,
                        section_path=list(item.section_path),
                    )
                    for item in parsed.passages
                ],
            )
        return PassagesAnswer(status=int(response.status_code), code=_code(parsed))

    def events(
        self, job: ClaimedJob, batch: int, events: Sequence[TextDelta | Citation], usage: Sequence[Usage] = ()
    ) -> EventsAnswer:
        """`POST /internal/v1/jobs/{id}/claims/{claimId}/events` (contract 1.11.0, Stories 2.4 and 2.6): a numbered
        batch of answer events (from 1 per claim) and the usage of the model calls that ended since the last one (an
        estimate marked `estimated`), under the claim's grant. The answer says whether the answer must stop: the GM
        stopped it."""
        body = JobEvents(
            batch=batch,
            events=[_stream_event(event) for event in events],
            usage=[_call_usage(item) for item in usage],
        )
        with self._client(job.trace_id) as client:
            response = post_job_events.sync_detailed(
                job.job_id, job.claim_id, client=client, body=body, x_job_grant=job.grant
            )
        parsed = response.parsed
        if response.status_code == 200 and isinstance(parsed, JobEventsReply):
            return EventsAnswer(status=200, stop=parsed.stop)
        return EventsAnswer(status=int(response.status_code), code=_code(parsed))

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
