from __future__ import annotations

import json
import re
from uuid import UUID

import httpx
import pytest

from gmshield_worker.backend.client import ClaimAnswer, JobsApi, Outcome, fetch_contract_version
from gmshield_worker.log import new_trace_id
from tests.conftest import StandIn


def test_asks_for_the_version_with_a_traceparent(stand_in: StandIn) -> None:
    trace_id = new_trace_id()
    answer = fetch_contract_version(stand_in.url, trace_id)
    assert answer.status == 200
    assert answer.version == "1.4.0"
    request = stand_in.requests[-1]
    assert request["path"] == "/internal/v1/version"
    assert re.fullmatch(rf"00-{trace_id}-[0-9a-f]{{16}}-01", request["traceparent"])


def test_every_request_has_a_new_span(stand_in: StandIn) -> None:
    trace_id = new_trace_id()
    fetch_contract_version(stand_in.url, trace_id)
    fetch_contract_version(stand_in.url, trace_id)
    first, second = (request["traceparent"] for request in stand_in.requests)
    assert first != second
    assert first.split("-")[1] == second.split("-")[1] == trace_id


def test_a_problem_answer_reports_its_status(stand_in: StandIn) -> None:
    stand_in.status = 503
    stand_in.body = (
        b'{"type":"urn:gmshield:problem:service.unavailable","title":"t","status":503,'
        b'"category":"service","code":"service.unavailable","params":{},"traceId":"x"}'
    )
    answer = fetch_contract_version(stand_in.url, new_trace_id())
    assert answer.status == 503
    assert answer.version is None


def test_connection_refused_raises_for_the_caller(stand_in: StandIn) -> None:
    url = stand_in.url
    with pytest.raises(httpx.ConnectError):
        fetch_contract_version(url.rsplit(":", 1)[0] + ":9", new_trace_id())


def test_proxy_settings_of_the_environment_are_ignored(stand_in: StandIn, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    assert fetch_contract_version(stand_in.url, new_trace_id()).version == "1.4.0"


def test_the_workers_credential_is_sent_when_given(stand_in: StandIn) -> None:
    stand_in.credential = "w" * 64
    assert fetch_contract_version(stand_in.url, new_trace_id()).status == 401
    assert fetch_contract_version(stand_in.url, new_trace_id(), credential="w" * 64).version == "1.4.0"
    assert stand_in.requests[-1]["authorization"] == "Bearer " + "w" * 64


def test_a_claim_answers_the_job_with_its_own_trace_and_never_shows_the_grant(stand_in: StandIn) -> None:
    stand_in.credential = "w" * 64
    job_trace = new_trace_id()
    job_id = stand_in.offer("probe", {"workMs": 5, "failTimes": 1}, job_trace)
    api = JobsApi(stand_in.url, "w" * 64)
    lane_trace = new_trace_id()

    answer = api.claim(UUID(job_id), lane_trace)

    assert answer.status == 201
    assert answer.definite
    job = answer.job
    assert job is not None
    assert job.attempt == 1
    assert job.kind == "probe"
    assert job.input == {"workMs": 5, "failTimes": 1}
    assert job.trace_id == job_trace
    assert job.grant not in repr(job)
    assert job.renew_at < job.lapses_at
    claim_request = stand_in.requests[-1]
    assert claim_request["path"] == f"/internal/v1/jobs/{job_id}/claims"
    assert claim_request["traceparent"].split("-")[1] == lane_trace
    assert "x-job-grant" not in claim_request

    second = api.claim(UUID(job_id), lane_trace)
    assert (second.status, second.code, second.job, second.definite) == (409, "job.claimed", None, True)

    renewal = api.renew(job)
    assert renewal.status == 200
    renew_request = stand_in.requests[-1]
    assert renew_request["x-job-grant"] == job.grant
    assert renew_request["traceparent"].split("-")[1] == job_trace
    assert renew_request["authorization"] == "Bearer " + "w" * 64

    assert api.report(job, Outcome.failed("probe.failed")).status == 204
    report_request = stand_in.requests[-1]
    assert json.loads(report_request["body"]) == {"outcome": "failed", "code": "probe.failed"}
    assert report_request["traceparent"].split("-")[1] == job_trace
    refused = api.report(job, Outcome.done({"workedMs": 5}))
    assert (refused.status, refused.code) == (403, "job.grant_refused")


def test_a_claim_without_a_definite_answer_is_not_definite(stand_in: StandIn) -> None:
    job_id = stand_in.offer()
    api = JobsApi(stand_in.url, "w" * 64)
    stand_in.claim_status = 503
    assert not api.claim(UUID(job_id), new_trace_id()).definite
    stand_in.claim_status = None
    assert api.claim(UUID(job_id), new_trace_id()).definite
    assert not ClaimAnswer(status=404).definite
    assert not ClaimAnswer(status=401).definite
