from __future__ import annotations

import re

import httpx
import pytest

from gmshield_worker.backend.client import fetch_contract_version
from gmshield_worker.log import new_trace_id
from tests.conftest import StandIn


def test_asks_for_the_version_with_a_traceparent(stand_in: StandIn) -> None:
    trace_id = new_trace_id()
    answer = fetch_contract_version(stand_in.url, trace_id)
    assert answer.status == 200
    assert answer.version == "1.0.0"
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
    assert fetch_contract_version(stand_in.url, new_trace_id()).version == "1.0.0"
