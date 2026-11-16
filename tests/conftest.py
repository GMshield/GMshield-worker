"""Shared fixtures: a local HTTP stand-in for the backend's internal API, and log capture.

The stand-in answers `GET /internal/v1/version` and, since Story 1.12, the job routes, with a
small model of the backend's rules: a job offered with `offer()` can be claimed once (201, then
409 `job.claimed` while the claim lives, 409 `job.not_offered` once it ended), a renewal or an
outcome needs the live claim's grant (else 403 `job.grant_refused`). `claim_status`,
`renew_status` and `outcome_status` force an answer instead (for example 503); `credential`, when
set, is required as `Authorization: Bearer` on every route.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

CLAIMS = re.compile(r"/internal/v1/jobs/([0-9a-f-]{36})/claims")
CLAIM_ROUTE = re.compile(r"/internal/v1/jobs/([0-9a-f-]{36})/claims/([0-9a-f-]{36})/(renewals|outcome)")


@dataclass
class StandInJob:
    job_id: str
    kind: str
    input: dict[str, Any]
    trace_id: str
    state: str = "offered"
    attempt: int = 0
    claim_id: str | None = None
    grant: str | None = None
    lapses_at: datetime | None = None
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    renewals: int = 0


@dataclass
class StandIn:
    """Answers `GET /internal/v1/version` with `version`, or with `status` and a raw `body`."""

    version: str = "1.1.0"
    status: int = 200
    body: bytes | None = None
    requests: list[dict[str, str]] = field(default_factory=list)
    url: str = ""
    credential: str | None = None
    jobs: dict[str, StandInJob] = field(default_factory=dict)
    claim_status: int | None = None
    renew_status: int | None = None
    outcome_status: int | None = None
    renew_seconds: float = 30.0
    lapse_seconds: float = 90.0
    max_attempts: int = 3
    lock: threading.Lock = field(default_factory=threading.Lock)

    def offer(self, kind: str = "probe", job_input: dict[str, Any] | None = None, trace_id: str | None = None) -> str:
        """Records an offered job, as the backend would before adding its message; returns its id."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = StandInJob(job_id, kind, job_input or {}, trace_id or secrets.token_hex(16))
        return job_id

    def times(self) -> tuple[str, str, datetime]:
        now = datetime.now(UTC)
        lapses = now + timedelta(seconds=self.lapse_seconds)
        renew = now + timedelta(seconds=self.renew_seconds)
        return renew.isoformat().replace("+00:00", "Z"), lapses.isoformat().replace("+00:00", "Z"), lapses

    def job_requests(self, job_id: str) -> list[dict[str, str]]:
        return [request for request in self.requests if f"/jobs/{job_id}/" in request["path"]]


def problem(status: int, code: str) -> tuple[int, dict[str, Any]]:
    return status, {
        "type": f"urn:gmshield:problem:{code}",
        "title": "t",
        "status": status,
        "category": code.split(".", 1)[0],
        "code": code,
        "params": {},
        "traceId": "0" * 32,
    }


def _claim(state: StandIn, job_id: str) -> tuple[int, dict[str, Any]]:
    if state.claim_status is not None:
        return problem(state.claim_status, "server.unavailable")
    job = state.jobs.get(job_id)
    if job is None:
        return problem(404, "request.not_found")
    if job.state == "running" and job.lapses_at is not None and job.lapses_at > datetime.now(UTC):
        return problem(409, "job.claimed")
    if job.state not in ("offered", "running"):
        return problem(409, "job.not_offered")
    renew_at, lapses_text, lapses_at = state.times()
    job.state, job.attempt = "running", job.attempt + 1
    job.claim_id, job.grant, job.lapses_at = str(uuid.uuid4()), secrets.token_urlsafe(32), lapses_at
    return 201, {
        "claimId": job.claim_id,
        "attempt": job.attempt,
        "renewAt": renew_at,
        "lapsesAt": lapses_text,
        "grant": job.grant,
        "kind": job.kind,
        "input": job.input,
        "traceparent": f"00-{job.trace_id}-{secrets.token_hex(8)}-01",
    }


def _granted(state: StandIn, job_id: str, claim_id: str, grant: str | None) -> StandInJob | None:
    job = state.jobs.get(job_id)
    if job is None or job.state != "running" or job.claim_id != claim_id or job.grant != grant:
        return None
    if job.lapses_at is None or job.lapses_at <= datetime.now(UTC):
        return None
    return job


def _jobs(state: StandIn, path: str, headers: dict[str, str], body: bytes) -> tuple[int, Any]:
    claims = CLAIMS.fullmatch(path)
    if claims:
        return _claim(state, claims[1])
    route = CLAIM_ROUTE.fullmatch(path)
    if route is None:
        return problem(404, "request.not_found")
    job_id, claim_id, action = route[1], route[2], route[3]
    forced = state.renew_status if action == "renewals" else state.outcome_status
    if forced is not None:
        return problem(forced, "server.unavailable" if forced >= 500 else "job.grant_refused")
    job = _granted(state, job_id, claim_id, headers.get("x-job-grant"))
    if job is None:
        return problem(403, "job.grant_refused")
    if action == "renewals":
        renew_at, lapses_text, lapses_at = state.times()
        job.lapses_at, job.renewals = lapses_at, job.renewals + 1
        return 200, {"renewAt": renew_at, "lapsesAt": lapses_text}
    outcome = json.loads(body)
    job.outcomes.append(outcome)
    if outcome["outcome"] == "failed":
        job.state = "offered" if job.attempt < state.max_attempts else "failed"
    else:
        job.state = "done"
    job.grant = None
    return 204, None


def _route(state: StandIn, method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, Any]:
    if state.credential is not None and headers.get("authorization") != f"Bearer {state.credential}":
        return problem(401, "service.unauthenticated")
    if method == "GET" and path == "/internal/v1/version":
        if state.body is not None:
            return state.status, state.body
        return state.status, {"version": state.version}
    if method != "POST":
        return problem(404, "request.not_found")
    with state.lock:
        return _jobs(state, path, headers, body)


@pytest.fixture
def stand_in() -> Iterator[StandIn]:
    state = StandIn()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, method: str) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            headers = {k.lower(): v for k, v in self.headers.items()}
            state.requests.append(
                {"method": method, "path": self.path, "body": body.decode("utf-8", "replace"), **headers}
            )
            status, answer = _route(state, method, self.path, headers, body)
            if answer is None:
                self.send_response(status)
                self.send_header("content-length", "0")
                self.end_headers()
            elif isinstance(answer, bytes):
                self._send(status, "application/json", answer)
            else:
                kind = "application/problem+json" if status >= 400 else "application/json"
                self._send(status, kind, json.dumps(answer).encode())

        def do_GET(self) -> None:
            self._answer("GET")

        def do_POST(self) -> None:
            self._answer("POST")

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def log_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]
