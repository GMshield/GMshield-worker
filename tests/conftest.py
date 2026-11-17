"""Shared fixtures: a local HTTP stand-in for the backend's internal API, and log capture.

The stand-in answers `GET /internal/v1/version` and, since Story 1.12, the job routes, with a
small model of the backend's rules: a job offered with `offer()` can be claimed once (201, then
409 `job.claimed` while the claim lives, 409 `job.not_offered` once it ended), a renewal or an
outcome needs the live claim's grant (else 403 `job.grant_refused`). `claim_status`,
`renew_status` and `outcome_status` force an answer instead (for example 503); `credential`, when
set, is required as `Authorization: Bearer` on every route.

Since Story 1.15 it also serves a claimed job's files: `GET .../inputs/{name}` answers the bytes
`offer(inputs=...)` gave for that name (404 for another name), and `POST .../outputs` keeps the body
in `outputs` under a new id and answers `{fileId, size, sha256}`; both need the live claim's grant.
`output_status` forces the output's answer, and `wrong_sha256` makes the answer's hash wrong.
A failed outcome with `retry: false` ends the job at once, as the backend does. A done `ingestion`
outcome is checked against the backend's rules for a step report (`ingestion_result_valid`) and
answered 400 `job.result_invalid` otherwise, so a renamed field fails the worker's tests. Since Story 1.18 that
includes a next step over every text file (`input: "texts"`), and the passages and outline a report carries; since
Story 1.19, the page views (`pageViews`), each picture an output of this job.

`main.run` makes the process non-dumpable; an autouse fixture makes the test process dumpable again
after each test, so one test never changes what the next one sees in `/proc`.
"""

from __future__ import annotations

import ctypes
import hashlib
import itertools
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

PR_SET_DUMPABLE = 4
PIPELINE_VERSION = re.compile(r"[A-Za-z0-9._-]{1,32}")
CLAIMS = re.compile(r"/internal/v1/jobs/([0-9a-f-]{36})/claims")
CLAIM_ROUTE = re.compile(r"/internal/v1/jobs/([0-9a-f-]{36})/claims/([0-9a-f-]{36})/(renewals|outcome|outputs)")
INPUT_ROUTE = re.compile(r"/internal/v1/jobs/([0-9a-f-]{36})/claims/([0-9a-f-]{36})/inputs/([^/]+)")


@dataclass(frozen=True)
class Raw:
    """An answer's body sent as it is, with its content type."""

    body: bytes
    content_type: str = "application/octet-stream"


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
    inputs: dict[str, bytes] = field(default_factory=dict)
    output_ids: set[str] = field(default_factory=set)


@dataclass
class StandIn:
    """Answers `GET /internal/v1/version` with `version`, or with `status` and a raw `body`."""

    version: str = "1.5.0"
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
    outputs: dict[str, bytes] = field(default_factory=dict)
    output_status: int | None = None
    wrong_sha256: bool = False

    def offer(
        self,
        kind: str = "probe",
        job_input: dict[str, Any] | None = None,
        trace_id: str | None = None,
        inputs: dict[str, bytes] | None = None,
    ) -> str:
        """Records an offered job, as the backend would before adding its message; returns its id."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = StandInJob(job_id, kind, job_input or {}, trace_id or secrets.token_hex(16))
        self.jobs[job_id].inputs = dict(inputs or {})
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


def _output(state: StandIn, job: StandInJob, body: bytes) -> tuple[int, Any]:
    if state.output_status is not None:
        return problem(state.output_status, "job.output_too_large")
    file_id = str(uuid.uuid4())
    state.outputs[file_id] = body
    job.output_ids.add(file_id)
    digest = hashlib.sha256(body).hexdigest()
    return 201, {"fileId": file_id, "size": len(body), "sha256": "0" * 64 if state.wrong_sha256 else digest}


def _input(state: StandIn, route: re.Match[str], headers: dict[str, str]) -> tuple[int, Any]:
    with state.lock:
        job = _granted(state, route[1], route[2], headers.get("x-job-grant"))
        if job is None:
            return problem(403, "job.grant_refused")
        if route[3] not in job.inputs:
            return problem(404, "request.not_found")
        return 200, Raw(job.inputs[route[3]])


def _jobs(state: StandIn, path: str, headers: dict[str, str], body: bytes) -> tuple[int, Any]:
    claims = CLAIMS.fullmatch(path)
    if claims:
        return _claim(state, claims[1])
    route = CLAIM_ROUTE.fullmatch(path)
    if route is None:
        return problem(404, "request.not_found")
    job_id, claim_id, action = route[1], route[2], route[3]
    forced = {"renewals": state.renew_status, "outcome": state.outcome_status}.get(action)
    if forced is not None:
        return problem(forced, "server.unavailable" if forced >= 500 else "job.grant_refused")
    job = _granted(state, job_id, claim_id, headers.get("x-job-grant"))
    if job is None:
        return problem(403, "job.grant_refused")
    if action == "outputs":
        return _output(state, job, body)
    return _renewal(state, job) if action == "renewals" else _outcome(state, job, body)


def _renewal(state: StandIn, job: StandInJob) -> tuple[int, Any]:
    renew_at, lapses_text, lapses_at = state.times()
    job.lapses_at, job.renewals = lapses_at, job.renewals + 1
    return 200, {"renewAt": renew_at, "lapsesAt": lapses_text}


def _page(value: object, total: int) -> bool:
    return type(value) is int and 1 <= value <= total


STAGE = re.compile(r"[a-z][a-z0-9_]{0,31}")
LAYERS = ("text", "image")
OUTCOMES = ("read_directly", "read_by_ocr", "unreadable")


def _ranges(items: object, total: int, field_name: str | None, values: tuple[str, ...]) -> list[tuple[int, int]] | None:
    """The ranges of a report's array, or None when one is not `{firstPage, lastPage[, field]}` within the total."""
    if items is None:
        return []
    if not isinstance(items, list):
        return None
    found = []
    for item in items:
        if (
            not isinstance(item, dict)
            or not _page(item.get("firstPage"), total)
            or not _page(item.get("lastPage"), total)
        ):
            return None
        if item["lastPage"] < item["firstPage"] or (field_name is not None and item.get(field_name) not in values):
            return None
        found.append((item["firstPage"], item["lastPage"]))
    return found


def _no_repeats(found: list[tuple[int, int]]) -> bool:
    ordered = sorted(found)
    return all(later[0] > earlier[1] for earlier, later in itertools.pairwise(ordered))


def _next_valid(job: StandInJob, following: object, total: int) -> bool:
    if following is None:
        return True
    if not isinstance(following, list):
        return False
    known = job.output_ids | {job.input.get("inputFileId")}
    for item in following:
        if not isinstance(item, dict) or not isinstance(item.get("stage"), str) or not STAGE.fullmatch(item["stage"]):
            return False
        first, last = item.get("firstPage"), item.get("lastPage")
        texts = item.get("input") == "texts"
        if (first is None) != (last is None) or (not texts and item.get("input") not in known):
            return False
        if first is not None and not (_page(first, total) and _page(last, total) and first <= last):
            return False
    return True


def _report_total(job: StandInJob, result: dict[str, Any]) -> int | None:
    """The report's page total, or the chain's from the claim input (the same when both are given); 0 when neither is
    known; None when the report's is not a page total or differs from the chain's."""
    version = result.get("pipelineVersion")
    if not isinstance(version, str) or not PIPELINE_VERSION.fullmatch(version):
        return None
    reported, chain = result.get("pageTotal"), job.input.get("pageTotal")
    if reported is not None and (not _page(reported, 1_000_000) or (chain is not None and reported != chain)):
        return None
    return int(reported) if reported is not None else int(chain) if type(chain) is int else 0


def _files_valid(job: StandInJob, result: dict[str, Any]) -> bool:
    original = result.get("original")
    if original is not None and (not isinstance(original, dict) or original.get("fileId") not in job.output_ids):
        return False
    files = [item for name in ("batches", "texts") for item in (result.get(name) or [])]
    views = result.get("pageViews") or []
    pictures = [view["picture"] for view in views if isinstance(view, dict) and view.get("picture") is not None]
    named = [item.get("fileId") for item in [*files, *pictures] if isinstance(item, dict)]
    return (
        all(isinstance(item, dict) for item in [*files, *pictures])
        and all(file_id in job.output_ids for file_id in named)
        and len(named) == len(set(named))
    )


def _pages_valid(result: dict[str, Any], total: int) -> bool:
    batches = _ranges(result.get("batches"), total, None, ())
    layers = _ranges(result.get("layers"), total, "layer", LAYERS)
    pages = _ranges(result.get("pages"), total, "outcome", OUTCOMES)
    texts = _ranges(result.get("texts"), total, None, ())
    if batches is None or layers is None or pages is None or texts is None:
        return False
    read = {
        page
        for item in result.get("pages") or []
        if item["outcome"] != "unreadable"
        for page in range(item["firstPage"], item["lastPage"] + 1)
    }
    covered = all(page in read for first, last in texts for page in range(first, last + 1))
    return covered and _no_repeats(layers) and _no_repeats(pages)


MAX_PASSAGE = 16_384
MAX_DEPTH = 32
MAX_TITLE = 1000


def _title(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= MAX_TITLE


def _place_valid(item: dict[str, Any]) -> bool:
    """Boxes (each four fractions) or a range (`[start, end)`, not empty), never both nor neither."""
    boxes, span = item.get("boxes"), item.get("range")
    if (boxes is None) == (span is None):
        return False
    if boxes is not None:
        return isinstance(boxes, list) and all(
            isinstance(box, list)
            and len(box) == 4
            and all(type(side) in (int, float) and 0 <= side <= 1 for side in box)
            for box in boxes
        )
    return (
        isinstance(span, list) and len(span) == 2 and all(type(end) is int for end in span) and 0 <= span[0] < span[1]
    )


def _passages_valid(result: dict[str, Any], total: int) -> bool:
    """Passages (page, text of 1 to 16,384 code points, a path of at most 32 titles, a place) and outline entries
    (level, title of 1 to 1,000 code points, page), every page within the total."""
    passages, outline = result.get("passages"), result.get("outline")
    if passages is None and outline is None:
        return True
    passages, outline = passages or [], outline or []
    if not isinstance(passages, list) or not isinstance(outline, list):
        return False
    for item in passages:
        if (
            not isinstance(item, dict)
            or not _page(item.get("page"), total)
            or not isinstance(item.get("text"), str)
            or not 1 <= len(item["text"]) <= MAX_PASSAGE
            or not isinstance(item.get("sectionPath"), list)
            or len(item["sectionPath"]) > MAX_DEPTH
            or not all(_title(name) for name in item["sectionPath"])
            or not _place_valid(item)
        ):
            return False
    return all(
        isinstance(entry, dict)
        and type(entry.get("level")) is int
        and 0 <= entry["level"] <= 1000
        and _title(entry.get("title"))
        and _page(entry.get("page"), total)
        for entry in outline
    )


MAX_SIDE = 10_000
MAX_LINES = 10_000
MAX_LINE = 4_096
MAX_PAGE_TEXT = 16_384


def _box(box: object) -> bool:
    return (
        isinstance(box, list) and len(box) == 4 and all(type(side) in (int, float) and 0 <= side <= 1 for side in box)
    )


def _view_valid(view: object, read: set[int], seen: set[int]) -> bool:
    """A page view on a page this report gives an outcome, once: a picture (1 to 10,000 pixels a side) with at most
    10,000 lines of 1 to 4,096 code points and a box of four fractions, or a text page's text of at most 16,384."""
    if not isinstance(view, dict) or type(view.get("page")) is not int or view["page"] not in read:
        return False
    if view["page"] in seen or (view.get("picture") is None) == (view.get("text") is None):
        return False
    seen.add(view["page"])
    if view.get("text") is not None:
        return isinstance(view["text"], str) and len(view["text"]) <= MAX_PAGE_TEXT and view.get("lines") is None
    picture, lines = view["picture"], view.get("lines")
    sides = [picture.get("width"), picture.get("height")] if isinstance(picture, dict) else [None]
    return (
        all(type(side) is int and 1 <= side <= MAX_SIDE for side in sides)
        and isinstance(lines, list)
        and len(lines) <= MAX_LINES
        and all(
            isinstance(line, dict)
            and isinstance(line.get("text"), str)
            and 1 <= len(line["text"]) <= MAX_LINE
            and _box(line.get("box"))
            for line in lines
        )
    )


def _page_views_valid(result: dict[str, Any]) -> bool:
    """The report's page views (Story 1.19), each on a page of this report's `pages`."""
    views = result.get("pageViews")
    if views is None:
        return True
    if not isinstance(views, list):
        return False
    read = {page for item in result.get("pages") or [] for page in range(item["firstPage"], item["lastPage"] + 1)}
    seen: set[int] = set()
    return all(_view_valid(view, read, seen) for view in views)


def ingestion_result_valid(job: StandInJob, result: object) -> bool:
    """The backend's rules for a done `ingestion` step (`IngestionSteps.Report.of` and its range checks): a pipeline
    version; the page total (the report's, or the chain's from the claim input, the same when both are given); an
    original, batches and texts that are outputs of this job; layers and outcomes within the total, no page twice;
    texts only over pages this report gives an outcome other than `unreadable`; next steps over a known file or every
    text file; passages and an outline within the backend's limits; page views on this report's pages, each picture
    an output of this job."""
    if not isinstance(result, dict):
        return False
    total = _report_total(job, result)
    return (
        total is not None
        and _files_valid(job, result)
        and _pages_valid(result, total)
        and _next_valid(job, result.get("next"), total)
        and _passages_valid(result, total)
        and _page_views_valid(result)
    )


def _outcome(state: StandIn, job: StandInJob, body: bytes) -> tuple[int, Any]:
    outcome = json.loads(body)
    if (
        outcome["outcome"] == "done"
        and job.kind == "ingestion"
        and not ingestion_result_valid(job, outcome.get("result"))
    ):
        return problem(400, "job.result_invalid")
    job.outcomes.append(outcome)
    if outcome["outcome"] == "failed":
        last = outcome.get("retry") is False or job.attempt >= state.max_attempts
        job.state = "failed" if last else "offered"
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
    inputs = INPUT_ROUTE.fullmatch(path)
    if method == "GET" and inputs is not None:
        return _input(state, inputs, headers)
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
            text = "" if "outputs" in self.path else body.decode("utf-8", "replace")
            state.requests.append({"method": method, "path": self.path, "body": text, **headers})
            status, answer = _route(state, method, self.path, headers, body)
            if answer is None:
                self.send_response(status)
                self.send_header("content-length", "0")
                self.end_headers()
            elif isinstance(answer, Raw):
                self._send(status, answer.content_type, answer.body)
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


@pytest.fixture(autouse=True)
def dumpable_again() -> Iterator[None]:
    yield
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_DUMPABLE, 1, 0, 0, 0)


def log_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]
