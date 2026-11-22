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
Story 1.19, the page views (`pageViews`), each picture an output of this job. Since Story 2.1 a claim's body, when
it has one, must state a recipe as the backend reads it (else 400), and each one is kept in `stated_recipes`; a done
`index.change` outcome must report `indexRecipe` (`index_result_valid`). An index change's passages are an input like
any other (`offer(inputs={"passages": ...})`).

Since Story 2.3 (contract 1.8.0) it serves `POST .../passages` as the backend's scope does: 1 to 100 ids (else 400
`job.passages_invalid`), each a passage of `passages` (`StandInPassage`) whose source is in one of the job input's
`tiers`, else 403 `scope.passage_refused` with nothing returned; each answered once, in the order asked, labelled
with the tier whose sources hold it. `passages_status` forces a refusal with `passages_code` instead.

Since Story 2.10 (contract 1.9.0) a claim may state `providers` (at most 8 words of `[a-z0-9-]{1,32}`, else 400), each
list kept in `stated_providers`. A job offered with `model=` calls a model: its claim answers that `model`, unless the
worker did not state its provider, when the job ends `failed` with `job.provider_unavailable` (`ended_code`), no
attempt used, and the claim is answered 409 `job.not_offered`, as the backend does.

Since Story 2.4 (contract 1.10.0) it serves `POST .../events` as the backend does: a batch number from 1 per claim, a
repeated number accepted and recorded once, a gap 409 `job.batch_out_of_order`; each accepted batch is kept in
`batches` under its job (`{claimId, batch, events, usage}`), and each call's usage once in `usage` by call id.
`events_status` forces an answer (for example 503) `events_failures` times (or always when `None`). A failed outcome of
a job with an accepted batch holding an event ends it at once (no splice), as the backend does.

Since Story 2.6 (contract 1.11.0) a usage may say `estimated` (a boolean, else 400). `stop_after` plays the GM's stop:
once a job has that many accepted batches, every later batch of it stores no event, only its usage, and is answered
`{"stop": true}`, as the backend does after a stop.

Since contract 1.12.0 it serves `PUT /internal/v1/index-recipe` as the backend does: a body `{indexRecipe}` with a
recipe as the backend reads it is answered 204 and kept in `recipe_statements`; anything else 400 `request.invalid`.
`recipe_status` forces an answer (for example 503).

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
CLAIM_ROUTE = re.compile(
    r"/internal/v1/jobs/([0-9a-f-]{36})/claims/([0-9a-f-]{36})/(renewals|outcome|outputs|passages|events)"
)
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
    model: dict[str, str] | None = None
    ended_code: str | None = None


@dataclass(frozen=True)
class StandInPassage:
    """A passage the backend holds: its source (a version id), text, section path and page."""

    source: str
    text: str
    section_path: list[str]
    page: int


@dataclass
class StandIn:
    """Answers `GET /internal/v1/version` with `version`, or with `status` and a raw `body`."""

    version: str = "1.13.0"
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
    stated_recipes: list[str | None] = field(default_factory=list)
    stated_providers: list[list[str] | None] = field(default_factory=list)
    passages: dict[str, StandInPassage] = field(default_factory=dict)
    passages_status: int | None = None
    passages_code: str = "scope.passage_refused"
    batches: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    events_status: int | None = None
    events_failures: int | None = None
    stop_after: int | None = None
    recipe_statements: list[str] = field(default_factory=list)
    recipe_status: int | None = None

    def offer(
        self,
        kind: str = "probe",
        job_input: dict[str, Any] | None = None,
        trace_id: str | None = None,
        inputs: dict[str, bytes] | None = None,
        model: dict[str, str] | None = None,
    ) -> str:
        """Records an offered job, as the backend would before adding its message; returns its id. `model` makes it a
        kind that calls one (`{"provider", "model", "tier"}`)."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = StandInJob(job_id, kind, job_input or {}, trace_id or secrets.token_hex(16))
        self.jobs[job_id].inputs = dict(inputs or {})
        self.jobs[job_id].model = model
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


RECIPE = re.compile(r"[A-Za-z0-9._-]{1,32}")
PROVIDER = re.compile(r"[a-z0-9-]{1,32}")


def _providers_valid(providers: object) -> bool:
    if providers is None:
        return True
    return (
        isinstance(providers, list)
        and len(providers) <= 8
        and all(isinstance(item, str) and PROVIDER.fullmatch(item) for item in providers)
    )


def _claim_refused(job: StandInJob, providers: list[str] | None) -> tuple[int, dict[str, Any]] | None:
    """409 for a job that is claimed or not offered; a model kind's job without its provider stated ends failed."""
    if job.state == "running" and job.lapses_at is not None and job.lapses_at > datetime.now(UTC):
        return problem(409, "job.claimed")
    if job.state not in ("offered", "running"):
        return problem(409, "job.not_offered")
    if job.model is not None and job.model["provider"] not in (providers or []):
        job.state, job.ended_code = "failed", "job.provider_unavailable"
        return problem(409, "job.not_offered")
    return None


def _claim(state: StandIn, job_id: str, body: bytes) -> tuple[int, dict[str, Any]]:
    request = json.loads(body) if body else {}
    stated, providers = request.get("indexRecipe"), request.get("providers")
    recipe_valid = stated is None or (isinstance(stated, str) and RECIPE.fullmatch(stated) is not None)
    if not recipe_valid or not _providers_valid(providers):
        return problem(400, "request.invalid")
    state.stated_recipes.append(stated)
    state.stated_providers.append(providers)
    if state.claim_status is not None:
        return problem(state.claim_status, "server.unavailable")
    job = state.jobs.get(job_id)
    if job is None:
        return problem(404, "request.not_found")
    refused = _claim_refused(job, providers)
    if refused is not None:
        return refused
    renew_at, lapses_text, lapses_at = state.times()
    job.state, job.attempt = "running", job.attempt + 1
    job.claim_id, job.grant, job.lapses_at = str(uuid.uuid4()), secrets.token_urlsafe(32), lapses_at
    answer: dict[str, Any] = {
        "claimId": job.claim_id,
        "attempt": job.attempt,
        "renewAt": renew_at,
        "lapsesAt": lapses_text,
        "grant": job.grant,
        "kind": job.kind,
        "input": job.input,
        "traceparent": f"00-{job.trace_id}-{secrets.token_hex(8)}-01",
    }
    if job.model is not None:
        answer["model"] = dict(job.model)
    return 201, answer


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
        return _claim(state, claims[1], body)
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
    answers = {"outputs": _output, "passages": _passages, "outcome": _outcome, "events": _events}
    return _renewal(state, job) if action == "renewals" else answers[action](state, job, body)


def _tier_of(job: StandInJob, source: str) -> str | None:
    tiers = job.input.get("tiers")
    for tier in tiers if isinstance(tiers, list) else []:
        if isinstance(tier, dict) and source in (tier.get("sources") or []):
            return str(tier.get("tier"))
    return None


def _passages(state: StandIn, job: StandInJob, body: bytes) -> tuple[int, Any]:
    ids = json.loads(body).get("ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 100 or any(not isinstance(item, str) for item in ids):
        return problem(400, "job.passages_invalid")
    if state.passages_status is not None:
        return problem(state.passages_status, state.passages_code)
    wanted = list(dict.fromkeys(ids))
    found = []
    for passage_id in wanted:
        passage = state.passages.get(passage_id)
        tier = _tier_of(job, passage.source) if passage is not None else None
        if passage is None or tier is None:
            return problem(403, "scope.passage_refused")
        found.append(
            {
                "id": passage_id,
                "tier": tier,
                "text": passage.text,
                "sectionPath": passage.section_path,
                "page": passage.page,
            }
        )
    return 200, {"passages": found}


EVENT_TYPES = ("delta", "citation")


def _events_valid(body: dict[str, Any]) -> bool:
    events, usage = body.get("events"), body.get("usage") or []
    if type(body.get("batch")) is not int or body["batch"] < 1 or not isinstance(events, list) or len(events) > 500:
        return False
    for event in events:
        if not isinstance(event, dict) or event.get("type") not in EVENT_TYPES:
            return False
        if event["type"] == "delta" and not (isinstance(event.get("text"), str) and event["text"]):
            return False
        if event["type"] == "citation" and not (type(event.get("offset")) is int and event["offset"] >= 0):
            return False
    keys = ("input", "cacheRead", "cacheWrite", "output", "reasoning")
    return isinstance(usage, list) and all(
        isinstance(item, dict)
        and isinstance(item.get("callId"), str)
        and all(type(item.get(k)) is int for k in keys)
        and type(item.get("estimated", False)) is bool
        for item in usage
    )


def _events(state: StandIn, job: StandInJob, body: bytes) -> tuple[int, Any]:
    if state.events_status is not None and (state.events_failures is None or state.events_failures > 0):
        if state.events_failures is not None:
            state.events_failures -= 1
        return problem(state.events_status, "server.unavailable" if state.events_status >= 500 else "job.refused")
    request = json.loads(body)
    if not _events_valid(request):
        return problem(400, "job.events_invalid")
    stored = state.batches.setdefault(job.job_id, [])
    if state.stop_after is not None and len(stored) >= state.stop_after:
        for item in request.get("usage") or []:
            state.usage.setdefault(item["callId"], item)
        return 200, {"stop": True}
    last = max((item["batch"] for item in stored if item["claimId"] == job.claim_id), default=0)
    if request["batch"] > last + 1:
        return problem(409, "job.batch_out_of_order")
    if request["batch"] == last + 1:
        stored.append({"claimId": job.claim_id, **request})
        for item in request.get("usage") or []:
            state.usage.setdefault(item["callId"], item)
    return 200, {"stop": False}


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


def index_result_valid(result: object) -> bool:
    """The backend's rule for a done `index.change` (`IndexChanges.completed`): the recipe it was indexed with."""
    return (
        isinstance(result, dict)
        and isinstance(result.get("indexRecipe"), str)
        and bool(RECIPE.fullmatch(result["indexRecipe"]))
    )


def _outcome(state: StandIn, job: StandInJob, body: bytes) -> tuple[int, Any]:
    outcome = json.loads(body)
    done = outcome["outcome"] == "done"
    if done and job.kind == "ingestion" and not ingestion_result_valid(job, outcome.get("result")):
        return problem(400, "job.result_invalid")
    if done and job.kind == "index.change" and not index_result_valid(outcome.get("result")):
        return problem(400, "job.result_invalid")
    job.outcomes.append(outcome)
    if outcome["outcome"] == "failed":
        delivered = any(batch["events"] for batch in state.batches.get(job.job_id, []))
        last = outcome.get("retry") is False or job.attempt >= state.max_attempts or delivered
        job.state = "failed" if last else "offered"
    else:
        job.state = "done"
    job.grant = None
    return 204, None


def _recipe(state: StandIn, body: bytes) -> tuple[int, Any]:
    if state.recipe_status is not None:
        return problem(state.recipe_status, "server.unavailable")
    try:
        statement = json.loads(body)
    except ValueError:
        return problem(400, "request.invalid")
    recipe = statement.get("indexRecipe") if isinstance(statement, dict) else None
    if not isinstance(recipe, str) or not RECIPE.fullmatch(recipe):
        return problem(400, "request.invalid")
    with state.lock:
        state.recipe_statements.append(recipe)
    return 204, None


def _route(state: StandIn, method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, Any]:
    if state.credential is not None and headers.get("authorization") != f"Bearer {state.credential}":
        return problem(401, "service.unauthenticated")
    if method == "GET" and path == "/internal/v1/version":
        return state.status, state.body if state.body is not None else {"version": state.version}
    inputs = INPUT_ROUTE.fullmatch(path)
    if method == "GET" and inputs is not None:
        return _input(state, inputs, headers)
    if method == "PUT" and path == "/internal/v1/index-recipe":
        return _recipe(state, body)
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

        def do_PUT(self) -> None:
            self._answer("PUT")

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
