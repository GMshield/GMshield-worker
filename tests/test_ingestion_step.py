"""One ingestion step end to end (Story 1.15), against the stand-in backend and the real sandbox: the clean stage of a
PDF with active content, a password, a broken file, a crash, the deadline and a bad manifest, the step folder erased
every time, and the outcome through a lane (`retry: false`, the result) under the job's trace id."""

from __future__ import annotations

import dataclasses
import io
import json
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

import pikepdf
import pytest

from gmshield_worker.backend.client import ClaimedJob, JobsApi, Outcome
from gmshield_worker.backend.files import JobFiles
from gmshield_worker.ingestion import PIPELINE_VERSION, layout, sandbox, step
from gmshield_worker.ingestion.errors import (
    FILE_UNREADABLE,
    PASSWORD_REQUIRED,
    PIPELINE_CHANGED,
    STAGE_UNKNOWN,
    STEP_FAILED,
    StepFailed,
)
from gmshield_worker.log import new_trace_id
from tests import pdfs
from tests.conftest import StandIn, StandInJob, ingestion_result_valid, log_lines
from tests.test_lanes import lane, wait_for

CREDENTIAL = "w" * 64
CLEAN_INPUT = {
    "versionId": "0195a3c0-0000-7000-8000-000000000001",
    "stage": "clean",
    "format": "pdf",
    "input": "source",
}


def claim(stand_in: StandIn, content: bytes, job_input: dict[str, Any] | None = None) -> ClaimedJob:
    stand_in.credential = CREDENTIAL
    job_id = stand_in.offer("ingestion", job_input or CLEAN_INPUT, trace_id=new_trace_id(), inputs={"source": content})
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    return answer.job


def handler(stand_in: StandIn, work: Path, **options: Any) -> step.IngestionStep:
    work.mkdir(exist_ok=True)
    return step.IngestionStep(JobFiles(stand_in.url, CREDENTIAL), str(work), **options)


def made(path: Path, build: Any) -> bytes:
    build(path)
    return path.read_bytes()


def test_a_pdf_with_active_content_is_cleaned_split_sent_back_and_reported(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    content = made(tmp_path / "linked.pdf", lambda path: pdfs.active(path, page_count=25))
    job = claim(stand_in, content)
    work = tmp_path / "work"

    outcome = handler(stand_in, work)(job, threading.Event())

    assert outcome.outcome == "done"
    result = outcome.result
    assert result is not None
    assert result["pipelineVersion"] == PIPELINE_VERSION
    assert result["pageTotal"] == 25
    assert [(batch["firstPage"], batch["lastPage"]) for batch in result["batches"]] == [(1, 20), (21, 25)]
    # One triage step per batch, over that batch.
    assert result["next"] == [
        {"stage": "triage", "firstPage": batch["firstPage"], "lastPage": batch["lastPage"], "input": batch["fileId"]}
        for batch in result["batches"]
    ]
    files = [result["original"]["fileId"], *(batch["fileId"] for batch in result["batches"])]
    assert sorted(files) == sorted(stand_in.outputs)
    for file_id in files:
        sent = stand_in.outputs[file_id]
        with pikepdf.Pdf.open(io.BytesIO(sent)) as pdf:
            assert not pdf.is_encrypted
            assert len(pdf.attachments) == 0
        assert pdfs.EMBEDDED_MARKER not in sent
    with pikepdf.Pdf.open(io.BytesIO(stand_in.outputs[result["original"]["fileId"]])) as original:
        assert len(original.pages) == 25
    assert list(work.iterdir()) == []
    lines = [line for line in log_lines(capsys.readouterr().out) if line.get("jobId") == str(job.job_id)]
    events = [line["event"] for line in lines]
    assert events == ["ingestion.step_started", "ingestion.input_downloaded", "ingestion.child_ended",
                      "ingestion.outputs_stored", "ingestion.step_done"]  # fmt: skip
    assert {line["traceId"] for line in lines} == {job.trace_id}
    done = lines[-1]
    assert isinstance(done["workerMaxRssKib"], int)
    assert isinstance(lines[2]["childMaxRssKib"], int)
    assert job.grant not in json.dumps(lines)


@pytest.mark.parametrize(
    ("build", "code"),
    [
        (lambda path: pdfs.encrypted(path, user="user-secret"), PASSWORD_REQUIRED),
        (pdfs.broken, FILE_UNREADABLE),
    ],
)
def test_a_password_or_an_unreadable_file_fails_at_once_and_leaves_nothing(
    stand_in: StandIn, tmp_path: Path, build: Any, code: str
) -> None:
    job = claim(stand_in, made(tmp_path / "input.pdf", build))
    work = tmp_path / "work"

    outcome = handler(stand_in, work)(job, threading.Event())

    assert outcome == Outcome.failed(code, retry=False)
    assert stand_in.outputs == {}
    assert list(work.iterdir()) == []


def test_a_child_past_its_deadline_is_killed_the_step_fails_for_a_retry_and_its_folder_goes(
    stand_in: StandIn, tmp_path: Path
) -> None:
    job = claim(stand_in, made(tmp_path / "input.pdf", lambda path: pdfs.active(path, page_count=3)))
    work = tmp_path / "work"

    outcome = handler(stand_in, work, deadline_seconds=0.0)(job, threading.Event())

    assert outcome == Outcome.failed(STEP_FAILED)
    assert outcome.retry is None
    assert stand_in.outputs == {}
    assert list(work.iterdir()) == []


def test_a_child_that_crashes_or_writes_a_bad_manifest_fails_the_step_and_sends_nothing(
    stand_in: StandIn, tmp_path: Path
) -> None:
    def crashed(_script: Path, _arguments: list[str], **_options: Any) -> sandbox.ChildResult:
        return sandbox.ChildResult(returncode=-9, timed_out=False, cancelled=False, max_rss_kib=1, seconds=0.1)

    def lying(_script: Path, _arguments: list[str], *, folder: Path, **_options: Any) -> sandbox.ChildResult:
        out = folder / "out"
        out.mkdir()
        (out / "original.pdf").symlink_to("/etc/hostname")
        (out / "manifest.json").write_text(
            json.dumps({"pageTotal": 1, "original": "original.pdf",
                        "batches": [{"file": "../source.pdf", "firstPage": 1, "lastPage": 1}]}),
            encoding="utf-8",
        )  # fmt: skip
        return sandbox.ChildResult(returncode=0, timed_out=False, cancelled=False, max_rss_kib=1, seconds=0.1)

    work = tmp_path / "work"
    for runner in (crashed, lying):
        job = claim(stand_in, b"%PDF-1.7\n")
        assert handler(stand_in, work, runner=runner)(job, threading.Event()) == Outcome.failed(STEP_FAILED)
        assert list(work.iterdir()) == []
    assert stand_in.outputs == {}


def test_a_manifest_must_cover_every_page_in_order(tmp_path: Path) -> None:
    def written(body: object) -> Path:
        (tmp_path / "manifest.json").write_text(json.dumps(body), encoding="utf-8")
        return tmp_path

    good = {"pageTotal": 3, "original": "original.pdf",
            "batches": [{"file": "batch-00001.pdf", "firstPage": 1, "lastPage": 2},
                        {"file": "batch-00003.pdf", "firstPage": 3, "lastPage": 3}]}  # fmt: skip
    assert step.read_manifest(written(good)).page_total == 3
    bad: object
    for bad in (
        {**good, "pageTotal": 4},
        {**good, "pageTotal": True},
        {**good, "original": "/etc/passwd"},
        {**good, "batches": [{"file": "batch-00001.pdf", "firstPage": 2, "lastPage": 3}]},
        {**good, "batches": [{"file": "batch-00001.pdf", "firstPage": 1, "lastPage": 3.0}]},
        {"pageTotal": 3},
        [],
    ):
        with pytest.raises(StepFailed):
            step.read_manifest(written(bad))


def test_a_stage_or_format_this_worker_does_not_run_fails_at_once(stand_in: StandIn, tmp_path: Path) -> None:
    for job_input in ({**CLEAN_INPUT, "stage": "read"}, {**CLEAN_INPUT, "format": "png"}):
        job = claim(stand_in, b"x", job_input)
        assert handler(stand_in, tmp_path / "work")(job, threading.Event()) == Outcome.failed(
            STAGE_UNKNOWN, retry=False
        )
    job = claim(stand_in, b"x", {**CLEAN_INPUT, "input": "../etc"})
    assert handler(stand_in, tmp_path / "work")(job, threading.Event()) == Outcome.failed(STEP_FAILED)


def test_a_refused_download_fails_the_step_for_a_retry(stand_in: StandIn, tmp_path: Path) -> None:
    job = claim(stand_in, b"%PDF-1.7\n", {**CLEAN_INPUT, "input": "missing"})
    work = tmp_path / "work"

    assert handler(stand_in, work)(job, threading.Event()) == Outcome.failed(STEP_FAILED)
    assert list(work.iterdir()) == []


def test_through_a_lane_a_password_is_reported_with_retry_false_and_a_clean_step_with_its_result(
    stand_in: StandIn, tmp_path: Path
) -> None:
    stand_in.credential = CREDENTIAL
    work = tmp_path / "work"
    work.mkdir()
    ingestion = step.IngestionStep(JobFiles(stand_in.url, CREDENTIAL), str(work))
    locked_pdf = made(tmp_path / "user.pdf", lambda path: pdfs.encrypted(path, user="u"))
    readable_pdf = made(tmp_path / "ok.pdf", lambda path: pdfs.active(path, page_count=2))
    locked = stand_in.offer("ingestion", CLEAN_INPUT, new_trace_id(), {"source": locked_pdf})
    readable = stand_in.offer("ingestion", CLEAN_INPUT, new_trace_id(), {"source": readable_pdf})

    with lane(stand_in, "ingestion", 1, {step.KIND: ingestion}) as (_lane, stream):
        stream.add(locked)
        stream.add(readable)
        wait_for(lambda: stand_in.jobs[locked].state == "failed" and stand_in.jobs[readable].state == "done", 30)

    assert stand_in.jobs[locked].outcomes == [{"outcome": "failed", "code": PASSWORD_REQUIRED, "retry": False}]
    assert stand_in.jobs[locked].attempt == 1
    reported = stand_in.jobs[readable].outcomes[0]
    assert reported["outcome"] == "done"
    assert reported["result"]["pageTotal"] == 2
    assert reported["result"]["original"]["fileId"] in stand_in.outputs
    assert list(work.iterdir()) == []


def test_the_stand_in_refuses_a_done_ingestion_result_the_backend_would_refuse(stand_in: StandIn) -> None:
    job = claim(stand_in, b"%PDF-1.7\n")
    api = JobsApi(stand_in.url, CREDENTIAL)
    good = {"pipelineVersion": PIPELINE_VERSION, "pageTotal": 2, "batches": [], "next": []}
    unknown = "0195a3c0-0000-7000-8000-000000000001"
    for bad in (
        {**good, "pipelineVersion": ""},
        {**good, "pageTotal": "2"},
        {**good, "next": "none"},
        {**good, "original": {"fileId": unknown}},
        {**good, "page_total": 2, "pageTotal": None, "layers": [{"firstPage": 1, "lastPage": 1, "layer": "text"}]},
        {**good, "layers": [{"firstPage": 1, "lastPage": 3, "layer": "text"}]},
        {**good, "layers": [{"firstPage": 1, "lastPage": 2, "layer": "text"}, {"firstPage": 2, "lastPage": 2,
                                                                               "layer": "image"}]},
        {**good, "pages": [{"firstPage": 1, "lastPage": 1, "outcome": "read"}]},
        {**good, "pages": [{"firstPage": 1, "lastPage": 1, "outcome": "unreadable"}],
         "texts": [{"fileId": unknown, "firstPage": 1, "lastPage": 1}]},
        {**good, "next": [{"stage": "layout", "firstPage": 1, "lastPage": 1, "input": unknown}]},
    ):  # fmt: skip
        answer = api.report(job, Outcome.done(bad))
        assert (answer.status, answer.code) == (400, "job.result_invalid"), bad
    assert stand_in.jobs[str(job.job_id)].state == "running"


# --- Story 1.16: the pipeline check, triage, layout and text

BATCH_ID = "0195a3c0-0000-7000-8000-0000000000b2"
SENTENCE = "The party rests at the inn for the night"


def step_input(stage: str, first: int, last: int, **extra: Any) -> dict[str, Any]:
    """A later step's claim input: a batch holding pages 21 to 25 of a 25-page version."""
    return {
        "versionId": CLEAN_INPUT["versionId"],
        "stage": stage,
        "format": "pdf",
        "input": "source",
        "pageTotal": 25,
        "pipelineVersion": PIPELINE_VERSION,
        "firstPage": first,
        "lastPage": last,
        "inputFileId": BATCH_ID,
        "inputFirstPage": 21,
        "inputLastPage": 25,
        **extra,
    }


def batch_pdf(tmp_path: Path) -> bytes:
    """Pages 21 to 25: 22 is blank and 23 shows a page number only, as scans do."""
    return made(tmp_path / "batch.pdf", lambda path: pdfs.with_text(path, [SENTENCE, "", "23", SENTENCE, SENTENCE]))


def retried(job: ClaimedJob) -> ClaimedJob:
    """The same claim seen as a second attempt."""
    return dataclasses.replace(job, attempt=2)


def test_a_step_of_a_chain_started_by_another_pipeline_fails_at_once_before_any_download(
    stand_in: StandIn, tmp_path: Path
) -> None:
    for stage in ("triage", "layout"):
        job = claim(stand_in, b"%PDF-1.7\n", step_input(stage, 21, 25, pipelineVersion="1"))

        outcome = handler(stand_in, tmp_path / "work")(job, threading.Event())

        assert outcome == Outcome.failed(PIPELINE_CHANGED, retry=False)
        assert not any("/inputs/" in request["path"] for request in stand_in.job_requests(str(job.job_id)))


def test_a_malformed_claim_input_fails_at_once(stand_in: StandIn, tmp_path: Path) -> None:
    for bad in (
        step_input("triage", 21, 26),
        step_input("triage", 20, 25),
        step_input("triage", 21, 25, inputFileId="../x"),
        step_input("triage", 21, 25, inputLastPage=None),
        step_input("triage", 22, 21),
        {key: value for key, value in step_input("layout", 21, 25).items() if key != "inputFileId"},
        {key: value for key, value in step_input("triage", 21, 25).items() if key != "firstPage"},
    ):
        job = claim(stand_in, b"%PDF-1.7\n", bad)
        assert handler(stand_in, tmp_path / "work")(job, threading.Event()) == Outcome.failed(STEP_FAILED, retry=False)


def test_triage_marks_each_page_and_asks_a_layout_step_for_each_run_of_text_pages(
    stand_in: StandIn, tmp_path: Path
) -> None:
    job = claim(stand_in, batch_pdf(tmp_path), step_input("triage", 21, 25))
    work = tmp_path / "work"

    outcome = handler(stand_in, work)(job, threading.Event())

    assert outcome.outcome == "done", outcome
    assert outcome.result == {
        "pipelineVersion": PIPELINE_VERSION,
        "pageTotal": 25,
        "layers": [
            {"firstPage": 21, "lastPage": 21, "layer": "text"},
            {"firstPage": 22, "lastPage": 23, "layer": "image"},
            {"firstPage": 24, "lastPage": 25, "layer": "text"},
        ],
        "pages": [],
        "next": [
            {"stage": "layout", "firstPage": 21, "lastPage": 21, "input": BATCH_ID},
            {"stage": "layout", "firstPage": 24, "lastPage": 25, "input": BATCH_ID},
        ],
    }
    assert stand_in.jobs[str(job.job_id)].outcomes == []
    assert list(work.iterdir()) == []


def test_a_retried_triage_runs_one_child_per_page_and_a_page_whose_child_fails_is_unreadable(
    stand_in: StandIn, tmp_path: Path
) -> None:
    job = claim(stand_in, batch_pdf(tmp_path), step_input("triage", 22, 25))
    calls: list[tuple[list[str], float]] = []

    def failing_on_24(script: Path, arguments: list[str], **options: Any) -> sandbox.ChildResult:
        calls.append((arguments, options["deadline_seconds"]))
        if arguments[3] == "24":
            return sandbox.ChildResult(returncode=-9, timed_out=True, cancelled=False, max_rss_kib=1, seconds=1.0)
        return sandbox.run_child(script, arguments, **options)

    outcome = handler(stand_in, tmp_path / "work", runner=failing_on_24, page_budget_seconds=40.0)(
        retried(job), threading.Event()
    )

    assert [(arguments[3], arguments[4]) for arguments, _deadline in calls] == [(str(p), str(p)) for p in range(22, 26)]
    assert {deadline for _arguments, deadline in calls} == {10.0}
    assert outcome.result is not None
    assert outcome.result["layers"] == [
        {"firstPage": 22, "lastPage": 23, "layer": "image"},
        {"firstPage": 25, "lastPage": 25, "layer": "text"},
    ]
    assert outcome.result["pages"] == [{"firstPage": 24, "lastPage": 24, "outcome": "unreadable"}]
    assert outcome.result["next"] == [{"stage": "layout", "firstPage": 25, "lastPage": 25, "input": BATCH_ID}]


def fake_layout(failing: set[int] | None = None, missing: set[int] | None = None) -> Any:
    """A layout child that writes a page record for each page asked for, except those it fails on or leaves out."""
    calls: list[list[str]] = []

    def run(_script: Path, arguments: list[str], *, folder: Path, **_options: Any) -> sandbox.ChildResult:
        calls.append(arguments)
        first, last = int(arguments[3]), int(arguments[4])
        if failing and first in failing:
            return sandbox.ChildResult(returncode=1, timed_out=False, cancelled=False, max_rss_kib=1, seconds=0.1)
        out = folder / arguments[1]
        out.mkdir()
        written = [page for page in range(first, last + 1) if not missing or page not in missing]
        for page in written:
            record = {"page": page, "lines": [{"text": f"line {page}", "box": [0, 0, 1, 0.1]}],
                      "blocks": [{"label": "text", "text": f"Page {page}.", "box": [0, 0, 1, 1], "headingLevel": None,
                                  "rows": None}]}  # fmt: skip
            (out / layout.page_file(page)).write_text(json.dumps(record), encoding="utf-8")
        (out / layout.MANIFEST).write_text(json.dumps({"pages": written}), encoding="utf-8")
        return sandbox.ChildResult(returncode=0, timed_out=False, cancelled=False, max_rss_kib=1, seconds=0.1)

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_layout_reads_its_run_into_one_json_file_and_reports_every_page_read_directly(
    stand_in: StandIn, tmp_path: Path
) -> None:
    job = claim(stand_in, batch_pdf(tmp_path), step_input("layout", 23, 25))
    runner = fake_layout()
    work = tmp_path / "work"

    outcome = handler(stand_in, work, runner=runner, artifacts="/models")(job, threading.Event())

    assert runner.calls == [["source.pdf", "out", "21", "23", "25", "/models", "2"]]
    assert outcome.result is not None
    assert outcome.result["pages"] == [{"firstPage": 23, "lastPage": 25, "outcome": "read_directly"}]
    [text_file] = outcome.result["texts"]
    assert (text_file["firstPage"], text_file["lastPage"]) == (23, 25)
    sent = json.loads(stand_in.outputs[text_file["fileId"]])
    assert [page["page"] for page in sent["pages"]] == [23, 24, 25]
    assert sent["pages"][0]["blocks"][0]["text"] == "Page 23."
    assert outcome.result["next"] == []
    assert stand_in.jobs[str(job.job_id)].state == "running"
    assert list(work.iterdir()) == []


def test_a_first_layout_attempt_that_misses_a_page_fails_for_a_retry(stand_in: StandIn, tmp_path: Path) -> None:
    for runner in (fake_layout(missing={24}), fake_layout(failing={23})):
        job = claim(stand_in, batch_pdf(tmp_path), step_input("layout", 23, 25))
        assert handler(stand_in, tmp_path / "work", runner=runner)(job, threading.Event()) == Outcome.failed(
            STEP_FAILED
        )
    assert stand_in.outputs == {}


def test_a_retried_layout_reads_page_by_page_and_leaves_the_failed_page_out_of_its_files(
    stand_in: StandIn, tmp_path: Path
) -> None:
    job = claim(stand_in, batch_pdf(tmp_path), step_input("layout", 21, 25))
    runner = fake_layout(failing={23})

    outcome = handler(stand_in, tmp_path / "work", runner=runner)(retried(job), threading.Event())

    assert [call[3:5] for call in runner.calls] == [[str(p), str(p)] for p in range(21, 26)]
    assert outcome.result is not None
    assert outcome.result["pages"] == [
        {"firstPage": 21, "lastPage": 22, "outcome": "read_directly"},
        {"firstPage": 23, "lastPage": 23, "outcome": "unreadable"},
        {"firstPage": 24, "lastPage": 25, "outcome": "read_directly"},
    ]
    assert [(item["firstPage"], item["lastPage"]) for item in outcome.result["texts"]] == [(21, 22), (24, 25)]
    for item in outcome.result["texts"]:
        pages = [page["page"] for page in json.loads(stand_in.outputs[item["fileId"]])["pages"]]
        assert pages == list(range(item["firstPage"], item["lastPage"] + 1))
    # The stand-in, like the backend, accepts it.
    assert JobsApi(stand_in.url, CREDENTIAL).report(job, outcome).status == 204


TEXT_INPUT = {"versionId": CLEAN_INPUT["versionId"], "stage": "text", "format": "markdown", "input": "source"}


def test_a_markdown_upload_comes_back_unchanged_with_its_pages_read_directly(stand_in: StandIn, tmp_path: Path) -> None:
    content = ("# A\n\nUnder A.\n\n## B\n\n" + " ".join(["palavra"] * 875) + "\n").encode("utf-8")
    job = claim(stand_in, content, TEXT_INPUT)
    work = tmp_path / "work"

    outcome = handler(stand_in, work)(job, threading.Event())

    assert outcome.result is not None
    result = outcome.result
    assert result["pageTotal"] == 5
    assert stand_in.outputs[result["original"]["fileId"]] == content
    assert result["pages"] == [{"firstPage": 1, "lastPage": 5, "outcome": "read_directly"}]
    [text_file] = result["texts"]
    pages = json.loads(stand_in.outputs[text_file["fileId"]])["pages"]
    assert [page["headingPath"] for page in pages] == [["A"], ["A", "B"], ["A", "B"], ["A", "B"], ["A", "B"]]
    assert all(len(page["text"]) <= 3000 for page in pages)
    assert list(work.iterdir()) == []
    assert JobsApi(stand_in.url, CREDENTIAL).report(job, outcome).status == 204


def test_an_empty_text_upload_fails_at_once(stand_in: StandIn, tmp_path: Path) -> None:
    job = claim(stand_in, b"", {**TEXT_INPUT, "format": "txt"})
    assert handler(stand_in, tmp_path / "work")(job, threading.Event()) == Outcome.failed(FILE_UNREADABLE, retry=False)


def test_the_stand_in_checks_texts_against_this_reports_outcomes_and_next_against_known_files() -> None:
    job = StandInJob("j", "ingestion", {"pageTotal": 3, "inputFileId": BATCH_ID}, "t", output_ids={"f"})
    good = {
        "pipelineVersion": PIPELINE_VERSION,
        "pages": [{"firstPage": 1, "lastPage": 2, "outcome": "read_directly"},
                  {"firstPage": 3, "lastPage": 3, "outcome": "unreadable"}],
        "texts": [{"fileId": "f", "firstPage": 1, "lastPage": 2}],
        "next": [{"stage": "layout", "firstPage": 1, "lastPage": 3, "input": BATCH_ID}],
    }  # fmt: skip
    assert ingestion_result_valid(job, good)
    assert not ingestion_result_valid(job, {**good, "texts": [{"fileId": "f", "firstPage": 1, "lastPage": 3}]})
    assert not ingestion_result_valid(job, {**good, "pageTotal": 4})
    assert not ingestion_result_valid(job, {**good, "next": [{"stage": "layout", "input": "other"}]})
