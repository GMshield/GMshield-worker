"""One ingestion step end to end (Story 1.15), against the stand-in backend and the real sandbox: the clean stage of a
PDF with active content, a password, a broken file, a crash, the deadline and a bad manifest, the step folder erased
every time, and the outcome through a lane (`retry: false`, the result) under the job's trace id. Then triage, layout
and text (Story 1.16), OCR (Story 1.17), the passages stage over every text file (Story 1.18), and the page views the
reading stages report (Story 1.19)."""

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
    # One triage step per batch, over that batch, then the passages step over every text file (held by the backend).
    assert result["next"] == [
        {"stage": "triage", "firstPage": batch["firstPage"], "lastPage": batch["lastPage"], "input": batch["fileId"]}
        for batch in result["batches"]
    ] + [{"stage": "passages", "firstPage": 1, "lastPage": 25, "input": "texts"}]
    assert ingestion_result_valid(stand_in.jobs[str(job.job_id)], result)
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
    # "pictures" is a stage this worker does not run: under another pipeline it is still a pipeline change.
    for stage in ("triage", "layout", "ocr", "passages", "pictures"):
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


def test_triage_marks_each_page_and_asks_layout_for_each_run_of_text_pages_then_ocr_for_each_run_of_image_pages(
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
            {"stage": "ocr", "firstPage": 22, "lastPage": 23, "input": BATCH_ID},
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
    assert outcome.result["next"] == [
        {"stage": "layout", "firstPage": 25, "lastPage": 25, "input": BATCH_ID},
        {"stage": "ocr", "firstPage": 22, "lastPage": 23, "input": BATCH_ID},
    ]


PICTURE = b"RIFF\x00\x00\x00\x00WEBPVP8 "


def write_picture(out: Path, page: int) -> dict[str, int]:
    """A page picture as a child writes it (the parent never parses one), and its manifest entry."""
    (out / layout.picture_file(page)).write_bytes(PICTURE + str(page).encode())
    return {"page": page, "width": 1545, "height": 2000}


def fake_layout(failing: set[int] | None = None, missing: set[int] | None = None) -> Any:
    """A layout child that writes a page record and a picture for each page asked for, except those it fails on or
    leaves out."""
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
        pictures = [write_picture(out, page) for page in written]
        (out / layout.MANIFEST).write_text(json.dumps({"pages": written, "pictures": pictures}), encoding="utf-8")
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
    # Matrix row "Digital page" (Story 1.19): each page's picture sent back, with its size and its lines.
    views = outcome.result["pageViews"]
    assert [view["page"] for view in views] == [23, 24, 25]
    assert views[0]["picture"]["width"] == 1545 and views[0]["picture"]["height"] == 2000
    assert stand_in.outputs[views[0]["picture"]["fileId"]] == PICTURE + b"23"
    assert views[2]["lines"] == [{"text": "line 25", "box": [0, 0, 1, 0.1]}]
    assert ingestion_result_valid(stand_in.jobs[str(job.job_id)], outcome.result)
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
    # Matrix row "Crashed page": the page whose child failed has no view.
    assert [view["page"] for view in outcome.result["pageViews"]] == [21, 22, 24, 25]
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
    # Matrix row "Text page" (Story 1.19): each page's text, no picture.
    assert result["pageViews"] == [{"page": page["page"], "text": page["text"]} for page in pages]
    assert result["next"] == [{"stage": "passages", "firstPage": 1, "lastPage": 5, "input": "texts"}]
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


# --- Story 1.17: OCR


def scan_input(stage: str, first: int, last: int) -> dict[str, Any]:
    """A later step's claim input over the first batch (pages 1 to 20) of a 25-page scan."""
    return step_input(stage, first, last, inputFirstPage=1, inputLastPage=20)


def test_triage_of_a_scanned_batch_cuts_its_image_pages_into_ocr_steps_of_ten(
    stand_in: StandIn, tmp_path: Path
) -> None:
    content = made(tmp_path / "scan.pdf", lambda path: pdfs.with_text(path, [""] * 20))
    job = claim(stand_in, content, scan_input("triage", 1, 20))

    outcome = handler(stand_in, tmp_path / "work")(job, threading.Event())

    assert outcome.result is not None
    assert outcome.result["layers"] == [{"firstPage": 1, "lastPage": 20, "layer": "image"}]
    assert outcome.result["next"] == [
        {"stage": "ocr", "firstPage": 1, "lastPage": 10, "input": BATCH_ID},
        {"stage": "ocr", "firstPage": 11, "lastPage": 20, "input": BATCH_ID},
    ]
    assert step.OCR_PAGES == 10
    assert step.cut([(1, 25), (30, 30), (41, 50)], 10) == [(1, 10), (11, 20), (21, 25), (30, 30), (41, 50)]


def fake_ocr(crashing: set[int] | None = None, blank: set[int] | None = None) -> Any:
    """An OCR child that reads every page asked for, except the blank ones it lists unreadable and those it crashes on
    (which, read one at a time, end with no manifest)."""
    calls: list[list[str]] = []

    def run(_script: Path, arguments: list[str], *, folder: Path, **_options: Any) -> sandbox.ChildResult:
        calls.append(arguments)
        first, last = int(arguments[3]), int(arguments[4])
        if crashing and any(page in crashing for page in range(first, last + 1)):
            return sandbox.ChildResult(returncode=-11, timed_out=False, cancelled=False, max_rss_kib=1, seconds=0.1)
        out = folder / arguments[1]
        out.mkdir()
        unreadable = [page for page in range(first, last + 1) if blank and page in blank]
        written = [page for page in range(first, last + 1) if page not in unreadable]
        for page in written:
            record = {"page": page, "lines": [{"text": "Capitulo", "box": [0, 0, 1, 0.1]}],
                      "blocks": [{"label": "text", "text": f"Scan {page}.", "box": [0, 0, 1, 1], "headingLevel": None,
                                  "rows": None}]}  # fmt: skip
            (out / layout.page_file(page)).write_text(json.dumps(record), encoding="utf-8")
        pictures = [write_picture(out, page) for page in range(first, last + 1)]
        manifest = {"pages": written, "unreadable": unreadable, "pictures": pictures}
        (out / layout.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        return sandbox.ChildResult(returncode=0, timed_out=False, cancelled=False, max_rss_kib=1, seconds=0.1)

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_ocr_reads_its_run_with_one_child_and_reports_pages_read_by_ocr_and_a_blank_one_unreadable(
    stand_in: StandIn, tmp_path: Path
) -> None:
    job = claim(stand_in, b"%PDF-1.7\n", scan_input("ocr", 1, 10))
    runner = fake_ocr(blank={7})
    work = tmp_path / "work"

    outcome = handler(stand_in, work, runner=runner, artifacts="/models")(job, threading.Event())

    assert runner.calls == [["source.pdf", "out", "1", "1", "10", "/models", "2"]]
    assert outcome.result is not None
    assert outcome.result["pages"] == [
        {"firstPage": 1, "lastPage": 6, "outcome": "read_by_ocr"},
        {"firstPage": 7, "lastPage": 7, "outcome": "unreadable"},
        {"firstPage": 8, "lastPage": 10, "outcome": "read_by_ocr"},
    ]
    assert [(item["firstPage"], item["lastPage"]) for item in outcome.result["texts"]] == [(1, 6), (8, 10)]
    sent = json.loads(stand_in.outputs[outcome.result["texts"][1]["fileId"]])
    assert [page["page"] for page in sent["pages"]] == [8, 9, 10]
    assert sent["pages"][0]["blocks"][0]["text"] == "Scan 8."
    assert outcome.result["next"] == []
    # Matrix row "Blank scan" (Story 1.19): every page has its picture; the blank one has no lines.
    views = {view["page"]: view for view in outcome.result["pageViews"]}
    assert sorted(views) == list(range(1, 11))
    assert views[7]["lines"] == []
    assert views[8]["lines"] == [{"text": "Capitulo", "box": [0, 0, 1, 0.1]}]
    assert stand_in.outputs[views[7]["picture"]["fileId"]] == PICTURE + b"7"
    assert list(work.iterdir()) == []
    assert JobsApi(stand_in.url, CREDENTIAL).report(job, outcome).status == 204


def test_a_first_ocr_attempt_that_crashes_or_misses_a_page_fails_for_a_retry(stand_in: StandIn, tmp_path: Path) -> None:
    def missing(_script: Path, arguments: list[str], *, folder: Path, **_options: Any) -> sandbox.ChildResult:
        out = folder / arguments[1]
        out.mkdir()
        (out / layout.MANIFEST).write_text(json.dumps({"pages": [], "unreadable": [1, 2]}), encoding="utf-8")
        return sandbox.ChildResult(returncode=0, timed_out=False, cancelled=False, max_rss_kib=1, seconds=0.1)

    for runner in (fake_ocr(crashing={3}), missing):
        job = claim(stand_in, b"%PDF-1.7\n", scan_input("ocr", 1, 3))
        assert handler(stand_in, tmp_path / "work", runner=runner)(job, threading.Event()) == Outcome.failed(
            STEP_FAILED
        )
    assert stand_in.outputs == {}


def test_a_retried_ocr_reads_page_by_page_with_a_share_of_the_budget_and_a_crashing_page_is_unreadable(
    stand_in: StandIn, tmp_path: Path
) -> None:
    job = claim(stand_in, b"%PDF-1.7\n", scan_input("ocr", 1, 5))
    runner = fake_ocr(crashing={3})
    deadlines: list[float] = []

    def timed(script: Path, arguments: list[str], **options: Any) -> sandbox.ChildResult:
        deadlines.append(options["deadline_seconds"])
        return runner(script, arguments, **options)  # type: ignore[no-any-return]

    outcome = handler(stand_in, tmp_path / "work", runner=timed, page_budget_seconds=540.0)(
        retried(job), threading.Event()
    )

    assert [call[3:5] for call in runner.calls] == [[str(p), str(p)] for p in range(1, 6)]
    assert set(deadlines) == {108.0}
    assert outcome.result is not None
    assert [view["page"] for view in outcome.result["pageViews"]] == [1, 2, 4, 5]
    assert outcome.result["pages"] == [
        {"firstPage": 1, "lastPage": 2, "outcome": "read_by_ocr"},
        {"firstPage": 3, "lastPage": 3, "outcome": "unreadable"},
        {"firstPage": 4, "lastPage": 5, "outcome": "read_by_ocr"},
    ]
    assert JobsApi(stand_in.url, CREDENTIAL).report(job, outcome).status == 204


def test_a_manifest_that_lists_a_page_twice_or_not_as_a_number_is_refused(tmp_path: Path) -> None:
    body: object
    for body in ({"pages": [1], "unreadable": [1]}, {"pages": [1], "unreadable": ["2"]}, {"pages": "1"}, []):
        (tmp_path / layout.MANIFEST).write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(StepFailed):
            step.read_pages(tmp_path, [1, 2])


# --- Story 1.18: passages


def passages_input(texts: list[tuple[str, int, int]], fmt: str = "pdf", total: int = 25) -> dict[str, Any]:
    """A passages step's claim input: every text file of the version, in page order."""
    return {
        "versionId": CLEAN_INPUT["versionId"],
        "stage": "passages",
        "format": fmt,
        "input": "texts",
        "attempt": 1,
        "pageTotal": total,
        "pipelineVersion": PIPELINE_VERSION,
        "firstPage": 1,
        "lastPage": total,
        "texts": [{"name": name, "firstPage": first, "lastPage": last} for name, first, last in texts],
    }


def text_name(index: int) -> str:
    return f"text-0195a3c0-0000-7000-8000-{index:012d}"


def claim_passages(stand_in: StandIn, job_input: dict[str, Any], files: dict[str, bytes]) -> ClaimedJob:
    stand_in.credential = CREDENTIAL
    job_id = stand_in.offer("ingestion", job_input, trace_id=new_trace_id(), inputs=files)
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    return answer.job


def pdf_records(*pages: tuple[int, list[dict[str, Any]]]) -> bytes:
    return json.dumps({"pages": [{"page": page, "lines": [], "blocks": blocks} for page, blocks in pages]}).encode()


def heading(content: str, level: int) -> dict[str, Any]:
    return {"label": "section_header", "text": content, "box": [0, 0, 1, 0.1], "headingLevel": level, "rows": None}


def para(content: str) -> dict[str, Any]:
    return {"label": "text", "text": content, "box": [0, 0.2, 1, 0.4], "headingLevel": None, "rows": None}


def test_passages_read_every_text_file_in_page_order_and_carry_the_path_across_them(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Matrix row "Path across batches": "Magic Items" on page 19 (first batch), "Vorpal Sword" on page 21 (second).
    first = pdf_records((19, [heading("Magic Items", 1), para("Wondrous things.")]), (20, [para("More.")]))
    second = pdf_records((21, [heading("Vorpal Sword", 2), para("A legendary blade.")]), (22, []))
    job = claim_passages(
        stand_in,
        passages_input([(text_name(1), 19, 20), (text_name(2), 21, 22)]),
        {text_name(1): first, text_name(2): second},
    )
    work = tmp_path / "work"

    outcome = handler(stand_in, work)(job, threading.Event())

    assert outcome.outcome == "done", outcome
    result = outcome.result
    assert result is not None
    assert result["pipelineVersion"] == PIPELINE_VERSION
    assert result["pageTotal"] == 25
    assert result["next"] == []
    assert [(item["page"], item["sectionPath"], item["text"]) for item in result["passages"]] == [
        (19, ["Magic Items"], "Magic Items\nWondrous things."),
        (20, ["Magic Items"], "Magic Items\nMore."),
        (21, ["Magic Items", "Vorpal Sword"], "Vorpal Sword\nA legendary blade."),
    ]
    assert result["passages"][0]["boxes"] == [[0, 0, 1, 0.1], [0, 0.2, 1, 0.4]]
    assert result["outline"] == [
        {"level": 1, "title": "Magic Items", "page": 19},
        {"level": 2, "title": "Vorpal Sword", "page": 21},
    ]
    # No child: two downloads, in page order, then the report.
    paths = [request["path"] for request in stand_in.job_requests(str(job.job_id)) if "/inputs/" in request["path"]]
    assert [path.rsplit("/", 1)[1] for path in paths] == [text_name(1), text_name(2)]
    assert list(work.iterdir()) == []
    assert JobsApi(stand_in.url, CREDENTIAL).report(job, outcome).status == 204
    lines = [line for line in log_lines(capsys.readouterr().out) if line.get("jobId") == str(job.job_id)]
    assert [line["event"] for line in lines] == ["ingestion.step_started", "ingestion.input_downloaded",
                                                 "ingestion.input_downloaded", "ingestion.step_done"]  # fmt: skip
    assert lines[-1]["passages"] == 3
    assert lines[-1]["sections"] == 2
    assert "Vorpal" not in json.dumps(lines)


def test_a_version_with_no_text_file_reports_no_passages(stand_in: StandIn, tmp_path: Path) -> None:
    # Matrix row "All unreadable": no text file; the step runs and reports none.
    job = claim_passages(stand_in, passages_input([]), {})

    outcome = handler(stand_in, tmp_path / "work")(job, threading.Event())

    assert outcome == Outcome.done(
        {"pipelineVersion": PIPELINE_VERSION, "pageTotal": 25, "passages": [], "outline": [], "next": []}
    )
    assert JobsApi(stand_in.url, CREDENTIAL).report(job, outcome).status == 204


def test_a_markdown_upload_goes_through_the_text_stage_then_the_passages_stage(
    stand_in: StandIn, tmp_path: Path
) -> None:
    # Matrix row "Markdown": `# A` / `## B`; the text stage's own files feed the passages stage.
    content = b"# A\n\nUnder A.\n\n## B\n\nUnder B.\n"
    text_job = claim(stand_in, content, TEXT_INPUT)
    text_outcome = handler(stand_in, tmp_path / "work")(text_job, threading.Event())
    assert text_outcome.result is not None
    [text_file] = text_outcome.result["texts"]
    name = "text-" + text_file["fileId"]
    job = claim_passages(
        stand_in,
        passages_input([(name, 1, 2)], fmt="markdown", total=2),
        {name: stand_in.outputs[text_file["fileId"]]},
    )

    outcome = handler(stand_in, tmp_path / "work")(job, threading.Event())

    assert outcome.result is not None
    assert outcome.result["passages"] == [
        {"page": 1, "sectionPath": ["A"], "text": "# A\n\nUnder A.", "range": [0, 13]},
        {"page": 2, "sectionPath": ["A", "B"], "text": "## B\n\nUnder B.", "range": [0, 14]},
    ]
    assert outcome.result["outline"] == [{"level": 1, "title": "A", "page": 1}, {"level": 2, "title": "B", "page": 2}]
    assert JobsApi(stand_in.url, CREDENTIAL).report(job, outcome).status == 204


def test_a_malformed_texts_input_or_text_file_fails_the_passages_step_at_once(
    stand_in: StandIn, tmp_path: Path
) -> None:
    good = passages_input([(text_name(1), 1, 20), (text_name(2), 21, 25)])
    for bad in (
        {**good, "texts": "all"},
        {**good, "texts": [{"name": "source", "firstPage": 1, "lastPage": 20}]},
        {**good, "texts": [{"name": text_name(1), "firstPage": 1, "lastPage": 26}]},
        {**good, "texts": [{"name": text_name(1), "firstPage": 5, "lastPage": 4}]},
        {**good, "texts": [{"name": text_name(2), "firstPage": 21, "lastPage": 25},
                           {"name": text_name(1), "firstPage": 1, "lastPage": 20}]},
        {key: value for key, value in good.items() if key != "pageTotal"},
    ):  # fmt: skip
        job = claim_passages(stand_in, bad, {})
        assert handler(stand_in, tmp_path / "work")(job, threading.Event()) == Outcome.failed(STEP_FAILED, retry=False)

    files = {text_name(1): pdf_records((3, [])), text_name(2): b"{}"}
    job = claim_passages(stand_in, good, files)
    work = tmp_path / "work"
    assert handler(stand_in, work)(job, threading.Event()) == Outcome.failed(STEP_FAILED, retry=False)
    assert list(work.iterdir()) == []


def test_a_passages_step_whose_text_file_cannot_be_downloaded_fails_for_a_retry(
    stand_in: StandIn, tmp_path: Path
) -> None:
    job = claim_passages(stand_in, passages_input([(text_name(1), 1, 25)]), {})
    assert handler(stand_in, tmp_path / "work")(job, threading.Event()) == Outcome.failed(STEP_FAILED)


def test_the_stand_in_refuses_passages_the_backend_would_refuse(stand_in: StandIn) -> None:
    job = claim(stand_in, b"%PDF-1.7\n", passages_input([]))
    api = JobsApi(stand_in.url, CREDENTIAL)
    passage = {"page": 1, "sectionPath": ["A"], "text": "a", "boxes": [[0, 0, 1, 1]]}
    good = {"pipelineVersion": PIPELINE_VERSION, "passages": [passage], "outline": [], "next": []}
    assert ingestion_result_valid(stand_in.jobs[str(job.job_id)], good)
    for bad in (
        {**good, "passages": [{**passage, "page": 0}]},
        {**good, "passages": [{**passage, "page": 26}]},
        {**good, "passages": [{**passage, "text": ""}]},
        {**good, "passages": [{**passage, "text": "a" * 16_385}]},
        {**good, "passages": [{**passage, "sectionPath": ["x"] * 33}]},
        {**good, "passages": [{**passage, "sectionPath": ["x" * 1001]}]},
        {**good, "passages": [{**passage, "range": [0, 1]}]},
        {**good, "passages": [{key: value for key, value in passage.items() if key != "boxes"}]},
        {**good, "outline": [{"level": 1, "title": "", "page": 1}]},
        {**good, "outline": [{"level": 1, "title": "A", "page": 26}]},
    ):
        answer = api.report(job, Outcome.done(bad))
        assert (answer.status, answer.code) == (400, "job.result_invalid"), bad
    assert stand_in.jobs[str(job.job_id)].state == "running"


# --- Story 1.19: page views


def test_a_picture_manifest_that_names_a_page_twice_another_page_a_bad_size_or_no_file_is_refused(
    tmp_path: Path,
) -> None:
    (tmp_path / layout.picture_file(1)).write_bytes(PICTURE)
    good = {"page": 1, "width": 10, "height": 20}
    found = None
    for pictures, refused in (
        ([good], False),
        ([good, good], True),
        ([{**good, "page": 3}], True),
        ([{**good, "page": 2}], True),  # no file
        ([{**good, "width": 0}], True),
        ([{**good, "height": 10_001}], True),
        ([{**good, "width": "10"}], True),
        ([{"page": 1}], True),
        ("all", True),
    ):
        (tmp_path / layout.MANIFEST).write_text(json.dumps({"pages": [1], "pictures": pictures}), encoding="utf-8")
        if refused:
            with pytest.raises(StepFailed):
                step.read_pictures(tmp_path, [1, 2])
        else:
            found = step.read_pictures(tmp_path, [1, 2])
    assert found == {1: step.Picture(tmp_path / layout.picture_file(1), 10, 20)}
    (tmp_path / layout.MANIFEST).write_text(json.dumps({"pages": [1]}), encoding="utf-8")
    assert step.read_pictures(tmp_path, [1]) == {}


def test_a_page_views_lines_keep_those_the_backend_takes_cut_to_its_limits(tmp_path: Path) -> None:
    lines: list[object] = [
        {"text": "kept", "box": [0, 0.5, 1, 1]},
        {"text": "", "box": [0, 0, 1, 1]},
        {"text": "no box"},
        {"text": "outside", "box": [0, 0, 1.5, 1]},
        {"text": "short", "box": [0, 0, 1]},
        {"text": "bool", "box": [True, 0, 1, 1]},
        "not a line",
        {"text": "ç" * 5000, "box": [0, 0, 1, 1]},
    ]
    record = tmp_path / "page.json"
    record.write_text(json.dumps({"page": 1, "lines": lines}), encoding="utf-8")
    kept = step.page_lines(record)
    assert [line["text"][:4] for line in kept] == ["kept", "çççç"]
    assert len(kept[1]["text"]) == step.MAX_LINE
    record.write_text(json.dumps({"page": 1, "lines": [{"text": "a", "box": [0, 0, 1, 1]}] * 10_005}), encoding="utf-8")
    assert len(step.page_lines(record)) == step.MAX_LINES
    record.write_text("{", encoding="utf-8")
    with pytest.raises(StepFailed):
        step.page_lines(record)


def test_a_text_files_pages_become_page_views_or_the_step_fails(tmp_path: Path) -> None:
    path = tmp_path / "text-00001.json"
    path.write_text(json.dumps({"pages": [{"page": 1, "text": "Um"}, {"page": 2, "text": "Dois"}]}), encoding="utf-8")
    assert step.text_views(path, 1, 2) == [{"page": 1, "text": "Um"}, {"page": 2, "text": "Dois"}]
    for first, last in ((1, 3), (2, 3)):
        with pytest.raises(StepFailed):
            step.text_views(path, first, last)
    path.write_text(json.dumps({"pages": [{"page": 1, "text": "a" * 16_385}]}), encoding="utf-8")
    with pytest.raises(StepFailed):
        step.text_views(path, 1, 1)
    path.write_text(json.dumps({"pages": [{"page": 1}]}), encoding="utf-8")
    with pytest.raises(StepFailed):
        step.text_views(path, 1, 1)


def test_the_stand_in_refuses_page_views_the_backend_would_refuse(stand_in: StandIn, tmp_path: Path) -> None:
    job = claim(stand_in, batch_pdf(tmp_path), step_input("layout", 21, 23))
    api = JobsApi(stand_in.url, CREDENTIAL)
    picture = str(JobFiles(stand_in.url, CREDENTIAL).upload(job, _written(tmp_path / "p.webp")).file_id)
    other = str(JobFiles(stand_in.url, CREDENTIAL).upload(job, _written(tmp_path / "q.webp")).file_id)
    frame: dict[str, Any] = {"fileId": picture, "width": 10, "height": 10}
    view: dict[str, Any] = {"page": 21, "picture": frame, "lines": [{"text": "a", "box": [0, 0, 1, 1]}]}
    good = {
        "pipelineVersion": PIPELINE_VERSION,
        "pages": [{"firstPage": 21, "lastPage": 23, "outcome": "unreadable"}],
        "pageViews": [view, {"page": 22, "text": "texto"}],
        "next": [],
    }
    assert ingestion_result_valid(stand_in.jobs[str(job.job_id)], good)
    for bad in (
        [{**view, "page": 24}],
        [view, {**view, "picture": {**frame, "fileId": other}}],
        [{**view, "picture": {**frame, "fileId": str(UUID(int=7))}}],
        [view, {**view, "page": 22}],
        [{**view, "picture": {**frame, "width": 0}}],
        [{**view, "lines": [{"text": "", "box": [0, 0, 1, 1]}]}],
        [{**view, "lines": [{"text": "a", "box": [0, 0, 2, 1]}]}],
        [{**view, "text": "both"}],
        [{"page": 21}],
        [{"page": 21, "text": "a" * 16_385}],
    ):
        answer = api.report(job, Outcome.done({**good, "pageViews": bad}))
        assert (answer.status, answer.code) == (400, "job.result_invalid"), bad
    assert api.report(job, Outcome.done(good)).status == 204


def _written(path: Path) -> Path:
    path.write_bytes(PICTURE)
    return path


# --- Story 1.21: an image set's images assembled into one PDF

# A 96 x 64 HEIC, red on its left half and blue on its right, committed for these tests (see tests/image/test_image.py
# for its provenance).
HEIC = Path(__file__).parent / "image" / "two-colours.heic"
RED = (220, 30, 30)
WHITE_PAGE = (255, 255, 255)


def image_name(index: int) -> str:
    return f"image-0195a3c0-0000-7000-8000-{index:012d}"


def images_input(formats: list[str]) -> dict[str, Any]:
    """An images step's claim input: the set's images, in their order."""
    return {
        "versionId": CLEAN_INPUT["versionId"],
        "stage": "images",
        "format": "images",
        "input": "images",
        "inputs": [{"name": image_name(index), "format": fmt} for index, fmt in enumerate(formats, start=1)],
    }


def encoded_image(picture: Any, fmt: str, **options: Any) -> bytes:
    buffer = io.BytesIO()
    picture.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def photo_set() -> list[tuple[str, bytes]]:
    """A JPEG lying on its side with its EXIF orientation (upright, 40 x 60 with a red block at its top-left), the HEIC,
    a WebP, a TIFF of two frames, a BMP, a HEIC that does not decode and a PNG with transparency: eight pages, the
    seventh unreadable. Each page has a size of its own."""
    from PIL import Image, ImageDraw

    upright = Image.new("RGB", (40, 60), WHITE_PAGE)
    ImageDraw.Draw(upright).rectangle([0, 0, 14, 14], fill=RED)
    exif = Image.Exif()
    exif[0x0112] = 6  # turn 90 degrees clockwise to show it
    lying = upright.rotate(90, expand=True)
    frames = [Image.new("L", (30, 20), 0), Image.new("L", (32, 20), 128)]
    tiff = io.BytesIO()
    frames[0].save(tiff, format="TIFF", save_all=True, append_images=frames[1:])
    transparent = Image.new("RGBA", (24, 36), (0, 0, 0, 0))
    return [
        ("jpeg", encoded_image(lying, "JPEG", quality=95, exif=exif.tobytes())),
        ("heic", HEIC.read_bytes()),
        ("webp", encoded_image(Image.new("RGB", (50, 20), (0, 200, 0)), "WEBP", lossless=True)),
        ("tiff", tiff.getvalue()),
        ("bmp", encoded_image(Image.new("RGB", (16, 16), (0, 0, 255)), "BMP")),
        ("heic", b"\x00\x00\x00\x18ftypheic" + bytes(600)),
        ("png", encoded_image(transparent, "PNG")),
    ]


def claim_images(stand_in: StandIn, images: list[tuple[str, bytes]]) -> ClaimedJob:
    stand_in.credential = CREDENTIAL
    files = {image_name(index): content for index, (_fmt, content) in enumerate(images, start=1)}
    job_id = stand_in.offer(
        "ingestion", images_input([fmt for fmt, _content in images]), trace_id=new_trace_id(), inputs=files
    )
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    return answer.job


def rendered(content: bytes) -> list[Any]:
    """Each page of a PDF rendered at 216 DPI, as the OCR stage renders it."""
    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    document = pdfium.PdfDocument(content)
    try:
        return [document[index].render(scale=3).to_pil().convert("RGB") for index in range(len(document))]
    finally:
        document.close()


def near(colour: tuple[int, int, int], expected: tuple[int, int, int]) -> bool:
    return all(abs(got - want) <= 12 for got, want in zip(colour, expected, strict=True))


def test_an_image_sets_images_become_its_pages_in_order_upright_and_an_undecodable_one_is_a_blank_page_in_place(
    stand_in: StandIn, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = claim_images(stand_in, photo_set())
    work = tmp_path / "work"

    outcome = handler(stand_in, work)(job, threading.Event())

    assert outcome.outcome == "done", outcome
    result = outcome.result
    assert result is not None
    assert result["pipelineVersion"] == PIPELINE_VERSION
    assert result["pageTotal"] == 8
    assert [(batch["firstPage"], batch["lastPage"]) for batch in result["batches"]] == [(1, 8)]
    # The undecodable image is a blank page that triage sends to OCR, which reports it unreadable as any blank scan.
    assert "pages" not in result
    batch = result["batches"][0]["fileId"]
    assert result["next"] == [
        {"stage": "triage", "firstPage": 1, "lastPage": 8, "input": batch},
        {"stage": "passages", "firstPage": 1, "lastPage": 8, "input": "texts"},
    ]
    assert ingestion_result_valid(stand_in.jobs[str(job.job_id)], result)
    pages = rendered(stand_in.outputs[result["original"]["fileId"]])
    # Each page keeps its picture's pixels at 216 DPI, in the set's order, a TIFF's frames in place.
    assert [page.size for page in pages[:6]] == [(40, 60), (96, 64), (50, 20), (30, 20), (32, 20), (16, 16)]
    assert pages[7].size == (24, 36)
    # The JPEG is upright: its red block at the top-left, white below it.
    assert near(pages[0].getpixel((5, 5)), RED)
    assert near(pages[0].getpixel((35, 55)), WHITE_PAGE)
    assert near(pages[1].getpixel((10, 30)), RED)
    assert near(pages[1].getpixel((80, 30)), (30, 30, 220))
    assert near(pages[2].getpixel((25, 10)), (0, 200, 0))
    assert near(pages[3].getpixel((5, 5)), (0, 0, 0))
    assert near(pages[4].getpixel((5, 5)), (128, 128, 128))
    assert near(pages[5].getpixel((5, 5)), (0, 0, 255))
    # The undecodable image is a blank page at its place; the transparent PNG lies on white.
    assert pages[6].getpixel((100, 100)) == WHITE_PAGE
    assert pages[7].getpixel((5, 5)) == WHITE_PAGE
    assert rendered(stand_in.outputs[batch]) and len(rendered(stand_in.outputs[batch])) == 8
    assert list(work.iterdir()) == []
    lines = [line for line in log_lines(capsys.readouterr().out) if line.get("jobId") == str(job.job_id)]
    assert [line["event"] for line in lines].count("ingestion.input_downloaded") == 7
    assert lines[-1]["event"] == "ingestion.step_done"
    assert lines[-1]["undecodedPages"] == 1
    assert JobsApi(stand_in.url, CREDENTIAL).report(job, outcome).status == 204


def test_a_retried_images_step_decodes_one_child_per_image_and_one_that_crashes_is_a_blank_page(
    stand_in: StandIn, tmp_path: Path
) -> None:
    images = photo_set()[:3]
    job = claim_images(stand_in, images)
    calls: list[tuple[list[str], float]] = []

    def runner(script: Path, arguments: list[str], **options: Any) -> sandbox.ChildResult:
        calls.append((arguments, options["deadline_seconds"]))
        if arguments[:2] == ["--decode", "image-00002"]:
            return sandbox.ChildResult(returncode=-9, timed_out=False, cancelled=False, max_rss_kib=1, seconds=0.1)
        return sandbox.run_child(script, arguments, **options)

    outcome = handler(stand_in, tmp_path / "work", runner=runner, page_budget_seconds=540.0)(
        retried(job), threading.Event()
    )

    assert [arguments for arguments, _deadline in calls] == [
        ["--decode", "image-00001", "decoded-00001"],
        ["--decode", "image-00002", "decoded-00002"],
        ["--decode", "image-00003", "decoded-00003"],
        ["--assemble", "out", "decoded-00001", "decoded-00002", "decoded-00003"],
    ]
    assert [deadline for _arguments, deadline in calls[:3]] == [180.0, 180.0, 180.0]
    assert outcome.result is not None
    assert outcome.result["pageTotal"] == 3
    pages = rendered(stand_in.outputs[outcome.result["original"]["fileId"]])
    assert [page.size for page in pages[::2]] == [(40, 60), (50, 20)]
    # The image whose child crashed is a blank page at its place.
    assert pages[1].getpixel((100, 100)) == WHITE_PAGE
    assert near(pages[0].getpixel((5, 5)), RED)
    assert list((tmp_path / "work").iterdir()) == []


def test_a_restarted_image_set_reads_its_stored_original_and_cleans_it_again(stand_in: StandIn, tmp_path: Path) -> None:
    content = made(tmp_path / "original.pdf", lambda path: pdfs.with_text(path, [SENTENCE, SENTENCE]))
    job = claim(stand_in, content, {**CLEAN_INPUT, "stage": "images", "format": "images", "pageTotal": 2})

    outcome = handler(stand_in, tmp_path / "work")(job, threading.Event())

    assert outcome.result is not None
    assert outcome.result["pageTotal"] == 2
    assert "pages" not in outcome.result
    assert [item["stage"] for item in outcome.result["next"]] == ["triage", "passages"]
    assert ingestion_result_valid(stand_in.jobs[str(job.job_id)], outcome.result)


def test_a_malformed_images_input_fails_at_once_before_any_download(stand_in: StandIn, tmp_path: Path) -> None:
    good = images_input(["jpeg", "png"])
    for inputs in (
        None,
        [],
        [{"name": "image-../x", "format": "jpeg"}],
        [{"name": image_name(1), "format": "pdf"}],
        [{"name": image_name(1), "format": "jpeg"}, {"name": image_name(1), "format": "png"}],
        [{"name": image_name(index), "format": "jpeg"} for index in range(1, 502)],
        ["image"],
    ):
        job = claim(stand_in, b"x", {**good, "inputs": inputs})
        outcome = handler(stand_in, tmp_path / "work")(job, threading.Event())
        assert outcome == Outcome.failed(STEP_FAILED, retry=False), inputs
        assert not any("/inputs/" in request["path"] for request in stand_in.job_requests(str(job.job_id)))
    # An images stage of another format is a stage this worker does not run.
    job = claim(stand_in, b"x", {**good, "format": "pdf"})
    assert handler(stand_in, tmp_path / "work")(job, threading.Event()) == Outcome.failed(STAGE_UNKNOWN, retry=False)


def test_an_images_manifest_lists_its_unreadable_pages_within_the_total_in_order_once(tmp_path: Path) -> None:
    good = {"pageTotal": 3, "original": "original.pdf", "batches": [{"file": "batch-00001.pdf", "firstPage": 1,
                                                                      "lastPage": 3}]}  # fmt: skip
    (tmp_path / "manifest.json").write_text(json.dumps({**good, "unreadable": [1, 3]}), encoding="utf-8")
    assert step.read_manifest(tmp_path).unreadable == (1, 3)
    for unreadable in ([4], [2, 2], [3, 1], ["1"], 1):
        (tmp_path / "manifest.json").write_text(json.dumps({**good, "unreadable": unreadable}), encoding="utf-8")
        with pytest.raises(StepFailed):
            step.read_manifest(tmp_path)
