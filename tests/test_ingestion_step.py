"""One ingestion step end to end (Story 1.15), against the stand-in backend and the real sandbox: the clean stage of a
PDF with active content, a password, a broken file, a crash, the deadline and a bad manifest, the step folder erased
every time, and the outcome through a lane (`retry: false`, the result) under the job's trace id."""

from __future__ import annotations

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
from gmshield_worker.ingestion import PIPELINE_VERSION, sandbox, step
from gmshield_worker.ingestion.errors import FILE_UNREADABLE, PASSWORD_REQUIRED, STAGE_UNKNOWN, STEP_FAILED, StepFailed
from gmshield_worker.log import new_trace_id
from tests import pdfs
from tests.conftest import StandIn, log_lines
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
    assert result["next"] == []
    assert [(batch["firstPage"], batch["lastPage"]) for batch in result["batches"]] == [(1, 20), (21, 25)]
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
    for bad in (
        {**good, "pipelineVersion": ""},
        {**good, "pageTotal": "2"},
        {**good, "next": None},
        {**good, "original": {"fileId": "0195a3c0-0000-7000-8000-000000000001"}},
        {**good, "page_total": 2, "pageTotal": None},
    ):
        answer = api.report(job, Outcome.done(bad))
        assert (answer.status, answer.code) == (400, "job.result_invalid"), bad
    assert stand_in.jobs[str(job.job_id)].state == "running"
