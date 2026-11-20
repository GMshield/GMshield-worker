"""The files of a claimed job over HTTP (Story 1.15), against the stand-in: the input streamed to a file, an output
streamed back and checked, the headers of every call, and the refusals."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from gmshield_worker.backend.client import ClaimedJob, JobsApi
from gmshield_worker.backend.files import CHUNK_BYTES, TIMEOUT, FileTransferError, JobFiles
from gmshield_worker.log import new_trace_id, parse_traceparent
from tests.conftest import StandIn

CREDENTIAL = "w" * 64


def claimed(stand_in: StandIn, inputs: dict[str, bytes]) -> ClaimedJob:
    stand_in.credential = CREDENTIAL
    job_id = stand_in.offer("ingestion", {"stage": "clean"}, trace_id=new_trace_id(), inputs=inputs)
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    return answer.job


def test_an_input_is_streamed_into_a_new_file_with_the_jobs_headers(stand_in: StandIn, tmp_path: Path) -> None:
    content = bytes(range(256)) * (CHUNK_BYTES // 256 + 3)
    job = claimed(stand_in, {"source": content})
    files = JobFiles(stand_in.url, CREDENTIAL)

    size = files.download(job, "source", tmp_path / "source.pdf")

    assert size == len(content)
    assert (tmp_path / "source.pdf").read_bytes() == content
    request = stand_in.requests[-1]
    assert request["method"] == "GET"
    assert request["path"] == f"/internal/v1/jobs/{job.job_id}/claims/{job.claim_id}/inputs/source"
    assert request["authorization"] == f"Bearer {CREDENTIAL}"
    assert request["x-job-grant"] == job.grant
    assert parse_traceparent(request["traceparent"]) == job.trace_id
    # An existing file is never overwritten.
    with pytest.raises(FileExistsError):
        files.download(job, "source", tmp_path / "source.pdf")


def test_an_unknown_input_or_a_wrong_grant_raises_with_the_status_and_code(stand_in: StandIn, tmp_path: Path) -> None:
    job = claimed(stand_in, {"source": b"%PDF-1.7"})
    files = JobFiles(stand_in.url, CREDENTIAL)

    with pytest.raises(FileTransferError) as missing:
        files.download(job, "other", tmp_path / "a")
    assert (missing.value.status, missing.value.code) == (404, "request.not_found")
    stolen = ClaimedJob(job.job_id, job.claim_id, job.attempt, job.renew_at, job.lapses_at, job.kind, job.input,
                        job.trace_id, "x" * 43)  # fmt: skip
    with pytest.raises(FileTransferError) as refused:
        files.download(stolen, "source", tmp_path / "b")
    assert (refused.value.status, refused.value.code) == (403, "job.grant_refused")
    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


def test_an_output_is_streamed_back_with_its_length_and_checked_against_the_answer(
    stand_in: StandIn, tmp_path: Path
) -> None:
    job = claimed(stand_in, {})
    content = bytes(range(251)) * (2 * CHUNK_BYTES // 251 + 7)
    (tmp_path / "original.pdf").write_bytes(content)
    files = JobFiles(stand_in.url, CREDENTIAL)

    stored = files.upload(job, tmp_path / "original.pdf")

    assert stored.size == len(content)
    assert stored.sha256 == hashlib.sha256(content).hexdigest()
    assert stand_in.outputs[str(stored.file_id)] == content
    request = stand_in.requests[-1]
    assert request["path"] == f"/internal/v1/jobs/{job.job_id}/claims/{job.claim_id}/outputs"
    assert request["content-type"] == "application/octet-stream"
    assert request["content-length"] == str(len(content))
    assert "transfer-encoding" not in request
    assert request["x-job-grant"] == job.grant


def test_an_answer_that_does_not_match_what_was_sent_or_a_refusal_raises(stand_in: StandIn, tmp_path: Path) -> None:
    job = claimed(stand_in, {})
    (tmp_path / "batch.pdf").write_bytes(b"%PDF-1.7 batch")
    files = JobFiles(stand_in.url, CREDENTIAL)

    stand_in.wrong_sha256 = True
    with pytest.raises(FileTransferError) as mismatch:
        files.upload(job, tmp_path / "batch.pdf")
    assert mismatch.value.code == "worker.output_mismatch"

    stand_in.wrong_sha256 = False
    stand_in.output_status = 413
    with pytest.raises(FileTransferError) as refused:
        files.upload(job, tmp_path / "batch.pdf")
    assert (refused.value.status, refused.value.code) == (413, "job.output_too_large")


def test_an_input_read_into_memory_keeps_to_its_limit_and_its_length(stand_in: StandIn) -> None:
    job = claimed(stand_in, {"passages": b"0123456789"})
    files = JobFiles(stand_in.url, CREDENTIAL)

    assert files.read(job, "passages", 10) == b"0123456789"
    # Content-Length over the limit: refused before the body is read.
    with pytest.raises(FileTransferError) as declared:
        files.read(job, "passages", 9)
    assert (declared.value.status, declared.value.code) == (200, "worker.input_too_large")
    with pytest.raises(FileTransferError) as missing:
        files.read(job, "other", 10)
    assert (missing.value.status, missing.value.code) == (404, "request.not_found")

    def answer(body: httpx.Response) -> JobFiles:
        return JobFiles("http://stand-in-backend:8080", CREDENTIAL, {"transport": httpx.MockTransport(lambda _: body)})

    # A body with no length that grows past the limit, and one shorter than its declared length.
    with pytest.raises(FileTransferError) as grown:
        answer(httpx.Response(200, content=iter([b"01234", b"56789", b"x"]))).read(job, "passages", 10)
    assert grown.value.code == "worker.input_too_large"
    with pytest.raises(FileTransferError) as short:
        answer(httpx.Response(200, headers={"content-length": "10"}, stream=httpx.ByteStream(b"01234"))).read(
            job, "passages", 10
        )
    assert (short.value.status, short.value.code) == (200, "worker.input_truncated")


def test_file_calls_have_their_own_timeouts() -> None:
    assert (TIMEOUT.connect, TIMEOUT.read, TIMEOUT.write) == (2.0, 60.0, 60.0)
    assert "secret-value" not in repr(JobFiles("http://b", "secret-value"))
