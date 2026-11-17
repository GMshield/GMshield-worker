"""The files of a claimed job (Story 1.15, internal contract 1.2.0): its inputs, downloaded, and its outputs, sent
back. Both stream in `CHUNK_BYTES` pieces and never hold a file whole in memory.

Each call builds its own httpx client, with the same headers and rules as `client.py` (the worker's credential, a
`traceparent` of the job's trace with a new span id, the claim's grant in `X-Job-Grant`, no redirect, no proxy from
the environment) and timeouts of its own: 2 seconds to connect, 60 seconds for each read or write. An answer that is
not the contract's raises `FileTransferError` with the status and the problem's code, never a body; a transport
failure raises httpx's own error. Nothing here logs a file's content, its name or the grant.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from gmshield_worker.backend.client import ClaimedJob
from gmshield_worker.log import traceparent

CHUNK_BYTES = 8 * 1024 * 1024
TIMEOUT = httpx.Timeout(connect=2.0, read=60.0, write=60.0, pool=2.0)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


class FileTransferError(Exception):
    """The backend refused or answered outside the contract. Holds the status and the problem's code only."""

    def __init__(self, status: int, code: str | None = None) -> None:
        super().__init__(f"{status} {code or ''}".strip())
        self.status = status
        self.code = code


@dataclass(frozen=True)
class StoredOutput:
    """An output the backend stored: its id (to name in the job's result), its size and its SHA-256."""

    file_id: UUID
    size: int
    sha256: str


def _problem_code(response: httpx.Response) -> str | None:
    try:
        body = json.loads(response.read())
    except ValueError:
        return None
    code = body.get("code") if isinstance(body, dict) else None
    return code if isinstance(code, str) else None


@dataclass(frozen=True)
class JobFiles:
    """`GET .../inputs/{name}` and `POST .../outputs` of the internal API, for one claimed job at a time."""

    base_url: str
    credential: str = field(repr=False)
    httpx_args: Mapping[str, Any] | None = None

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=TIMEOUT,
            follow_redirects=False,
            trust_env=False,
            **(self.httpx_args or {}),
        )

    def _headers(self, job: ClaimedJob) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.credential}",
            "traceparent": traceparent(job.trace_id),
            "x-job-grant": job.grant,
        }

    def _path(self, job: ClaimedJob, tail: str) -> str:
        return f"/internal/v1/jobs/{job.job_id}/claims/{job.claim_id}/{tail}"

    def download(self, job: ClaimedJob, name: str, target: Path) -> int:
        """Streams the input `name` into a new file `target`; its size. A short or long body raises."""
        headers = {**self._headers(job), "accept": "application/octet-stream"}
        written = 0
        with (
            self._client() as client,
            client.stream("GET", self._path(job, f"inputs/{name}"), headers=headers) as response,
        ):
            if response.status_code != 200:
                raise FileTransferError(response.status_code, _problem_code(response))
            expected = int(response.headers.get("content-length", "-1"))
            with target.open("xb") as out:
                for chunk in response.iter_raw(CHUNK_BYTES):
                    out.write(chunk)
                    written += len(chunk)
        if expected >= 0 and written != expected:
            raise FileTransferError(200, "worker.input_truncated")
        return written

    def upload(self, job: ClaimedJob, source: Path) -> StoredOutput:
        """Streams `source` back as an output of the job; the backend's id for it, checked against its size and hash."""
        size = source.stat().st_size
        digest = hashlib.sha256()

        def chunks() -> Iterator[bytes]:
            with source.open("rb") as stream:
                while chunk := stream.read(CHUNK_BYTES):
                    digest.update(chunk)
                    yield chunk

        headers = {
            **self._headers(job),
            "accept": "application/json",
            "content-type": "application/octet-stream",
            "content-length": str(size),
        }
        with self._client() as client:
            response = client.post(self._path(job, "outputs"), content=chunks(), headers=headers)
        if response.status_code != 201:
            raise FileTransferError(response.status_code, _problem_code(response))
        try:
            body = response.json()
            stored = StoredOutput(UUID(body["fileId"]), int(body["size"]), str(body["sha256"]))
        except (ValueError, KeyError, TypeError) as error:
            raise FileTransferError(response.status_code, "worker.answer_invalid") from error
        if stored.size != size or not _SHA256.fullmatch(stored.sha256) or stored.sha256 != digest.hexdigest():
            raise FileTransferError(response.status_code, "worker.output_mismatch")
        return stored
