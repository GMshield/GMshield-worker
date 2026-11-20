"""Test mode only: the `test.question` job kind (Story 2.3). It proves the search end to end before any production
kind carries a question (Story 2.4): its claim input is the backend's question input (`Scopes.questionInput`), it runs
`retrieval.search.find_passages`, and its result is the handed-over ids, `{"ids": [...]}`, in order. It calls no model.

An input that is not the backend's fails with `retrieval.input_invalid` and `retry: false`; a search failure with its
code, retried by the backend only when another attempt could change it (Ollama, Qdrant, the passages route down).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from gmshield_worker.backend.client import ClaimedJob, Outcome
from gmshield_worker.log import TraceLog
from gmshield_worker.retrieval import search
from gmshield_worker.retrieval.embedding import Embedder
from gmshield_worker.retrieval.indexing import StoreFactory

KIND = "test.question"
UNIT = "worker.retrieval"


@dataclass
class QuestionProbe:
    """The `test.question` handler: `(job, cancel) -> Outcome`, as the lanes call it."""

    reader: search.PassageReader
    embedder: Embedder
    stores: StoreFactory

    def __call__(self, job: ClaimedJob, cancel: threading.Event, /) -> Outcome:
        log = TraceLog(UNIT, job.trace_id)
        job_id = str(job.job_id)
        try:
            checked = search.question_input(job.input)
            log.info("retrieval.search_started", jobId=job_id, attempt=job.attempt)
            handed = search.find_passages(
                job, checked, embedder=self.embedder, stores=self.stores, reader=self.reader, log=log
            )
        except search.InputInvalid:
            log.warn("retrieval.search_failed", jobId=job_id, code=search.INPUT_INVALID, retry=False)
            return Outcome.failed(search.INPUT_INVALID, retry=False)
        except search.SearchFailed as failed:
            error_type = type(failed.__cause__).__name__ if failed.__cause__ is not None else type(failed).__name__
            log.warn(
                "retrieval.search_failed", jobId=job_id, code=failed.code, retry=failed.retry, errorType=error_type
            )
            return Outcome.failed(failed.code, retry=None if failed.retry else False)
        return Outcome.done({"ids": [item.passage_id for item in handed]})
