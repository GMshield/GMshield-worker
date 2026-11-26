"""The handlers by job kind. A handler runs one claimed attempt and says how it ended; the lane
claims, renews and reports. `cancel` is set when the work must stop (the claim was refused or
lapsed, or the worker is stopping): a handler checks it and returns early, and the lane then
reports nothing.

`ingestion` (Story 1.15, `gmshield_worker.ingestion.step`; its records steps call a model from Story 4.2),
`index.change` and `index.removal` (Story 2.1, `gmshield_worker.retrieval.indexing`; `index.record_removal` from Story
4.5 part B), `question.answer` (Story 2.4,
`gmshield_worker.lanes.answer`) and
`library.search` (Story 3.9, `gmshield_worker.lanes.search`) run in every mode; `probe`, `test.question` (Story 2.3,
`gmshield_worker.lanes.question`) and `test.model` (Story 2.10, `gmshield_worker.lanes.model_probe`) in test mode
only. The index and question handlers open a Qdrant client of
their own for each job.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from functools import partial
from typing import Protocol

from gmshield_worker.backend.client import ClaimedJob, JobsApi, Outcome
from gmshield_worker.backend.files import JobFiles
from gmshield_worker.ingestion import step
from gmshield_worker.lanes import answer, model_probe, probe, question, search
from gmshield_worker.retrieval import indexing, store
from gmshield_worker.retrieval.embedding import Embedder
from gmshield_worker.settings import Settings


class Handler(Protocol):
    def __call__(self, job: ClaimedJob, cancel: threading.Event, /) -> Outcome: ...


def handlers_for(settings: Settings) -> Mapping[str, Handler]:
    """The handlers this worker runs: ingestion steps, the index, the answer and the library search always, the probe,
    the question probe and the model probe in test mode only."""
    files = JobFiles(settings.backend_url, settings.worker_credential)
    stores = partial(store.passage_store, settings.index)
    embedder = Embedder(settings.index.ollama_url)
    reader = JobsApi(settings.backend_url, settings.worker_credential)
    found: dict[str, Handler] = {
        step.KIND: step.IngestionStep(files, settings.work_dir, keys=settings.models),
        indexing.CHANGE: indexing.IndexChange(files, embedder, stores),
        indexing.REMOVAL: indexing.IndexRemoval(stores),
        indexing.RECORD_REMOVAL: indexing.IndexRecordRemoval(stores),
        answer.KIND: answer.AnswerJob(reader, embedder, stores, settings.models),
        search.KIND: search.SearchJob(reader, embedder, stores),
    }
    if settings.test_mode:
        found[probe.KIND] = probe.run
        found[question.KIND] = question.QuestionProbe(reader, embedder, stores)
        found[model_probe.KIND] = model_probe.ModelProbe(reader, settings.models)
    return found
