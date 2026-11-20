"""The dense vector of a passage (Story 2.1, AD-10): embeddinggemma through Ollama, never cut silently.

`POST {GMSHIELD_OLLAMA_URL}/api/embed` with the model `MODEL`, `truncate: false`, and each passage as
`title: {title or "none"} | text: {text}` (embeddinggemma's document prompt, its section title in the title slot), in
batches of `BATCH` (4: Ollama embeds one request at a time, about 0.6 s a passage, so a question waits behind at most
one small batch). Plain httpx: no `ollama` package, no proxy from the environment, no redirect.

Questions first: within the worker process, no index call to Ollama (a batch, or one passage or one part of a split
passage) starts while a question's embedding waits or runs (`QUESTIONS_FIRST`, shared by every `Embedder`, so by both
lanes). A call waits at most `INDEX_WAIT_SECONDS` (10 s) for the questions to end, then goes, so a steady flow of
questions never stops indexing. The gate is per process: the bound assumes one worker process per Ollama, as deployed.
The batch size and the gate are not part of the index recipe: the vectors do not change.

When Ollama refuses a batch because an input is longer than the model's context, each of its passages is embedded on
its own, and a passage refused alone is split in two by `ingestion.passages.split` (at a blank line, a line end or a
space near its middle), each half embedded with the same title and split again if refused, at most `MAX_SPLITS` times
deep. Its vector is the L2-normalized mean of its parts' vectors, so every word counts and the point is still one per
passage. A passage still refused at that depth fails the job with `PASSAGE_TOO_LONG`, which no attempt can change; any
other failure of Ollama (refused, unreachable, an answer that is not the contract's) with `EMBEDDING_UNAVAILABLE`,
which the backend retries. Nothing here logs: the caller logs counts.

A question (Story 2.3) is embedded by `Embedder.embed_query` with the same model, `truncate: false` and normalization,
as embeddinggemma's query prompt `task: search result | query: {question}`. A question is never split: one the model
refuses as too long fails with `QUESTION_TOO_LONG`, which no attempt can change; any other failure with
`EMBEDDING_UNAVAILABLE`.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from gmshield_worker.ingestion.passages import split

MODEL = "embeddinggemma:300m"
DIMENSIONS = 768
BATCH = 4
# The longest an index batch waits for the questions embedding before it, then goes.
INDEX_WAIT_SECONDS = 10.0
MAX_SPLITS = 4
EMBEDDING_UNAVAILABLE = "index.embedding_unavailable"
PASSAGE_TOO_LONG = "index.passage_too_long"
QUESTION_TOO_LONG = "retrieval.question_too_long"
# Ollama's answer (HTTP 400) when an input is longer than the context and `truncate` is false.
_CONTEXT_REFUSAL = "exceeds the context length"
TIMEOUT = httpx.Timeout(connect=2.0, read=120.0, write=30.0, pool=2.0)

type Vector = list[float]


class EmbeddingFailed(Exception):
    """The passages could not be embedded: the code, and whether another attempt could change that."""

    def __init__(self, code: str, *, retry: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retry = retry


class _TooLong(Exception):
    """Ollama refused an input as longer than the model's context."""


def embedding_input(title: str, text: str) -> str:
    """embeddinggemma's document prompt: the section title in its title slot ("none" without one)."""
    return f"title: {title or 'none'} | text: {text}"


def query_input(question: str) -> str:
    """embeddinggemma's query prompt for retrieval (Story 2.3)."""
    return f"task: search result | query: {question}"


def normalized(vector: Sequence[float]) -> Vector:
    """The vector scaled to length 1; refused when it has no length."""
    length = math.sqrt(sum(value * value for value in vector))
    # A square root is never negative: `<= 0.0` refuses a zero vector without a float equality (python:S1244).
    if not math.isfinite(length) or length <= 0.0:
        raise EmbeddingFailed(EMBEDDING_UNAVAILABLE, retry=True)
    return [value / length for value in vector]


def mean(vectors: Sequence[Sequence[float]]) -> Vector:
    """The mean of vectors of one size."""
    return [sum(values) / len(vectors) for values in zip(*vectors, strict=True)]


def halves(text: str) -> tuple[str, str] | None:
    """The text cut in two near its middle by `passages.split`; None when it cannot be cut into two non-empty parts."""
    front, rest = split(text, max(1, len(text) // 2))
    if not front or not rest:
        return None
    return front, rest


class QuestionsFirst:
    """The gate that puts questions before index batches at Ollama: `question()` holds it while a question's embedding
    waits or runs; `before_batch(wait)` waits until no question holds it, at most `wait` seconds: True when no question
    holds it, False when the wait ran out."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._questions = 0

    @contextmanager
    def question(self) -> Iterator[None]:
        with self._changed:
            self._questions += 1
        try:
            yield
        finally:
            with self._changed:
                self._questions -= 1
                self._changed.notify_all()

    def before_batch(self, wait: float = INDEX_WAIT_SECONDS) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: self._questions == 0, timeout=wait)


# The worker's one gate: every `Embedder` shares it unless a test gives its own.
QUESTIONS_FIRST = QuestionsFirst()


@dataclass
class Embedded:
    """One passage's vector, and how many parts it was embedded in (1 when it fitted whole)."""

    vector: Vector
    parts: int = 1


@dataclass(frozen=True)
class Embedder:
    """The embeddings of passages, one `Embedded` per `(title, text)`, in order, and of a question."""

    base_url: str
    timeout: httpx.Timeout = field(default_factory=lambda: TIMEOUT)
    transport: httpx.BaseTransport | None = field(default=None, repr=False)
    gate: QuestionsFirst = field(default=QUESTIONS_FIRST, repr=False, compare=False)
    index_wait: float = INDEX_WAIT_SECONDS

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
            transport=self.transport,
        )

    def embed(self, passages: Sequence[tuple[str, str]]) -> list[Embedded]:
        with self._client() as client:
            found: list[Embedded] = []
            for start in range(0, len(passages), BATCH):
                batch = passages[start : start + BATCH]
                # Questions first: a question waiting or running at Ollama goes before this batch (at most index_wait).
                self.gate.before_batch(self.index_wait)
                try:
                    vectors = self._call(client, [embedding_input(title, text) for title, text in batch])
                    found.extend(Embedded(normalized(vector)) for vector in vectors)
                except _TooLong:
                    for title, text in batch:
                        parts = self._parts(client, title, text, depth=0)
                        found.append(Embedded(normalized(mean(parts)), len(parts)))
            return found

    def embed_query(self, question: str) -> Vector:
        """The question's vector, under the query prompt: one call, never split, holding the questions-first gate from
        before the client opens until the answer is read."""
        with self.gate.question(), self._client() as client:
            try:
                return normalized(self._call(client, [query_input(question)])[0])
            except _TooLong:
                raise EmbeddingFailed(QUESTION_TOO_LONG, retry=False) from None

    def _parts(self, client: httpx.Client, title: str, text: str, *, depth: int) -> list[Vector]:
        """The vectors of the passage's parts: itself when it fits, else its halves' parts, `MAX_SPLITS` deep. Each
        call waits for the questions first, as a batch does."""
        self.gate.before_batch(self.index_wait)
        try:
            return self._call(client, [embedding_input(title, text)])
        except _TooLong:
            cut = halves(text) if depth < MAX_SPLITS else None
            if cut is None:
                raise EmbeddingFailed(PASSAGE_TOO_LONG, retry=False) from None
            return [part for half in cut for part in self._parts(client, title, half, depth=depth + 1)]

    def _call(self, client: httpx.Client, inputs: list[str]) -> list[Vector]:
        try:
            response = client.post("/api/embed", json={"model": MODEL, "input": inputs, "truncate": False})
        except httpx.HTTPError as error:
            raise EmbeddingFailed(EMBEDDING_UNAVAILABLE, retry=True) from error
        if response.status_code == 400 and _CONTEXT_REFUSAL in _error_of(response):
            raise _TooLong
        if response.status_code != 200:
            raise EmbeddingFailed(EMBEDDING_UNAVAILABLE, retry=True)
        return _vectors(response, len(inputs))


def _error_of(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, str) else ""


def _vectors(response: httpx.Response, count: int) -> list[Vector]:
    """The answer's `embeddings`: `count` vectors of `DIMENSIONS` finite numbers, or `EMBEDDING_UNAVAILABLE`."""
    try:
        body: Any = response.json()
    except ValueError as error:
        raise EmbeddingFailed(EMBEDDING_UNAVAILABLE, retry=True) from error
    vectors = body.get("embeddings") if isinstance(body, dict) else None
    if not isinstance(vectors, list) or len(vectors) != count:
        raise EmbeddingFailed(EMBEDDING_UNAVAILABLE, retry=True)
    found: list[Vector] = []
    for vector in vectors:
        if (
            not isinstance(vector, list)
            or len(vector) != DIMENSIONS
            or not all(type(value) in (int, float) and math.isfinite(value) for value in vector)
        ):
            raise EmbeddingFailed(EMBEDDING_UNAVAILABLE, retry=True)
        found.append([float(value) for value in vector])
    return found
