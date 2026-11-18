"""Story 2.1: the index. Keyword vectors, embeddings through a fake Ollama (httpx `MockTransport`), the collection in
Qdrant's local mode (`QdrantClient(":memory:")`), and the two job handlers against the backend's stand-in."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import httpx
import pytest
from qdrant_client import QdrantClient, models

from gmshield_worker.backend.client import ClaimedJob, JobsApi
from gmshield_worker.backend.files import JobFiles
from gmshield_worker.lanes import handlers
from gmshield_worker.log import new_trace_id
from gmshield_worker.retrieval import embedding, indexing, keywords, store
from gmshield_worker.retrieval.recipe import INDEX_RECIPE
from gmshield_worker.settings import IndexSettings, check_settings
from tests.conftest import StandIn, index_result_valid, log_lines

CREDENTIAL = "w" * 64
KEY = bytes(range(32))
OTHER_KEY = bytes(range(1, 33))
OWNER = "0195a3c0-0000-7000-8000-0000000000aa"
SOURCE = "0195a3c0-0000-7000-8000-0000000000bb"
OTHER_SOURCE = "0195a3c0-0000-7000-8000-0000000000cc"


# --- a fake Ollama


@dataclass
class FakeOllama:
    """`/api/embed`: a vector derived from each input, or Ollama's context refusal for an input whose text part is
    longer than `limit` code points; `status` forces another answer."""

    limit: int = 10_000
    status: int | None = None
    dimensions: int = embedding.DIMENSIONS
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        if request.url.path != "/api/embed":
            return httpx.Response(404)
        if self.status is not None:
            return httpx.Response(self.status, json={"error": "model unavailable"})
        if any(len(text.split(" | text: ", 1)[1]) > self.limit for text in body["input"]):
            return httpx.Response(400, json={"error": "the input length exceeds the context length"})
        vectors = [self.raw(text) for text in body["input"]]
        return httpx.Response(200, json={"model": body["model"], "embeddings": vectors})

    def raw(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [((digest[i % 32] + i) % 17) - 8.0 for i in range(self.dimensions)]

    def vector(self, title: str, text: str) -> list[float]:
        return self.raw(embedding.embedding_input(title, text))


def embedder(fake: FakeOllama) -> embedding.Embedder:
    return embedding.Embedder("http://stand-in-ollama:11434", transport=httpx.MockTransport(fake))


def close(first: list[float], second: list[float]) -> bool:
    return all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(first, second, strict=True))


# --- keywords


def test_keyword_tokens_fold_width_case_and_diacritics_into_unicode_words() -> None:
    assert keywords.tokens("Espada Vórpal, ESPADA vorpal!") == ["espada", "vorpal", "espada", "vorpal"]
    assert keywords.tokens("\ufb01re \uff36\uff4f\uff52\uff50\uff41\uff4c Straße ação") == [
        "fire",
        "vorpal",
        "strasse",
        "acao",
    ]
    assert keywords.tokens("1d20+5 — perícia_x") == ["1d20", "5", "pericia_x"]
    assert keywords.tokens("한국어 마법") == ["한국어", "마법"]
    assert keywords.tokens("—!?") == []


def test_a_keyword_vector_numbers_tokens_by_the_owners_key_and_counts_them() -> None:
    vector = keywords.keyword_vector("Vorpal vórpal espada", KEY)
    vorpal = keywords.token_number(KEY, "vorpal")
    espada = keywords.token_number(KEY, "espada")
    assert vorpal == int.from_bytes(__import__("hmac").new(KEY, b"vorpal", hashlib.sha256).digest()[:4], "big")
    assert dict(zip(vector.indices, vector.values, strict=True)) == {vorpal: 2.0, espada: 1.0}
    assert vector.indices == sorted(vector.indices)
    assert all(0 <= index < 2**32 for index in vector.indices)
    # The same text under the same key gives the same numbers; another owner's key gives others.
    assert keywords.keyword_vector("Vorpal vórpal espada", KEY) == vector
    other = keywords.keyword_vector("Vorpal vórpal espada", OTHER_KEY)
    assert set(other.indices).isdisjoint(vector.indices)
    assert sorted(other.values) == sorted(vector.values)
    assert keywords.keyword_vector("", KEY) == keywords.Sparse([], [])
    with pytest.raises(ValueError, match="32 bytes"):
        keywords.keyword_vector("x", b"short")


# --- embedding


def test_passages_are_embedded_in_batches_of_16_with_the_title_slot_and_no_truncation() -> None:
    fake = FakeOllama()
    passages = [(f"Seção {i}" if i % 2 else "", f"Texto {i}") for i in range(40)]

    found = embedder(fake).embed(passages)

    assert [len(call["input"]) for call in fake.calls] == [16, 16, 8]
    assert all(call["model"] == "embeddinggemma:300m" and call["truncate"] is False for call in fake.calls)
    assert fake.calls[0]["input"][:2] == ["title: none | text: Texto 0", "title: Seção 1 | text: Texto 1"]
    assert len(found) == 40
    for (title, text), item in zip(passages, found, strict=True):
        assert item.parts == 1
        assert close(item.vector, embedding.normalized(fake.vector(title, text)))
        assert math.isclose(math.sqrt(sum(value * value for value in item.vector)), 1.0)


def test_a_passage_the_model_refuses_is_split_and_its_vector_is_the_normalized_mean_of_its_parts() -> None:
    fake = FakeOllama(limit=120)
    words = [f"palavra{i:03d}" for i in range(40)]
    long_text = " ".join(words[:20]) + "\n\n" + " ".join(words[20:])
    passages = [("A", "curto"), ("Vorpal", long_text), ("B", "também curto")]

    found = embedder(fake).embed(passages)

    # The batch was refused, then each passage went alone, and the long one in parts.
    assert len(fake.calls[0]["input"]) == 3
    accepted = [
        text.split(" | text: ", 1)[1]
        for call in fake.calls[1:]
        for text in call["input"]
        if len(text.split(" | text: ", 1)[1]) <= fake.limit
    ]
    parts = [text for text in accepted if text not in ("curto", "também curto")]
    assert len(parts) == found[1].parts > 1
    assert " ".join(" ".join(parts).split()) == " ".join(long_text.split())  # nothing cut silently
    alone = [call["input"][0] for call in fake.calls[1:] if "palavra" in call["input"][0]]
    assert alone
    assert all(text.startswith("title: Vorpal | text: ") for text in alone)
    expected = embedding.normalized(embedding.mean([fake.vector("Vorpal", part) for part in parts]))
    assert close(found[1].vector, expected)
    assert close(found[0].vector, embedding.normalized(fake.vector("A", "curto")))
    assert (found[0].parts, found[2].parts) == (1, 1)


def test_a_passage_still_refused_after_four_splits_fails_without_a_retry() -> None:
    fake = FakeOllama(limit=5)

    with pytest.raises(embedding.EmbeddingFailed) as failed:
        embedder(fake).embed([("T", "uma frase longa demais " * 40)])

    assert (failed.value.code, failed.value.retry) == ("index.passage_too_long", False)
    # 1 batch, then the passage alone and its halves down to depth 4: never more than 31 single calls.
    assert 1 < len(fake.calls) <= 1 + 31


@pytest.mark.parametrize(
    "fake",
    [FakeOllama(status=500), FakeOllama(status=404), FakeOllama(status=400), FakeOllama(dimensions=3)],
)
def test_any_other_answer_of_ollama_is_retryable_unavailability(fake: FakeOllama) -> None:
    with pytest.raises(embedding.EmbeddingFailed) as failed:
        embedder(fake).embed([("T", "texto")])
    assert (failed.value.code, failed.value.retry) == ("index.embedding_unavailable", True)


def test_an_unreachable_ollama_is_retryable_unavailability() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    unreachable = embedding.Embedder("http://stand-in-ollama:11434", transport=httpx.MockTransport(refuse))
    with pytest.raises(embedding.EmbeddingFailed) as failed:
        unreachable.embed([("T", "texto")])
    assert (failed.value.code, failed.value.retry) == ("index.embedding_unavailable", True)


# --- the store


class Recording:
    """A local client that also records the payload indexes asked for (local mode keeps none)."""

    def __init__(self) -> None:
        self.client = QdrantClient(":memory:")
        self.indexes: list[tuple[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def create_payload_index(self, collection: str, name: str, *, field_schema: Any, wait: bool) -> None:
        self.indexes.append((name, field_schema))


def point(passage_id: str, source: str = SOURCE, page: int = 1) -> store.Point:
    return store.Point(
        passage_id=passage_id,
        dense=embedding.normalized([1.0] * embedding.DIMENSIONS),
        sparse=keywords.keyword_vector("espada vorpal", KEY),
        owner=OWNER,
        source=source,
        kind="rulebook",
        page=page,
        language=None,
    )


def test_the_collection_has_its_two_vectors_and_its_tenant_and_source_indexes() -> None:
    recording = Recording()
    passages = store.PassageStore(recording)  # type: ignore[arg-type]

    passages.ensure_collection()
    passages.ensure_collection()

    params = recording.client.get_collection("passages").config.params
    assert isinstance(params.vectors, dict)
    assert params.vectors["dense"].size == 768
    assert params.vectors["dense"].distance == models.Distance.COSINE
    assert params.sparse_vectors is not None
    assert params.sparse_vectors["sparse"].modifier == models.Modifier.IDF
    # Local mode keeps no payload index, so both are asked for again: each with its own parameters.
    assert recording.indexes[:2] == [
        ("owner", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True)),
        ("source", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD)),
    ]


def test_a_points_payload_is_exactly_its_facts_and_an_upsert_again_keeps_one_point() -> None:
    client = QdrantClient(":memory:")
    passages = store.PassageStore(client)
    passages.ensure_collection()
    first = str(uuid.uuid4())

    passages.upsert([point(first), point(str(uuid.uuid4()), page=2)])
    passages.upsert([point(first)])

    assert passages.count(OWNER, SOURCE) == 2
    stored = client.retrieve("passages", [first], with_payload=True, with_vectors=True)[0]
    assert stored.id == first
    assert stored.payload == {"owner": OWNER, "source": SOURCE, "kind": "rulebook", "page": 1, "language": None}
    assert isinstance(stored.vector, dict)
    assert set(stored.vector) == {"dense", "sparse"}
    record = dataclasses_replace(point(first), record_type="0195a3c0-0000-7000-8000-0000000000dd")
    assert set(record.payload()) == {"owner", "source", "kind", "page", "language", "record_type"}


def dataclasses_replace(value: store.Point, **changes: Any) -> store.Point:
    import dataclasses

    return dataclasses.replace(value, **changes)


def test_a_removal_by_owner_and_source_leaves_another_sources_points_and_is_safe_to_repeat() -> None:
    passages = store.PassageStore(QdrantClient(":memory:"))
    passages.ensure_collection()
    passages.upsert([point(str(uuid.uuid4())) for _ in range(3)])
    passages.upsert([point(str(uuid.uuid4()), source=OTHER_SOURCE) for _ in range(2)])

    passages.remove(OWNER, SOURCE)
    passages.remove(OWNER, SOURCE)

    assert passages.count(OWNER, SOURCE) == 0
    assert passages.count(OWNER, OTHER_SOURCE) == 2


def test_an_unreachable_qdrant_fails_as_store_unavailable() -> None:
    settings = IndexSettings(qdrant_url="http://127.0.0.1:9", qdrant_api_key=None)
    with store.passage_store(settings) as passages, pytest.raises(store.StoreFailed) as failed:
        passages.ensure_collection()
    assert failed.value.code == "index.store_unavailable"


def test_an_https_qdrant_needs_the_private_authoritys_certificate(tmp_path: Any) -> None:
    missing = IndexSettings(qdrant_url="https://qdrant:6333", qdrant_api_key="k", ca_file=str(tmp_path / "none.crt"))
    with pytest.raises(store.StoreFailed), store.passage_store(missing):
        pass


# --- the handlers, against the backend's stand-in


@dataclass
class Indexed:
    job: ClaimedJob
    passages: list[dict[str, Any]]


def lines(passages: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(item).encode() + b"\n" for item in passages)


def offer_change(stand_in: StandIn, count: int, **overrides: Any) -> Indexed:
    stand_in.credential = CREDENTIAL
    passages = [
        {
            "id": str(uuid.uuid4()),
            "ordinal": 200 + i,
            "page": 3 + i // 10,
            "title": f"Espada Vorpal {i}" if i % 3 else "",
            "text": f"Texto secreto número {i}: a espada vórpal corta.",
        }
        for i in range(count)
    ]
    job_input = {
        "source": SOURCE,
        "owner": OWNER,
        "kind": "rulebook",
        "language": "pt-BR",
        "tokenKey": base64.b64encode(KEY).decode(),
        "passages": "passages",
        **overrides,
    }
    job_id = stand_in.offer("index.change", job_input, new_trace_id(), inputs={"passages": lines(passages)})
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    return Indexed(answer.job, passages)


@contextmanager
def local(client: QdrantClient) -> Iterator[store.PassageStore]:
    yield store.PassageStore(client)


def change_handler(stand_in: StandIn, fake: FakeOllama, client: QdrantClient) -> indexing.IndexChange:
    return indexing.IndexChange(JobFiles(stand_in.url, CREDENTIAL), embedder(fake), lambda: local(client))


def test_an_index_change_writes_one_point_per_passage_and_reports_its_recipe(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeOllama()
    client = QdrantClient(":memory:")
    indexed = offer_change(stand_in, 37)

    outcome = change_handler(stand_in, fake, client)(indexed.job, threading.Event())

    assert outcome.outcome == "done"
    assert outcome.result == {"indexRecipe": INDEX_RECIPE, "points": 37}
    assert index_result_valid(outcome.result)
    assert stand_in.stated_recipes == [INDEX_RECIPE]
    assert [len(call["input"]) for call in fake.calls] == [16, 16, 5]
    ids = [item["id"] for item in indexed.passages]
    stored = client.retrieve("passages", ids, with_payload=True, with_vectors=True)
    assert sorted(str(found.id) for found in stored) == sorted(ids)
    by_id = {str(found.id): found for found in stored}
    for item in indexed.passages:
        found = by_id[item["id"]]
        assert found.payload == {
            "owner": OWNER,
            "source": SOURCE,
            "kind": "rulebook",
            "page": item["page"],
            "language": "pt-BR",
        }
        assert isinstance(found.vector, dict)
        sparse = found.vector["sparse"]
        assert isinstance(sparse, models.SparseVector)
        expected = keywords.keyword_vector(item["text"], KEY)
        assert dict(zip(sparse.indices, sparse.values, strict=True)) == dict(
            zip(expected.indices, expected.values, strict=True)
        )
    # Ids and counts only: no text, title, token or key in a log line, and the key never in a repr.
    captured = capsys.readouterr()
    logged = captured.out + captured.err
    done = [line for line in log_lines(captured.out) if line["event"] == "index.change_done"]
    assert done[0]["points"] == 37
    for secret in ("secreto", "Espada", "vórpal", "vorpal", base64.b64encode(KEY).decode(), KEY.hex()):
        assert secret not in logged
    checked = indexing.change_input(indexed.job.input)
    assert base64.b64encode(KEY).decode() not in repr(checked)
    assert str(KEY) not in repr(checked)
    assert "secreto" not in repr(indexing.read_passages(lines(indexed.passages)))


def test_a_run_again_after_qdrant_was_down_upserts_the_same_points(stand_in: StandIn) -> None:
    fake = FakeOllama()
    client = QdrantClient(":memory:")
    down = IndexSettings(qdrant_url="http://127.0.0.1:9")

    # Matrix row "Qdrant down": the job fails, retryable.
    first = offer_change(stand_in, 5)
    failed = indexing.IndexChange(
        JobFiles(stand_in.url, CREDENTIAL), embedder(fake), lambda: store.passage_store(down)
    )(first.job, threading.Event())
    assert (failed.outcome, failed.code, failed.retry) == ("failed", "index.store_unavailable", None)

    handler = change_handler(stand_in, fake, client)
    for _ in range(2):
        again = offer_change(stand_in, 0)
        retried = dataclasses_job(again.job, first.job.input)
        stand_in.jobs[str(retried.job_id)].inputs = {"passages": lines(first.passages)}
        assert handler(retried, threading.Event()).outcome == "done"
    assert store.PassageStore(client).count(OWNER, SOURCE) == 5


def dataclasses_job(job: ClaimedJob, job_input: dict[str, Any]) -> ClaimedJob:
    import dataclasses

    return dataclasses.replace(job, input=job_input)


def test_a_passage_too_long_and_an_unavailable_ollama_fail_as_the_backend_expects(stand_in: StandIn) -> None:
    client = QdrantClient(":memory:")
    too_long = change_handler(stand_in, FakeOllama(limit=3), client)(offer_change(stand_in, 2).job, threading.Event())
    assert (too_long.outcome, too_long.code, too_long.retry) == ("failed", "index.passage_too_long", False)
    down = change_handler(stand_in, FakeOllama(status=503), client)(offer_change(stand_in, 2).job, threading.Event())
    assert (down.outcome, down.code, down.retry) == ("failed", "index.embedding_unavailable", None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tokenKey": base64.b64encode(b"short").decode()},
        {"tokenKey": "not base64!"},
        {"source": "not-a-uuid"},
        {"owner": None},
        {"kind": "a kind with spaces"},
        {"passages": "../passages"},
    ],
)
def test_an_input_that_is_not_the_backends_fails_without_a_retry(stand_in: StandIn, overrides: dict[str, Any]) -> None:
    outcome = change_handler(stand_in, FakeOllama(), QdrantClient(":memory:"))(
        offer_change(stand_in, 1, **overrides).job, threading.Event()
    )
    assert (outcome.outcome, outcome.code, outcome.retry) == ("failed", "index.input_invalid", False)


@pytest.mark.parametrize(
    "body",
    [
        b"not json\n",
        b'{"id": "x", "ordinal": 0, "page": 1, "title": "", "text": "t"}\n',
        b'{"id": "0195a3c0-0000-7000-8000-000000000001", "ordinal": 0, "page": 0, "title": "", "text": "t"}\n',
        b'{"id": "0195a3c0-0000-7000-8000-000000000001", "ordinal": 0, "page": 1, "title": "", "text": ""}\n',
        b'{"id": "0195a3c0-0000-7000-8000-000000000001", "ordinal": 1, "page": 1, "title": "", "text": "t"}\n'
        b'{"id": "0195a3c0-0000-7000-8000-000000000002", "ordinal": 1, "page": 1, "title": "", "text": "t"}\n',
        b"\xff\n",
    ],
)
def test_passages_that_are_not_the_backends_are_refused(body: bytes) -> None:
    with pytest.raises(indexing.InputInvalid):
        indexing.read_passages(body)


def test_a_cancelled_change_stops_before_its_next_batch(stand_in: StandIn) -> None:
    fake = FakeOllama()
    cancel = threading.Event()
    cancel.set()
    outcome = change_handler(stand_in, fake, QdrantClient(":memory:"))(offer_change(stand_in, 3).job, cancel)
    assert (outcome.outcome, outcome.code) == ("failed", "index.cancelled")
    assert fake.calls == []


def test_a_missing_passages_input_is_retryable(stand_in: StandIn) -> None:
    indexed = offer_change(stand_in, 1, passages="other")
    outcome = change_handler(stand_in, FakeOllama(), QdrantClient(":memory:"))(indexed.job, threading.Event())
    assert (outcome.outcome, outcome.code, outcome.retry) == ("failed", "index.input_unavailable", None)


def test_an_index_removal_deletes_its_sources_points_only_and_is_safe_to_repeat(stand_in: StandIn) -> None:
    client = QdrantClient(":memory:")
    passages = store.PassageStore(client)
    change = change_handler(stand_in, FakeOllama(), client)
    assert change(offer_change(stand_in, 4).job, threading.Event()).outcome == "done"
    passages.upsert([point(str(uuid.uuid4()), source=OTHER_SOURCE)])
    removal = indexing.IndexRemoval(lambda: local(client))

    for _ in range(2):
        job_id = stand_in.offer("index.removal", {"source": SOURCE, "owner": OWNER}, new_trace_id())
        answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
        assert answer.job is not None
        claimed = answer.job
        assert removal(claimed, threading.Event()).outcome == "done"

    assert passages.count(OWNER, SOURCE) == 0
    assert passages.count(OWNER, OTHER_SOURCE) == 1
    bad = dataclasses_job(claimed, {"source": SOURCE})
    assert removal(bad, threading.Event()).code == "index.input_invalid"
    down = indexing.IndexRemoval(lambda: store.passage_store(IndexSettings(qdrant_url="http://127.0.0.1:9")))
    assert down(claimed, threading.Event()).code == "index.store_unavailable"


def test_the_index_handlers_run_in_every_mode() -> None:
    found = handlers.handlers_for(check_settings({"GMSHIELD_MODE": "test"}))
    assert {"index.change", "index.removal"} <= set(found)
