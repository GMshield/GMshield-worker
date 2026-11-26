"""Story 2.1: the index. Keyword vectors, embeddings through a fake Ollama (httpx `MockTransport`), the collection in
Qdrant's local mode (`QdrantClient(":memory:")`), and the two job handlers against the backend's stand-in."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
import time
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


def close_32(stored: list[Any], expected: list[float]) -> bool:
    """A vector as Qdrant keeps it (32-bit floats) against the expected one."""
    return all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(stored, expected, strict=True))


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


def bm25(tf: int, length: int) -> float:
    norm = 1.2 * (1 - 0.75 + 0.75 * length / 250)
    return tf * 2.2 / (tf + norm)


def test_a_keyword_vector_numbers_tokens_by_the_owners_key_and_weighs_them_by_bm25() -> None:
    vector = keywords.keyword_vector("Vorpal vórpal espada", KEY)
    vorpal = keywords.token_number(KEY, "vorpal")
    espada = keywords.token_number(KEY, "espada")
    assert vorpal == int.from_bytes(__import__("hmac").new(KEY, b"vorpal", hashlib.sha256).digest()[:4], "big")
    values = dict(zip(vector.indices, vector.values, strict=True))
    assert values.keys() == {vorpal, espada}
    assert values[vorpal] == pytest.approx(bm25(2, 3))
    assert values[espada] == pytest.approx(bm25(1, 3))
    assert math.isclose(keywords.K1, 1.2)
    assert math.isclose(keywords.B, 0.75)
    assert keywords.KEYWORD_AVG_TOKENS == 250
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


def test_a_question_vector_has_the_same_numbers_each_valued_one() -> None:
    question = keywords.query_vector("Onde fica a Universidade? A universidade de Coimbra", KEY)
    passage = keywords.keyword_vector("Onde fica a Universidade? A universidade de Coimbra", KEY)
    assert question.indices == passage.indices
    assert question.values == pytest.approx([1.0] * 6)
    assert keywords.query_vector("", KEY) == keywords.Sparse([], [])
    with pytest.raises(ValueError, match="32 bytes"):
        keywords.query_vector("x", b"short")


def test_a_long_page_of_common_words_no_longer_outweighs_a_short_passage_on_them() -> None:
    # Matrix row "Long passage, common words": a 2,800-character page with "de" 37 times against a 500-character
    # passage with "de" 3 times. Raw counts made the page's "de" 12 times the passage's; BM25 saturates and normalizes.
    filler = [f"pal{index:03d}" for index in range(400)]
    long_words = filler[:363] + ["de"] * 37
    short_words = filler[:72] + ["de"] * 3
    long_text, short_text = " ".join(long_words), " ".join(short_words)
    assert 2600 <= len(long_text) <= 3000
    assert 450 <= len(short_text) <= 600
    de = keywords.token_number(KEY, "de")

    def value(text: str) -> float:
        vector = keywords.keyword_vector(text, KEY)
        return vector.values[vector.indices.index(de)]

    assert value(long_text) == pytest.approx(bm25(37, 400))
    assert value(short_text) == pytest.approx(bm25(3, 75))
    assert value(long_text) / value(short_text) < 1.5
    # Never past k1 + 1, however often a word recurs.
    assert value(" ".join(["de"] * 1000)) < 2.2


def raw_counts(text: str) -> keywords.Sparse:
    """Recipe "1"'s keyword vector: each token number valued by its count."""
    counts: dict[int, int] = {}
    for token in keywords.tokens(text):
        number = keywords.token_number(KEY, token)
        counts[number] = counts.get(number, 0) + 1
    numbers = sorted(counts)
    return keywords.Sparse(numbers, [float(counts[number]) for number in numbers])


def sparse_ranking(vector: Any, texts: list[str], question: str) -> list[int]:
    """The texts' indices as the collection's sparse prefetch ranks them for the question (Qdrant's `idf` modifier)."""
    client = QdrantClient(":memory:")
    passages = store.PassageStore(client)
    passages.ensure_collection()
    ids = [str(uuid.uuid4()) for _ in texts]
    dense = [1.0] + [0.0] * (embedding.DIMENSIONS - 1)
    passages.upsert(
        [
            store.Point(passage_id, dense, vector(text), OWNER, "0195a3c0-0000-7000-8000-0000000000e1", None, 1, None)
            for passage_id, text in zip(ids, texts, strict=True)
        ]
    )
    asked = keywords.query_vector(question, KEY)
    found = client.query_points(
        store.COLLECTION,
        query=models.SparseVector(indices=asked.indices, values=asked.values),
        using=store.SPARSE,
        limit=len(texts),
    ).points
    return [ids.index(str(point.id)) for point in found]


def test_a_short_passage_with_the_rare_question_words_outranks_a_long_page_of_common_ones() -> None:
    # The Coimbra founding question: a long page full of "de", "foi", "onde" and "universidade" against the short
    # passage that names the founding. Under recipe "2" the short passage comes first; under raw counts it did not.
    question = "Quando e onde foi fundada a Universidade de Coimbra"
    sentence = (
        "A universidade foi onde o reitor de Lisboa e o bispo de Braga viveram, e foi de facto a sede de um estudo"
    )
    long_page = " ".join([sentence] * 26)
    assert 2700 <= len(long_page) <= 2900
    short = "Universidade de Coimbra fundada em 1290 por D. Dinis."
    fillers = [
        f"Capitulo {i}: a cidade de Lisboa e o rio de onde veio o rei, que foi a universidade." for i in range(8)
    ]
    fillers += [f"Nota {i} sobre o clero e a nobreza do reino de Portugal." for i in range(8)]
    texts = [long_page, short, *fillers]

    assert sparse_ranking(lambda text: keywords.keyword_vector(text, KEY), texts, question)[0] == 1
    assert sparse_ranking(raw_counts, texts, question)[0] == 0


def test_the_index_recipe_is_2() -> None:
    assert INDEX_RECIPE == "2"


# --- embedding


def test_passages_are_embedded_in_batches_of_4_with_the_title_slot_and_no_truncation() -> None:
    fake = FakeOllama()
    passages = [(f"Seção {i}" if i % 2 else "", f"Texto {i}") for i in range(40)]

    found = embedder(fake).embed(passages)

    assert embedding.BATCH == 4
    assert [len(call["input"]) for call in fake.calls] == [4] * 10
    assert all(call["model"] == "embeddinggemma:300m" and call["truncate"] is False for call in fake.calls)
    assert fake.calls[0]["input"][:2] == ["title: none | text: Texto 0", "title: Seção 1 | text: Texto 1"]
    assert len(found) == 40
    for (title, text), item in zip(passages, found, strict=True):
        assert item.parts == 1
        assert close(item.vector, embedding.normalized(fake.vector(title, text)))
        assert math.isclose(math.sqrt(sum(value * value for value in item.vector)), 1.0)


class GatedOllama(FakeOllama):
    """`FakeOllama`, recording when each call starts; a query waits for `release` before it answers."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.query_started = threading.Event()
        self.order: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        query = body["input"][0].startswith("task: search result")
        self.order.append("query" if query else "batch")
        if query:
            self.query_started.set()
            self.release.wait(timeout=5)
            return httpx.Response(200, json={"model": body["model"], "embeddings": [self.raw(body["input"][0])]})
        return self.index(request)

    def index(self, request: httpx.Request) -> httpx.Response:
        return super().__call__(request)


def test_an_index_batch_waits_while_a_question_embeds_and_starts_once_it_ends() -> None:
    fake = GatedOllama()
    gate = embedding.QuestionsFirst()
    shared = embedding.Embedder("http://stand-in-ollama:11434", transport=httpx.MockTransport(fake), gate=gate)
    question = threading.Thread(target=shared.embed_query, args=("Qual é a CA do tarrasque?",))
    question.start()
    assert fake.query_started.wait(timeout=5)
    found: list[embedding.Embedded] = []
    batch = threading.Thread(target=lambda: found.extend(shared.embed([("", "Texto 0"), ("", "Texto 1")])))
    batch.start()

    # The question runs at Ollama: the batch waits, and has not called Ollama.
    batch.join(timeout=0.3)
    assert batch.is_alive()
    assert fake.order == ["query"]
    fake.release.set()
    question.join(timeout=5)
    batch.join(timeout=5)

    assert fake.order == ["query", "batch"]
    assert len(found) == 2


def test_an_index_batch_goes_after_waiting_at_most_its_cap_for_the_questions() -> None:
    fake = FakeOllama()
    gate = embedding.QuestionsFirst()
    capped = embedding.Embedder(
        "http://stand-in-ollama:11434", transport=httpx.MockTransport(fake), gate=gate, index_wait=0.2
    )
    assert embedding.INDEX_WAIT_SECONDS == 10.0
    assert embedding.Embedder("http://x").gate is embedding.QUESTIONS_FIRST  # one gate for the whole worker

    # A question holds the gate and never lets go: each batch waits its cap, then goes.
    with gate.question():
        started = time.monotonic()
        found = capped.embed([("", f"Texto {i}") for i in range(5)])
        spent = time.monotonic() - started

    assert len(found) == 5
    assert [len(call["input"]) for call in fake.calls] == [4, 1]
    assert 0.4 <= spent < 3.0  # two batches, each after its 0.2 s wait
    assert gate.before_batch(0.0)  # free again once the question ended


class AskDuringSplit(GatedOllama):
    """`GatedOllama` with a context `limit`, whose first index call (refused as too long) starts `asking`, a question,
    and answers only once that question runs at Ollama."""

    def __init__(self) -> None:
        super().__init__()
        self.limit = 120
        self.asking: threading.Thread | None = None

    def index(self, request: httpx.Request) -> httpx.Response:
        answer = super().index(request)
        if self.order == ["batch"] and self.asking is not None:
            self.asking.start()
            assert self.query_started.wait(timeout=5)
        return answer


def test_a_question_arriving_during_a_split_goes_before_the_next_index_call() -> None:
    fake = AskDuringSplit()
    gate = embedding.QuestionsFirst()
    shared = embedding.Embedder("http://stand-in-ollama:11434", transport=httpx.MockTransport(fake), gate=gate)
    fake.asking = threading.Thread(target=shared.embed_query, args=("Qual é a CA do tarrasque?",))
    long_text = " ".join(f"palavra{i:03d}" for i in range(40))
    found: list[embedding.Embedded] = []
    indexing = threading.Thread(target=lambda: found.extend(shared.embed([("A", long_text)])))
    indexing.start()
    assert fake.query_started.wait(timeout=5)

    # The question runs at Ollama: the passage's next call (alone, then in parts) waits for it.
    indexing.join(timeout=0.3)
    assert indexing.is_alive()
    assert fake.order == ["batch", "query"]
    fake.release.set()
    fake.asking.join(timeout=5)
    indexing.join(timeout=5)

    assert fake.order[:2] == ["batch", "query"]
    assert set(fake.order[2:]) == {"batch"}
    assert len(fake.order) > 3  # refused alone, then split
    assert len(found) == 1
    assert found[0].parts > 1


def test_a_failed_question_lets_go_of_the_gate() -> None:
    gate = embedding.QuestionsFirst()
    down = embedding.Embedder(
        "http://stand-in-ollama:11434", transport=httpx.MockTransport(FakeOllama(status=500)), gate=gate
    )

    with pytest.raises(embedding.EmbeddingFailed):
        down.embed_query("Qual é a CA do tarrasque?")

    assert gate.before_batch(0.0)


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


def test_a_vector_with_no_length_is_refused_as_retryable_unavailability() -> None:
    for vector in ([0.0] * embedding.DIMENSIONS, [math.inf] + [0.0] * (embedding.DIMENSIONS - 1)):
        with pytest.raises(embedding.EmbeddingFailed) as refused:
            embedding.normalized(vector)
        assert (refused.value.code, refused.value.retry) == (embedding.EMBEDDING_UNAVAILABLE, True)


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
    # Local mode keeps no payload index, so all three are asked for again: each with its own parameters.
    assert recording.indexes[:3] == [
        ("owner", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True)),
        ("source", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD)),
        ("record_type", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD)),
    ]
    assert recording.indexes[3:6] == recording.indexes[:3]


class WithIndexes(Recording):
    """A local client whose collection already has the payload indexes named in `present`."""

    def __init__(self, present: set[str]) -> None:
        super().__init__()
        self.present = present

    def get_collection(self, collection: str) -> Any:
        found = self.client.get_collection(collection)
        keyword = models.PayloadIndexInfo(data_type=models.PayloadSchemaType.KEYWORD, points=0)
        schema = dict.fromkeys(self.present, keyword)
        return found.model_copy(update={"payload_schema": schema})


def test_a_collection_from_before_records_gets_only_its_record_type_index() -> None:
    recording = WithIndexes({"owner", "source"})
    passages = store.PassageStore(recording)  # type: ignore[arg-type]

    passages.ensure_collection()

    assert recording.indexes == [("record_type", models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD))]


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
    record = dataclasses_replace(point(first), record_type="spell")
    assert record.payload() == {
        "owner": OWNER,
        "source": SOURCE,
        "kind": "rulebook",
        "page": 1,
        "language": None,
        "record_type": "spell",
    }


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
    assert [len(call["input"]) for call in fake.calls] == [4] * 9 + [1]
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


def test_a_passage_holding_unicode_line_breaks_is_one_line() -> None:
    text = "Primeira linha\u2028segunda\u0085terceira\u2029fim\r"
    body = json.dumps(
        {"id": "0195a3c0-0000-7000-8000-000000000001", "ordinal": 0, "page": 1, "title": "T\u2028x", "text": text},
        ensure_ascii=False,
    ).encode()
    assert "\u2028".encode() in body

    found = indexing.read_passages(body + b"\n")

    assert [(passage.text, passage.title) for passage in found] == [(text, "T\u2028x")]


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
    assert {"index.change", "index.removal", "index.record_removal"} <= set(found)
    assert isinstance(found["index.record_removal"], indexing.IndexRecordRemoval)


# --- Story 4.5 part B: a deleted record's point


OTHER_OWNER = "0195a3c0-0000-7000-8000-0000000000dd"


def test_a_points_removal_by_id_takes_only_the_owners_points_of_that_source_and_is_safe_to_repeat() -> None:
    client = QdrantClient(":memory:")
    passages = store.PassageStore(client)
    passages.ensure_collection()
    deleted, kept, elsewhere, theirs = (str(uuid.uuid4()) for _ in range(4))
    passages.upsert(
        [
            dataclasses_replace(point(deleted), record_type="spell"),
            dataclasses_replace(point(kept), record_type="spell"),
            dataclasses_replace(point(elsewhere, source=OTHER_SOURCE), record_type="spell"),
            dataclasses_replace(point(theirs), owner=OTHER_OWNER, record_type="spell"),
        ]
    )

    # Every id is named, but only the owner's point of that source goes.
    for _ in range(2):
        passages.remove_points(OWNER, SOURCE, [deleted, elsewhere, theirs, str(uuid.uuid4())])
    passages.remove_points(OWNER, SOURCE, [])

    remaining = {str(found.id) for found in client.retrieve("passages", [deleted, kept, elsewhere, theirs])}
    assert remaining == {kept, elsewhere, theirs}


def offer_record_removal(stand_in: StandIn, job_input: dict[str, Any]) -> ClaimedJob:
    stand_in.credential = CREDENTIAL
    job_id = stand_in.offer("index.record_removal", job_input, new_trace_id())
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    return answer.job


def test_a_record_removal_deletes_its_records_points_and_leaves_the_rest(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    client = QdrantClient(":memory:")
    passages = store.PassageStore(client)
    passages.ensure_collection()
    deleted, kept, passage_id = (str(uuid.uuid4()) for _ in range(3))
    passages.upsert(
        [
            dataclasses_replace(point(deleted), record_type="spell"),
            dataclasses_replace(point(kept), record_type="item"),
            point(passage_id),
        ]
    )
    removal = indexing.IndexRecordRemoval(lambda: local(client))

    # Matrix row "After removal": the deleted record's point goes; a repeat is done too.
    for _ in range(2):
        job = offer_record_removal(stand_in, {"owner": OWNER, "source": SOURCE, "ids": [deleted.upper()]})
        outcome = removal(job, threading.Event())
        assert (outcome.outcome, outcome.result) == ("done", {})

    assert passages.count(OWNER, SOURCE) == 2
    assert {str(found.id) for found in client.retrieve("passages", [deleted, kept, passage_id])} == {kept, passage_id}
    done = [line for line in log_lines(capsys.readouterr().out) if line["event"] == "index.record_removal_done"]
    assert done and done[-1]["ids"] == 1 and done[-1]["source"] == SOURCE


@pytest.mark.parametrize(
    "job_input",
    [
        {"owner": OWNER, "source": SOURCE},
        {"owner": OWNER, "source": SOURCE, "ids": []},
        {"owner": OWNER, "source": SOURCE, "ids": "0195a3c0-0000-7000-8000-0000000000ee"},
        {"owner": OWNER, "source": SOURCE, "ids": ["not-an-id"]},
        {"owner": OWNER, "source": SOURCE, "ids": [None]},
        {"owner": OWNER, "source": SOURCE, "ids": [SOURCE, SOURCE.upper()]},
        {"owner": OWNER, "source": SOURCE, "ids": [str(uuid.uuid4()) for _ in range(101)]},
        {"owner": "not-an-id", "source": SOURCE, "ids": [SOURCE]},
        {"source": SOURCE, "ids": [SOURCE]},
        {"owner": OWNER, "ids": [SOURCE]},
    ],
)
def test_a_record_removal_input_that_is_not_the_backends_fails_without_a_retry(
    stand_in: StandIn, job_input: dict[str, Any]
) -> None:
    client = QdrantClient(":memory:")
    passages = store.PassageStore(client)
    passages.ensure_collection()
    passages.upsert([dataclasses_replace(point(SOURCE), record_type="spell")])
    outcome = indexing.IndexRecordRemoval(lambda: local(client))(
        offer_record_removal(stand_in, job_input), threading.Event()
    )
    assert (outcome.outcome, outcome.code, outcome.retry) == ("failed", "index.input_invalid", False)
    assert passages.count(OWNER, SOURCE) == 1


def test_a_record_removal_with_qdrant_down_is_retryable(stand_in: StandIn) -> None:
    down = IndexSettings(qdrant_url="http://127.0.0.1:9")
    removal = indexing.IndexRecordRemoval(lambda: store.passage_store(down))

    # Matrix row "After removal", its error handling: the backend retries it, and the record stays hidden meanwhile.
    job = offer_record_removal(stand_in, {"owner": OWNER, "source": SOURCE, "ids": [str(uuid.uuid4())]})
    outcome = removal(job, threading.Event())
    assert (outcome.outcome, outcome.code, outcome.retry) == ("failed", "index.store_unavailable", None)


# --- Story 4.2 part B: the records run


def offer_records(stand_in: StandIn, items: list[dict[str, Any]], **overrides: Any) -> Indexed:
    stand_in.credential = CREDENTIAL
    job_input = {
        "source": SOURCE,
        "owner": OWNER,
        "kind": "rulebook",
        "language": "pt-BR",
        "tokenKey": base64.b64encode(KEY).decode(),
        "records": "records",
        **overrides,
    }
    job_id = stand_in.offer("index.change", job_input, new_trace_id(), inputs={"records": lines(items)})
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    return Indexed(answer.job, items)


def spells(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": str(uuid.uuid4()),
            "page": 12 + i // 4,
            "type": ("spell", "item", "rule")[i % 3],
            "name": f"Bola de Fogo secreta {i}",
            "fields": [{"name": "Nível", "value": str(i)}, {"name": "Alcance", "value": "45 m"}] if i % 2 else [],
        }
        for i in range(count)
    ]


def test_a_records_run_writes_one_point_per_record_its_id_the_records_with_exactly_six_payload_keys(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeOllama()
    client = QdrantClient(":memory:")
    indexed = offer_records(stand_in, spells(9))

    outcome = change_handler(stand_in, fake, client)(indexed.job, threading.Event())

    # Matrix row "Accept with records": one point per record, its id the record's id.
    assert outcome.outcome == "done"
    assert outcome.result == {"indexRecipe": INDEX_RECIPE, "points": 9}
    assert index_result_valid(outcome.result)
    ids = [item["id"] for item in indexed.passages]
    stored = {str(found.id): found for found in client.retrieve("passages", ids, with_payload=True, with_vectors=True)}
    assert sorted(stored) == sorted(ids)
    for item in indexed.passages:
        found = stored[item["id"]]
        assert found.payload == {
            "owner": OWNER,
            "source": SOURCE,
            "kind": "rulebook",
            "page": item["page"],
            "language": "pt-BR",
            "record_type": item["type"],
        }
        lines_text = "\n".join(f"{field['name']}: {field['value']}" for field in item["fields"])
        text = lines_text or item["name"]
        assert isinstance(found.vector, dict)
        # Qdrant keeps 32-bit floats.
        dense = embedding.normalized(fake.vector(item["name"], text))
        stored_dense = found.vector["dense"]
        assert isinstance(stored_dense, list)
        assert close_32(stored_dense, dense)
        sparse = found.vector["sparse"]
        assert isinstance(sparse, models.SparseVector)
        expected = keywords.keyword_vector(f"{item['name']}\n{lines_text}" if lines_text else item["name"], KEY)
        assert dict(zip(sparse.indices, sparse.values, strict=True)) == dict(
            zip(expected.indices, expected.values, strict=True)
        )
    # The embedding's title slot holds the name; its text the fields' lines, or the name alone without fields.
    sent = [text for call in fake.calls for text in call["input"]]
    assert "title: Bola de Fogo secreta 0 | text: Bola de Fogo secreta 0" in sent
    assert "title: Bola de Fogo secreta 1 | text: Nível: 1\nAlcance: 45 m" in sent
    # Ids and counts only: no name, field, token or key in a log line or a repr.
    captured = capsys.readouterr()
    logged = captured.out + captured.err
    done = [line for line in log_lines(captured.out) if line["event"] == "index.change_done"]
    assert (done[0]["records"], done[0]["points"]) == (9, 9)
    for secret in ("secreta", "Bola", "Nível", "Alcance", "45 m", base64.b64encode(KEY).decode()):
        assert secret not in logged
    assert "secreta" not in repr(indexing.read_records(lines(indexed.passages)))
    assert "Alcance" not in repr(indexing.read_records(lines(indexed.passages)))


def test_a_huge_records_embedded_and_keyword_text_is_cut_to_16384_code_points_and_its_point_written(
    stand_in: StandIn,
) -> None:
    fake = FakeOllama(limit=100_000)
    client = QdrantClient(":memory:")
    name = "Tomo" + "é" * 196
    fields = [{"name": f"Campo {i}", "value": "palavra" + "ç" * 7_990} for i in range(4)]
    record_id = str(uuid.uuid4())
    record: dict[str, Any] = {"id": record_id, "page": 3, "type": "item", "name": name, "fields": fields}
    title, text = indexing.record_text(indexing.read_records(lines([record]))[0])
    whole = "\n".join(f"{field['name']}: {field['value']}" for field in fields)
    assert len(whole) > 16_384

    # Matrix row "Huge record": name first, then the fields' lines, cut together to 16,384 code points.
    assert title == name
    assert len(title) + len(text) == indexing.RECORD_TEXT_LIMIT == 16_384
    assert text == whole[: 16_384 - len(name)]
    outcome = change_handler(stand_in, fake, client)(offer_records(stand_in, [record]).job, threading.Event())

    assert (outcome.outcome, outcome.result) == ("done", {"indexRecipe": INDEX_RECIPE, "points": 1})
    assert fake.calls[0]["input"] == [embedding.embedding_input(title, text)]
    found = client.retrieve("passages", [record_id], with_vectors=True)[0]
    assert isinstance(found.vector, dict)
    sparse = found.vector["sparse"]
    assert isinstance(sparse, models.SparseVector)
    expected = keywords.keyword_vector(f"{title}\n{text}", KEY)
    assert sparse.indices == expected.indices
    # A name past the limit is cut to it and leaves no room for a text.
    long_name = indexing.Record(record_id, 1, "spell", "n" * 20_000, ())
    assert [len(part) for part in indexing.record_text(long_name)] == [16_384, 0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"passages": "passages"},  # both names
        {"records": None},  # the name is null
        {"records": "../records"},
    ],
)
def test_a_records_input_that_is_not_the_backends_fails_without_a_retry(
    stand_in: StandIn, overrides: dict[str, Any]
) -> None:
    outcome = change_handler(stand_in, FakeOllama(), QdrantClient(":memory:"))(
        offer_records(stand_in, spells(1), **overrides).job, threading.Event()
    )
    assert (outcome.outcome, outcome.code, outcome.retry) == ("failed", "index.input_invalid", False)


def test_an_input_naming_neither_passages_nor_records_is_refused() -> None:
    raw = {"source": SOURCE, "owner": OWNER, "kind": None, "language": None, "tokenKey": base64.b64encode(KEY).decode()}
    with pytest.raises(indexing.InputInvalid):
        indexing.change_input(raw)
    assert indexing.change_input({**raw, "records": "records"}).records
    assert not indexing.change_input({**raw, "passages": "passages"}).records


RECORD_ID = "0195a3c0-0000-7000-8000-000000000001"


def record_line(**changes: Any) -> bytes:
    item: dict[str, Any] = {"id": RECORD_ID, "page": 1, "type": "spell", "name": "Luz", "fields": [], **changes}
    return json.dumps(item).encode() + b"\n"


@pytest.mark.parametrize(
    "body",
    [
        b"not json\n",
        b"[]\n",
        record_line(id="x"),
        record_line(page=0),
        record_line(page=1_000_001),
        record_line(page="1"),
        record_line(page=True),
        record_line(type="Monster"),
        record_line(type="3d6"),
        record_line(type=""),
        record_line(type="a" * 33),
        record_line(type=["spell"]),
        record_line(type=None),
        record_line(name=""),
        record_line(name="   "),
        record_line(name=7),
        record_line(fields=None),
        record_line(fields=[["Nível", "1"]]),
        record_line(fields=[{"name": "Nível"}]),
        record_line(fields=[{"name": "Nível", "value": 1}]),
        record_line(fields=[{"name": "Nível", "value": "1", "extra": "x"}]),
        record_line() + record_line(),
        b"\xff\n",
    ],
)
def test_records_that_are_not_the_backends_are_refused(body: bytes) -> None:
    with pytest.raises(indexing.InputInvalid):
        indexing.read_records(body)


CORE_KEYS = ("rule", "spell", "item", "creature", "character_sheet", "encounter", "dungeon", "note")


def test_every_core_type_and_an_approved_key_are_read_and_fields_keep_their_order() -> None:
    # Story 4.4 part B: any record type key, a core key or an approved type's (`talento`, `t_3d6_tabela`).
    keys = [*CORE_KEYS, "talento", "t_3d6_tabela", "a" + "b" * 31]
    body = b"".join(
        record_line(id=f"0195a3c0-0000-7000-8000-0000000000{index:02d}", type=key, page=index + 1)
        for index, key in enumerate(keys)
    )
    assert [record.record_type for record in indexing.read_records(body)] == keys
    ordered = indexing.read_records(
        record_line(fields=[{"name": "B", "value": "2"}, {"name": "A", "value": "1"}, {"name": "", "value": ""}])
    )
    assert ordered[0].fields == (("B", "2"), ("A", "1"), ("", ""))


def test_passage_search_never_answers_a_records_point_and_a_removal_takes_both(stand_in: StandIn) -> None:
    client = QdrantClient(":memory:")
    passages = store.PassageStore(client)
    passages.ensure_collection()
    passage_id, record_id = str(uuid.uuid4()), str(uuid.uuid4())
    # The same vectors for both: only the filter tells them apart.
    passages.upsert([point(passage_id), dataclasses_replace(point(record_id), record_type="spell")])
    dense = embedding.normalized([1.0] * embedding.DIMENSIONS)

    # Matrix row "Search": the version has record points, and search returns passage ids only.
    found = passages.search(OWNER, [SOURCE], dense, keywords.query_vector("espada vorpal", KEY), 8)
    assert [item.passage_id for item in found] == [passage_id]
    assert passages.count(OWNER, SOURCE) == 2

    # Matrix row "Remove / discard": the removal by owner and source takes the record's point with the passage's.
    removal = indexing.IndexRemoval(lambda: local(client))
    job_id = stand_in.offer("index.removal", {"source": SOURCE, "owner": OWNER}, new_trace_id())
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    assert removal(answer.job, threading.Event()).outcome == "done"
    assert passages.count(OWNER, SOURCE) == 0


def test_a_records_run_with_qdrant_down_is_retryable(stand_in: StandIn) -> None:
    down = IndexSettings(qdrant_url="http://127.0.0.1:9")
    handler = indexing.IndexChange(
        JobFiles(stand_in.url, CREDENTIAL), embedder(FakeOllama()), lambda: store.passage_store(down)
    )

    # Matrix row "Records run fails": retried by the backend like a passages run.
    failed = handler(offer_records(stand_in, spells(2)).job, threading.Event())
    assert (failed.outcome, failed.code, failed.retry) == ("failed", "index.store_unavailable", None)
    ollama_down = change_handler(stand_in, FakeOllama(status=503), QdrantClient(":memory:"))
    unavailable = ollama_down(offer_records(stand_in, spells(2)).job, threading.Event())
    assert (unavailable.outcome, unavailable.code, unavailable.retry) == ("failed", "index.embedding_unavailable", None)


# --- Story 5.1 part B: a GM's artifact


ARTIFACT = "0195a3c0-0000-7000-8000-0000000000ee"


def test_an_artifact_run_writes_one_point_its_source_and_id_the_artifact_and_a_removal_deletes_it(
    stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeOllama()
    client = QdrantClient(":memory:")
    passages = store.PassageStore(client)
    stand_in.credential = CREDENTIAL
    item = {
        "id": ARTIFACT,
        "page": 1,
        "type": "item",
        "name": "Espada de Kael",
        "fields": [
            {"name": "Dano", "value": "1d8 cortante"},
            {"name": "content", "value": "Forjada nas montanhas do norte."},
        ],
    }
    job_input = {
        "source": ARTIFACT,
        "owner": OWNER,
        "kind": "artifact",
        "language": None,
        "tokenKey": base64.b64encode(KEY).decode(),
        "records": "records",
    }
    job_id = stand_in.offer(indexing.ARTIFACT, job_input, new_trace_id(), inputs={"records": lines([item])})
    answer = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(job_id), new_trace_id())
    assert answer.job is not None
    handler = handlers.handlers_for(check_settings({"GMSHIELD_MODE": "test"}))[indexing.ARTIFACT]
    assert isinstance(handler, indexing.IndexChange)

    outcome = change_handler(stand_in, fake, client)(answer.job, threading.Event())

    # Matrix row "Index": the one records line becomes one point, its id and its source the artifact's id.
    assert outcome.outcome == "done"
    assert outcome.result == {"indexRecipe": INDEX_RECIPE, "points": 1}
    found = client.retrieve("passages", [ARTIFACT], with_payload=True)
    assert [str(one.id) for one in found] == [ARTIFACT]
    assert found[0].payload == {
        "owner": OWNER,
        "source": ARTIFACT,
        "kind": "artifact",
        "page": 1,
        "language": None,
        "record_type": "item",
    }
    # The content is embedded as the last field's line, after the others.
    sent = [text for call in fake.calls for text in call["input"]]
    assert sent == ["title: Espada de Kael | text: Dano: 1d8 cortante\ncontent: Forjada nas montanhas do norte."]
    logged = capsys.readouterr().out
    for secret in ("Espada", "Kael", "Forjada", "cortante"):
        assert secret not in logged

    # Its removal deletes the artifact's points by owner and source, and leaves another source's.
    passages.upsert([point(str(uuid.uuid4()), source=OTHER_SOURCE)])
    removal = handlers.handlers_for(check_settings({"GMSHIELD_MODE": "test"}))[indexing.ARTIFACT_REMOVAL]
    assert isinstance(removal, indexing.IndexRemoval)
    removal_id = stand_in.offer(indexing.ARTIFACT_REMOVAL, {"source": ARTIFACT, "owner": OWNER}, new_trace_id())
    claimed = JobsApi(stand_in.url, CREDENTIAL).claim(UUID(removal_id), new_trace_id())
    assert claimed.job is not None
    assert indexing.IndexRemoval(lambda: local(client))(claimed.job, threading.Event()).outcome == "done"
    assert passages.count(OWNER, ARTIFACT) == 0
    assert passages.count(OWNER, OTHER_SOURCE) == 1
