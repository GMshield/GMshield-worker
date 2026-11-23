"""The Qdrant collection `passages` (Story 2.1, AD-10): one point per passage, its id the passage's id, and (Story 4.2
part B) one point per record of an accepted version, its id the record's id.

The collection is created when missing: the named vectors `dense` (768 dimensions, cosine) and `sparse` (Qdrant's
`idf` modifier), a tenant keyword index on `owner`, and keyword indexes on `source` and `record_type` (whichever index
is missing is added to an existing collection). A passage's point has the payload exactly `{owner, source, kind, page,
language}`; a record's point has exactly `{owner, source, kind, page, language, record_type}`, its page the record's
first page and `record_type` its core type's key: no text, no title, no name, no field, nothing a GM can switch. An
upsert of the same passage or record writes the same point again, so a retried job leaves one point per item. A
removal deletes by a filter on owner and source, never by a list of ids, so it is safe to repeat, takes a version's
record points with its passage points, and also takes what an index change wrote before it was cancelled.

A search (Story 2.3) is one `query_points` call per tier: a dense and a sparse prefetch (`PREFETCH` each) fused by
Qdrant's reciprocal rank fusion (`FusionQuery(Fusion.RRF)`, whose constant is `RRF_K`), `limit` points, with the filter
`owner == the job's owner AND source IN the tier's sources` and the kinds' condition on the query and on each prefetch,
every time. The kinds (`Kinds`) say which points a search may answer:

- `PASSAGES` (a question's answer passages): `record_type is empty` too. Qdrant's `is_empty` matches a point that has
  no `record_type` (or a null or empty one), so this search never gets a record's id.
- `RECORDS` (Story 4.3 part C, an answer's records search): `must_not record_type is empty`, so only record points, of
  any type, and no passage. An answer runs it beside its passages search, so records never take a passage's slot.
- `EVERYTHING` (Story 4.3 part B, the library search): no condition on `record_type`, so passages and records come back
  mixed by their fused rank.
- `records_of(key)` (a library search with a record-type filter): `record_type == key`, so only records of that core
  type, and no passage.

It answers ids and fused scores, and whether each point is a record (only `record_type` of its payload is read, and
only when records may come back; no vector). A question with no keyword token has no sparse prefetch.

The client speaks REST (no gRPC) to `GMSHIELD_QDRANT_URL` with the API key, an https URL verified against the private
authority's certificate, ignoring proxy settings from the environment. Any failure of Qdrant raises `StoreFailed`
with `STORE_UNAVAILABLE`, which the backend retries; nothing here logs, and an error's content is never kept.
"""

from __future__ import annotations

import ssl
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException

from gmshield_worker.retrieval.embedding import DIMENSIONS
from gmshield_worker.retrieval.keywords import Sparse
from gmshield_worker.settings import IndexSettings

COLLECTION = "passages"
DENSE = "dense"
SPARSE = "sparse"
OWNER = "owner"
SOURCE = "source"
RECORD_TYPE = "record_type"
STORE_UNAVAILABLE = "index.store_unavailable"
TIMEOUT_SECONDS = 30
PREFETCH = 40
# Qdrant's constant k of `Fusion.RRF`: a point at rank r (from 0) of a prefetch scores 1 / (r + RRF_K) for it.
RRF_K = 2


class StoreFailed(Exception):
    """Qdrant refused or could not be reached. Holds the code only."""

    def __init__(self) -> None:
        super().__init__(STORE_UNAVAILABLE)
        self.code = STORE_UNAVAILABLE


@dataclass(frozen=True)
class Found:
    """A point a search found: its id (a passage's or, from Story 4.3 part B, a record's), its fused score, and
    whether it is a record's point."""

    passage_id: str
    score: float
    record: bool = False


@dataclass(frozen=True)
class Kinds:
    """Which points a search may answer, as the module says: passages only (`records` false), passages and records,
    records only (`passages` false), or only the records of one core type (`record_type`)."""

    records: bool = False
    record_type: str | None = None
    passages: bool = True


PASSAGES = Kinds()
RECORDS = Kinds(records=True, passages=False)
EVERYTHING = Kinds(records=True)


def records_of(key: str) -> Kinds:
    """Only the records of the core type `key`."""
    return Kinds(records=True, record_type=key)


@dataclass(frozen=True)
class Point:
    """One point: its id (`passage_id`, a passage's or a record's id), its two vectors and the facts of its payload;
    `record_type` only for a record's point."""

    passage_id: str
    dense: list[float]
    sparse: Sparse
    owner: str
    source: str
    kind: str | None
    page: int
    language: str | None
    record_type: str | None = None

    def payload(self) -> dict[str, Any]:
        found: dict[str, Any] = {
            OWNER: self.owner,
            SOURCE: self.source,
            "kind": self.kind,
            "page": self.page,
            "language": self.language,
        }
        if self.record_type is not None:
            found[RECORD_TYPE] = self.record_type
        return found


def _source_filter(owner: str, source: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key=OWNER, match=models.MatchValue(value=owner)),
            models.FieldCondition(key=SOURCE, match=models.MatchValue(value=source)),
        ]
    )


def _sources_filter(owner: str, sources: Sequence[str], kinds: Kinds = PASSAGES) -> models.Filter:
    """The owner's points of these sources that `kinds` allows: passage points only (a record's point has a
    `record_type`), every point, every record point, or the record points of one core type."""
    must: list[models.Condition] = [
        models.FieldCondition(key=OWNER, match=models.MatchValue(value=owner)),
        models.FieldCondition(key=SOURCE, match=models.MatchAny(any=list(sources))),
    ]
    if not kinds.records:
        must.append(models.IsEmptyCondition(is_empty=models.PayloadField(key=RECORD_TYPE)))
    elif kinds.record_type is not None:
        must.append(models.FieldCondition(key=RECORD_TYPE, match=models.MatchValue(value=kinds.record_type)))
    elif not kinds.passages:
        return models.Filter(
            must=must, must_not=[models.IsEmptyCondition(is_empty=models.PayloadField(key=RECORD_TYPE))]
        )
    return models.Filter(must=must)


@dataclass
class PassageStore:
    """The collection `passages` through one client."""

    client: QdrantClient

    def ensure_collection(self) -> None:
        """Creates the collection when it does not exist, and whichever of its three payload indexes is missing (a start
        that stopped between them is finished by the next job, and a collection made before Story 4.2 part B gets its
        `record_type` index)."""
        indexes = {
            OWNER: models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
            SOURCE: models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
            RECORD_TYPE: models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
        }
        try:
            if not self.client.collection_exists(COLLECTION):
                self.client.create_collection(
                    COLLECTION,
                    vectors_config={DENSE: models.VectorParams(size=DIMENSIONS, distance=models.Distance.COSINE)},
                    sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
                )
            schema = self.client.get_collection(COLLECTION).payload_schema
            for name, params in indexes.items():
                if name not in schema:
                    self.client.create_payload_index(COLLECTION, name, field_schema=params, wait=True)
        except (ApiException, httpx.HTTPError, ValueError) as error:
            raise StoreFailed from error

    def upsert(self, points: Sequence[Point]) -> None:
        """Writes the points (a point already there is replaced), and returns once they are applied."""
        if not points:
            return
        try:
            self.client.upsert(
                COLLECTION,
                points=[
                    models.PointStruct(
                        id=point.passage_id,
                        vector={
                            DENSE: point.dense,
                            SPARSE: models.SparseVector(indices=point.sparse.indices, values=point.sparse.values),
                        },
                        payload=point.payload(),
                    )
                    for point in points
                ],
                wait=True,
            )
        except (ApiException, httpx.HTTPError, ValueError) as error:
            raise StoreFailed from error

    def remove(self, owner: str, source: str) -> None:
        """Deletes every point of the owner's source, by a filter on both (its passages' and its records' points);
        nothing when there is none."""
        try:
            self.client.delete(
                COLLECTION, points_selector=models.FilterSelector(filter=_source_filter(owner, source)), wait=True
            )
        except (ApiException, httpx.HTTPError, ValueError) as error:
            raise StoreFailed from error

    def search(
        self,
        owner: str,
        sources: Sequence[str],
        dense: Sequence[float],
        sparse: Sparse,
        limit: int,
        *,
        kinds: Kinds = PASSAGES,
    ) -> list[Found]:
        """The owner's points of these sources that `kinds` allows nearest the question, fused by RRF, best first;
        nothing, and no call, without a source."""
        if not sources:
            return []
        allowed = _sources_filter(owner, sources, kinds)
        prefetch = [models.Prefetch(query=list(dense), using=DENSE, limit=PREFETCH, filter=allowed)]
        if sparse.indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE,
                    limit=PREFETCH,
                    filter=allowed,
                )
            )
        try:
            answer = self.client.query_points(
                COLLECTION,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=allowed,
                limit=limit,
                with_payload=models.PayloadSelectorInclude(include=[RECORD_TYPE]) if kinds.records else False,
                with_vectors=False,
            )
        except (ApiException, httpx.HTTPError, ValueError) as error:
            raise StoreFailed from error
        return [
            Found(str(point.id), float(point.score), bool((point.payload or {}).get(RECORD_TYPE)))
            for point in answer.points
        ]

    def count(self, owner: str, source: str) -> int:
        """How many points the owner's source has."""
        try:
            return self.client.count(COLLECTION, count_filter=_source_filter(owner, source), exact=True).count
        except (ApiException, httpx.HTTPError, ValueError) as error:
            raise StoreFailed from error


def qdrant_client(settings: IndexSettings) -> QdrantClient:
    """A REST client of the configured Qdrant: https verified against the private authority, no proxy."""
    extra: dict[str, Any] = {"trust_env": False}
    if urlsplit(settings.qdrant_url).scheme == "https":
        extra["verify"] = ssl.create_default_context(cafile=settings.ca_file)
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=TIMEOUT_SECONDS,
        prefer_grpc=False,
        check_compatibility=False,
        **extra,
    )


@contextmanager
def passage_store(settings: IndexSettings) -> Iterator[PassageStore]:
    """A store over a new client for one job, closed after it."""
    try:
        client = qdrant_client(settings)
    except (OSError, ValueError) as error:
        raise StoreFailed from error
    try:
        yield PassageStore(client)
    finally:
        client.close()
