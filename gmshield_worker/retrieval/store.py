"""The Qdrant collection `passages` (Story 2.1, AD-10): one point per passage, its id the passage's id.

The collection is created when missing: the named vectors `dense` (768 dimensions, cosine) and `sparse` (Qdrant's
`idf` modifier), a tenant keyword index on `owner` and a keyword index on `source`. A point's payload is exactly
`{owner, source, kind, page, language}`, plus `record_type` only when a record has one (never in Story 2.1): no text,
no title, no name, nothing a GM can switch. An upsert of the same passage writes the same point again, so a retried
job leaves one point per passage. A removal deletes by a filter on owner and source, never by a list of ids, so it is
safe to repeat and also takes what an index change wrote before it was cancelled.

A search (Story 2.3) is one `query_points` call per tier: a dense and a sparse prefetch (`PREFETCH` each) fused by
Qdrant's reciprocal rank fusion (`FusionQuery(Fusion.RRF)`, whose constant is `RRF_K`), `limit` points, with the filter
`owner == the job's owner AND source IN the tier's sources` on the query and on each prefetch, every time. It answers
ids and fused scores only (no payload, no vector). A question with no keyword token has no sparse prefetch.

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
    """A point a search found: the passage's id and its fused score."""

    passage_id: str
    score: float


@dataclass(frozen=True)
class Point:
    """One passage's point: its id, its two vectors and the facts of its payload."""

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
            found["record_type"] = self.record_type
        return found


def _source_filter(owner: str, source: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key=OWNER, match=models.MatchValue(value=owner)),
            models.FieldCondition(key=SOURCE, match=models.MatchValue(value=source)),
        ]
    )


def _sources_filter(owner: str, sources: Sequence[str]) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key=OWNER, match=models.MatchValue(value=owner)),
            models.FieldCondition(key=SOURCE, match=models.MatchAny(any=list(sources))),
        ]
    )


@dataclass
class PassageStore:
    """The collection `passages` through one client."""

    client: QdrantClient

    def ensure_collection(self) -> None:
        """Creates the collection when it does not exist, and whichever of its two payload indexes is missing (a start
        that stopped between the two is finished by the next job)."""
        indexes = {
            OWNER: models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
            SOURCE: models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
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
        """Deletes every point of the owner's source, by a filter on both; nothing when there is none."""
        try:
            self.client.delete(
                COLLECTION, points_selector=models.FilterSelector(filter=_source_filter(owner, source)), wait=True
            )
        except (ApiException, httpx.HTTPError, ValueError) as error:
            raise StoreFailed from error

    def search(
        self, owner: str, sources: Sequence[str], dense: Sequence[float], sparse: Sparse, limit: int
    ) -> list[Found]:
        """The owner's points of these sources nearest the question, fused by RRF, best first; nothing, and no call,
        without a source."""
        if not sources:
            return []
        allowed = _sources_filter(owner, sources)
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
                with_payload=False,
                with_vectors=False,
            )
        except (ApiException, httpx.HTTPError, ValueError) as error:
            raise StoreFailed from error
        return [Found(str(point.id), float(point.score)) for point in answer.points]

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
