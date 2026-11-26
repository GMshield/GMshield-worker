"""The retrieval fixtures (Story 2.3): SRD 5.2.1 passages (CC-BY-4.0, see `fixtures/retrieval/ATTRIBUTION.txt`) and
Portuguese passages written for the tests, the fixed questions, the library searches' queries (Story 3.9), and the real
embeddinggemma vectors of all of them, recorded
once through the Ollama tunnel by `tools/record_embeddings.py` and committed.

`RecordedOllama` answers `/api/embed` from those vectors only: an input that was not recorded (a changed passage,
question or prompt) fails the test, so the retrieval tests can never drift from what the real model said.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from gmshield_worker.retrieval import embedding

FOLDER = Path(__file__).resolve().parent / "fixtures" / "retrieval"
PASSAGES = FOLDER / "passages.json"
QUESTIONS = FOLDER / "questions.json"
VECTORS = FOLDER / "vectors.json"


@dataclass(frozen=True)
class FixturePassage:
    passage_id: str
    key: str
    language: str
    srd: bool
    page: int
    section_path: list[str]
    title: str
    text: str


@dataclass(frozen=True)
class FixtureQuestion:
    key: str
    case: str
    language: str
    question: str
    expected: str


@dataclass(frozen=True)
class FixtureSearch:
    """A library search's query (Story 3.9) and the passages whose text holds it verbatim, in fixture order."""

    key: str
    query: str
    expected: list[str]


@dataclass(frozen=True)
class FixtureTermPair:
    """A fixed pair of Story 4.6: a Portuguese question naming a term the GM approved (`term`, with the English record's
    name merged in as one of its `aliases`), the record type keys the backend resolves it to, and the English SRD
    passage whose record the answer must find."""

    key: str
    question: str
    term: str
    aliases: list[str]
    record_types: list[str]
    expected: str


def passages() -> list[FixturePassage]:
    raw: dict[str, Any] = json.loads(PASSAGES.read_text(encoding="utf-8"))
    return [
        FixturePassage(
            passage_id=item["id"],
            key=item["key"],
            language=item["language"],
            srd=item["srd"],
            page=item["page"],
            section_path=list(item["sectionPath"]),
            title=item["title"],
            text=item["text"],
        )
        for item in raw["passages"]
    ]


def questions() -> list[FixtureQuestion]:
    raw: dict[str, Any] = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    return [FixtureQuestion(**item) for item in raw["questions"]]


def searches() -> list[FixtureSearch]:
    raw: dict[str, Any] = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    return [FixtureSearch(item["key"], item["query"], list(item["expected"])) for item in raw["searches"]]


def term_pairs() -> list[FixtureTermPair]:
    raw: dict[str, Any] = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    return [
        FixtureTermPair(
            key=item["key"],
            question=item["question"],
            term=item["term"],
            aliases=list(item["aliases"]),
            record_types=list(item["recordTypes"]),
            expected=item["expected"],
        )
        for item in raw["termPairs"]
    ]


def inputs() -> list[str]:
    """Every embedding input of the fixtures, in order: each passage's document prompt, then each question's query
    prompt, then each library search's query prompt (Story 3.9), then each term pair's question (Story 4.6), exactly as
    the worker sends them."""
    return (
        [embedding.embedding_input(item.title, item.text) for item in passages()]
        + [embedding.query_input(item.question) for item in questions()]
        + [embedding.query_input(item.query) for item in searches()]
        + [embedding.query_input(item.question) for item in term_pairs()]
    )


def recorded() -> dict[str, list[float]]:
    """The recorded vectors by embedding input."""
    raw: dict[str, Any] = json.loads(VECTORS.read_text(encoding="utf-8"))
    return {item["input"]: [float(value) for value in item["vector"]] for item in raw["inputs"]}


def write_recorded(vectors: list[tuple[str, list[float]]]) -> None:
    """Writes the vectors, one input per line, each value to 7 decimals (well within the 1e-4 check)."""
    lines = [
        json.dumps({"input": text, "vector": [round(value, 7) for value in vector]}, ensure_ascii=False)
        for text, vector in vectors
    ]
    body = ",\n    ".join(lines)
    VECTORS.write_text(
        f'{{\n  "model": {json.dumps(embedding.MODEL)},\n  "inputs": [\n    {body}\n  ]\n}}\n', encoding="utf-8"
    )


@dataclass
class RecordedOllama:
    """`/api/embed` answered from the recorded vectors; an unrecorded input, another model or a truncating request
    fails the test. `status` forces another answer (Ollama down)."""

    vectors: dict[str, list[float]] = field(default_factory=recorded)
    status: int | None = None
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.status is not None:
            return httpx.Response(self.status, json={"error": "model unavailable"})
        body = json.loads(request.content)
        assert request.url.path == "/api/embed"
        assert body["model"] == embedding.MODEL
        assert body["truncate"] is False
        missing = [index for index, text in enumerate(body["input"]) if text not in self.vectors]
        assert not missing, f"unrecorded embedding input(s) at {missing}: run tools/record_embeddings.py"
        self.calls.append(list(body["input"]))
        return httpx.Response(
            200, json={"model": body["model"], "embeddings": [self.vectors[t] for t in body["input"]]}
        )

    def embedder(self) -> embedding.Embedder:
        return embedding.Embedder("http://stand-in-ollama:11434", transport=httpx.MockTransport(self))
