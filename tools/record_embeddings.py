"""Record the real embeddinggemma vectors of the retrieval fixtures (Story 2.3).

The retrieval tests run without Ollama, from vectors the real model gave once for every fixture input
(`tests/fixtures/retrieval/passages.json` and `questions.json`, embedded as the worker embeds them: each passage under
the document prompt, each question under the query prompt). Re-record after changing a fixture, the prompts or the
model; the tests fail on any input that was not recorded.

Run from the repository root with the Ollama tunnel up (`ssh roberto-server` forwards it to localhost:11434):

    uv run python tools/record_embeddings.py          # (re)write tests/fixtures/retrieval/vectors.json
    uv run python tools/record_embeddings.py --check  # the recorded vectors match the live model within 1e-4

Ollama is `GMSHIELD_OLLAMA_URL`, else `http://localhost:11434`. The vectors are stored as Ollama answers them, each
value rounded to 7 decimals (well within the 1e-4 check).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gmshield_worker.retrieval import embedding  # noqa: E402
from tests import retrieval_fixtures  # noqa: E402

DEFAULT_URL = "http://localhost:11434"
TOLERANCE = 1e-4


def live(url: str, inputs: list[str]) -> list[list[float]]:
    """The model's vectors of the inputs, as Ollama answers them, `embedding.BATCH` at a time."""
    found: list[list[float]] = []
    with httpx.Client(base_url=url, timeout=embedding.TIMEOUT, trust_env=False, follow_redirects=False) as client:
        for start in range(0, len(inputs), embedding.BATCH):
            batch = inputs[start : start + embedding.BATCH]
            response = client.post("/api/embed", json={"model": embedding.MODEL, "input": batch, "truncate": False})
            response.raise_for_status()
            vectors = response.json()["embeddings"]
            if len(vectors) != len(batch) or any(len(vector) != embedding.DIMENSIONS for vector in vectors):
                raise ValueError("Ollama's answer is not one 768-dimension vector per input")
            found.extend([float(value) for value in vector] for vector in vectors)
    return found


def check(url: str) -> int:
    inputs = retrieval_fixtures.inputs()
    recorded = retrieval_fixtures.recorded()
    if list(recorded) != inputs:
        print("The recorded inputs differ from the fixtures: re-record.", file=sys.stderr)
        return 1
    worst = 0.0
    for text, vector in zip(inputs, live(url, inputs), strict=True):
        worst = max([worst, *(abs(a - b) for a, b in zip(recorded[text], vector, strict=True))])
    print(f"{len(inputs)} inputs, largest difference {worst:.2e} (tolerance {TOLERANCE:.0e})")
    return 0 if worst <= TOLERANCE else 1


def record(url: str) -> int:
    inputs = retrieval_fixtures.inputs()
    retrieval_fixtures.write_recorded(list(zip(inputs, live(url, inputs), strict=True)))
    print(f"{len(inputs)} inputs recorded in {retrieval_fixtures.VECTORS.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the real embeddinggemma vectors of the retrieval fixtures.")
    parser.add_argument("--check", action="store_true", help="compare the recorded vectors with the live model")
    arguments = parser.parse_args()
    url = os.environ.get("GMSHIELD_OLLAMA_URL", DEFAULT_URL)
    return check(url) if arguments.check else record(url)


if __name__ == "__main__":
    sys.exit(main())
