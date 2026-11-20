"""Record the real Azure OpenAI streams of the model fixtures (Story 2.10).

The model tests replay each provider's HTTP stream from `tests/fixtures/models/` (`tests/model_fixtures.py`). This tool
records Azure OpenAI's `answer` and `length` cases from the real deployment: it sends the synthetic prompt of
`model_fixtures` (an invented passage, never library text) to the Chat Completions API of the v1 endpoint, exactly as
the worker does (no `api-version`, `store: false`, usage streamed), and keeps the body only. Before anything is written
it drops every header, replaces each response id with `model_fixtures.OPENAI_RESPONSE_ID`, and refuses to write a body
that holds the key or the resource's host. The `refusal`, `refusal_delta` and `outage` cases stay hand-built: a
synthetic prompt causes none of them. Anthropic's cases are hand-built from its documented format (no Anthropic key
exists).

The key never leaves the server and is never printed. Run it on roberto-server, inside the worker image, with the
worker's secrets (the paths are the ones Dalton's deploy uses; the folder holds a copy of `tools/` and `tests/`):

    docker run --rm --env-file ~/gmshield-ci/secrets/worker.env --user "$(id -u):$(id -g)" \\
        -v ~/gm2-10-record:/record -w /record <worker image> python tools/record_model_responses.py
    ... python tools/record_model_responses.py --check   # today's live streams have the recorded shape

then copy only `tests/fixtures/models/azure-openai.json` back. It reads `GMSHIELD_AZURE_OPENAI_API_KEY` and
`GMSHIELD_AZURE_OPENAI_BASE_URL` (`https://<resource>.openai.azure.com/openai/v1/`) and nothing else. It needs only
the standard library and httpx, so the production image runs it.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import model_fixtures  # noqa: E402

PROVIDER = "azure-openai"
KEY = "GMSHIELD_AZURE_OPENAI_API_KEY"
BASE_URL = "GMSHIELD_AZURE_OPENAI_BASE_URL"
AZURE_PATH = "/openai/v1/"
DEFAULT_MODEL = "gpt-6.1-terra"
TIMEOUT_SECONDS = 120.0
ATTEMPTS = 3
# The answer has room to finish; the length case is cut at once.
MAX_TOKENS = {"answer": 2048, "length": 16}


class RecordingFailed(Exception):
    """The provider's answer cannot become a fixture. The message names no key, host or prompt."""


def settings() -> tuple[str, str]:
    """The key and the v1 base URL, or `RecordingFailed` naming the variable."""
    key = (os.environ.get(KEY) or "").strip()
    url = (os.environ.get(BASE_URL) or "").strip()
    if not key:
        raise RecordingFailed(f"{KEY} is not set")
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.path.rstrip("/") + "/" != AZURE_PATH or parts.query:
        raise RecordingFailed(f"{BASE_URL} is not https://<resource>.openai.azure.com/openai/v1/")
    return key, f"https://{parts.netloc}{AZURE_PATH}"


def request_body(model: str, max_tokens: int) -> dict[str, Any]:
    """The worker's Chat Completions request (`models.chat`): streamed, usage included, storage off, no tool."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": model_fixtures.SYSTEM_PROMPT},
            {"role": "user", "content": model_fixtures.USER_PROMPT},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "store": False,
        "max_completion_tokens": max_tokens,
    }


def clean(lines: list[str], secrets: list[str]) -> list[str]:
    """The event stream without comments, with each response id replaced; refused when a secret is in it."""
    kept: list[str] = []
    for line in lines:
        if line.startswith(":"):
            continue
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                data = json.loads(line[5:])
            except ValueError as error:
                raise RecordingFailed("a stream line is not JSON") from error
            if isinstance(data, dict) and data.get("id"):
                data["id"] = model_fixtures.OPENAI_RESPONSE_ID
            kept.append("data: " + json.dumps(data, separators=(",", ":"), ensure_ascii=False))
        else:
            kept.append(line)
    text = "\n".join(kept)
    if any(secret and secret in text for secret in secrets):
        raise RecordingFailed("the stream holds the key or the resource's host: nothing written")
    return kept


def live_case(client: httpx.Client, key: str, base_url: str, model: str, case: str) -> dict[str, Any]:
    """One live call of a case, as a fixture case."""
    host = urlsplit(base_url).hostname or ""
    for _ in range(ATTEMPTS):
        response = client.post(
            base_url + "chat/completions",
            headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
            json=request_body(model, MAX_TOKENS[case]),
        )
        if response.status_code != 200:
            raise RecordingFailed(f"{case}: the provider answered {response.status_code}")
        found = {
            "source": "recorded",
            "recorded": datetime.datetime.now(datetime.UTC).date().isoformat(),
            "status": 200,
            "contentType": response.headers.get("content-type", "text/event-stream"),
            "body": clean(response.text.splitlines(), [key, host]),
        }
        finishes = {
            choice.get("finish_reason")
            for data in model_fixtures.stream_data(found)
            for choice in data.get("choices") or []
        }
        text = model_fixtures.streamed_text(PROVIDER, found)
        if case == "length" and "length" in finishes:
            return found
        if case == "answer" and "stop" in finishes and f"[[{model_fixtures.PASSAGE_ID}]]" in text:
            return found
    raise RecordingFailed(f"{case}: {ATTEMPTS} answers without the expected stop or citation marker")


def shape(case: dict[str, Any]) -> dict[str, Any]:
    """What a replay depends on: the keys of each kind of chunk, the stop reasons and the usage fields."""
    chunks: set[str] = set()
    finishes: set[str] = set()
    usage: set[str] = set()
    for data in model_fixtures.stream_data(case):
        chunks.add(",".join(sorted(data)))
        for choice in data.get("choices") or []:
            if choice.get("finish_reason"):
                finishes.add(choice["finish_reason"])
        if data.get("usage"):
            usage |= set(data["usage"])
    return {"chunks": sorted(chunks), "finishes": sorted(finishes), "usage": sorted(usage)}


def run(check: bool, model: str) -> int:
    key, base_url = settings()
    fixture = model_fixtures.load(PROVIDER)
    with httpx.Client(timeout=TIMEOUT_SECONDS, trust_env=False, follow_redirects=False) as client:
        live = {case: live_case(client, key, base_url, model, case) for case in MAX_TOKENS}
    if check:
        differing = [case for case in live if shape(live[case]) != shape(fixture["cases"][case])]
        for case in live:
            print(f"{case}: {'differs from' if case in differing else 'matches'} the recorded shape")
        return 1 if differing else 0
    fixture["model"] = model
    fixture["note"] = (
        "Chat Completions streams of the Azure OpenAI v1 API. answer and length were recorded by"
        " tools/record_model_responses.py on roberto-server (headers dropped, response ids replaced); refusal,"
        " refusal_delta (a Chat Completions refusal delta) and outage are hand-built from the documented format (a"
        " synthetic prompt causes none of them)."
    )
    fixture["cases"].update(live)
    model_fixtures.write(PROVIDER, fixture)
    for case, found in live.items():
        print(f"{case}: recorded, {len(model_fixtures.stream_data(found))} chunks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the real Azure OpenAI streams of the model fixtures.")
    parser.add_argument("--check", action="store_true", help="compare today's live streams with the recorded shape")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the Azure deployment name (default gpt-6.1-terra)")
    arguments = parser.parse_args()
    try:
        return run(arguments.check, arguments.model)
    except RecordingFailed as failed:
        print(f"record_model_responses: {failed}", file=sys.stderr)
        return 1
    except httpx.HTTPError as error:
        print(f"record_model_responses: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
