"""The recorded model responses (Story 2.10): one file per provider in `tests/fixtures/models/`, and a local server that
replays them over HTTP, so each real chat model is tested on the bytes its provider sends, without a key or a network.

Each file holds cases (`answer`, `refusal`, `outage`, `length`, and for Anthropic `stream_error`). A case is the HTTP
status, the content type and the body (a server-sent event stream, one line per item, or a JSON error), and says where
it came from: `recorded` (by `tools/record_model_responses.py`, against the real provider) or `hand-built` (written from
the provider's documented format). Recorded cases keep no header, no key and no response id: the ids are replaced by
`OPENAI_RESPONSE_ID` and `ANTHROPIC_RESPONSE_ID`.

The prompt is synthetic, with an invented passage (no library text): `SYSTEM_PROMPT` and `USER_PROMPT`, whose passage
id is `PASSAGE_ID`. This module uses the standard library only, so the recording tool can run in the production image.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

FOLDER = Path(__file__).resolve().parent / "fixtures" / "models"
PROVIDER_FILES = {"azure-openai": "azure-openai.json", "anthropic": "anthropic.json"}

PASSAGE_ID = "019a5d3c-0000-7000-8000-00000000c0de"
SYSTEM_PROMPT = (
    "Answer only from the passages in the user turn. After each statement drawn from a passage, write "
    "[[<passage id>]] with that passage's id. Answer in one short sentence."
)
USER_PROMPT = (
    f"Passage {PASSAGE_ID}: In the town of Lowmere, the glow-moss on the old bridge shines a pale blue at night.\n\n"
    "Question: What colour does the glow-moss on the Lowmere bridge shine?"
)
OPENAI_RESPONSE_ID = "chatcmpl-recorded"
ANTHROPIC_RESPONSE_ID = "msg_recorded"
MARKER = re.compile(r"\[\[([A-Za-z0-9-]{1,64})\]\]", re.ASCII)


def path_of(provider: str) -> Path:
    return FOLDER / PROVIDER_FILES[provider]


def load(provider: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path_of(provider).read_text(encoding="utf-8"))
    return loaded


def write(provider: str, fixture: dict[str, Any]) -> None:
    path_of(provider).write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def body_bytes(case: dict[str, Any]) -> bytes:
    """A case's body as sent: the lines of an event stream joined, or the JSON error as it is."""
    body = case["body"]
    text = "\n".join(body) + "\n" if isinstance(body, list) else json.dumps(body)
    return text.encode("utf-8")


def stream_data(case: dict[str, Any]) -> list[dict[str, Any]]:
    """The JSON of every `data:` line of an event stream case (`[DONE]` left out)."""
    found = []
    for line in case["body"]:
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            found.append(json.loads(line[5:]))
    return found


def streamed_text(provider: str, case: dict[str, Any]) -> str:
    """The text the model streamed in a case, markers included, read straight from the provider's format."""
    pieces = []
    for data in stream_data(case):
        if provider == "anthropic":
            delta = data.get("delta") or {}
            if data.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                pieces.append(delta["text"])
        else:
            for choice in data.get("choices") or []:
                pieces.append((choice.get("delta") or {}).get("content") or "")
    return "".join(pieces)


def display_and_citations(text: str) -> tuple[str, list[tuple[str, int]]]:
    """The text without its markers, and each marker's id at its code-point offset in that text."""
    display, citations, last = [], [], 0
    for match in MARKER.finditer(text):
        display.append(text[last : match.start()])
        citations.append((match[1], len("".join(display))))
        last = match.end()
    display.append(text[last:])
    return "".join(display), citations


@dataclass
class Replay:
    """A local server answering every POST with the current case of a provider's fixture."""

    fixture: dict[str, Any]
    case: str = "answer"
    url: str = ""
    requests: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def replay(provider: str) -> Iterator[Replay]:
    state = Replay(load(provider))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            headers = {name.lower(): value for name, value in self.headers.items()}
            state.requests.append({"path": self.path, "headers": headers, "body": json.loads(body or b"{}")})
            case = state.fixture["cases"][state.case]
            answer = body_bytes(case)
            self.send_response(int(case["status"]))
            self.send_header("content-type", case["contentType"])
            self.send_header("content-length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
