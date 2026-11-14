"""Shared fixtures: a local HTTP stand-in for the backend's internal API, and log capture."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@dataclass
class StandIn:
    """Answers `GET /internal/v1/version` with `version`, or with `status` and a raw `body`."""

    version: str = "1.0.0"
    status: int = 200
    body: bytes | None = None
    requests: list[dict[str, str]] = field(default_factory=list)
    url: str = ""


@pytest.fixture
def stand_in() -> Iterator[StandIn]:
    state = StandIn()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state.requests.append({"path": self.path, **{k.lower(): v for k, v in self.headers.items()}})
            if self.path != "/internal/v1/version":
                self._send(404, "application/problem+json", b'{"code":"request.not_found"}')
            elif state.body is not None:
                self._send(state.status, "application/json", state.body)
            else:
                self._send(state.status, "application/json", json.dumps({"version": state.version}).encode())

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def log_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]
