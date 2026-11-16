"""One JSON object per line: `timestamp`, `level`, `traceId`, `unit` and `event`, then the event's
own fields. ERROR lines go to standard error, every other line to standard output.

Fields are ids, numbers, versions, status codes, setting names and error types: never content,
never a secret, never a setting's value, never a URL with credentials. W3C trace context helpers
live here too: every backend request carries a `traceparent` with this trace id and a new span id.

A job has a trace id of its own, the one of the request that created it (the claim's `traceparent`):
`TraceLog` writes every line about that job under it, so the backend's and the worker's lines about
one job carry one trace id.
"""

from __future__ import annotations

import json
import re
import secrets
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

type Level = Literal["DEBUG", "INFO", "WARN", "ERROR"]
type Field = str | int | float | bool | list[str] | None

NO_TRACE_ID = "0" * 32
_NO_SPAN_ID = "0" * 16
_FIXED = frozenset({"timestamp", "level", "traceId", "unit", "event"})


def _random_hex(size: int, invalid: str) -> str:
    while True:
        value = secrets.token_hex(size)
        if value != invalid:
            return value


def new_trace_id() -> str:
    """A new, valid W3C trace id (32 lowercase hex digits, not all zero)."""
    return _random_hex(16, NO_TRACE_ID)


def traceparent(trace_id: str) -> str:
    """A `traceparent` for one outgoing request of this trace: the same trace id, a new span id."""
    return f"00-{trace_id}-{_random_hex(8, _NO_SPAN_ID)}-01"


_TRACEPARENT = re.compile(r"([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})(-.*)?", re.ASCII)


def parse_traceparent(value: object) -> str | None:
    """The trace id of a valid W3C `traceparent`, or `None` (as the backend reads it)."""
    if not isinstance(value, str):
        return None
    match = _TRACEPARENT.fullmatch(value.strip())
    if match is None:
        return None
    version, trace_id, parent_id, _flags, extra = match.groups()
    if (
        version == "ff"
        or (version == "00" and extra is not None)
        or trace_id == NO_TRACE_ID
        or parent_id == _NO_SPAN_ID
    ):
        return None
    return trace_id


def log_line(level: Level, unit: str, event: str, trace_id: str, /, **fields: Field) -> dict[str, Field]:
    line: dict[str, Field] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": level,
        "traceId": trace_id,
        "unit": unit,
        "event": event,
    }
    for name, value in fields.items():
        # An event field never replaces a fixed one.
        if name not in _FIXED:
            line[name] = value
    return line


def log_event(level: Level, unit: str, event: str, trace_id: str, /, **fields: Field) -> None:
    stream = sys.stderr if level == "ERROR" else sys.stdout
    stream.write(json.dumps(log_line(level, unit, event, trace_id, **fields), separators=(",", ":")) + "\n")
    stream.flush()


@dataclass(frozen=True)
class TraceLog:
    """Writes the lines of one unit under one trace id: a lane's own, or a job's."""

    unit: str
    trace_id: str

    def info(self, event: str, /, **fields: Field) -> None:
        log_event("INFO", self.unit, event, self.trace_id, **fields)

    def warn(self, event: str, /, **fields: Field) -> None:
        log_event("WARN", self.unit, event, self.trace_id, **fields)

    def error(self, event: str, /, **fields: Field) -> None:
        log_event("ERROR", self.unit, event, self.trace_id, **fields)
