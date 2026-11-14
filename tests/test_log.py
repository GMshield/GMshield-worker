from __future__ import annotations

import json
import re

import pytest

from gmshield_worker.log import NO_TRACE_ID, log_event, log_line, new_trace_id, traceparent


def test_trace_ids_are_valid_and_new() -> None:
    first, second = new_trace_id(), new_trace_id()
    assert re.fullmatch(r"[0-9a-f]{32}", first)
    assert first not in (second, NO_TRACE_ID)


def test_traceparent_keeps_the_trace_and_changes_the_span() -> None:
    trace_id = new_trace_id()
    first, second = traceparent(trace_id), traceparent(trace_id)
    assert re.fullmatch(rf"00-{trace_id}-[0-9a-f]{{16}}-01", first)
    assert first != second


def test_fields_never_replace_fixed_ones() -> None:
    line = log_line("INFO", "unit", "event.name", "a" * 32, event="other", traceId="b", count=3)
    assert line["event"] == "event.name"
    assert line["traceId"] == "a" * 32
    assert line["count"] == 3


def test_one_json_line_per_event(capsys: pytest.CaptureFixture[str]) -> None:
    log_event("INFO", "unit", "one", "a" * 32)
    log_event("ERROR", "unit", "two", "a" * 32)
    out, err = capsys.readouterr()
    assert json.loads(out)["event"] == "one"
    assert json.loads(err)["event"] == "two"
