from __future__ import annotations

import json
import re

import pytest

from gmshield_worker.log import NO_TRACE_ID, TraceLog, log_event, log_line, new_trace_id, parse_traceparent, traceparent


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


def test_a_traceparent_gives_back_its_trace_id_as_the_backend_reads_it() -> None:
    trace_id = new_trace_id()
    assert parse_traceparent(traceparent(trace_id)) == trace_id
    assert parse_traceparent(f" 00-{trace_id}-00f067aa0ba902b7-01 ") == trace_id
    assert parse_traceparent(f"01-{trace_id}-00f067aa0ba902b7-01-extra") == trace_id
    for bad in (
        None,
        42,
        "",
        f"ff-{trace_id}-00f067aa0ba902b7-01",
        f"00-{trace_id}-00f067aa0ba902b7-01-extra",
        f"00-{NO_TRACE_ID}-00f067aa0ba902b7-01",
        f"00-{trace_id}-0000000000000000-01",
        f"00-{trace_id.upper()}-00f067aa0ba902b7-01",
    ):
        assert parse_traceparent(bad) is None, bad


def test_a_trace_log_writes_under_its_trace_id(capsys: pytest.CaptureFixture[str]) -> None:
    trace_id = new_trace_id()
    log = TraceLog("worker.lane", trace_id)
    log.info("job.claimed", jobId="j")
    log.warn("job.abandoned")
    log.error("job.handler_failed")
    captured = capsys.readouterr()
    out = [json.loads(line) for line in captured.out.splitlines()]
    err = [json.loads(line) for line in captured.err.splitlines()]
    assert [(line["level"], line["event"], line["traceId"]) for line in out] == [
        ("INFO", "job.claimed", trace_id),
        ("WARN", "job.abandoned", trace_id),
    ]
    assert err[0]["level"] == "ERROR"
    assert err[0]["traceId"] == trace_id
