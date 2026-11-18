"""The ingestion lane: one ingestion step at a time (2 cores, 11 GiB; Story 1.15 onward). The
backend offers one ingestion job at a time, alternating between owners; this lane runs one."""

from __future__ import annotations

import threading
from collections.abc import Mapping

from gmshield_worker.lanes.consumer import JobRoutes, Lane
from gmshield_worker.lanes.handlers import Handler
from gmshield_worker.lanes.streams import WorkStream

NAME = "ingestion"
STREAM = "gmshield:work:ingestion"
CAPACITY = 1


def lane(
    stream: WorkStream,
    *,
    api: JobRoutes,
    handlers: Mapping[str, Handler],
    consumer: str,
    trace_id: str,
    stop: threading.Event,
) -> Lane:
    return Lane(
        NAME, stream, capacity=CAPACITY, api=api, handlers=handlers, consumer=consumer, trace_id=trace_id, stop=stop
    )
