"""The interactive lane: questions and other work a person waits for (Epic 2 onward). It has
capacity of its own (`GMSHIELD_INTERACTIVE_CAPACITY`, default 2) and starts first, so it is served
first and never waits behind ingestion."""

from __future__ import annotations

import threading
from collections.abc import Mapping

from gmshield_worker.lanes.consumer import JobRoutes, Lane
from gmshield_worker.lanes.handlers import Handler
from gmshield_worker.lanes.streams import WorkStream

NAME = "interactive"
STREAM = "gmshield:work:interactive"


def lane(
    stream: WorkStream,
    *,
    capacity: int,
    api: JobRoutes,
    handlers: Mapping[str, Handler],
    consumer: str,
    trace_id: str,
    stop: threading.Event,
) -> Lane:
    return Lane(
        NAME, stream, capacity=capacity, api=api, handlers=handlers, consumer=consumer, trace_id=trace_id, stop=stop
    )
