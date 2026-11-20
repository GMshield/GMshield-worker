"""Work lanes (Story 1.12): the consumers of the interactive and ingestion Redis streams.

Each lane reads only its own stream (`XREADGROUP`, group `workers`, one consumer name per
process), claims each job over the internal API before working, renews the claim in the
background and reports the outcome. The interactive lane has capacity of its own
(`GMSHIELD_INTERACTIVE_CAPACITY`, default 2) and starts first, so an interactive job is claimed at
once even while the ingestion lane (one job at a time) is busy. A message is acknowledged only after
a definite answer to its claim (2xx or 409); the backend alone decides retries.

`build_lanes` wires both lanes to Redis and the backend; `consumer.py` holds the shared loop,
`renewal.py` the renewal timer, `handlers.py` the handlers by kind and `streams.py` Redis.
"""

from __future__ import annotations

import os
import secrets
import socket
import threading
from dataclasses import dataclass, field

from gmshield_worker.backend.client import JobsApi
from gmshield_worker.lanes import ingestion, interactive
from gmshield_worker.lanes.consumer import Lane
from gmshield_worker.lanes.handlers import handlers_for
from gmshield_worker.lanes.streams import RedisWorkStream, connect
from gmshield_worker.settings import Settings


@dataclass
class Lanes:
    """Both lanes: started interactive first, stopped together."""

    lanes: list[Lane]
    stop_event: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        for lane in self.lanes:
            lane.start()

    def stop(self) -> None:
        self.stop_event.set()
        for lane in self.lanes:
            lane.stop()


def consumer_name() -> str:
    """One consumer name per process: the host (the container), the process and a random part."""
    return f"{socket.gethostname()[:40]}-{os.getpid()}-{secrets.token_hex(4)}"


def build_lanes(settings: Settings, trace_id: str) -> Lanes:
    """Both lanes on one Redis client and the backend's job routes. Connects nothing until started."""
    redis = connect(settings.redis)
    api = JobsApi(settings.backend_url, settings.worker_credential, providers=settings.models.providers)
    handlers = handlers_for(settings)
    consumer = consumer_name()
    stop = threading.Event()
    return Lanes(
        lanes=[
            interactive.lane(
                RedisWorkStream(redis, interactive.STREAM),
                capacity=settings.interactive_capacity,
                api=api,
                handlers=handlers,
                consumer=consumer,
                trace_id=trace_id,
                stop=stop,
            ),
            ingestion.lane(
                RedisWorkStream(redis, ingestion.STREAM),
                api=api,
                handlers=handlers,
                consumer=consumer,
                trace_id=trace_id,
                stop=stop,
            ),
        ],
        stop_event=stop,
    )
