"""The work streams in Redis, through redis-py.

The worker only reads: `XREADGROUP` (its own stream, group `workers`), `XACK` and, at start,
`XAUTOCLAIM`, which are what its Redis account allows (with `AUTH` and `PING`). It never runs
`XADD`, `XTRIM` or `XGROUP`: the backend creates the group and the messages. The client speaks RESP2
with `AUTH` and sends no `CLIENT SETINFO`, so it needs no other command. A message holds one field,
`job`, the job's id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import redis

from gmshield_worker.settings import RedisSettings

GROUP = "workers"
FIELD = "job"
# Longer than the longest blocking read, so a blocked read is never cut short by the socket.
SOCKET_TIMEOUT_SECONDS = 10.0
CONNECT_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class Message:
    """One stream message: its id and the job id it carries (`None` when it has none)."""

    message_id: str
    job_id: str | None


class WorkStream(Protocol):
    """What a lane needs of its stream."""

    @property
    def name(self) -> str: ...

    def read(self, consumer: str, block_ms: int) -> list[Message]:
        """At most one new message for this consumer, waiting up to `block_ms` for one."""
        ...

    def ack(self, message_id: str) -> None: ...

    def take_over(self, consumer: str, min_idle_ms: int) -> list[Message]:
        """Every message pending longer than `min_idle_ms` for any consumer, now this consumer's."""
        ...


def connect(settings: RedisSettings) -> redis.Redis:
    """A client (a pool, safe across threads). Connects on the first command."""
    options: dict[str, Any] = {
        "host": settings.host,
        "port": settings.port,
        "username": settings.username,
        "password": settings.password,
        "decode_responses": True,
        "protocol": 2,
        "driver_info": None,
        "socket_timeout": SOCKET_TIMEOUT_SECONDS,
        "socket_connect_timeout": CONNECT_TIMEOUT_SECONDS,
    }
    if settings.tls:
        options.update(ssl=True, ssl_ca_certs=settings.ca_file, ssl_cert_reqs="required", ssl_check_hostname=True)
    return redis.Redis(**options)


def _messages(entries: object) -> list[Message]:
    found: list[Message] = []
    for entry in cast("list[Any]", entries or []):
        message_id, fields = entry[0], entry[1]
        if message_id is None:
            continue
        job = fields.get(FIELD) if isinstance(fields, dict) else None
        found.append(Message(str(message_id), job if isinstance(job, str) else None))
    return found


@dataclass(frozen=True)
class RedisWorkStream:
    client: redis.Redis
    name: str

    def read(self, consumer: str, block_ms: int) -> list[Message]:
        answer = cast(
            "list[Any] | None", self.client.xreadgroup(GROUP, consumer, {self.name: ">"}, count=1, block=block_ms)
        )
        found: list[Message] = []
        for _stream, entries in answer or []:
            found.extend(_messages(entries))
        return found

    def ack(self, message_id: str) -> None:
        self.client.xack(self.name, GROUP, message_id)

    def take_over(self, consumer: str, min_idle_ms: int) -> list[Message]:
        taken: list[Message] = []
        start = "0-0"
        while True:
            answer: list[Any] = self.client.xautoclaim(
                self.name, GROUP, consumer, min_idle_ms, start_id=start, count=100
            )
            taken.extend(_messages(answer[1]))
            start = str(answer[0])
            if start == "0-0":
                return taken
