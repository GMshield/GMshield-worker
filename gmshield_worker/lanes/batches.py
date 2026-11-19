"""Posting an answer's events in numbered batches (Story 2.4, internal contract 1.11.0): `EventBatcher`.

The handler hands every model event to `emit` as it comes and never waits: a sender thread of the batcher posts them
to `POST .../events` under the claim's grant, numbered from 1 per claim.

- The first text delta goes out at once, alone (with any citation before it, at most `max_events` events), so the GM
  sees the first words as soon as they exist. After it, a batch goes out at most every `interval` (100 ms) or as soon
  as `max_events` (32) events wait, whichever comes first, with at most `max_events` events.
- A call's usage rides in the next batch; at the end, what is left goes out, a usage-only batch included.
- A 5xx or a transport error is retried with the same number, backing off, while the claim lives (`cancel` is set
  once it lapsed, and then the rest is dropped). Any other refusal (4xx) stops the batcher: `refused` names its code.
- A batch the backend accepts counts in `accepted`; its answer's `stop` sets `stop` (Story 2.6: the GM stopped the
  answer). From then on no event is posted, only the usage still to report: events already waiting and any emitted
  later are dropped (the backend would store none of them).

`close` sends what is left and waits for the sender, so the final batch is posted before the outcome. Lines carry ids,
numbers and counts only, never a text.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from gmshield_worker.backend.client import ClaimedJob, EventsAnswer
from gmshield_worker.log import TraceLog
from gmshield_worker.models.output import Citation, ModelEvent, TextDelta, Usage

BATCH_INTERVAL_SECONDS = 0.1
MAX_EVENTS = 32
RETRY_INITIAL_SECONDS = 0.2
RETRY_MAX_SECONDS = 2.0
EVENTS_REFUSED = "worker.events_refused"

type StreamEvent = TextDelta | Citation


class EventPoster(Protocol):
    """Posting a batch under the claim's grant (`backend.client.JobsApi`, or a stand-in in tests)."""

    def events(
        self, job: ClaimedJob, batch: int, events: Sequence[StreamEvent], usage: Sequence[Usage] = ()
    ) -> EventsAnswer: ...


class EventBatcher:
    """The answer events of one claim, posted in numbered batches by a sender thread of its own."""

    def __init__(
        self,
        api: EventPoster,
        job: ClaimedJob,
        cancel: threading.Event,
        log: TraceLog,
        *,
        interval: float = BATCH_INTERVAL_SECONDS,
        max_events: int = MAX_EVENTS,
        retry_initial: float = RETRY_INITIAL_SECONDS,
        retry_max: float = RETRY_MAX_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api = api
        self._job = job
        self._cancel = cancel
        self._log = log
        self._interval = interval
        self._max_events = max_events
        self._retry_initial = retry_initial
        self._retry_max = retry_max
        self._clock = clock
        self._condition = threading.Condition()
        self._events: list[StreamEvent] = []
        self._usage: list[Usage] = []
        self._closing = False
        self._first_sent = False
        self._last_sent = 0.0
        self._number = 0
        self.accepted = 0
        self.posted_events = 0
        self.stop = False
        self.refused: str | None = None
        self._thread = threading.Thread(target=self._run, name=f"events-{job.job_id}", daemon=True)

    # ---- the handler's side

    def start(self) -> None:
        self._thread.start()

    def emit(self, event: ModelEvent) -> None:
        """Takes one model event: a delta or a citation to post, a usage to ride along; a failure is not posted."""
        with self._condition:
            if isinstance(event, TextDelta | Citation):
                if self.stop:
                    return
                self._events.append(event)
            elif isinstance(event, Usage):
                self._usage.append(event)
            else:
                return
            self._condition.notify_all()

    @property
    def delivered(self) -> bool:
        """Whether the backend accepted an event (usage alone shows the GM nothing): from then on the answer must never
        be retried."""
        return self.posted_events > 0

    @property
    def halted(self) -> bool:
        """Whether the answer must stop: the backend said so, or refused a batch."""
        return self.stop or self.refused is not None

    def close(self, timeout: float | None = None) -> None:
        """Sends what is left, then waits for the sender (at most `timeout` seconds)."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout)

    # ---- the sender

    def _due(self) -> tuple[list[StreamEvent], list[Usage]] | None:
        """Under the lock: waits until a batch is due and takes it; `None` once there is nothing more to send."""
        while True:
            if self.stop or self.refused is not None or self._cancel.is_set():
                return self._usage_after_stop()
            if not self._first_sent:
                first = next((index for index, item in enumerate(self._events) if isinstance(item, TextDelta)), None)
                if first is not None:
                    return self._take(min(first + 1, self._max_events), with_usage=False)
            elif len(self._events) >= self._max_events:
                return self._take(self._max_events, with_usage=True)
            elif self._events and self._clock() - self._last_sent >= self._interval:
                return self._take(len(self._events), with_usage=True)
            if self._closing:
                if self._events or self._usage:
                    return self._take(min(len(self._events), self._max_events), with_usage=True)
                return None
            wait = None
            if self._first_sent and self._events:
                wait = max(self._interval - (self._clock() - self._last_sent), 0.001)
            self._condition.wait(wait)

    def _usage_after_stop(self) -> tuple[list[StreamEvent], list[Usage]] | None:
        """Under the lock, once the GM stopped the answer: no event goes out any more, only the usage still to report;
        `None` once there is none and the batcher is closing. After a refusal or a lapsed claim, nothing more goes out
        (`None` at once)."""
        self._events.clear()
        while not self._usage and not self._closing and self.refused is None and not self._cancel.is_set():
            self._condition.wait()
        if self._usage and self.refused is None and not self._cancel.is_set():
            return self._take(0, with_usage=True)
        return None

    def _take(self, count: int, *, with_usage: bool) -> tuple[list[StreamEvent], list[Usage]]:
        events, self._events = self._events[:count], self._events[count:]
        usage: list[Usage] = []
        if with_usage:
            usage, self._usage = self._usage, []
        self._first_sent = self._first_sent or any(isinstance(item, TextDelta) for item in events)
        self._last_sent = self._clock()
        return events, usage

    def _run(self) -> None:
        while True:
            with self._condition:
                due = self._due()
            if due is None:
                return
            events, usage = due
            self._number += 1
            self._post(self._number, events, usage)

    def _post(self, number: int, events: list[StreamEvent], usage: list[Usage]) -> None:
        job_id = str(self._job.job_id)
        delay = self._retry_initial
        while True:
            try:
                answer: EventsAnswer | None = self._api.events(self._job, number, events, usage)
            except Exception as error:  # no answer: logged by type, retried while the claim lives
                self._log.warn("events.unanswered", jobId=job_id, batch=number, errorType=type(error).__name__)
                answer = None
            if answer is not None and answer.status == 200:
                with self._condition:
                    self.accepted += 1
                    if not answer.stop:
                        # A batch answered stop had its events dropped by the backend: they reached no one.
                        self.posted_events += len(events)
                    self.stop = self.stop or answer.stop
                self._log.info("events.posted", jobId=job_id, batch=number, events=len(events), usage=len(usage))
                return
            if answer is not None and answer.status < 500:
                with self._condition:
                    self.refused = answer.code or EVENTS_REFUSED
                    self._condition.notify_all()
                self._log.warn("events.refused", jobId=job_id, batch=number, status=answer.status, code=answer.code)
                return
            if answer is not None:
                self._log.warn("events.unanswered", jobId=job_id, batch=number, status=answer.status)
            if self._cancel.wait(delay):
                self._log.warn("events.abandoned", jobId=job_id, batch=number)
                return
            delay = min(delay * 2, self._retry_max)
