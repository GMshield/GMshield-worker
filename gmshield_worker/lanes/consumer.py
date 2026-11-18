"""The consumer loop that both lanes share.

One reader thread per lane takes one message at a time, and only when the lane has a free slot
(its capacity), so a busy lane leaves its messages in the stream for another worker. For each
message it claims the job; a definite answer (2xx or 409) acknowledges the message, anything else
(no answer, a 5xx) leaves it pending for a takeover. A claimed job runs in the lane's pool: the
handler works while the renewal timer keeps the claim alive, then the outcome is reported, retried
while the claim lives. Every line about a job is written under the job's own trace id.

At start, the lane takes over the messages that other consumers left pending for more than two
minutes (`XAUTOCLAIM`): a worker that died between reading and claiming.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from uuid import UUID

from gmshield_worker.backend.client import Answer, ClaimAnswer, ClaimedJob, Outcome, RenewalAnswer
from gmshield_worker.lanes.handlers import Handler
from gmshield_worker.lanes.renewal import Renewal
from gmshield_worker.lanes.streams import Message, WorkStream
from gmshield_worker.log import TraceLog

UNIT = "worker.lane"
# These timings are the worker's own and need not follow the backend's GMSHIELD_JOB_* settings. The
# takeover, at start only, rescues a message a dead worker read but never claimed; the backend re-offers
# an unclaimed job after GMSHIELD_JOB_REOFFER_AFTER regardless, and a second message for a claimed job
# gets 409, which is acknowledged. The outcome's retries already end when the claim lapses
# (GMSHIELD_JOB_CLAIM_LAPSE, through the renewal's `cancel`); REPORT_GIVE_UP_SECONDS only bounds them.
TAKE_OVER_AFTER_MS = 120_000
READ_BLOCK_MS = 2_000
BACKOFF_INITIAL_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
REPORT_RETRY_MAX_SECONDS = 5.0
# An outcome still unanswered after this long is given up: the renewal stops, the claim lapses and the
# backend decides (an outcome that always fails, for example a handler of the backend that throws, would
# otherwise hold the job and the lane's slot for ever).
REPORT_GIVE_UP_SECONDS = 120.0


class JobRoutes(Protocol):
    """The internal API's job routes (`backend.client.JobsApi`, or a stand-in in tests)."""

    def claim(self, job_id: UUID, trace_id: str) -> ClaimAnswer: ...

    def renew(self, job: ClaimedJob) -> RenewalAnswer: ...

    def report(self, job: ClaimedJob, outcome: Outcome) -> Answer: ...


class Lane:
    def __init__(
        self,
        name: str,
        stream: WorkStream,
        *,
        capacity: int,
        api: JobRoutes,
        handlers: Mapping[str, Handler],
        consumer: str,
        trace_id: str,
        stop: threading.Event,
        read_block_ms: int = READ_BLOCK_MS,
        take_over_after_ms: int = TAKE_OVER_AFTER_MS,
        backoff_initial: float = BACKOFF_INITIAL_SECONDS,
        renewal_retry_seconds: float = 2.0,
        report_give_up_seconds: float = REPORT_GIVE_UP_SECONDS,
        renewal_factory: Callable[..., Renewal] = Renewal,
    ) -> None:
        if capacity < 1:
            raise ValueError("a lane needs a capacity of at least 1")
        self.name = name
        self.stream = stream
        self.capacity = capacity
        self.api = api
        self.handlers = handlers
        self.consumer = consumer
        self.stop_event = stop
        self.log = TraceLog(UNIT, trace_id)
        self._trace_id = trace_id
        self._read_block_ms = read_block_ms
        self._take_over_after_ms = take_over_after_ms
        self._backoff_initial = backoff_initial
        self._renewal_retry = renewal_retry_seconds
        self._report_give_up = report_give_up_seconds
        self._renewal_factory = renewal_factory
        self._slots = threading.BoundedSemaphore(capacity)
        self._pool = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix=f"lane-{name}")
        self._active: dict[UUID, threading.Event] = {}
        self._active_lock = threading.Lock()
        self._thread = threading.Thread(target=self.run, name=f"lane-{name}-reader", daemon=True)

    # ---- lifecycle

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stops reading, stops the running jobs (unreported: the backend takes them back), waits."""
        self.stop_event.set()
        with self._active_lock:
            for cancel in self._active.values():
                cancel.set()
        if self._thread.is_alive():
            self._thread.join(timeout=15)
        self._pool.shutdown(wait=True, cancel_futures=True)

    def run(self) -> None:
        self.log.info("lane.started", lane=self.name, capacity=self.capacity, stream=self.stream.name)
        failures = 0
        taken_over = False
        while not self.stop_event.is_set():
            if not taken_over:
                try:
                    pending = self.stream.take_over(self.consumer, self._take_over_after_ms)
                except Exception as error:
                    failures += 1
                    self._stream_failed(error, failures)
                    continue
                taken_over = True
                failures = 0
                if pending:
                    self.log.info("lane.took_over", lane=self.name, messages=len(pending))
                for message in pending:
                    if not self._acquire():
                        break
                    self._dispatch(message)
                continue
            if not self._acquire():
                continue
            try:
                messages = self.stream.read(self.consumer, self._read_block_ms)
            except Exception as error:
                self._slots.release()
                failures += 1
                self._stream_failed(error, failures)
                continue
            failures = 0
            if not messages:
                self._slots.release()
                continue
            self._dispatch(messages[0])
        self.log.info("lane.stopped", lane=self.name)

    def _acquire(self) -> bool:
        """Waits for a free slot; false when the lane is stopping."""
        while not self.stop_event.is_set():
            if self._slots.acquire(timeout=0.2):
                return True
        return False

    def _stream_failed(self, error: Exception, failures: int) -> None:
        # Redis down or restarting, or the group not created yet (NOGROUP): wait, then try again.
        self.log.warn("stream.unavailable", lane=self.name, attempt=failures, errorType=type(error).__name__)
        delay = min(BACKOFF_MAX_SECONDS, self._backoff_initial * 2.0 ** min(failures - 1, 10))
        self.stop_event.wait(delay)

    def _dispatch(self, message: Message) -> None:
        try:
            self._pool.submit(self._handle_and_release, message)
        except RuntimeError:  # the pool is shutting down
            self._slots.release()

    def _handle_and_release(self, message: Message) -> None:
        try:
            self.handle(message)
        except Exception as error:  # never lose the slot; logged by type only
            self.log.error("lane.failed", lane=self.name, errorType=type(error).__name__)
        finally:
            self._slots.release()

    # ---- one message

    def handle(self, message: Message) -> None:
        """Claims the message's job, acknowledges on a definite answer, and runs what was claimed."""
        try:
            job_id = UUID(message.job_id) if message.job_id else None
        except ValueError:
            job_id = None
        if job_id is None:
            self.log.warn("message.invalid", lane=self.name, messageId=message.message_id)
            return
        try:
            answer = self.api.claim(job_id, self._trace_id)
        except Exception as error:  # no answer: the message stays pending
            self.log.warn("claim.unanswered", lane=self.name, jobId=str(job_id), errorType=type(error).__name__)
            return
        if answer.definite:
            try:
                self.stream.ack(message.message_id)
            except Exception as error:
                self.log.warn("message.ack_failed", lane=self.name, jobId=str(job_id), errorType=type(error).__name__)
        if answer.job is None:
            event = "claim.refused" if answer.definite else "claim.unanswered"
            self.log.info(event, lane=self.name, jobId=str(job_id), status=answer.status, code=answer.code)
            return
        self.run_job(answer.job)

    def run_job(self, job: ClaimedJob) -> None:
        log = TraceLog(UNIT, job.trace_id)
        job_id = str(job.job_id)
        log.info("job.claimed", lane=self.name, jobId=job_id, attempt=job.attempt, kind=job.kind)
        cancel = threading.Event()
        with self._active_lock:
            self._active[job.job_id] = cancel
            if self.stop_event.is_set():
                cancel.set()
        renewal = self._renewal_factory(self.api.renew, job, cancel, log, retry_seconds=self._renewal_retry)
        renewal.start()
        try:
            outcome = self._work(job, cancel, log)
            if cancel.is_set():
                log.warn("job.abandoned", lane=self.name, jobId=job_id, reason=renewal.reason or "worker_stopping")
                return
            self._report(job, outcome, cancel, log)
        finally:
            renewal.stop()
            with self._active_lock:
                self._active.pop(job.job_id, None)

    def _work(self, job: ClaimedJob, cancel: threading.Event, log: TraceLog) -> Outcome:
        handler = self.handlers.get(job.kind)
        if handler is None:
            log.warn("job.unknown_kind", lane=self.name, jobId=str(job.job_id), kind=job.kind)
            return Outcome.failed("worker.unknown_kind")
        try:
            return handler(job, cancel)
        except Exception as error:  # the handler failed: reported as a failed attempt, by type only
            log.error("job.handler_failed", lane=self.name, jobId=str(job.job_id), errorType=type(error).__name__)
            return Outcome.failed("worker.handler_failed")

    def _report(self, job: ClaimedJob, outcome: Outcome, cancel: threading.Event, log: TraceLog) -> None:
        job_id = str(job.job_id)
        delay = 0.5
        give_up_at = time.monotonic() + self._report_give_up
        while True:
            try:
                answer = self.api.report(job, outcome)
            except Exception as error:
                log.warn("outcome.unanswered", lane=self.name, jobId=job_id, errorType=type(error).__name__)
            else:
                if 200 <= answer.status < 300:
                    log.info(
                        "job.reported",
                        lane=self.name,
                        jobId=job_id,
                        attempt=job.attempt,
                        outcome=outcome.outcome,
                        code=outcome.code,
                    )
                    return
                if answer.status < 500:
                    log.warn("outcome.refused", lane=self.name, jobId=job_id, status=answer.status, code=answer.code)
                    return
                log.warn("outcome.unanswered", lane=self.name, jobId=job_id, status=answer.status)
            # Retried while the claim lives (the renewal timer sets `cancel` once it lapsed), for a bounded time.
            remaining = give_up_at - time.monotonic()
            if remaining <= 0 or cancel.wait(min(delay, remaining)):
                log.warn("job.abandoned", lane=self.name, jobId=job_id, reason="outcome_unanswered")
                return
            delay = min(delay * 2, REPORT_RETRY_MAX_SECONDS)
