"""The renewal timer of one claim: renews at the claim's `renewAt`, and again at each new one.

A renewal the backend refuses (any 4xx: the grant is no longer the live claim's) stops the work at
once. A renewal that gets no answer is retried every few seconds until the claim's `lapsesAt`; past
it the work stops too, because the backend has taken the job back. Either way `cancel` is set and
`reason` says why.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from gmshield_worker.backend.client import ClaimedJob, RenewalAnswer
from gmshield_worker.log import TraceLog

RETRY_SECONDS = 2.0

type Renew = Callable[[ClaimedJob], RenewalAnswer]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Renewal:
    def __init__(
        self,
        renew: Renew,
        job: ClaimedJob,
        cancel: threading.Event,
        log: TraceLog,
        *,
        now: Callable[[], datetime] = _utc_now,
        retry_seconds: float = RETRY_SECONDS,
    ) -> None:
        self._renew = renew
        self._job = job
        self._cancel = cancel
        self._log = log
        self._now = now
        self._retry = timedelta(seconds=retry_seconds)
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"renewal-{job.job_id}", daemon=True)
        self.reason: str | None = None
        self.renewals = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)

    def _give_up(self, reason: str) -> None:
        self.reason = reason
        self._cancel.set()

    def _run(self) -> None:
        renew_at, lapses_at = self._job.renew_at, self._job.lapses_at
        while True:
            wait = max(0.0, (renew_at - self._now()).total_seconds())
            if self._stopped.wait(wait) or self._cancel.is_set():
                return
            try:
                answer: RenewalAnswer | None = self._renew(self._job)
            except Exception as error:  # no answer: logged by type, retried until the lapse
                self._log.warn("renewal.unanswered", jobId=str(self._job.job_id), errorType=type(error).__name__)
                answer = None
            if answer is not None and answer.status == 200 and answer.renew_at and answer.lapses_at:
                renew_at, lapses_at = answer.renew_at, answer.lapses_at
                self.renewals += 1
                self._log.info("job.renewed", jobId=str(self._job.job_id), attempt=self._job.attempt)
                continue
            if answer is not None and 400 <= answer.status < 500:
                self._log.warn("renewal.refused", jobId=str(self._job.job_id), status=answer.status, code=answer.code)
                self._give_up("renewal_refused")
                return
            if answer is not None:
                self._log.warn("renewal.unanswered", jobId=str(self._job.job_id), status=answer.status)
            now = self._now()
            if now >= lapses_at:
                self._log.warn("job.claim_lapsed", jobId=str(self._job.job_id), attempt=self._job.attempt)
                self._give_up("claim_lapsed")
                return
            renew_at = min(now + self._retry, lapses_at)
