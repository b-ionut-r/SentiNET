"""Background watchlist monitor.

Every tick (default 60 s) it refreshes, one at a time, each watched ticker (and
each ticker with an enabled alert rule) whose newest snapshot is older than
`settings.monitor_interval_minutes`. Staleness — not a fixed clock — drives the
schedule, so restarts neither stampede the APIs nor skip a cycle. Each refresh
is a normal analysis, which stores a snapshot and evaluates alert rules; the
monitor then retries undelivered webhooks and prunes very old snapshots.

A ticker whose refresh fails is not retried every tick (a failed run stores no
snapshot, so it would look "due" forever and fan out to ~30 provider calls a
minute): it backs off exponentially — interval × 2^(failures−1), capped at
24 h — and an unknown/delisted symbol waits the full 24 h straight away. A run
that returns but has no evidence at all (every source and feed failed: an
outage, or a laptop waking before its Wi-Fi) is not stored, so it counts as a
failure too.
Failing tickers are reported in `Monitor.status` (`GET /api/monitor`).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.services.errors import NoEvidence, UnknownSymbol
from app.services.tasks import describe_error
from app.storage import db

logger = logging.getLogger(__name__)

MIN_INTERVAL_MINUTES = 5  # floor: free APIs deserve politeness
MAX_BACKOFF = timedelta(hours=24)
PRUNE_EVERY = timedelta(hours=24)
SNAPSHOT_RETENTION_DAYS = 400


@dataclass
class TickerFailure:
    """Consecutive failed refreshes of one ticker and when it is next tried."""

    failures: int
    last_attempt: datetime
    retry_at: datetime
    error: str
    unknown: bool = False  # the symbol does not exist (typo / delisted)


@dataclass
class MonitorStatus:
    running: bool = False
    interval_minutes: int = 0
    last_cycle_at: datetime | None = None
    last_refreshed: list[str] = field(default_factory=list)
    last_error: str | None = None
    failing: dict[str, TickerFailure] = field(default_factory=dict)


class Monitor:
    """Staleness-driven refresher for watched tickers."""

    def __init__(
        self,
        interval_minutes: int | None = None,
        *,
        tick_seconds: float = 60.0,
        pause_seconds: float = 2.0,
        startup_delay: float = 15.0,
    ) -> None:
        minutes = interval_minutes if interval_minutes is not None else settings.monitor_interval_minutes
        self.interval = timedelta(minutes=max(MIN_INTERVAL_MINUTES, minutes))
        self.tick_seconds = tick_seconds
        self.pause_seconds = pause_seconds
        self.startup_delay = startup_delay
        self.status = MonitorStatus(interval_minutes=int(self.interval.total_seconds() // 60))
        self._task: asyncio.Task[None] | None = None
        self._last_prune: datetime | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self.running:
            self._task = asyncio.create_task(self._loop(), name="watchlist-monitor")
            self.status.running = True

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self.status.running = False

    def backoff(self, failures: int, unknown: bool) -> timedelta:
        """Wait before retrying a ticker after `failures` consecutive failed refreshes."""
        if unknown:
            return MAX_BACKOFF
        return min(MAX_BACKOFF, self.interval * 2 ** max(0, failures - 1))

    async def due_tickers(self, now: datetime) -> list[str]:
        """Watched/alerted tickers whose newest snapshot is older than the interval (minus backoff)."""
        tickers = list(dict.fromkeys([*await db.watch_tickers(), *await db.alert_tickers()]))
        failing = self.status.failing
        for gone in set(failing) - set(tickers):  # unwatched since: forget it
            del failing[gone]
        slack = timedelta(seconds=min(60.0, self.tick_seconds))  # don't miss a cycle by a hair
        due = []
        for t in tickers:
            last = await db.latest_snapshot(t)
            fail = failing.get(t)
            if fail is not None and last is not None and last.at > fail.last_attempt:
                del failing[t]  # analyzed successfully since (e.g. by a user): healthy again
                fail = None
            if fail is not None:
                if now >= fail.retry_at - slack:
                    due.append(t)
            elif last is None or now - last.at >= self.interval - slack:
                due.append(t)
        return due

    def _record_failure(self, ticker: str, now: datetime, exc: Exception) -> None:
        prev = self.status.failing.get(ticker)
        failures = (prev.failures if prev else 0) + 1
        unknown = isinstance(exc, UnknownSymbol)
        wait = self.backoff(failures, unknown)
        error = describe_error(exc)
        self.status.failing[ticker] = TickerFailure(failures=failures, last_attempt=now, retry_at=now + wait,
                                                    error=error, unknown=unknown)
        self.status.last_error = f"{ticker}: {error}"
        logger.warning("monitor refresh of %s failed (%d in a row; next try in %s): %s",
                       ticker, failures, wait, error)

    async def run_once(self, now: datetime | None = None) -> list[str]:
        """One cycle: refresh due tickers sequentially; returns those refreshed successfully."""
        from app.services import alerts, analyzer

        now = now or datetime.now(UTC)
        refreshed: list[str] = []
        for i, ticker in enumerate(await self.due_tickers(now)):
            if i and self.pause_seconds:
                await asyncio.sleep(self.pause_seconds)
            try:
                result = await analyzer.analyze(ticker, refresh=True)
                if not analyzer.has_evidence(result):
                    raise NoEvidence("every source and data feed failed; nothing stored")
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the cycle
                self._record_failure(ticker, now, exc)
                continue
            self.status.failing.pop(ticker, None)
            refreshed.append(ticker)
        try:
            await alerts.retry_undelivered(now)
            if self._last_prune is None or now - self._last_prune >= PRUNE_EVERY:
                await db.prune_snapshots(SNAPSHOT_RETENTION_DAYS)
                self._last_prune = now
        except Exception as exc:  # noqa: BLE001
            logger.warning("monitor housekeeping failed: %s", describe_error(exc))
        self.status.last_cycle_at = now
        self.status.last_refreshed = refreshed
        return refreshed

    async def _loop(self) -> None:
        await asyncio.sleep(self.startup_delay)
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                self.status.last_error = describe_error(exc)
                logger.exception("monitor cycle failed")
            await asyncio.sleep(self.tick_seconds)
