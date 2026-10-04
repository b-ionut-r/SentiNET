"""Background watchlist monitor.

Every tick (default 60 s) it refreshes, one at a time, each watched ticker (and
each ticker with an enabled alert rule) whose newest snapshot is older than
`settings.monitor_interval_minutes`. Staleness — not a fixed clock — drives the
schedule, so restarts neither stampede the APIs nor skip a cycle. Each refresh
is a normal analysis, which stores a snapshot and evaluates alert rules; the
monitor then retries undelivered webhooks and prunes very old snapshots.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.services.tasks import describe_error
from app.storage import db

logger = logging.getLogger(__name__)

MIN_INTERVAL_MINUTES = 5  # floor: free APIs deserve politeness
PRUNE_EVERY = timedelta(hours=24)
SNAPSHOT_RETENTION_DAYS = 400


@dataclass
class MonitorStatus:
    running: bool = False
    interval_minutes: int = 0
    last_cycle_at: datetime | None = None
    last_refreshed: list[str] = field(default_factory=list)
    last_error: str | None = None


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

    async def due_tickers(self, now: datetime) -> list[str]:
        """Watched/alerted tickers whose newest snapshot is older than the interval."""
        tickers = list(dict.fromkeys([*await db.watch_tickers(), *await db.alert_tickers()]))
        slack = timedelta(seconds=min(60.0, self.tick_seconds))  # don't miss a cycle by a hair
        due = []
        for t in tickers:
            last = await db.latest_snapshot(t)
            if last is None or now - last.at >= self.interval - slack:
                due.append(t)
        return due

    async def run_once(self, now: datetime | None = None) -> list[str]:
        """One cycle: refresh due tickers sequentially; returns those refreshed successfully."""
        from app.services import alerts, analyzer

        now = now or datetime.now(UTC)
        refreshed: list[str] = []
        for i, ticker in enumerate(await self.due_tickers(now)):
            if i and self.pause_seconds:
                await asyncio.sleep(self.pause_seconds)
            try:
                await analyzer.analyze(ticker, refresh=True)
                refreshed.append(ticker)
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the cycle
                self.status.last_error = f"{ticker}: {describe_error(exc)}"
                logger.warning("monitor refresh of %s failed: %s", ticker, describe_error(exc))
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
