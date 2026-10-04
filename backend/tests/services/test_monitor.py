"""Watchlist monitor: staleness-driven refresh, resilience, lifecycle."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from app.schemas import AlertRuleIn
from app.services import analyzer
from app.services.monitor import Monitor
from app.storage import db
from tests.services.fakes import FakeWorld


async def test_due_tickers_by_staleness(world: FakeWorld):
    await db.add_watch("NVDA")
    await db.add_watch("AAPL")
    await db.create_rule(AlertRuleIn(ticker="TSLA", kind="score_above", threshold=70))
    await analyzer.analyze("NVDA")  # fresh snapshot
    mon = Monitor(interval_minutes=30, tick_seconds=60)
    now = datetime.now(UTC)
    assert await mon.due_tickers(now) == ["AAPL", "TSLA"]
    assert await mon.due_tickers(now + timedelta(minutes=31)) == ["NVDA", "AAPL", "TSLA"]


async def test_run_once_refreshes_due_and_survives_failures(world: FakeWorld, monkeypatch):
    await db.add_watch("NVDA")
    await db.add_watch("BAD")
    await db.add_watch("AAPL")
    seen: list[tuple[str, bool]] = []
    real = analyzer.analyze

    async def fake_analyze(ticker: str, refresh: bool = False, progress=None):
        seen.append((ticker, refresh))
        if ticker == "BAD":
            raise RuntimeError("provider meltdown")
        return await real(ticker, refresh=refresh)

    monkeypatch.setattr(analyzer, "analyze", fake_analyze)
    mon = Monitor(interval_minutes=30, pause_seconds=0)
    refreshed = await mon.run_once()
    assert refreshed == ["NVDA", "AAPL"]
    assert seen == [("NVDA", True), ("BAD", True), ("AAPL", True)]
    assert "BAD" in mon.status.last_error
    assert mon.status.last_cycle_at is not None
    assert len(await db.list_snapshots("NVDA", 5)) == 1
    # Immediately after, nothing (except the failing ticker) is due.
    seen.clear()
    await mon.run_once()
    assert seen == [("BAD", True)]


def test_interval_floor():
    assert Monitor(interval_minutes=1).interval == timedelta(minutes=5)
    assert Monitor(interval_minutes=45).status.interval_minutes == 45


async def test_start_stop(world: FakeWorld):
    mon = Monitor(interval_minutes=30, startup_delay=0.01, tick_seconds=0.05)
    await db.add_watch("NVDA")
    mon.start()
    assert mon.running
    for _ in range(100):
        if await db.list_snapshots("NVDA", 1):
            break
        await asyncio.sleep(0.02)
    await mon.stop()
    assert not mon.running and len(await db.list_snapshots("NVDA", 5)) == 1
