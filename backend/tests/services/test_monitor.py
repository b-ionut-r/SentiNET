"""Watchlist monitor: staleness-driven refresh, resilience, lifecycle."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from app.config import settings
from app.schemas import AlertRuleIn
from app.services import alerts, analyzer
from app.services.monitor import MAX_BACKOFF, Monitor
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
    # Immediately after, nothing is due — not even the failing ticker (it backs off).
    seen.clear()
    await mon.run_once()
    assert seen == []
    assert mon.status.failing["BAD"].failures == 1


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


async def test_failing_ticker_backs_off_instead_of_retrying_every_tick(world: FakeWorld):
    """Reviewer repro: a failing run stores no snapshot, so the ticker used to be due on every 60 s tick."""
    await db.add_watch("NVDA")
    world.build_error = RuntimeError("analytics missing")
    mon = Monitor(interval_minutes=30, pause_seconds=0)
    t0 = datetime.now(UTC)
    attempts = []
    for k in range(6):  # six ticks over five minutes
        before = len(world.calls.get("quote", []))
        await mon.run_once(t0 + timedelta(seconds=61 * k))
        attempts.append(len(world.calls.get("quote", [])) - before)
    assert attempts == [1, 0, 0, 0, 0, 0]
    fail = mon.status.failing["NVDA"]
    assert fail.failures == 1 and fail.retry_at == t0 + timedelta(minutes=30) and not fail.unknown

    # Exponential: 30 min, then 60, then 120 … capped at 24 h.
    await mon.run_once(t0 + timedelta(minutes=30))
    assert mon.status.failing["NVDA"].failures == 2
    assert await mon.due_tickers(t0 + timedelta(minutes=30 + 58)) == []  # (1 tick of slack)
    assert await mon.due_tickers(t0 + timedelta(minutes=30 + 60)) == ["NVDA"]
    assert mon.backoff(20, unknown=False) == MAX_BACKOFF

    # Recovery clears the failure state.
    world.build_error = None
    assert await mon.run_once(t0 + timedelta(minutes=95)) == ["NVDA"]
    assert "NVDA" not in mon.status.failing


async def test_unknown_symbol_waits_a_day(world: FakeWorld):
    from app.sources.base import CompanyRef

    await db.create_rule(AlertRuleIn(ticker="ZZZZ9", kind="score_above", threshold=70))
    world.resolve = CompanyRef(ticker="ZZZZ9", name="ZZZZ9", short_name="ZZZZ9")
    for key in ("profile", "quote", "technicals", "analysts", "tone", "wiki"):
        world.intel[key] = None
    world.sources = []
    mon = Monitor(interval_minutes=30, pause_seconds=0)
    t0 = datetime.now(UTC)
    assert await mon.run_once(t0) == []
    fail = mon.status.failing["ZZZZ9"]
    assert fail.unknown and fail.retry_at == t0 + MAX_BACKOFF
    assert await mon.due_tickers(t0 + timedelta(hours=23)) == []
    # Rule deleted → the ticker is forgotten.
    await db.delete_rule((await db.list_rules())[0].id)
    assert await mon.due_tickers(t0 + timedelta(hours=25)) == [] and mon.status.failing == {}


async def test_user_analysis_clears_monitor_failure(world: FakeWorld):
    await db.add_watch("NVDA")
    world.build_error = RuntimeError("transient")
    mon = Monitor(interval_minutes=30, pause_seconds=0)
    t0 = datetime.now(UTC) - timedelta(minutes=5)
    await mon.run_once(t0)
    world.build_error = None
    await analyzer.analyze("NVDA", refresh=True)  # a user ran it successfully since
    assert await mon.due_tickers(t0 + timedelta(minutes=6)) == [] and "NVDA" not in mon.status.failing


async def test_monitor_cycle_posts_each_alert_once(world: FakeWorld, monkeypatch):
    """Reviewer repro: the post-analysis delivery and the cycle's retry used to race and double-post."""
    import app.core.http as http

    monkeypatch.setattr(settings, "alert_webhook_url", "https://discord.example/webhook")
    posts: list[str] = []

    async def slow_fetch(url, **kw):
        await asyncio.sleep(0.2)
        posts.append(kw["json"]["content"])

    monkeypatch.setattr(http, "fetch", slow_fetch)
    await db.add_watch("NVDA")
    await db.create_rule(alerts.normalize_rule(AlertRuleIn(ticker="NVDA", kind="score_above", threshold=60)))
    mon = Monitor(interval_minutes=30, pause_seconds=0)
    await mon.run_once()
    await alerts.drain()
    assert len(posts) == 1 and posts[0].startswith("**NVDA SentiNET 64 ≥ 60**")
    assert (await db.list_alert_events(5))[0].delivered is True


async def test_evidence_free_run_counts_as_a_failure_and_backs_off(world: FakeWorld):
    """Secops repro: an all-providers-down run was counted as a success (stored, alerted, no backoff)."""
    from tests.services.fakes import FakeSource, Sentinel

    await db.add_watch("NVDA")
    world.evidence = None
    world.sources = [(FakeSource("google_news", error=RuntimeError("ConnectError")), "enabled")]
    for key in ("profile", "quote", "technicals", "analysts", "insiders", "earnings", "tone", "wiki"):
        world.intel[key] = Sentinel(exc=RuntimeError("ConnectError"))
    mon = Monitor(interval_minutes=30, pause_seconds=0)
    t0 = datetime.now(UTC)
    assert await mon.run_once(t0) == []
    fail = mon.status.failing["NVDA"]
    assert fail.failures == 1 and not fail.unknown and "every source and data feed failed" in fail.error
    assert await db.list_snapshots("NVDA", 5) == [] and await db.latest_record("NVDA") is None
    assert await mon.due_tickers(t0 + timedelta(minutes=5)) == []  # backs off, no retry every tick
