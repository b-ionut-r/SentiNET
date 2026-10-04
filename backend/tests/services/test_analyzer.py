"""Orchestrator: fan-out, time boxes, degradation, single-flight, progress, caching, persistence."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.config import settings
from app.schemas import Analysis, AnalystAction, ProgressEvent, Quote
from app.services import analyzer, tasks
from app.services.errors import AnalysisFailed, InvalidTicker, UnknownSymbol
from app.sources.base import CompanyRef
from app.storage import db
from tests.services.fakes import FakeSource, FakeWorld, Sentinel


class Recorder:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    async def __call__(self, ev: ProgressEvent) -> None:
        self.events.append(ev)

    def by_key(self, key: str) -> list[str]:
        return [e.status for e in self.events if e.key == key]


async def test_happy_path_builds_inputs_and_result(world: FakeWorld):
    rec = Recorder()
    a = await analyzer.analyze("$nvda", progress=rec)
    assert isinstance(a, Analysis)
    assert a.ticker == "NVDA" and a.cached is False and a.elapsed_ms >= 0

    (inputs,) = world.inputs
    assert inputs.company.name == "NVIDIA Corporation"
    assert inputs.now.tzinfo is not None
    assert inputs.engine_name == "sentinel"
    assert inputs.quote.price == 180.0 and inputs.technicals.trend == "uptrend"
    assert inputs.tone.tone_7d == 1.2 and inputs.wiki_views[0][1] == 15000.0
    runs = {r.source.key: r for r in inputs.source_runs}
    assert runs["google_news"].status == "ok" and len(runs["google_news"].batch.signals) == 3
    assert runs["stocktwits"].batch.metrics["stocktwits_bullish"] == 30
    assert runs["finnhub"].status == "unconfigured" and runs["finnhub"].batch is None
    assert inputs.intel_status["quote"] == "ok"
    assert inputs.intel_status["earnings"] == "empty"  # provider answered "nothing"
    assert inputs.previous is None


async def test_analysts_receive_live_price(world: FakeWorld):
    world.intel["quote"] = Quote(price=160.0)
    a = await analyzer.analyze("NVDA")
    assert world.calls["analysts"] == [("NVDA", 160.0)]
    assert a.analysts.upside_pct == pytest.approx(25.0)


async def test_progress_events_are_complete_and_ordered(world: FakeWorld):
    rec = Recorder()
    await analyzer.analyze("NVDA", progress=rec)
    assert rec.by_key("resolve") == ["running", "ok"]
    for key in ("google_news", "stocktwits", "quote", "profile", "tone", "analysts"):
        statuses = rec.by_key(key)
        assert statuses[0] == "running" and statuses[-1] in ("ok", "empty"), (key, statuses)
    assert rec.by_key("finnhub") == ["skipped"]
    assert rec.by_key("synthesis") == ["running", "ok"]
    assert rec.events[-1].stage == "done" and rec.events[-1].status == "ok"
    st = next(e for e in rec.events if e.key == "stocktwits" and e.status == "ok")
    assert st.count == 2 and "75% bulls of 40 tagged" in st.detail
    q = next(e for e in rec.events if e.key == "quote" and e.status == "ok")
    assert q.detail.startswith("180.00 USD +1.50%") and q.ms is not None


async def test_cache_and_refresh(world: FakeWorld):
    first = await analyzer.analyze("NVDA")
    rec = Recorder()
    second = await analyzer.analyze("nvda", progress=rec)
    assert second.cached is True and first.cached is False
    assert len(world.inputs) == 1
    assert [e.status for e in rec.events] == ["ok"] and rec.events[0].stage == "done"
    third = await analyzer.analyze("NVDA", refresh=True)
    assert third.cached is False and len(world.inputs) == 2
    assert analyzer.cached_analysis("NVDA").cached is True
    assert analyzer.latest_analysis("NVDA") is not None


async def test_single_flight_shares_one_run_and_replays_progress(world: FakeWorld):
    world.sources[0][0].delay = 0.2
    early, late = Recorder(), Recorder()
    first = asyncio.create_task(analyzer.analyze("NVDA", progress=early))
    await asyncio.sleep(0.05)  # the run is in flight; some events already emitted
    second = asyncio.create_task(analyzer.analyze("NVDA", refresh=True, progress=late))
    a, b = await asyncio.gather(first, second)
    assert len(world.inputs) == 1 and world.sources[0][0].calls == 1
    assert a.generated_at == b.generated_at
    assert [(e.key, e.status) for e in late.events] == [(e.key, e.status) for e in early.events]


async def test_failing_and_slow_sources_degrade_gracefully(world: FakeWorld, monkeypatch):
    monkeypatch.setattr(settings, "source_timeout", 0.2)
    world.sources = [
        (FakeSource("ok_news"), "enabled"),
        (FakeSource("broken", error=RuntimeError("HTTP 500 for https://x.io/api?token=sk_live_ABCDEF123&q=1")), "enabled"),
        (FakeSource("slow", delay=5), "enabled"),
    ]
    rec = Recorder()
    a = await analyzer.analyze("NVDA", progress=rec)
    runs = {r.source.key: r for r in world.inputs[0].source_runs}
    assert runs["ok_news"].status == "ok"
    assert runs["broken"].status == "error"
    assert "sk_live_ABCDEF123" not in runs["broken"].error and "token=***" in runs["broken"].error
    assert runs["slow"].status == "error" and runs["slow"].error == "timed out after 0.2s"
    assert runs["slow"].latency_ms is not None and runs["slow"].latency_ms < 2000
    assert rec.by_key("slow") == ["running", "error"]
    assert a.verdict.score == world.score


async def test_intel_failures_are_reported_not_raised(world: FakeWorld, monkeypatch):
    monkeypatch.setattr(settings, "intel_timeout", 0.2)
    world.intel["insiders"] = Sentinel(exc=ValueError("bad table"))
    world.intel["tone"] = Sentinel(delay=0.6, value=None)
    world.intel["profile"] = {"not": "a Profile"}
    await analyzer.analyze("NVDA")
    inputs = world.inputs[0]
    assert inputs.intel_status["insiders"] == "error: ValueError: bad table"
    assert inputs.intel_status["tone"] == "error: still loading after 0.2s; ready on next refresh"
    assert inputs.intel_status["profile"].startswith("error: unexpected payload")
    assert inputs.insiders is None and inputs.tone is None and inputs.profile is None
    # The slow GDELT call keeps running in the background to warm its cache…
    assert any(t.get_name() == "bounded:intel:tone" for t in tasks._background)
    await asyncio.sleep(0.5)
    assert not any(t.get_name() == "bounded:intel:tone" for t in tasks._background)  # …and is reaped
    await tasks.cancel_background()


async def test_not_applicable_intel_is_skipped_for_crypto(world: FakeWorld):
    world.company = CompanyRef(ticker="BTC-USD", name="Bitcoin USD", short_name="Bitcoin",
                               quote_type="CRYPTOCURRENCY")
    rec = Recorder()
    await analyzer.analyze("BTC-USD", progress=rec)
    for key in ("analysts", "insiders", "earnings", "calendar", "filings"):
        assert key not in world.calls, key
        assert rec.by_key(key) == ["skipped"]
        assert key not in world.inputs[0].intel_status
    assert "quote" in world.calls and "tone" in world.calls


async def test_filings_skipped_without_cik(world: FakeWorld):
    world.company = CompanyRef(ticker="SHOP-TO", name="Shopify Inc.", short_name="Shopify")
    rec = Recorder()
    await analyzer.analyze("SHOP-TO", progress=rec)
    assert "filings" not in world.calls
    skipped = next(e for e in rec.events if e.key == "filings")
    assert skipped.status == "skipped" and "SEC" in skipped.detail


async def test_invalid_ticker(world: FakeWorld):
    for bad in ("", "   ", "NV DA!", "X" * 40):
        with pytest.raises(InvalidTicker) as exc:
            await analyzer.analyze(bad)
        assert "not a valid ticker" in str(exc.value)
    assert world.inputs == []


async def test_unknown_symbol_when_every_provider_says_nothing(world: FakeWorld):
    world.resolve = CompanyRef(ticker="ZZZZZ", name="ZZZZZ", short_name="ZZZZZ")
    world.sources = [(FakeSource("google_news", signals=0), "enabled")]
    for key in ("profile", "quote", "technicals", "analysts", "tone", "wiki"):
        world.intel[key] = None
    with pytest.raises(UnknownSymbol):
        await analyzer.analyze("ZZZZZ")
    assert world.inputs == []
    assert analyzer.cached_analysis("ZZZZZ") is None


async def test_outage_is_not_mistaken_for_unknown_symbol(world: FakeWorld):
    world.resolve = CompanyRef(ticker="ZZZZZ", name="ZZZZZ", short_name="ZZZZZ")
    world.sources = [(FakeSource("google_news", signals=0), "enabled")]
    for key in ("profile", "technicals", "analysts", "tone", "wiki"):
        world.intel[key] = None
    world.intel["quote"] = Sentinel(exc=RuntimeError("HTTP 429"))
    a = await analyzer.analyze("ZZZZZ")  # degraded, honestly reported
    assert world.inputs[0].intel_status["quote"].startswith("error")
    assert a.ticker == "ZZZZZ"


async def test_resolver_failure_falls_back_to_bare_symbol(world: FakeWorld):
    world.resolve = Sentinel(exc=RuntimeError("SEC down"))
    rec = Recorder()
    await analyzer.analyze("NVDA", progress=rec)
    assert rec.by_key("resolve") == ["running", "error"]
    assert world.inputs[0].company.name == "NVDA"
    assert analyzer.bare_company("BTC-USD").quote_type == "CRYPTOCURRENCY"
    assert analyzer.bare_company("^VIX").quote_type == "INDEX"


async def test_build_failure_is_clean_and_not_cached(world: FakeWorld):
    world.build_error = ZeroDivisionError("division by zero")
    rec = Recorder()
    with pytest.raises(AnalysisFailed) as exc:
        await analyzer.analyze("NVDA", progress=rec)
    assert exc.value.status_code == 503 and "ZeroDivisionError" in str(exc.value)
    assert rec.by_key("synthesis") == ["running", "error"]
    assert analyzer.cached_analysis("NVDA") is None
    world.build_error = None
    assert (await analyzer.analyze("NVDA")).cached is False  # next call retries


async def test_broken_listener_does_not_break_the_run(world: FakeWorld):
    calls = 0

    async def flaky(ev: ProgressEvent) -> None:
        nonlocal calls
        calls += 1
        raise ConnectionResetError("client went away")

    a = await analyzer.analyze("NVDA", progress=flaky)
    assert a.ticker == "NVDA" and calls == 1  # dropped after the first failure


async def test_cancelled_caller_does_not_abort_shared_run(world: FakeWorld):
    world.sources[0][0].delay = 0.2
    caller = asyncio.create_task(analyzer.analyze("NVDA"))
    await asyncio.sleep(0.05)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    await asyncio.sleep(0.4)
    assert analyzer.cached_analysis("NVDA") is not None  # finished and cached anyway


async def test_snapshot_persisted_and_previous_respects_min_age(world: FakeWorld):
    await analyzer.analyze("NVDA")
    snaps = await db.list_snapshots("NVDA", 10)
    assert len(snaps) == 1 and snaps[0].sentinel_score == world.score

    await analyzer.analyze("NVDA", refresh=True)
    assert world.inputs[-1].previous is None  # the stored one is < 15 min old

    old = world.build(world.inputs[0]).model_copy(update={
        "generated_at": datetime.now(UTC) - timedelta(hours=3), "verdict": world.build(world.inputs[0]).verdict})
    await db.save_snapshot(old.model_copy(update={"ticker": "NVDA"}))
    await analyzer.analyze("NVDA", refresh=True)
    prev = world.inputs[-1].previous
    assert prev is not None and datetime.now(UTC) - prev.at > timedelta(hours=2)


async def test_alert_rules_evaluated_after_each_analysis(world: FakeWorld):
    from app.schemas import AlertRuleIn

    await db.create_rule(AlertRuleIn(ticker="NVDA", kind="score_above", threshold=60))
    await analyzer.analyze("NVDA")
    events = await db.list_alert_events(10)
    assert len(events) == 1 and events[0].title == "NVDA SentiNET 64 ≥ 60"


async def test_storage_failure_does_not_fail_analysis(world: FakeWorld, monkeypatch):
    async def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(db, "save_snapshot", boom)
    a = await analyzer.analyze("NVDA")
    assert a.ticker == "NVDA"


async def test_analyst_actions_flow_into_inputs(world: FakeWorld):
    world.analyst_actions = [AnalystAction(date=datetime.now(UTC), firm="Mizuho", action="up", to_grade="Buy")]
    a = await analyzer.analyze("NVDA")
    assert a.analysts.actions[0].firm == "Mizuho"


async def test_shutdown_cancels_inflight_runs(world: FakeWorld):
    world.sources[0][0].delay = 5
    task = asyncio.create_task(analyzer.analyze("NVDA"))
    await asyncio.sleep(0.05)
    await analyzer.shutdown()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_intel_budget_caps_waiting_but_not_the_provider_call(world: FakeWorld, monkeypatch):
    monkeypatch.setattr(analyzer, "INTEL_BUDGET", 0.3)
    monkeypatch.setattr(analyzer, "MIN_TIME_BOX", 0.1)
    world.intel["tone"] = Sentinel(delay=0.8, value=None)
    rec = Recorder()
    a = await analyzer.analyze("NVDA", progress=rec)
    assert a.elapsed_ms < 700  # did not wait for the slow provider
    tone = next(e for e in rec.events if e.key == "tone" and e.status != "running")
    assert tone.status == "error" and "ready on next refresh" in tone.detail
    assert any(t.get_name() == "bounded:intel:tone" for t in tasks._background)  # still finishing
    await tasks.cancel_background()


async def test_refresh_spam_is_served_from_cache(world: FakeWorld, monkeypatch):
    monkeypatch.setattr(analyzer, "MIN_REFRESH_SECONDS", 45.0)
    await analyzer.analyze("NVDA")
    again = await analyzer.analyze("NVDA", refresh=True)
    assert again.cached is True and len(world.inputs) == 1


async def test_unknown_symbol_emits_terminal_event(world: FakeWorld):
    world.resolve = CompanyRef(ticker="ZZZZZ", name="ZZZZZ", short_name="ZZZZZ")
    world.sources = [(FakeSource("google_news", signals=0), "enabled")]
    for key in ("profile", "quote", "technicals", "analysts", "tone", "wiki"):
        world.intel[key] = None
    rec = Recorder()
    with pytest.raises(UnknownSymbol):
        await analyzer.analyze("ZZZZZ", progress=rec)
    assert rec.events[-1].stage == "done" and rec.events[-1].status == "error"
    assert "No market data" in rec.events[-1].detail


async def test_concurrent_runs_are_capped_and_queued(world: FakeWorld, monkeypatch):
    monkeypatch.setattr(analyzer, "MAX_CONCURRENT_RUNS", 2)
    monkeypatch.setattr(analyzer, "_slots", None)
    world.sources = [(FakeSource("news", delay=0.2), "enabled")]
    active = peak = 0
    real_fetch = world.sources[0][0].fetch

    async def tracking_fetch(company):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            return await real_fetch(company)
        finally:
            active -= 1

    world.sources[0][0].fetch = tracking_fetch
    recs = {t: Recorder() for t in ("AAA", "BBB", "CCC")}
    await asyncio.gather(*(analyzer.analyze(t, progress=r) for t, r in recs.items()))
    assert peak == 2 and len(world.inputs) == 3
    queued = [t for t, r in recs.items() if r.by_key("queue")]
    assert queued == ["CCC"] and recs["CCC"].by_key("queue") == ["running", "ok"]


async def test_unknown_symbol_despite_name_search_failures(world: FakeWorld):
    """Live-observed shape: bare echo profile, GDELT/Wikipedia failing, Yahoo says no quote."""
    from app.schemas import Profile

    world.resolve = CompanyRef(ticker="QZXWV", name="QZXWV", short_name="QZXWV")
    world.sources = [(FakeSource("google_news", signals=0), "enabled"),
                     (FakeSource("stocktwits", "social", signals=0), "enabled")]
    world.intel.update(profile=Profile(symbol="QZXWV", name="QZXWV"), quote=None, technicals=None,
                       analysts=None, tone=Sentinel(exc=RuntimeError("HTTP 429")),
                       wiki=Sentinel(exc=RuntimeError("Wikipedia HTTP 403")))
    world.intel["tone"] = Sentinel(delay=5)  # slow provider must not delay the verdict
    started = asyncio.get_running_loop().time()
    with pytest.raises(UnknownSymbol):
        await analyzer.analyze("QZXWV")
    assert asyncio.get_running_loop().time() - started < 1.0
    assert not any(t.get_name() == "bounded:intel:tone" for t in tasks._background)  # cancelled, not kept
