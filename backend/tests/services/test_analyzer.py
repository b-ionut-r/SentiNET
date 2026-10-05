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
    assert inputs.intel_status["tone"] == "error: still loading after 0.2s; continuing in the background (reload to include)"
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


async def test_run_budget_caps_waiting_but_not_the_provider_call(world: FakeWorld, monkeypatch):
    monkeypatch.setattr(analyzer, "RUN_BUDGET", 0.3)
    monkeypatch.setattr(analyzer, "MIN_TIME_BOX", 0.1)
    world.intel["quote"] = Sentinel(delay=0.8, value=Quote(price=1.0))  # a slow *core* task: budget applies
    rec = Recorder()
    a = await analyzer.analyze("NVDA", progress=rec)
    assert a.elapsed_ms < 700  # did not wait for the slow provider
    quote = next(e for e in rec.events if e.key == "quote" and e.status != "running")
    assert quote.status == "error" and "continuing in the background" in quote.detail
    assert any(t.get_name() == "bounded:intel:quote" for t in tasks._background)  # still finishing
    await tasks.cancel_background()


async def test_budget_counts_symbol_resolution(world: FakeWorld, monkeypatch):
    monkeypatch.setattr(analyzer, "RUN_BUDGET", 0.6)
    monkeypatch.setattr(analyzer, "MIN_TIME_BOX", 0.1)
    world.resolve = Sentinel(delay=0.4, value=world.company)
    world.intel["technicals"] = Sentinel(delay=2.0, value=None)
    a = await analyzer.analyze("NVDA")
    assert a.elapsed_ms < 900  # 0.4 s resolve + what was left of the 0.6 s budget, not 0.4 + 0.6
    await tasks.cancel_background()


async def test_tail_rule_stops_idling_for_slow_name_search_intel(world: FakeWorld, monkeypatch):
    monkeypatch.setattr(analyzer, "TAIL_GRACE", 0.2)
    world.intel["tone"] = Sentinel(delay=0.8, value=world.intel["tone"])
    world.intel["wiki"] = Sentinel(delay=30, value=None)
    rec = Recorder()
    a = await analyzer.analyze("NVDA", progress=rec)
    assert a.elapsed_ms < 600  # core done at ~0 s + 0.2 s grace; not the 11 s budget
    tone = next(e for e in rec.events if e.key == "tone" and e.status != "running")
    assert tone.status == "error" and "still loading" in tone.detail
    assert world.inputs[-1].tone is None and world.inputs[-1].intel_status["tone"].startswith("error: still loading")

    # The straggler lands later with data: the cached run is superseded, the next load recomputes with it.
    assert (await analyzer.analyze("NVDA")).cached is True
    await asyncio.sleep(0.8)
    assert analyzer.cached_analysis("NVDA") is None
    world.intel["tone"] = world.intel["tone"].value  # provider cache is warm now: instant
    fresh = await analyzer.analyze("NVDA")
    assert fresh.cached is False and world.inputs[-1].tone is not None
    await tasks.cancel_background()


async def test_failed_straggler_keeps_the_cached_result(world: FakeWorld, monkeypatch):
    monkeypatch.setattr(analyzer, "TAIL_GRACE", 0.1)
    world.intel["tone"] = Sentinel(delay=0.4, exc=RuntimeError("gdelt 429"))
    await analyzer.analyze("NVDA")
    await asyncio.sleep(0.5)
    assert analyzer.cached_analysis("NVDA") is not None  # nothing better arrived: no pointless re-run


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


async def test_keyword_search_hits_do_not_prove_a_symbol_exists(world: FakeWorld):
    """Live-observed: 'APPL' (AAPL typo) and delisted 'SIVB' match plenty of keyword-search news."""
    world.resolve = CompanyRef(ticker="APPL", name="APPL", short_name="APPL")
    world.sources = [(FakeSource("bing_news", signals=19, ticker_specific=False), "enabled"),
                     (FakeSource("google_news", signals=1, ticker_specific=False), "enabled"),
                     (FakeSource("bluesky", "social", signals=1, ticker_specific=False), "enabled"),
                     (FakeSource("stocktwits", "social", signals=0), "enabled")]
    for key in ("profile", "quote", "technicals", "analysts", "tone", "wiki"):
        world.intel[key] = None
    with pytest.raises(UnknownSymbol, match="mistyped or delisted"):
        await analyzer.analyze("APPL")
    assert world.inputs == [] and analyzer.cached_analysis("APPL") is None


@pytest.mark.parametrize("source", [
    FakeSource("stocktwits", "social", signals=0, metrics={"stocktwits_messages": 30, "stocktwits_watchers": 900}),
    FakeSource("apewisdom", "social", signals=0, metrics={"reddit_mentions": 12, "reddit_rank": 80}),
    FakeSource("seeking_alpha", signals=2, ticker_specific=True),
])
async def test_symbol_keyed_coverage_counts_as_existence(world: FakeWorld, source: FakeSource):
    """No Yahoo quote (e.g. an OTC name), but the symbol's own stream / issuer-tagged feed has it."""
    world.resolve = CompanyRef(ticker="OTCX", name="OTCX", short_name="OTCX")
    world.sources = [(FakeSource("bing_news", signals=5, ticker_specific=False), "enabled"), (source, "enabled")]
    for key in ("profile", "quote", "technicals", "analysts", "tone", "wiki"):
        world.intel[key] = None
    a = await analyzer.analyze("OTCX")
    assert a.ticker == "OTCX"


@pytest.mark.parametrize(("quote_type", "asset"), [("FUTURE", "futures"), ("CURRENCY", "currencies")])
async def test_issuer_intel_skipped_for_futures_and_currencies(world: FakeWorld, quote_type: str, asset: str):
    world.resolve = CompanyRef(ticker="GC=F", name="Gold Dec 26", short_name="Gold", quote_type=quote_type)
    rec = Recorder()
    await analyzer.analyze("GC=F", progress=rec)
    for key in ("analysts", "insiders", "earnings", "calendar", "filings"):
        assert rec.by_key(key) == ["skipped"] and key not in world.calls
    assert next(e for e in rec.events if e.key == "filings").detail == f"n/a for {asset}"
    assert set(world.inputs[0].intel_status) == {"profile", "quote", "technicals", "tone", "wiki"}


async def test_ensure_known(world: FakeWorld):
    # Resolver recognizes it (SEC CIK): no further calls.
    assert (await analyzer.ensure_known("NVDA")).name == "NVIDIA Corporation"
    assert "quote" not in world.calls
    # Unresolvable and Yahoo positively has no quote → 404.
    world.resolve = CompanyRef(ticker="TWTR", name="TWTR", short_name="TWTR")
    world.intel["quote"] = None
    with pytest.raises(UnknownSymbol):
        await analyzer.ensure_known("TWTR")
    # Quote lookup failing is an outage, not proof: accept.
    world.intel["quote"] = Sentinel(exc=RuntimeError("HTTP 429"))
    assert (await analyzer.ensure_known("TWTR")).ticker == "TWTR"


async def test_unknown_symbol_is_negatively_cached(world: FakeWorld):
    world.resolve = CompanyRef(ticker="QZXWV", name="QZXWV", short_name="QZXWV")
    world.sources = [(FakeSource("bing_news", signals=4, ticker_specific=False), "enabled")]
    for key in ("profile", "quote", "technicals", "analysts", "tone", "wiki"):
        world.intel[key] = None
    with pytest.raises(UnknownSymbol):
        await analyzer.analyze("QZXWV")
    rec = Recorder()
    with pytest.raises(UnknownSymbol, match="mistyped"):
        await analyzer.analyze("QZXWV", progress=rec)  # instant: no second fan-out
    assert len(world.calls["quote"]) == 1 and world.sources[0][0].calls == 1
    assert rec.events[-1].stage == "done" and rec.events[-1].status == "error"
    world.intel["quote"] = Quote(price=3.2)  # it lists now; an explicit refresh re-checks
    a = await analyzer.analyze("QZXWV", refresh=True)
    assert a.ticker == "QZXWV"


async def test_engine_label_never_queues_on_the_cpu_pool_once_built(world: FakeWorld, monkeypatch):
    pool_calls: list[str] = []
    real_run_cpu = analyzer.run_cpu

    async def counting(fn, *args):
        pool_calls.append(getattr(fn, "__name__", "?"))
        return await real_run_cpu(fn, *args)

    monkeypatch.setattr(analyzer, "run_cpu", counting)
    assert await analyzer._engine_label() == "sentinel"
    assert pool_calls == ["_engine_name"]  # first lookup may build the engine: off the loop
    await analyzer.analyze("NVDA")
    assert "_engine_name" not in pool_calls[1:]  # afterwards read directly (synthesis still uses the pool)
    assert world.inputs[-1].engine_name == "sentinel"


async def test_engine_label_is_time_boxed_when_the_pool_is_saturated(world: FakeWorld, monkeypatch):
    async def stuck(fn, *args):
        await asyncio.sleep(10)

    monkeypatch.setattr(analyzer, "run_cpu", stuck)
    monkeypatch.setattr(analyzer, "ENGINE_TIMEOUT", 0.05)
    monkeypatch.setattr(settings, "sentiment_engine", "finbert")
    assert await asyncio.wait_for(analyzer._engine_label(), 2) == "finbert"  # configured name, not a hang
    assert "finbert" not in analyzer._engines_built  # retried on the next run


# ---- run quality: evidence-free and degraded runs ------------------------------------------ #
def _all_down(world: FakeWorld) -> None:
    """Every source and every feed fails (an outage, or a laptop waking before its Wi-Fi)."""
    world.evidence = None  # decided like the real composite
    world.sources = [(FakeSource(k, error=RuntimeError("ConnectError")), "enabled")
                     for k in ("google_news", "bing_news", "stocktwits")]
    for key in ("profile", "quote", "technicals", "analysts", "insiders", "earnings", "tone", "wiki"):
        world.intel[key] = Sentinel(exc=RuntimeError("ConnectError"))


async def test_no_evidence_run_is_returned_but_never_stored_alerted_or_kept(world: FakeWorld, monkeypatch):
    from app.schemas import AlertRuleIn

    await db.create_rule(AlertRuleIn(ticker="NVDA", kind="score_below", threshold=55))
    _all_down(world)
    rec = Recorder()
    a = await analyzer.analyze("NVDA", progress=rec)
    assert a.verdict.score == 50 and a.sentiment.n == 0 and not analyzer.has_evidence(a)
    assert await db.list_snapshots("NVDA", 10) == []
    assert await db.list_alert_events(10) == []
    done = rec.events[-1]
    assert done.key == "done" and "not stored" in done.detail
    # Served from cache only briefly: the next load past SHORT_CACHE_TTL retries the providers.
    assert (await analyzer.analyze("NVDA")).cached is True
    monkeypatch.setattr(analyzer, "SHORT_CACHE_TTL", 0.0)
    await analyzer.analyze("NVDA", refresh=True)
    assert (await analyzer.analyze("NVDA")).cached is False
    # It proves nothing about the symbol either.
    assert not analyzer.has_evidence(analyzer.latest_analysis("NVDA"))


def test_run_quality_counts_failed_score_inputs():
    from app.analytics.inputs import SourceRun

    def runs(*statuses):
        return [SourceRun(source=FakeSource(f"s{i}"), status=st) for i, st in enumerate(statuses)]

    world = FakeWorld()
    from app.analytics.inputs import AnalysisInputs

    a = world.build(AnalysisInputs(company=world.company, now=datetime.now(UTC), engine_name="sentinel"))
    ok_intel = {"analysts": "ok", "insiders": "empty", "technicals": "ok", "tone": "error: still loading"}
    sound = analyzer.run_quality(a, runs("ok", "ok", "error", "empty", "unconfigured"), ok_intel)
    assert not sound.degraded and sound.failed == ("S2",) and sound.attempted == 7 and sound.note is None
    three = analyzer.run_quality(a, runs("ok", "error", "error", "ok", "ok", "ok", "ok"),
                                 {**ok_intel, "analysts": "error: timed out after 9s"})
    assert three.degraded and len(three.failed) == 3 and "3 of 10 inputs failed" in three.note
    small = analyzer.run_quality(a, runs("ok", "error", "error"), {"technicals": "ok"})  # crypto-sized
    assert small.degraded
    # A slow GDELT tail is not a failure; a disabled source was never tried.
    assert not analyzer.run_quality(a, runs("ok", "disabled", "disabled", "disabled"), ok_intel).degraded


async def test_degraded_run_is_stored_flagged_and_not_a_baseline(world: FakeWorld, monkeypatch):
    world.sources = [(FakeSource(k), "enabled") for k in ("google_news", "bing_news", "nasdaq", "stocktwits")]
    await analyzer.analyze("NVDA")
    sound = await db.latest_record("NVDA")
    assert not sound.degraded
    # Push the sound snapshot back in time so it qualifies as "previous" (≥ 15 min old).
    def age(conn):
        with conn:
            conn.execute("UPDATE snapshots SET at = ?", (db.to_db_time(datetime.now(UTC) - timedelta(hours=1)),))
    await db._run(age)

    for source, _ in world.sources[1:]:
        source.error = TimeoutError()  # 3 of 4 sources time out (a CPU-starved server)
    monkeypatch.setattr(analyzer, "SHORT_CACHE_TTL", 0.0)  # degraded results are cached only briefly
    rec = Recorder()
    await analyzer.analyze("NVDA", refresh=True, progress=rec)
    assert "degraded" in rec.events[-1].detail
    latest = await db.latest_record("NVDA")
    assert latest.degraded and latest.id != sound.id
    assert len(await db.list_snapshots("NVDA", 10)) == 1  # the score history shows readings only

    # Past the short TTL the next load re-runs, and it diffs against the sound snapshot.
    for source, _ in world.sources:
        source.error = None
    await analyzer.analyze("NVDA")
    assert len(world.inputs) == 3
    assert world.inputs[-1].previous is not None and world.inputs[-1].previous.at == (
        await db.latest_record("NVDA", sound_only=True, before=datetime.now(UTC) - timedelta(minutes=15))).at


# ---- outages are not "unknown symbol" ------------------------------------------------------ #
def _offline(world: FakeWorld, symbol: str) -> None:
    """Live repro shape (HTTPS_PROXY to a dead port): yfinance swallows ConnectionError and answers
    "no data" (ok, None); the resolver falls back to a bare ref; every httpx source raises."""
    import httpx

    world.resolve = CompanyRef(ticker=symbol, name=symbol, short_name=symbol)
    world.sources = [(FakeSource(k, error=httpx.ConnectError("proxy refused")), "enabled")
                     for k in ("google_news", "stocktwits", "apewisdom")]
    for key in ("profile", "quote", "technicals", "analysts", "insiders", "earnings", "tone", "wiki"):
        world.intel[key] = None
    world.witness = None  # SPY: "possibly delisted" too


async def test_outage_with_swallowed_errors_is_not_a_404(world: FakeWorld):
    _offline(world, "MSFT")
    world.evidence = None
    a = await analyzer.analyze("MSFT")  # no 404: an honest "No read" (not stored)
    assert a.ticker == "MSFT" and not analyzer.has_evidence(a)
    assert analyzer._unknown.get("MSFT") is None
    assert (await analyzer.ensure_known("MSFT")).ticker == "MSFT"  # watch/alert still accepted


async def test_yahoo_only_outage_is_not_a_404_for_foreign_listings(world: FakeWorld):
    _offline(world, "SHOP-TO")
    world.sources = [(FakeSource("google_news", signals=3, ticker_specific=False), "enabled")]  # news is up
    a = await analyzer.analyze("SHOP.TO")
    assert a.ticker == "SHOP-TO" and analyzer._unknown.get("SHOP-TO") is None
    assert world.calls["witness"]  # Yahoo was checked before concluding anything


async def test_typo_is_still_a_404_when_providers_answer(world: FakeWorld):
    _offline(world, "QZXWV")
    world.witness = Quote(price=600.0)  # Yahoo answers: SPY quoted, QZXWV not
    world.sources = [(FakeSource("apewisdom", "social", signals=0), "enabled"),  # board answered: not listed
                     (FakeSource("stocktwits", error=RuntimeError("HTTP 404 not found")), "enabled")]
    with pytest.raises(UnknownSymbol):
        await analyzer.analyze("QZXWV")
    world.resolve = CompanyRef(ticker="QZXWW", name="QZXWW", short_name="QZXWW")
    with pytest.raises(UnknownSymbol):
        await analyzer.ensure_known("QZXWW")
    # Every source erroring (with Yahoo's witness cached as up) concludes nothing either.
    world.sources = [(FakeSource("apewisdom", "social", error=RuntimeError("ConnectError")), "enabled")]
    world.resolve = CompanyRef(ticker="QZXWZ", name="QZXWZ", short_name="QZXWZ")
    assert (await analyzer.analyze("QZXWZ")).ticker == "QZXWZ"
