"""History service: degradation, and sharing what it fetches with the analysis cache."""
from __future__ import annotations

import asyncio

from app.config import settings
from app.services import analyzer, history, tasks
from tests.services.fakes import FakeSource, FakeWorld, Sentinel


async def test_tone_fetched_by_history_supersedes_a_toneless_cached_run(world: FakeWorld):
    good = world.intel["tone"]
    world.intel["tone"] = Sentinel(exc=RuntimeError("GDELT rate limited"))
    await analyzer.analyze("NVDA")
    assert analyzer.cached_analysis("NVDA") is not None
    world.intel["tone"] = good
    out = await history.get_history("NVDA", 90)
    assert out.status["tone"] == "ok"
    assert analyzer.cached_analysis("NVDA") is None  # next load recomputes, now with tone
    fresh = await analyzer.analyze("NVDA")
    assert fresh.cached is False and fresh.tone is not None


async def test_history_without_tone_keeps_the_cached_run(world: FakeWorld):
    world.intel["tone"] = Sentinel(exc=RuntimeError("GDELT rate limited"))
    await analyzer.analyze("NVDA")
    out = await history.get_history("NVDA", 90)
    assert out.status["tone"].startswith("error")
    assert analyzer.cached_analysis("NVDA") is not None  # nothing better: no pointless re-run


async def test_tone_landing_while_the_run_is_in_flight(world: FakeWorld):
    """The race of one page load: the run's GDELT call already failed when history gets the tone."""
    good = world.intel["tone"]
    world.intel["tone"] = Sentinel(exc=RuntimeError("GDELT rate limited"))
    world.sources = [(FakeSource("google_news", delay=0.3), "enabled")]
    run = asyncio.create_task(analyzer.analyze("NVDA"))
    await asyncio.sleep(0.1)  # tone has failed; the sources are still fetching
    world.intel["tone"] = good
    assert (await history.get_history("NVDA", 90)).status["tone"] == "ok"
    toneless = await run
    assert toneless.tone is None
    assert analyzer.cached_analysis("NVDA") is None  # superseded as it was stored
    assert (await analyzer.analyze("NVDA")).tone is not None


async def test_history_straggler_landing_later_supersedes(world: FakeWorld, monkeypatch):
    good = world.intel["tone"]
    world.intel["tone"] = Sentinel(exc=RuntimeError("GDELT rate limited"))
    await analyzer.analyze("NVDA")
    monkeypatch.setattr(settings, "intel_timeout", 0.2)  # history waits 0.3 s for tone
    world.intel["tone"] = Sentinel(delay=0.6, value=good)
    out = await history.get_history("NVDA", 90)
    assert out.status["tone"].startswith("error: still loading")
    assert analyzer.cached_analysis("NVDA") is not None
    await asyncio.sleep(0.6)  # the kept-alive call lands in the provider cache
    assert analyzer.cached_analysis("NVDA") is None
    await tasks.cancel_background()


async def test_wiki_over_a_longer_window_is_not_the_analyzers_call(world: FakeWorld):
    world.intel["wiki"] = Sentinel(exc=RuntimeError("Wikimedia 429"))
    await analyzer.analyze("NVDA")
    world.intel["wiki"] = [(world.intel["tone"].series[0].date, 1.0)]
    await history.get_history("NVDA", 365)  # 365-day pageviews: another cache entry
    assert analyzer.cached_analysis("NVDA") is not None
    await history.get_history("NVDA", 90)  # the analyzer's own 90-day call
    assert analyzer.cached_analysis("NVDA") is None


async def test_intel_landed_only_supersedes_what_the_run_lacked(world: FakeWorld):
    world.intel["wiki"] = None  # empty, not failed: still something the run went without
    await analyzer.analyze("NVDA")
    assert analyzer.intel_landed("NVDA", "tone") is False  # the run had tone
    assert analyzer.cached_analysis("NVDA") is not None
    assert analyzer.intel_landed("NVDA", "wiki") is True
    assert analyzer.cached_analysis("NVDA") is None
    assert analyzer.intel_landed("NVDA", "wiki") is False  # already superseded
    assert analyzer.intel_landed("MSFT", "tone") is False  # nothing cached
