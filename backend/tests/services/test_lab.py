"""Sentiment lab: input bounds and admission control (user-submitted CPU work never stalls analyses)."""
from __future__ import annotations

import asyncio
import threading

import pytest

from app.schemas import ScoreRequest
from app.services import analyzer, lab
from app.services.errors import Busy, InvalidInput
from tests.services.fakes import FakeWorld


def test_total_character_budget_is_enforced():
    per_text = lab.MAX_TEXT_CHARS
    fits = ["x" * per_text] * (lab.MAX_TOTAL_CHARS // per_text)
    assert lab.validate_texts(fits) == fits
    with pytest.raises(InvalidInput, match="250,000 characters"):
        lab.validate_texts(fits + ["y"])


async def test_lab_scores_on_its_own_thread_never_the_shared_cpu_pool(world: FakeWorld, monkeypatch):
    threads: list[str] = []
    real = lab._score_sync

    def spy(texts, company):
        threads.append(threading.current_thread().name)
        return real(texts, company)

    monkeypatch.setattr(lab, "_score_sync", spy)
    res = await lab.score_texts(ScoreRequest(texts=["Nvidia beats estimates"]))
    assert res.summary.n == 1
    assert threads and threads[0].startswith("lab") and not threads[0].startswith("cpu")


async def test_one_at_a_time_and_busy_beyond_the_queue(world: FakeWorld, monkeypatch):
    release = threading.Event()
    running: list[int] = []
    active = 0
    peak = 0
    lock = threading.Lock()
    real = lab._score_sync

    def slow(texts, company):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        running.append(len(texts))
        release.wait(5)
        with lock:
            active -= 1
        return real(texts, company)

    monkeypatch.setattr(lab, "_score_sync", slow)
    first = asyncio.create_task(lab.score_texts(ScoreRequest(texts=["a"])))
    await asyncio.sleep(0.05)
    queued = [asyncio.create_task(lab.score_texts(ScoreRequest(texts=["b"]))) for _ in range(lab.MAX_WAITING)]
    await asyncio.sleep(0.05)
    with pytest.raises(Busy) as busy:
        await lab.score_texts(ScoreRequest(texts=["c"]))
    assert busy.value.status_code == 503 and busy.value.retry_after

    # While the lab is saturated, an analysis (CPU pool synthesis) is not held up.
    a = await asyncio.wait_for(analyzer.analyze("NVDA"), 5)
    assert a.ticker == "NVDA" and not release.is_set()

    release.set()
    results = await asyncio.gather(first, *queued)
    assert all(r.summary.n == 1 for r in results)
    assert peak == 1  # never two lab jobs at once
    # The queue drains: a new request is admitted again.
    assert (await lab.score_texts(ScoreRequest(texts=["d"]))).summary.n == 1


async def test_abandoned_request_keeps_its_slot_until_the_work_ends(world: FakeWorld, monkeypatch):
    release = threading.Event()
    real = lab._score_sync

    def slow(texts, company):
        release.wait(5)
        return real(texts, company)

    monkeypatch.setattr(lab, "_score_sync", slow)
    first = asyncio.create_task(lab.score_texts(ScoreRequest(texts=["a"])))
    await asyncio.sleep(0.05)
    first.cancel()  # client went away; its CPU work still runs
    await asyncio.sleep(0.05)
    assert lab._slot().locked()
    release.set()
    for _ in range(100):
        if not lab._slot().locked():
            break
        await asyncio.sleep(0.02)
    assert not lab._slot().locked()


# ---- unknown tickers and texts about something else (QA repro: ZZZZQQ → relevance 0, "neutral 0.0") ---- #
async def test_unknown_ticker_is_a_404_not_a_silent_zero_relevance(world: FakeWorld):
    from app.schemas import Quote
    from app.services.errors import UnknownSymbol
    from app.sources.base import CompanyRef

    world.resolve = CompanyRef(ticker="ZZZZQQ", name="ZZZZQQ", short_name="ZZZZQQ")
    world.intel["quote"] = None  # Yahoo answers "no such symbol" while it quotes SPY
    with pytest.raises(UnknownSymbol, match="ZZZZQQ"):
        await lab.score_texts(ScoreRequest(texts=["Apple beats estimates"], ticker="zzzzqq"))
    # A provider outage proves nothing: the bare symbol is accepted.
    world.witness = None
    res = await lab.score_texts(ScoreRequest(texts=["Apple beats estimates"], ticker="ZZZZQQ"))
    assert res.results[0].relevance is not None
    # A resolved company scores with its resolved name (no quote call needed).
    world.resolve, world.witness, world.intel["quote"] = None, Quote(price=600.0), Quote(price=180.0)
    calls = len(world.calls.get("quote", []))
    res = await lab.score_texts(ScoreRequest(texts=["Nvidia beats estimates"], ticker="NVDA"))
    assert res.results[0].relevance == 1.0 and len(world.calls.get("quote", [])) == calls


def test_summary_agrees_with_its_counts_when_no_text_is_about_the_ticker(world: FakeWorld, monkeypatch):
    import sys

    from app.sources.base import CompanyRef

    monkeypatch.setattr(sys.modules["app.nlp.relevance"], "relevance", lambda text, company: 0.0)
    msft = CompanyRef(ticker="MSFT", name="Microsoft Corporation", short_name="Microsoft")
    res = lab._score_sync(["Apple beats estimates", "Apple upgrade on iPhone demand"], msft)
    assert [r.relevance for r in res.results] == [0.0, 0.0]
    s = res.summary
    assert s.bullish == 2 and s.label == "bullish" and s.score == pytest.approx(0.6)  # not a fabricated neutral 0
    # With one relevant text, the irrelevant ones carry no weight at all.
    monkeypatch.setattr(sys.modules["app.nlp.relevance"], "relevance",
                        lambda text, company: 1.0 if "Microsoft" in text else 0.0)
    res = lab._score_sync(["Apple beats estimates", "Microsoft hit with lawsuit"], msft)
    assert res.summary.score == pytest.approx(-0.5) and res.summary.label == "bearish"
