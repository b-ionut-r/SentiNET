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
