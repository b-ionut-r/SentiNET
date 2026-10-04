"""Live smoke tests against the real keyless providers (`pytest -m live tests/sources`)."""
from __future__ import annotations

import asyncio
import re

import pytest

from app.sources.registry import get_source
from tests.sources.conftest import company

pytestmark = pytest.mark.live

TEXT_SOURCES = ["google_news", "bing_news", "yahoo_news", "seeking_alpha", "nasdaq", "stocktwits", "bluesky", "hackernews"]


async def _fetch(key: str, ticker: str):
    return await asyncio.wait_for(get_source(key).fetch(company(ticker)), timeout=30)


@pytest.mark.parametrize("key", TEXT_SOURCES)
async def test_text_source_returns_relevant_items_for_nvda(key):
    batch = await _fetch(key, "NVDA")
    assert batch.signals, f"{key} returned nothing"
    for s in batch.signals:
        assert s.title.strip()
        assert s.timestamp is None or s.timestamp.utcoffset().total_seconds() == 0
    mentions = [s for s in batch.signals if re.search(r"(?i)nvidia|nvda", f"{s.title} {s.body or ''}")]
    assert len(mentions) >= max(1, len(batch.signals) // 4), f"{key}: too few on-topic items"


@pytest.mark.parametrize("key", ["google_news", "bing_news", "seeking_alpha", "stocktwits", "bluesky"])
async def test_text_source_handles_crypto(key):
    batch = await _fetch(key, "BTC-USD")
    assert batch.signals and any(re.search(r"(?i)bitcoin|btc", s.title) for s in batch.signals)


@pytest.mark.parametrize("key", ["google_news", "bing_news", "seeking_alpha", "stocktwits", "bluesky"])
async def test_text_source_handles_etf_theme(key):
    batch = await _fetch(key, "SPY")
    assert batch.signals and any(re.search(r"(?i)s&p|\bspy\b", s.title) for s in batch.signals)


async def test_stocktwits_metrics_live():
    batch = await _fetch("stocktwits", "NVDA")
    m = batch.metrics
    assert m["stocktwits_messages"] > 0 and m["stocktwits_watchers"] > 100_000
    assert m["stocktwits_bullish"] + m["stocktwits_bearish"] <= m["stocktwits_messages"]


async def test_apewisdom_live_board():
    batch = await _fetch("apewisdom", "SPY")  # SPY is a perennial top-ranked ticker
    assert batch.metrics["reddit_rank"] >= 1 and batch.metrics["reddit_tracked"] > 100


async def test_tradestie_live_never_reports_frozen_sentiment():
    from app.sources import tradestie

    batch = await _fetch("tradestie", "SPY")
    live = await tradestie.sentiment_gate()
    if batch.metrics and not live:
        assert "wsb_sentiment" not in batch.metrics
