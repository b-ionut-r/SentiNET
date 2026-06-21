"""Source adapter parsing tests against recorded fixtures (offline).

These exercise the JSON/RSS parsing logic without any network access by
monkeypatching the shared HTTP client.
"""
import json

import pytest

from app.sources.reddit import RedditSource
from app.sources.tradestie import TradestieSource


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    async def get(self, *args, **kwargs):
        return _FakeResponse(self._payload)


@pytest.mark.asyncio
async def test_reddit_parsing(monkeypatch):
    payload = {
        "data": {
            "children": [
                {
                    "data": {
                        "title": "AAPL to the moon",
                        "selftext": "earnings beat",
                        "created_utc": 1_700_000_000,
                        "permalink": "/r/stocks/abc",
                        "author": "user1",
                        "score": 10,
                        "num_comments": 5,
                    }
                }
            ]
        }
    }
    monkeypatch.setattr("app.sources.reddit.get_client", lambda: _FakeClient(payload))
    signals = await RedditSource().fetch("AAPL", "Apple")
    assert len(signals) == 1
    assert "moon" in signals[0].text
    assert signals[0].engagement == 15
    assert signals[0].url.endswith("/r/stocks/abc")


@pytest.mark.asyncio
async def test_tradestie_matches_ticker(monkeypatch):
    payload = [
        {"ticker": "AAPL", "no_of_comments": 42, "sentiment": "Bullish", "sentiment_score": 0.4},
        {"ticker": "TSLA", "no_of_comments": 10, "sentiment": "Bearish", "sentiment_score": -0.3},
    ]
    monkeypatch.setattr("app.sources.tradestie.get_client", lambda: _FakeClient(payload))
    signals = await TradestieSource().fetch("AAPL", "Apple")
    assert len(signals) == 1
    assert signals[0].engagement == 42
    assert signals[0].prelabeled_score == 0.4


@pytest.mark.asyncio
async def test_tradestie_no_match_returns_empty(monkeypatch):
    payload = [{"ticker": "TSLA", "no_of_comments": 10, "sentiment": "Bearish"}]
    monkeypatch.setattr("app.sources.tradestie.get_client", lambda: _FakeClient(payload))
    signals = await TradestieSource().fetch("AAPL", "Apple")
    assert signals == []
