"""End-to-end API test through the real pipeline with realistic injected data.

The live network is irrelevant here: we replace the source fan-out with a set
of realistic RawSignals, then drive the actual /api/analyze route so the whole
normalize -> VADER score -> weighted aggregate -> JSON path is exercised exactly
as it runs in production. This proves everything downstream of the network.
"""
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.cache import store
from app.main import app
from app.sources.base import RawSignal
from app.sources.registry import SourceResult


class _S:
    def __init__(self, name, weight, kind):
        self.name, self.label, self.weight, self.kind = name, name, weight, kind

    async def fetch(self, ticker, company):
        return []


@pytest.fixture(autouse=True)
def _clear_cache():
    store.clear()
    yield
    store.clear()


def test_analyze_end_to_end(monkeypatch):
    now = datetime.now(timezone.utc)
    realistic = [
        SourceResult(
            _S("yahoo", 1.1, "news"),
            [
                RawSignal(text="Apple beats earnings expectations, stock soars to record high",
                          url="https://finance.yahoo.com/news/a", author="Reuters", timestamp=now),
                RawSignal(text="Apple unveils new product line to strong analyst praise",
                          url="https://finance.yahoo.com/news/b", author="Bloomberg", timestamp=now),
            ],
            "ok",
        ),
        SourceResult(
            _S("reddit", 0.7, "social"),
            [
                RawSignal(text="$AAPL to the moon after that earnings beat!",
                          author="wsb_user", timestamp=now, engagement=120),
                RawSignal(text="Worried AAPL is overvalued here, might dump my shares",
                          author="bear_user", timestamp=now, engagement=8),
            ],
            "ok",
        ),
        SourceResult(_S("stocktwits", 0.8, "social"), [], "error"),
    ]

    async def fake_gather(ticker, company):
        return realistic

    monkeypatch.setattr("app.api.routes_analyze.gather_signals", fake_gather)

    client = TestClient(app)
    resp = client.get("/api/analyze/AAPL")
    assert resp.status_code == 200
    data = resp.json()

    assert data["ticker"] == "AAPL"
    assert data["total_signals"] == 4
    assert data["active_sources"] == 2            # yahoo + reddit produced signals
    assert data["overall_label"] == "bullish"     # bullish news + social dominate
    assert data["overall_score"] > 0.1
    # Honest degradation: the failed source is reported, not hidden.
    statuses = {s["source"]: s["status"] for s in data["sources"]}
    assert statuses["stocktwits"] == "error"
    # News feed carries real URLs for the UI.
    assert any(s["url"] for s in data["signals"])

    # Second call should be served from cache.
    resp2 = client.get("/api/analyze/AAPL")
    assert resp2.json()["cached"] is True
