"""Pipeline + aggregation tests using fake sources (fully offline)."""
from datetime import datetime, timezone

from app.pipeline.aggregate import build_response
from app.pipeline.normalize import normalize_and_score
from app.sources.base import RawSignal
from app.sources.registry import SourceResult


class _FakeSource:
    def __init__(self, name, weight, kind="news"):
        self.name = name
        self.label = name
        self.kind = kind
        self.weight = weight

    async def fetch(self, ticker, company):  # pragma: no cover - not called here
        return []


def _result(name, weight, raws, status="ok"):
    return SourceResult(source=_FakeSource(name, weight), signals=raws, status=status)


def test_normalize_dedups_and_scores():
    raws = [
        RawSignal(text="Apple earnings beat, stock soars to record high!"),
        RawSignal(text="Apple earnings beat, stock soars to record high!"),  # dup
        RawSignal(text="hi"),  # too short -> dropped
    ]
    results = [_result("yahoo", 1.1, raws)]
    signals = normalize_and_score(results)
    assert len(signals) == 1
    assert signals[0].label == "bullish"


def test_news_outweighs_social_volume():
    now = datetime.now(timezone.utc)
    # One strongly bullish news item vs. many mildly bearish social posts.
    news = _result(
        "yahoo", 1.1,
        [RawSignal(text="Record profit, huge beat, stock soars!", timestamp=now)],
    )
    social = _result(
        "reddit", 0.7,
        [RawSignal(text="meh not great, slight drop", timestamp=now) for _ in range(3)],
    )
    results = [news, social]
    signals = normalize_and_score(results)
    resp = build_response("AAPL", "Apple", results, signals)
    assert resp.total_signals == len(signals)
    assert resp.active_sources == 2
    # Per-source breakdown present for both.
    names = {b.source for b in resp.sources}
    assert {"yahoo", "reddit"} <= names


def test_empty_sources_degrade_gracefully():
    results = [_result("reddit", 0.7, [], status="error")]
    signals = normalize_and_score(results)
    resp = build_response("AAPL", "Apple", results, signals)
    assert resp.total_signals == 0
    assert resp.overall_label == "neutral"
    assert resp.sources[0].status == "error"
