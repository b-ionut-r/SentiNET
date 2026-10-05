"""Market regime and market-wide headline narratives."""
from __future__ import annotations

from app.analytics.market import build_market_overview, market_regime
from app.schemas import FearGreed, IndexQuote, TrendingTicker
from tests.analytics.factories import NOW, raw


def fg(score: float, rating: str, month_ago: float | None = None) -> FearGreed:
    return FearGreed(score=score, rating=rating, month_ago=month_ago)


def spy(ret_pct: float, off_high: float = 0.0) -> IndexQuote:
    start = 100.0
    end = start * (1 + ret_pct / 100)
    peak = end / (1 + off_high / 100)
    spark = [start, (start + peak) / 2, peak, end]
    spark = [start + (end - start) * i / 20 for i in range(20)] if off_high == 0 else spark + [end] * 2
    return IndexQuote(symbol="SPY", name="S&P 500", price=end, spark=spark)


def vix(level: float) -> IndexQuote:
    return IndexQuote(symbol="^VIX", name="VIX", price=level)


def test_risk_on_and_risk_off() -> None:
    label, detail = market_regime(fg(78, "extreme greed", 60), [spy(4.0), vix(12.5)])
    assert label == "Risk-on: Extreme Greed"
    assert "CNN Fear & Greed 78 (Extreme Greed, up from 60 a month ago)" in detail and "VIX 12.5 (calm)" in detail
    label, detail = market_regime(fg(18, "extreme fear"), [spy(-6.0, -7.0), vix(31.0)])
    assert label == "Risk-off: Extreme Fear" and "VIX 31.0 (stressed)" in detail and "−6.0% over 1M" in detail


def test_sentiment_price_disagreements() -> None:
    assert market_regime(fg(30, "fear"), [spy(3.0), vix(16.0)])[0] == "Wall of worry: Fear, prices firm"
    assert market_regime(fg(72, "greed"), [spy(-2.5, -3.0), vix(17.0)])[0] == "Complacent: Greed, prices slipping"


def test_neutral_and_partial_and_missing() -> None:
    label, detail = market_regime(fg(31, "fear", 46), [spy(0.6, -0.5), vix(15.3)])
    assert label == "Neutral: Fear"
    assert "down from 46 a month ago" in detail and "0.5% below its 1M high" in detail
    assert market_regime(None, [vix(35.0)])[0] == "Risk-off"
    assert market_regime(None, []) == ("Unknown", "Market gauges are unavailable right now.")


def test_overview_scores_and_clusters_headlines() -> None:
    headlines = [
        raw("Stocks surge as inflation cools faster than expected", 2, "Reuters"),
        raw("Stocks surge as inflation cools faster than expected", 3, "CNBC"),
        raw("Inflation cools faster than expected, stocks surge to record", 4, "Bloomberg"),
        raw("Oil plunges as OPEC output dispute deepens", 5, "Reuters"),
        raw("Oil plunges further as OPEC output dispute deepens", 6, "MarketWatch"),
        raw("Is it time to buy small caps?", 7, "The Motley Fool"),
    ]
    trending = [TrendingTicker(symbol="NVDA", source="reddit", rank=1, mentions=120)]
    m = build_market_overview(fg(55, "neutral"), fg(70, "greed"), [spy(1.5), vix(14.0)], trending, headlines,
                              {"fear_greed": "ok", "headlines": "ok"}, NOW)
    assert m.generated_at == NOW and m.trending == trending and m.status["headlines"] == "ok"
    assert m.crypto_fear_greed.score == 70
    assert m.headlines.n == 5  # the syndicated copy collapsed
    heads = [n.headline for n in m.narratives]
    assert heads[0].startswith(("Stocks surge", "Inflation cools")) and m.narratives[0].label == "bullish"
    assert any(h.startswith("Oil plunges") for h in heads)
    assert not any(h.startswith("Is it time") for h in heads)


def test_overview_without_any_inputs() -> None:
    m = build_market_overview(None, None, [], [], [], {"fear_greed": "error: timeout"}, NOW)
    assert m.regime == "Unknown" and m.narratives == [] and m.headlines.n == 0
