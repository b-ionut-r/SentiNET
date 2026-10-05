"""Analytics on real captured headlines with the real NLP modules (no fakes).

Uses textintel's hand-labeled Google News capture for NVDA (2026-10-04): the
'story' labels let us check that the top narratives are the stories a human
would pick. Skipped if the NLP package cannot be imported.
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import UTC, datetime

import pytest

from app.analytics.build import build_analysis
from app.sources.base import CompanyRef, RawSignal
from tests.analytics.factories import GOOGLE, STOCKTWITS, inputs, run
from tests.conftest import load_json_fixture

pytestmark = pytest.mark.real_nlp
pytest.importorskip("app.nlp.pipeline")
pytest.importorskip("app.nlp.narratives")

CAPTURED = datetime(2026, 10, 4, 23, 0, tzinfo=UTC)


def nvda_inputs():
    news = load_json_fixture("nlp/headlines_nvda.json")
    posts = load_json_fixture("nlp/stocktwits_nvda.json")
    c = news["company"]
    company = CompanyRef(ticker=c["ticker"], name=c["name"], short_name=c["short_name"], aliases=c["aliases"])
    headlines = [RawSignal(title=i["title"], publisher=i.get("publisher"),
                           timestamp=datetime.fromisoformat(i["timestamp"]), url=f"https://news.example/{n}")
                 for n, i in enumerate(news["items"])]
    social = [RawSignal(title=p["title"], publisher="StockTwits", timestamp=datetime.fromisoformat(
                  p["timestamp"].replace("Z", "+00:00")), user_label=p.get("user_label"),
                        ticker_specific=p.get("symbols") == ["NVDA"], extra={"symbols": len(p.get("symbols") or [])},
                        url=f"https://stocktwits.example/{n}")
              for n, p in enumerate(posts["items"])]
    stories = {f"https://news.example/{n}": i.get("story") for n, i in enumerate(news["items"])}
    built = inputs(company, [run(GOOGLE, headlines), run(STOCKTWITS, social)], now=CAPTURED)
    return built, stories


def test_real_headlines_produce_a_sound_analysis() -> None:
    built, stories = nvda_inputs()
    t0 = time.perf_counter()
    a = build_analysis(built)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0
    assert a.sentiment.n >= 60 and len(a.narratives) >= 3
    assert all(s.relevance >= 0.35 for s in a.signals)
    news = next(c for c in a.verdict.components if c.key == "news")
    assert news.available and news.score > 50  # a week of buyback and record-high coverage

    by_id = {s.id: s for s in a.signals}

    def dominant(n) -> str | None:
        labels = Counter(stories.get(by_id[i].url) for i in n.signal_ids if i in by_id)
        labels.pop("?", None)
        labels.pop(None, None)
        return labels.most_common(1)[0][0] if labels else None

    top3 = [dominant(n) for n in a.narratives[:3]]
    assert "buyback" in top3, top3
    buyback = next(n for n in a.narratives if dominant(n) == "buyback")
    assert buyback.count >= 5 and len(buyback.publishers) >= 4 and buyback.label == "bullish"


def test_real_gamestop_ebay_bid_is_a_deal_in_play() -> None:
    # Live GME (2026-10-04): the $56B eBay bid (~4.5x GameStop's cap) was the #2 story at impact 0.58,
    # with no insight; the real NLP keeps m_and_a only when GameStop is a party.
    from tests.analytics.factories import quote

    rows = [
        ("GME CEO Ryan Cohen May Reportedly Withdraw GameStop’s $56B eBay Bid — Here’s What He’s Considering Instead",
         "Stocktwits", "2026-10-04T12:35"),
        ("GME’s Ryan Cohen Isn’t Done Chasing eBay, Remains Committed To Cracking A Deal: Report",
         "Stocktwits", "2026-10-04T09:54"),
        ("GME Stock Rises After Hours — GameStop Shareholders Back Bigger Share Count To Support Proposed eBay "
         "Acquisition", "Stocktwits", "2026-10-02T22:22"),
        ("GME Reportedly Wants To Buy eBay But Retail Wonders How; eBay Stock Soars", "Stocktwits", "2026-10-03T00:29"),
        ("GameStop Steps Up EBAY Exposure To 6.5% As Ryan Cohen Pushes $56B Takeover Vision",
         "Stocktwits", "2026-10-02T23:06"),
        ("The Clock Is Ticking on GameStop’s eBay Acquisition Play as Warrants Near Expiration",
         "24/7 Wall St.", "2026-09-22T13:05"),
        ("GameStop CEO Cohen Continues Buying Spree With $10.6 Million Stock Purchase", "Barron's", "2026-10-02T17:43"),
        ("GameStop director Nat Turner buys $254,540 in stock", "Investing.com", "2026-10-02T02:23"),
    ]
    company = CompanyRef(ticker="GME", name="GameStop Corp.", short_name="GameStop")
    raws = [RawSignal(title=t, publisher=p, timestamp=datetime.fromisoformat(ts + ":00+00:00"), url=f"https://x/{n}")
            for n, (t, p, ts) in enumerate(rows)]
    a = build_analysis(inputs(company, [run(GOOGLE, raws)], quote=quote(price=24.7, market_cap=12.46e9),
                              now=datetime(2026, 10, 5, 6, 30, tzinfo=UTC)))
    deal = next(i for i in a.insights if i.kind == "deal")
    assert deal.severity == "alert" and deal.title == "Deal in play: $56B, 4.5× its market cap"
    assert "eBay" in deal.detail and "$10.6 Million" not in deal.detail  # it quotes a deal article
    assert a.verdict.headline.endswith("a $56B deal (4.5× its market cap) is in play.")
