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


def test_real_apple_headline_only_initiation_is_a_catalyst() -> None:
    # Live AAPL (2026-10-05): Citi's Buy/$365 initiation was detected in a Moomoo headline but appeared
    # nowhere; Morgan Stanley's target cut is in the ratings feed and must not be listed twice.
    from tests.analytics.factories import analysts as analyst_view
    from app.schemas import AnalystAction

    rows = [
        ("Citi Initiates Apple(AAPL.US) With Buy Rating, Announces Target Price $365", "Moomoo", "2026-10-03T16:37"),
        ("Morgan Stanley lowers Apple stock price target on limited upside", "Investing.com", "2026-10-01T12:38"),
        ("Apple stock gains 1.02 percent as Morgan Stanley trims target", "AD HOC NEWS", "2026-10-04T08:37"),
        ("Morgan Stanley Maintains Apple(AAPL.US) With Buy Rating, Cuts Target Price to $355", "Moomoo",
         "2026-10-01T13:30"),
        ("Apple: I Was Wrong, Margin Math Is Now In Its Favor (Rating Upgrade)", "Seeking Alpha", "2026-09-28T12:56"),
    ]
    company = CompanyRef(ticker="AAPL", name="Apple Inc.", short_name="Apple")
    raws = [RawSignal(title=t, publisher=p, timestamp=datetime.fromisoformat(ts + ":00+00:00"), url=f"https://x/{n}")
            for n, (t, p, ts) in enumerate(rows)]
    ms = AnalystAction(date=datetime(2026, 10, 1, tzinfo=UTC), firm="Morgan Stanley", action="main",
                       to_grade="Overweight", price_target=355.0, prior_target=360.0)
    a = build_analysis(inputs(company, [run(GOOGLE, raws)], analysts=analyst_view(actions=[ms]),
                              now=datetime(2026, 10, 5, 7, 0, tzinfo=UTC)))
    found = [c for c in a.catalysts if c.kind == "analyst" and "from the headlines" in (c.detail or "")]
    assert [c.title for c in found] == ["Citi Initiates Apple(AAPL.US) With Buy Rating, Announces Target Price $365"]
    assert found[0].polarity == "bull" and "by Citi · PT $365.00 · Moomoo" in (found[0].detail or "")


def test_real_bystander_headline_is_not_lifted_by_its_snippet() -> None:
    # Live LULU: title relevance 0.40 (a bystander of Nike's news), signal relevance 0.64 after the snippet lift.
    from app.analytics.narratives import MIN_RELEVANCE  # the story bar
    from app.analytics.prepare import prepare

    company = CompanyRef(ticker="LULU", name="lululemon athletica inc.", short_name="Lululemon")
    title = "Nike Sinks 8% as Weak Outlook and Layoffs Follow Revenue Miss; Lululemon and On Holding Remain Flat"
    raw = RawSignal(title=title, body="Lululemon stock slips as investors weigh Nike read-through  MarketWatch",
                    publisher="Yahoo Finance", timestamp=datetime(2026, 10, 2, 13, 1, tzinfo=UTC), url="https://x/1")
    own = RawSignal(title="Lululemon stock slips as investors weigh Nike read-through", publisher="MarketWatch",
                    timestamp=datetime(2026, 10, 2, 14, 0, tzinfo=UTC), url="https://x/2")
    items = {it.url: it for it in prepare(company, [run(GOOGLE, [raw, own])], datetime(2026, 10, 5, tzinfo=UTC)).items}
    assert items["https://x/1"].relevance <= 0.4 < MIN_RELEVANCE
    assert items["https://x/2"].relevance >= 0.8
