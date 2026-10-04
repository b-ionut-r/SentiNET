"""Keyless news sources: Google News, Bing News, Seeking Alpha, Nasdaq, Yahoo (offline, real fixtures)."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

from app.core.http import UpstreamError
from app.sources import bing_news, google_news, nasdaq, seeking_alpha, yahoo_news
from app.sources.query import search_terms
from tests.conftest import load_fixture, load_json_fixture
from tests.sources.conftest import company


def raw(name: str) -> bytes:
    return load_fixture(f"sources/{name}").encode("utf-8")


# --------------------------------------------------------------------------- Google News
def test_google_query_shapes():
    def q(ticker):
        return google_news.build_query(search_terms(company(ticker)), company(ticker))

    assert q("NVDA") == "(intitle:Nvidia OR intitle:NVDA) (stock OR shares OR investors OR analyst OR earnings)"
    # Everyday-word name: anchored on ticker / legal name, never on "stock" (avoids "price target" floods).
    assert q("TGT") == '(intitle:Target OR intitle:TGT) (TGT OR "Target Corporation")'
    assert q("SPY") == 'intitle:"S&P 500" (stocks OR index OR market OR ETF)'
    assert q("BTC-USD") == "(intitle:Bitcoin OR intitle:BTC)"
    assert q("SOFI").startswith("intitle:SoFi (")


def test_google_parse_splits_publisher_and_is_utc():
    signals = google_news.parse_feed(raw("google_news_nvda.xml"))
    assert len(signals) == 8  # 10 live items, 2 were Yahoo quote pages (NVDA.TO, NVD.F)
    assert not any("Quote" in s.title for s in signals)
    for s in signals:
        assert s.publisher and not s.title.endswith(f" - {s.publisher}")
        assert s.timestamp is not None and s.timestamp.tzinfo == timezone.utc
        assert s.url and s.url.startswith("https://news.google.com/")
        assert s.ticker_specific is False


def test_google_parse_drops_quote_and_option_chain_pages():
    signals = google_news.parse_feed(raw("google_news_tgt_listing_pages.xml"))
    titles = [s.title for s in signals]
    assert len(titles) == 6  # 9 items in the fixture, 3 are listing pages
    assert not any("Stock Chart" in t or "Earnings History" in t for t in titles)


@respx.mock
async def test_google_fetch_merges_fresh_and_week_and_tolerates_one_failure():
    feed = load_fixture("sources/google_news_nvda.xml")

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.params["q"].endswith("when:1d"):
            return httpx.Response(200, text=feed)
        return httpx.Response(200, text="<html>unexpected error page")

    respx.get(google_news.URL).mock(side_effect=route)
    batch = await google_news.GoogleNewsSource().fetch(company("NVDA"))
    assert len(batch.signals) == 8 and not batch.metrics
    times = [s.timestamp for s in batch.signals]
    assert times == sorted(times, reverse=True)


@respx.mock
async def test_google_fetch_raises_when_every_query_fails(monkeypatch):
    monkeypatch.setattr("app.core.http.asyncio.sleep", _no_sleep)  # skip the polite retry delay
    respx.get(google_news.URL).mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        await google_news.GoogleNewsSource().fetch(company("NVDA"))


# --------------------------------------------------------------------------- Bing News
def test_bing_unwraps_click_tracking_links():
    link = (
        "http://www.bing.com/news/apiclick.aspx?ref=FexRss&aid=&tid=x"
        "&url=https%3a%2f%2fwww.msn.com%2fen-us%2fmoney%2fstory%2far-AA1&c=1&mkt=en-us"
    )
    assert bing_news.unwrap_link(link) == "https://www.msn.com/en-us/money/story/ar-AA1"
    assert bing_news.unwrap_link("https://example.com/a") == "https://example.com/a"


def test_bing_pubdate_is_pacific_wall_time():
    # Labelled GMT, really PDT in October (UTC-7) and PST in January (UTC-8).
    assert bing_news.parse_pubdate("Sun, 04 Oct 2026 14:14:00 GMT") == datetime(2026, 10, 4, 21, 14, tzinfo=timezone.utc)
    assert bing_news.parse_pubdate("Mon, 05 Jan 2026 10:00:00 GMT") == datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc)


def test_bing_queries_are_always_multi_word():
    for ticker in ("NVDA", "TGT", "SPY", "BTC-USD", "XLK"):
        for query in bing_news.build_queries(search_terms(company(ticker))):
            assert len(query.split()) >= 2, query


def test_bing_parse_has_snippets_publishers_and_real_urls():
    signals = bing_news.parse_feed(raw("bing_news_nvda.xml"))
    assert len(signals) == 10
    assert all(s.url and "bing.com" not in s.url for s in signals)
    assert all(s.publisher and not s.publisher.endswith(" on MSN") for s in signals)
    assert sum(1 for s in signals if s.body) >= 8


@respx.mock
async def test_bing_retries_once_on_empty_body(monkeypatch):
    monkeypatch.setattr(bing_news.asyncio, "sleep", _no_sleep)
    feed = load_fixture("sources/bing_news_nvda.xml")
    route = respx.get(bing_news.URL).mock(
        side_effect=[httpx.Response(200, text=""), httpx.Response(200, text=feed), httpx.Response(200, text=feed)]
    )
    batch = await bing_news.BingNewsSource().fetch(company("NVDA"))
    assert route.call_count == 3 and len(batch.signals) == 10  # two queries, identical results deduped


async def _no_sleep(*_args, **_kwargs):
    return None


# --------------------------------------------------------------------------- Seeking Alpha
def test_seeking_alpha_parse_rebuilds_urls_and_is_honest_about_specificity():
    signals = seeking_alpha.parse_feed(raw("seeking_alpha_nvda.xml"), "NVDA")
    assert len(signals) == 10
    assert all(s.url and ("/news/" in s.url or "/article/" in s.url) for s in signals)
    specific = [s for s in signals if s.ticker_specific]
    assert 1 <= len(specific) < len(signals)
    assert all(s.extra["symbols"] == 1 for s in specific)
    assert {s.extra["type"] for s in signals} <= {"news", "analysis"}


def test_seeking_alpha_crypto_is_never_ticker_specific():
    signals = seeking_alpha.parse_feed(raw("seeking_alpha_btc.xml"), "BTC-USD", crypto=True)
    assert signals and not any(s.ticker_specific for s in signals)


def test_seeking_alpha_symbol_mapping():
    assert seeking_alpha.feed_symbol(company("BRK-B")) == "BRK.B"
    assert seeking_alpha.feed_symbol(company("BTC-USD")) == "BTC-USD"
    assert not seeking_alpha.SeekingAlphaSource().supports(company("SHOP.TO"))


@respx.mock
async def test_seeking_alpha_unknown_symbol_is_empty_not_error():
    respx.get(seeking_alpha.URL.format(symbol="NVDA")).mock(return_value=httpx.Response(404))
    batch = await seeking_alpha.SeekingAlphaSource().fetch(company("NVDA"))
    assert not batch


@respx.mock
async def test_seeking_alpha_fetch_drops_stale_items(frozen_now):
    respx.get(seeking_alpha.URL.format(symbol="NVDA")).mock(
        return_value=httpx.Response(200, content=raw("seeking_alpha_nvda.xml"))
    )
    batch = await seeking_alpha.SeekingAlphaSource().fetch(company("NVDA"))
    assert batch.signals and all((frozen_now - s.timestamp).days <= 14 for s in batch.signals)


# --------------------------------------------------------------------------- Nasdaq
def test_nasdaq_parse_marks_single_tag_items_specific():
    signals = nasdaq.parse_feed(raw("nasdaq_nvda.xml"), "NVDA")
    assert len(signals) == 10
    assert all(s.publisher for s in signals)
    for s in signals:
        assert s.ticker_specific == (s.extra["symbols"] == 1)


def test_nasdaq_generic_fallback_feed_is_discarded():
    # Unknown symbols get the site-wide "Latest Article Feed" — not about the symbol at all.
    assert nasdaq.parse_feed(raw("nasdaq_unknown_symbol.xml"), "ZZZQX") == []


def test_nasdaq_does_not_support_crypto_or_foreign_listings():
    src = nasdaq.NasdaqSource()
    assert not src.supports(company("BTC-USD")) and not src.supports(company("SHOP.TO"))
    assert src.supports(company("SPY"))


# --------------------------------------------------------------------------- Yahoo
def test_yahoo_parse_search_shape_drops_items_tagged_elsewhere():
    rows = load_json_fixture("sources/yahoo_search_news_nvda.json")
    signals = [s for s in (yahoo_news.parse_item(r, "NVDA") for r in rows) if s]
    assert len(signals) == len([r for r in rows if "NVDA" in r.get("relatedTickers", [])])
    assert all(s.publisher and s.timestamp and not s.ticker_specific for s in signals)


def test_yahoo_parse_get_news_shape():
    rows = load_json_fixture("sources/yahoo_get_news_shape_handmade.json")
    signals = [s for s in (yahoo_news.parse_item(r, "NVDA") for r in rows) if s]
    assert len(signals) == 1  # the AMD-tagged item is dropped
    assert signals[0].publisher == "Example Wire" and signals[0].body == "Synthetic summary."
    assert signals[0].timestamp == datetime(2026, 10, 4, 18, 0, tzinfo=timezone.utc)


async def test_yahoo_falls_back_to_get_news(monkeypatch, frozen_now):
    monkeypatch.setattr(yahoo_news, "_search_news", lambda symbol: [])
    monkeypatch.setattr(
        yahoo_news, "_ticker_news", lambda symbol: load_json_fixture("sources/yahoo_get_news_shape_handmade.json")
    )
    batch = await yahoo_news.YahooNewsSource().fetch(company("NVDA"))
    assert [s.title for s in batch.signals] == ["Example: Nvidia example headline"]


async def test_yahoo_reports_search_error_when_fallback_also_fails(monkeypatch):
    def search(symbol):
        raise UpstreamError("yahoo down")

    def ticker_news(symbol):
        raise RuntimeError("fallback down")

    monkeypatch.setattr(yahoo_news, "_search_news", search)
    monkeypatch.setattr(yahoo_news, "_ticker_news", ticker_news)
    with pytest.raises(UpstreamError, match="yahoo down"):
        await yahoo_news.YahooNewsSource().fetch(company("NVDA"))


def test_yahoo_query_plan_per_asset_class():
    assert yahoo_news.search_queries(company("NVDA")) == ["NVDA", "Nvidia stock"]
    assert yahoo_news.search_queries(company("BTC-USD")) == ["Bitcoin"]
    assert yahoo_news.search_queries(company("SPY")) == ["SPY", "S&P 500"]


async def test_yahoo_merges_both_searches_and_dedupes(monkeypatch, frozen_now):
    rows = load_json_fixture("sources/yahoo_search_news_nvda.json")
    seen: list[str] = []

    def search(query):
        seen.append(query)
        return rows

    monkeypatch.setattr(yahoo_news, "_search_news", search)
    batch = await yahoo_news.YahooNewsSource().fetch(company("NVDA"))
    assert sorted(seen) == ["NVDA", "Nvidia stock"]
    assert len(batch.signals) == len({r["link"] for r in rows if "NVDA" in r.get("relatedTickers", [])})


def test_yahoo_theme_searches_do_not_require_the_fund_tag():
    row = {"title": "S&P 500 hits a record", "link": "u", "publisher": "X", "relatedTickers": ["^GSPC"]}
    assert yahoo_news.parse_item(row, "SPY", require_tag=True) is None
    assert yahoo_news.parse_item(row, "SPY", require_tag=False) is not None


@respx.mock
async def test_bing_discards_generic_trending_fallback_pages():
    generic = (
        '<?xml version="1.0" encoding="utf-8" ?><rss version="2.0"><channel><title>x</title>'
        "<item><title>Jaguars beat Bengals in Week 4</title><link>https://example.com/a</link>"
        "<pubDate>Sun, 04 Oct 2026 14:14:00 GMT</pubDate></item></channel></rss>"
    )
    respx.get(bing_news.URL).mock(return_value=httpx.Response(200, text=generic))
    assert not await bing_news.BingNewsSource().fetch(company("NVDA"))
