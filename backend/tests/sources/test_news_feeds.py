"""Keyless news sources: Google News, Bing News, Seeking Alpha, Nasdaq, Yahoo (offline, real fixtures)."""
from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx
import pytest
import respx

from app.core.http import UpstreamError
from app.sources import bing_news, google_news, nasdaq, seeking_alpha, yahoo_news
from app.sources.base import CompanyRef
from app.sources.query import Mentions, issuer_symbols, search_terms
from tests.conftest import load_fixture, load_json_fixture
from tests.sources.conftest import CAPTURE_NOW, company


def raw(name: str) -> bytes:
    return load_fixture(f"sources/{name}").encode("utf-8")


# --------------------------------------------------------------------------- Google News
def test_google_query_shapes():
    def q(ticker):
        return [query.split(" -")[0] for query in google_news.build_queries(search_terms(company(ticker)), company(ticker))]

    finance = "(stock OR shares OR investors OR analyst OR earnings)"
    assert q("NVDA") == [f"intitle:Nvidia {finance} when:1d", f"intitle:Nvidia {finance} when:7d", "intitle:NVDA Nvidia when:7d"]
    # Everyday-word name: anchored on ticker / legal name / exact phrases, never on "stock" alone.
    # The ticker is a *separate* name-anchored query, never OR-ed into the headline subject
    # (that admitted "Sell Gold … TGT 147800" commodity calls).
    tgt = q("TGT")
    assert tgt[0] == 'intitle:Target (TGT OR "Target Corporation" OR "Target stock" OR "Target shares") when:1d'
    assert tgt[2] == "intitle:TGT Target when:7d"
    assert q("ICE")[2] == 'intitle:ICE "Intercontinental Exchange" when:7d'
    # Self-evident themes need no context words ("REITs": 68 -> 100 results live)…
    assert q("SPY") == ['intitle:"S&P 500" when:1d', 'intitle:"S&P 500" when:7d']
    # …word-like ones do; near-identical variants collapse ("Nasdaq 100" OR "Nasdaq-100" returned 0).
    qqq = CompanyRef(ticker="QQQ", name="Invesco QQQ Trust", short_name="Nasdaq 100", aliases=["Nasdaq-100"], quote_type="ETF")
    assert google_news.build_queries(search_terms(qqq), qqq)[0].startswith(
        "(intitle:\"Nasdaq 100\" OR intitle:Nasdaq) (stocks OR index OR market OR ETF) when:1d"
    )
    long_alias = CompanyRef(ticker="VNQ", name="Vanguard Real Estate Index Fund ETF Shares", short_name="Real Estate",
                            aliases=["Vanguard Real Estate Index Fund ETF Shares"], quote_type="ETF")  # fmt: skip
    assert "Vanguard" not in google_news.build_queries(search_terms(long_alias), long_alias)[0]
    assert q("BTC-USD")[0] == "intitle:Bitcoin when:1d"  # distinctive coin: no anchor needed
    assert q("TRUMP-USD")[0].startswith('(intitle:"Trump memecoin" OR intitle:"Trump meme coin"')
    avax = CompanyRef(ticker="AVAX-USD", name="Avalanche USD", short_name="Avalanche", quote_type="CRYPTOCURRENCY")
    assert google_news.build_queries(search_terms(avax), avax)[0].startswith("intitle:Avalanche (crypto OR token OR coin OR price)")


@respx.mock
async def test_google_drops_lowercase_homonyms_of_word_names():
    # Real titles returned for the TGT name query on 2026-10-04: Google matches intitle:Target
    # case-insensitively, so Korean broker notes ("target stock price") and sports "target" leak in.
    titles = [
        "Target (TGT) Surges 21.1% in Q3 Amid Consumer Shift; Dividend Sustainability in Focus",
        "Hana Securities raised its target stock price from 350,000 won to 400,000 won on the 2nd",
        "Rangers target shares untold story behind summer transfer collapse",
        "Bank of America halves FICO price target, stock falls below $600 amid mortgage scoring changes",
        "TGT Stock Has Rallied Over 50% This Year: Why Does TD Cowen See More Upside For The Retail Giant?",
    ]
    items = "".join(
        f"<item><title>{t} - Example</title><link>https://news.google.com/{i}</link>"
        f"<pubDate>Sun, 04 Oct 2026 1{i}:00:00 GMT</pubDate></item>"
        for i, t in enumerate(titles)
    )
    feed = f'<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>{items}</channel></rss>'
    respx.get(google_news.URL).mock(return_value=httpx.Response(200, text=feed.replace("&", "&amp;")))
    batch = await google_news.GoogleNewsSource().fetch(company("TGT"))
    assert [s.title for s in batch.signals] == [titles[4], titles[0]]


@respx.mock
async def test_google_drops_exchange_tag_only_matches_for_index_funds():
    """QQQ's `intitle:Nasdaq` query: 21 of 100 live results (2026-10-05) named Nasdaq only as a listing."""
    titles = [
        "Nasdaq Hits Record High as Treasury Yields Retreat",
        'Sprouts Farmers Market, Inc. (NASDAQ:SFM) Stock Now Rated "Hold" by Sell-Side Analysts',
        "Citigroup (NYSE: C) Doubles Price Target On Strategy (NASDAQ: MSTR) To $240 As Stock Surges 30% In A Month",
        "Invesco QQQ (NASDAQ:QQQ) Sets New 12-Month High - Here's What Happened",
        "NeuroSense Provides Update on Nasdaq Listing Compliance",
        "Nasdaq 100 Forecast: NDX rises as oil eases and tech gains",
    ]
    items = "".join(
        f"<item><title>{t} - Example</title><link>https://news.google.com/{i}</link>"
        f"<pubDate>Mon, 05 Oct 2026 0{i}:00:00 GMT</pubDate></item>"
        for i, t in enumerate(titles)
    )
    feed = f'<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>{items}</channel></rss>'
    respx.get(google_news.URL).mock(return_value=httpx.Response(200, text=feed.replace("&", "&amp;")))
    qqq = CompanyRef(ticker="QQQ", name="Invesco QQQ Trust", short_name="Nasdaq 100",
                     aliases=["Nasdaq-100", "Invesco QQQ", "Nasdaq"], quote_type="ETF")
    batch = await google_news.GoogleNewsSource().fetch(qqq)
    assert {s.title for s in batch.signals} == {titles[0], titles[3], titles[5]}


def test_google_queries_exclude_listing_pages_server_side():
    for query in google_news.build_queries(search_terms(company("NVDA")), company("NVDA")):
        assert query.endswith(google_news.EXCLUDE) and '-"quote & history"' in query


def test_google_parse_splits_publisher_and_is_utc():
    signals = google_news.parse_feed(raw("google_news_nvda.xml"))
    assert len(signals) == 8  # 10 live items, 2 were Yahoo quote pages (NVDA.TO, NVD.F)
    assert not any("Quote" in s.title for s in signals)
    for s in signals:
        assert s.publisher and not s.title.endswith(f" - {s.publisher}")
        assert s.timestamp is not None and s.timestamp.tzinfo == UTC
        assert s.url and s.url.startswith("https://news.google.com/")
        assert s.ticker_specific is False


def test_google_parse_drops_quote_and_option_chain_pages():
    signals = google_news.parse_feed(raw("google_news_tgt_listing_pages.xml"))
    titles = [s.title for s in signals]
    assert len(titles) == 5  # 9 items in the fixture, 4 are listing/forecast pages
    assert not any("Stock Chart" in t or "Earnings History" in t or "Should You Buy TGT" in t for t in titles)


@respx.mock
async def test_google_fetch_merges_queries_and_tolerates_failures():
    feed = load_fixture("sources/google_news_nvda.xml")

    def route(request: httpx.Request) -> httpx.Response:
        if "when:1d" in request.url.params["q"]:
            return httpx.Response(200, text=feed)
        return httpx.Response(200, text="<html>unexpected error page")

    route_mock = respx.get(google_news.URL).mock(side_effect=route)
    batch = await google_news.GoogleNewsSource().fetch(company("NVDA"))
    assert route_mock.call_count == 3  # 24 h + 7 d name queries + the ticker query
    assert len(batch.signals) == 8 and not batch.metrics
    times = [s.timestamp for s in batch.signals]
    assert times == sorted(times, reverse=True)


@respx.mock
async def test_google_slow_query_cannot_sink_the_others(monkeypatch):
    """A 503 + retry on one query used to run past the source's time box and lose every result."""
    import asyncio

    from app.config import settings

    monkeypatch.setattr(settings, "source_timeout", 1.0)
    feed = load_fixture("sources/google_news_nvda.xml")

    async def route(request: httpx.Request) -> httpx.Response:
        if "when:7d" in request.url.params["q"] and request.url.params["q"].startswith("intitle:Nvidia"):
            await asyncio.sleep(5)  # stalled upstream (or 503 + back-off + slow retry)
        return httpx.Response(200, text=feed)

    respx.get(google_news.URL).mock(side_effect=route)
    started = asyncio.get_running_loop().time()
    batch = await asyncio.wait_for(google_news.GoogleNewsSource().fetch(company("NVDA")), settings.source_timeout)
    assert asyncio.get_running_loop().time() - started < settings.source_timeout
    assert len(batch.signals) == 8  # the 24 h and ticker queries answered: their items survive


@respx.mock
async def test_google_every_query_timing_out_is_a_clear_error(monkeypatch):
    import asyncio

    from app.config import settings
    from app.core.http import UpstreamError

    monkeypatch.setattr(settings, "source_timeout", 1.0)

    async def stall(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, text="")

    respx.get(google_news.URL).mock(side_effect=stall)
    with pytest.raises(UpstreamError, match="timed out"):
        await google_news.GoogleNewsSource().fetch(company("NVDA"))


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
    assert bing_news.parse_pubdate("Sun, 04 Oct 2026 14:14:00 GMT") == datetime(2026, 10, 4, 21, 14, tzinfo=UTC)
    assert bing_news.parse_pubdate("Mon, 05 Jan 2026 10:00:00 GMT") == datetime(2026, 1, 5, 18, 0, tzinfo=UTC)
    # A real numeric offset is trusted as-is (no double shift).
    assert bing_news.parse_pubdate("Sun, 04 Oct 2026 14:14:00 -0700") == datetime(2026, 10, 4, 21, 14, tzinfo=UTC)


def test_bing_feed_falls_back_to_raw_times_if_correction_lands_in_the_future():
    feed = (
        b'<?xml version="1.0" encoding="utf-8" ?><rss version="2.0"><channel><title>x</title>'
        b"<item><title>Nvidia shares rise</title><link>https://example.com/a</link>"
        b"<pubDate>Sun, 04 Oct 2026 21:00:00 GMT</pubDate></item></channel></rss>"
    )
    now = datetime(2026, 10, 4, 21, 5, tzinfo=UTC)  # if Bing sent true UTC, +7 h would be in the future
    assert bing_news.parse_feed(feed, now=now)[0].timestamp == datetime(2026, 10, 4, 21, 0, tzinfo=UTC)
    later = datetime(2026, 10, 5, 6, 0, tzinfo=UTC)
    assert bing_news.parse_feed(feed, now=later)[0].timestamp == datetime(2026, 10, 5, 4, 0, tzinfo=UTC)


def test_bing_queries_are_always_multi_word():
    for ticker in ("NVDA", "TGT", "SPY", "BTC-USD", "XLK"):
        for query in bing_news.build_queries(search_terms(company(ticker))):
            assert len(query.split()) >= 2, query


def test_bing_parse_has_snippets_publishers_and_real_urls():
    signals = bing_news.parse_feed(raw("bing_news_nvda.xml"), now=CAPTURE_NOW)
    assert len(signals) == 10
    assert all(s.url and "bing.com" not in s.url for s in signals)
    assert all(s.publisher and not s.publisher.endswith(" on MSN") for s in signals)
    assert sum(1 for s in signals if s.body) >= 8


@respx.mock
async def test_bing_keeps_only_items_that_name_the_asset(frozen_now):
    # Real page for '"AT&T" shares' (2026-10-04): Bing half-ignored the query — celebrity
    # "shares her…" stories, NFL blogs. None names AT&T, so nothing survives.
    respx.get(bing_news.URL).mock(return_value=httpx.Response(200, content=raw("bing_news_att_shares.xml")))
    assert not await bing_news.BingNewsSource().fetch(company("T"))


@respx.mock
async def test_bing_drops_items_older_than_two_weeks(frozen_now):
    feed = load_fixture("sources/bing_news_nvda.xml").replace("Oct 2026", "Sep 2025")
    respx.get(bing_news.URL).mock(return_value=httpx.Response(200, text=feed))
    assert not await bing_news.BingNewsSource().fetch(company("NVDA"))


@respx.mock
async def test_bing_retries_once_on_empty_body(monkeypatch, frozen_now):
    monkeypatch.setattr(bing_news.asyncio, "sleep", _no_sleep)
    feed = load_fixture("sources/bing_news_nvda.xml")
    route = respx.get(bing_news.URL).mock(
        side_effect=[httpx.Response(200, text=""), httpx.Response(200, text=feed), httpx.Response(200, text=feed)]
    )
    batch = await bing_news.BingNewsSource().fetch(company("NVDA"))
    assert route.call_count == 3
    assert len(batch.signals) == 10  # two queries, identical results deduped; all 10 name Nvidia
    assert all(re.search(r"(?i)nvidia|nvda", f"{s.title} {s.body}") for s in batch.signals)


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


def test_seeking_alpha_share_class_siblings_count_as_the_issuer():
    # Real GOOGL feed: single-company Alphabet stories are tagged (GOOG, GOOGL).
    plain = seeking_alpha.parse_feed(raw("seeking_alpha_googl.xml"), "GOOGL")
    aware = seeking_alpha.parse_feed(raw("seeking_alpha_googl.xml"), "GOOGL", issuer=issuer_symbols(company("GOOGL")))
    assert not any(s.ticker_specific for s in plain)
    specific = [s.title for s in aware if s.ticker_specific]
    assert specific == ["Alphabet: Delivering And Getting Cheaper", "Google seeks EU court halt to search-data sharing order"]


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


def test_nasdaq_primary_tag_needs_the_headline_to_name_the_company():
    mentions = Mentions(search_terms(company("NVDA")))
    signals = {s.title: s.ticker_specific for s in nasdaq.parse_feed(raw("nasdaq_nvda.xml"), "NVDA", mentions=mentions)}
    assert signals["This Under-the-Radar Supplier Could Be Nvidia's Secret Weapon"] is True  # NVDA,NVDA,MU,ASML
    # "NVDA,NVDA,IONQ": primary tag, but a two-stock listicle that never names Nvidia.
    assert signals["2 Millionaire-Maker Quantum Computing Stocks to Buy Hand Over Fist"] is False
    assert signals["AMD Reaches a $1 Trillion Market Cap. Can It Finally Dethrone Nvidia?"] is False  # AMD primary


def test_nasdaq_share_class_siblings():
    googl = company("GOOGL")
    signals = nasdaq.parse_feed(raw("nasdaq_googl.xml"), "GOOGL", issuer_symbols(googl), Mentions(search_terms(googl)))
    specific = [s.title for s in signals if s.ticker_specific]
    assert any(t.startswith("Alphabet Reported Negative Free Cash Flow") for t in specific)  # GOOG,GOOG,GOOGL
    assert not any(t.startswith("What Is Market Cap?") for t in specific)  # GOOGL primary, headline doesn't name it


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
    assert signals[0].timestamp == datetime(2026, 10, 4, 18, 0, tzinfo=UTC)


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


def test_yahoo_share_class_tags_are_the_same_issuer():
    # Real yf.Search payloads: Yahoo tags every Alphabet story GOOG, never GOOGL.
    rows = load_json_fixture("sources/yahoo_search_news_googl.json")
    assert not any(yahoo_news.parse_item(r, {"GOOGL"}) for r in rows)
    kept = [yahoo_news.parse_item(r, issuer_symbols(company("GOOGL"))) for r in rows]
    assert all(kept) and len(kept) == 12


def test_yahoo_safety_valve_skips_the_tag_filter_when_it_would_gut_a_name_search(caplog):
    rows = load_json_fixture("sources/yahoo_search_news_alphabet.json")  # all tagged GOOG
    stream = load_json_fixture("sources/yahoo_search_news_nvda.json")
    # A hypothetical unknown sibling symbol: the name search would lose > 80%, so its tags are ignored...
    kept = yahoo_news.parse_results([stream, rows], {"ABCX"}, require_tag=True)
    assert len(kept) == len(rows)
    # ...but the ticker stream (query 0) is never exempt.
    assert yahoo_news.parse_results([rows], {"ABCX"}, require_tag=True) == []


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
    # Real stream: all 12 items are tagged NVDA, but 7 never name Nvidia (Motley Fool disclosure
    # tags on Pfizer/Netflix/dividend pieces) and are dropped.
    titles = [s.title for s in batch.signals]
    assert len(titles) == 5 and all(re.search(r"(?i)nvidia", t) for t in titles)
    assert "Is Netflix (NFLX) Stock a Buy?" not in titles


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
