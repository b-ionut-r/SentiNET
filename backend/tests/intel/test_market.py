"""Market-wide intel: Fear & Greed, trending tickers, merged headlines."""
from __future__ import annotations

from html import escape

from datetime import datetime, timedelta, UTC

import httpx
import pytest
import respx

from app.core.http import UpstreamError
from app.intel import market
from app.intel.market import (
    merge_headlines,
    parse_apewisdom,
    parse_cnn,
    parse_crypto_fng,
    parse_feed,
    parse_stocktwits_trending,
    stocktwits_symbol,
)
from app.sources.base import RawSignal
from tests.intel.helpers import load_json, load_text

CAPTURED = datetime(2026, 10, 4, 22, 40, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _frozen_clock_and_fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import http

    monkeypatch.setattr(market, "_now", lambda: CAPTURED)
    monkeypatch.setattr(http, "_retry_delay", lambda resp, attempt: 0.0)


def test_parse_cnn_fear_greed() -> None:
    fg = parse_cnn(load_json("market/cnn_fear_greed.json"))
    assert fg is not None and 0 <= fg.score <= 100
    assert fg.rating in {"Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"}
    assert fg.previous_close is not None and fg.week_ago is not None and fg.year_ago is not None
    assert [c.key for c in fg.components] == list(market.CNN_COMPONENTS)
    assert all(c.score is None or 0 <= c.score <= 100 for c in fg.components)
    assert len(fg.history) >= 200 and fg.history[0].t < fg.history[-1].t
    assert parse_cnn({}) is None and parse_cnn("nope") is None


def test_parse_crypto_fng() -> None:
    fg = parse_crypto_fng(load_json("market/crypto_fng.json"))
    assert fg is not None and 0 <= fg.score <= 100 and fg.rating
    assert fg.week_ago is not None and fg.month_ago is not None and fg.year_ago is not None
    assert len(fg.history) <= 366 and fg.history[-1].v == fg.score
    assert fg.history[-1].t - fg.history[-8].t == timedelta(days=7) and fg.history[-8].v == fg.week_ago
    assert parse_crypto_fng({"data": []}) is None


def test_parse_apewisdom() -> None:
    rows = parse_apewisdom(load_json("market/apewisdom.json"))
    assert len(rows) == 25 and rows[0].rank == 1 and rows[0].source == "reddit"
    assert all("&amp;" not in (r.name or "") for r in rows)  # HTML entities decoded
    for r in rows:
        if r.mentions is not None and r.mentions_prev and r.mentions_prev >= market.MIN_CHANGE_BASE:
            assert r.change_pct == pytest.approx((r.mentions - r.mentions_prev) / r.mentions_prev * 100, abs=0.1)
    by_symbol = {r.symbol: r for r in rows}
    assert by_symbol["META"].mentions_prev == 1 and by_symbol["META"].change_pct is None  # not "+1100%"
    assert by_symbol["TLT"].name == "iShares 20+ Year Trea…"  # custodian prefix dropped, cut marked
    assert by_symbol["SGOV"].name == "iShares 0-3 Month Treasury Bond ETF"


def test_reddit_board_drops_ticker_words() -> None:
    payload = {"results": [
        {"rank": 1, "ticker": "SPY", "name": "SPDR S&amp;P 500 ETF Trust", "mentions": 42, "mentions_24h_ago": 40},
        {"rank": 2, "ticker": "DTE", "name": "DTE Energy", "mentions": 17, "mentions_24h_ago": 24},
        {"rank": 3, "ticker": "MU", "name": "Micron Technology", "mentions": 32, "mentions_24h_ago": 16},
    ]}
    rows = parse_apewisdom(payload, limit=2, skip=market._crowd_word())
    assert [r.symbol for r in rows] == ["SPY", "MU"]  # "0 DTE" is options slang, not DTE Energy
    assert rows[0].name == "SPDR S&P 500 ETF Trust" and rows[1].change_pct == 100.0


def test_headline_filters_drop_junk_foreign_and_off_topic() -> None:
    def item(title: str, source: str = "", url: str = "https://x/1") -> str:
        src = f'<source url="https://s">{source}</source>' if source else ""
        return (f"<item><title>{title}</title><link>{url}</link>{src}"
                "<pubDate>Sun, 04 Oct 2026 18:00:00 GMT</pubDate></item>")

    rss = ('<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
           + item("Stock market dips 0.52% in holiday-shortened trading week - punchng.com", "punchng.com")
           + item("(AMHE) Stock Market Analysis (AMHE:CA) - news.stocktradersdaily.com", "news.stocktradersdaily.com")
           + item("상장폐지 stock market delistings nearly doubled - 매일경제", "매일경제")
           + item("S&amp;P 500 earnings season set to impress, Goldman says - Seeking Alpha", "Seeking Alpha")
           + "</channel></rss>")
    titles = [s.title for s in parse_feed(rss, "google_news", None, "any")]
    assert titles == ["S&P 500 earnings season set to impress, Goldman says"]
    top = ('<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
           + item("In crude Ohio rally speech, Trump says he may not help if Democrats win")
           + item("Chick-fil-A wants to stay a family business as it expands abroad")
           + item("OPEC+ agrees to keep November oil output targets steady")
           + "</channel></rss>")
    assert [s.title for s in parse_feed(top, "cnbc_top", "CNBC", "title")] == [
        "OPEC+ agrees to keep November oil output targets steady"]


def test_parse_stocktwits_trending() -> None:
    rows = parse_stocktwits_trending(load_json("market/stocktwits_trending.json"))
    assert rows and all(r.source == "stocktwits" for r in rows)
    assert [r.rank for r in rows] == sorted(r.rank for r in rows)
    assert all(not r.symbol.endswith(".X") for r in rows)
    assert stocktwits_symbol("BTC.X") == "BTC-USD" and stocktwits_symbol("T.TSX") == "T.TO"


@pytest.mark.parametrize("key", ["cnbc_top", "cnbc_finance", "marketwatch", "google_news", "bing_news", "bing_wallstreet"])
def test_parse_each_feed(key: str) -> None:
    url_pub = {k: (pub, filt) for k, _u, pub, filt in market.FEEDS}
    pub, filt = url_pub[key]
    sigs = parse_feed(load_text(f"market/feed_{key}.xml"), key, pub, filt)
    assert sigs, key
    for s in sigs:
        assert s.title and s.timestamp and s.timestamp.tzinfo is not None and s.publisher
        assert not s.ticker_specific and s.extra["feed"] == key
        assert "bing.com/news/apiclick" not in (s.url or "")
    if key == "google_news":
        assert all(not s.title.endswith(f" - {s.publisher}") for s in sigs)  # publisher split off
    if key.startswith("bing"):
        assert all(not (s.publisher or "").endswith(" on MSN") for s in sigs)
    if filt:
        assert all(not s.title.startswith(("I’m", "I'm", "My ")) for s in sigs)  # advice columns dropped


def test_merge_headlines_dedupes_and_orders() -> None:
    now = CAPTURED
    a = RawSignal(title="Stocks rally as yields fall", url="https://a/1", timestamp=now - timedelta(hours=1))
    dup = RawSignal(title="Stocks rally as yields fall!", url="https://b/2", timestamp=now - timedelta(hours=2))
    old = RawSignal(title="Fed minutes due next week", url="https://c/3", timestamp=now - timedelta(days=6))
    newer = RawSignal(title="Oil jumps on supply fears", url="https://d/4", timestamp=now - timedelta(minutes=5))
    merged = merge_headlines([[a, old], [dup, newer]], now=now)
    assert [s.title for s in merged] == ["Oil jumps on supply fears", "Stocks rally as yields fall"]


def test_merge_real_feeds() -> None:
    batches = [parse_feed(load_text(f"market/feed_{k}.xml"), k, pub, filt) for k, _u, pub, filt in market.FEEDS
               if k != "cnbc_economy"]
    merged = merge_headlines(batches, now=CAPTURED)
    assert 25 <= len(merged) <= 120
    keys = [market._dedupe_key(s.title) for s in merged]
    assert len(keys) == len(set(keys))
    assert [s.timestamp for s in merged] == sorted((s.timestamp for s in merged), reverse=True)


async def test_get_market_headlines_tolerates_dead_feeds() -> None:
    with respx.mock as mock:
        for key, url, *_ in market.FEEDS:
            if key == "cnbc_top":
                mock.get(url).mock(return_value=httpx.Response(200, text=load_text("market/feed_cnbc_top.xml")))
            else:
                mock.get(url).mock(return_value=httpx.Response(503))
        heads = await market.get_market_headlines()
    assert heads and all(h.publisher == "CNBC" for h in heads)


async def test_get_market_headlines_all_dead_raises() -> None:
    with respx.mock as mock:
        mock.route().mock(return_value=httpx.Response(503))
        with pytest.raises(UpstreamError):
            await market.get_market_headlines()


async def test_get_trending_partial_and_total_failure() -> None:
    with respx.mock as mock:
        mock.get(market.APEWISDOM_URL).mock(return_value=httpx.Response(200, json=load_json("market/apewisdom.json")))
        mock.get(market.STOCKTWITS_TRENDING_URL).mock(return_value=httpx.Response(403))
        rows = await market.get_trending()
    assert len(rows) == 25 and {r.source for r in rows} == {"reddit"}
    market.get_trending.cache.clear()  # type: ignore[attr-defined]
    with respx.mock as mock:
        mock.route().mock(return_value=httpx.Response(500))
        with pytest.raises(UpstreamError):
            await market.get_trending()


async def test_cnn_request_looks_like_cnn_dot_com() -> None:
    with respx.mock as mock:
        route = mock.get(market.CNN_URL).mock(return_value=httpx.Response(200, json=load_json("market/cnn_fear_greed.json")))
        fg = await market.get_cnn_fear_greed()
        sent = route.calls[0].request.headers
    assert fg is not None and sent["Origin"] == "https://www.cnn.com" and "Mozilla" in sent["User-Agent"]


def test_headline_filters_keep_us_market_news() -> None:
    rss = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
    <item><title>Bangladesh stock market turnover falls 36%</title><link>https://x/1</link>
      <pubDate>Sun, 04 Oct 2026 18:00:00 GMT</pubDate></item>
    <item><title>ASX set to rise as Wall Street rallies on softer jobs data</title><link>https://x/2</link>
      <pubDate>Sun, 04 Oct 2026 18:00:00 GMT</pubDate></item>
    <item><title>I'm 71 and still working. Am I doing the right thing with my stocks?</title><link>https://x/3</link>
      <pubDate>Sun, 04 Oct 2026 18:00:00 GMT</pubDate></item>
    <item><title>Supreme Court justice weighs retirement as Senate control hangs in balance</title>
      <link>https://x/4</link><pubDate>Sun, 04 Oct 2026 18:00:00 GMT</pubDate></item>
    <item><title>Treasury yields jump as Fed minutes loom</title><link>https://x/5</link>
      <pubDate>Sun, 04 Oct 2026 18:00:00 GMT</pubDate></item>
    </channel></rss>"""
    titles = [s.title for s in parse_feed(rss, "bing_news", None, True)]
    assert titles == ["ASX set to rise as Wall Street rallies on softer jobs data", "Treasury yields jump as Fed minutes loom"]


def _rss(*titles: tuple[str, str]) -> str:
    """Google-News-style RSS: (title, outlet) pairs."""
    items = "".join(f'<item><title>{escape(t)} - {src}</title><link>https://x/{i}</link>'
                    f'<source url="https://s">{src}</source>'
                    f"<pubDate>Sun, 04 Oct 2026 18:00:00 GMT</pubDate></item>" for i, (t, src) in enumerate(titles))
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>{items}</channel></rss>'


def test_curated_feeds_keep_us_movers_that_name_a_country() -> None:
    """Review sample: CNBC stories about oil, Lilly (Indianapolis) and Asia were dropped as "foreign"."""
    movers = ["Oil jumps 3% after Saudi Arabia signals deeper output cuts",
              "Eli Lilly shares rise as Indianapolis drugmaker raises outlook",
              "Japan's Nikkei plunges 5% as yen surges",
              "Bitcoin tumbles as South Korea bans crypto exchanges",
              "Trump threatens 50% tariffs on India over Russian oil",
              "My Biggest Warning For Anyone Who Owns The S&P 500"]
    feed = _rss(*((t, "CNBC") for t in movers)).replace(" - CNBC</title>", "</title>")
    assert [s.title for s in parse_feed(feed, "cnbc_finance", "CNBC", "any")] == movers


def test_search_feeds_drop_foreign_local_markets_listicles_and_unknown_outlets() -> None:
    feed = _rss(("Taiwan Stock Market Surges Over 1,000 Points", "Focus Taiwan"),
                ("Stock Market Drops by N813bn on Profit-taking in BUA Foods", "THISDAYLIVE"),
                ("New week stock market: Expecting a recovery from the support zone", "Laodong.vn"),
                ("3 Stocks to Buy and Hold Forever", "The Motley Fool"),
                ("Here's How Many Shares of VOO You'd Need for $500 in Monthly Dividends", "The Motley Fool"),
                ("Broadcom's 2028 Projection Makes the Stock a Screaming Buy", "The Motley Fool"),
                ("If a Stock Market Crash Is Coming, These 3 ETFs Could Be Smart Buys Right Now", "The Motley Fool"),
                ("SCHD ETF outperforms S&P 500 in 2026 with risin...", "Pluang"),
                ("How Proshares S&P 500 Ex-technology Etf (SPXT) Affects Rotational Strategy Timing",
                 "Stock Traders Daily"),
                ("Stock Market Today: Gift Nifty, US Jobs Data To Oil Prices For Sensex", "NDTV Profit"),
                ("Oil jumps as Saudi Arabia cuts output; stocks slip", "Reuters"),
                ("Stock market today: Dow, S&P 500 slip as Fed minutes loom", "Yahoo Finance"),
                ("Stocks waver as investors weigh jobs data", "Reuters"),
                ("Indiana manufacturers brace for tariff hit as stocks slide", "Reuters"))
    titles = [s.title for s in parse_feed(feed, "google_news", None, "any")]
    assert titles == ["Oil jumps as Saudi Arabia cuts output; stocks slip",
                      "Stock market today: Dow, S&P 500 slip as Fed minutes loom",
                      "Stocks waver as investors weigh jobs data",
                      "Indiana manufacturers brace for tariff hit as stocks slide"]
    assert market.foreign_local_market("India's Sensex hits record as foreign funds return")
    assert not market.foreign_local_market("Indianapolis drugmaker Lilly shares jump")


def test_merge_caps_each_outlet_from_search_feeds() -> None:
    now = CAPTURED
    fool = [RawSignal(title=f"Market story number {i} about stocks", url=f"https://f/{i}", publisher="The Motley Fool",
                      timestamp=now - timedelta(minutes=i), extra={"feed": "google_news"}) for i in range(8)]
    cnbc = [RawSignal(title=f"CNBC market wrap {i}", url=f"https://c/{i}", publisher="CNBC",
                      timestamp=now - timedelta(minutes=i), extra={"feed": "cnbc_top"}) for i in range(6)]
    merged = merge_headlines([cnbc, fool], now=now)
    assert sum(s.publisher == "The Motley Fool" for s in merged) == market.PER_PUBLISHER_CAP
    assert sum(s.publisher == "CNBC" for s in merged) == 6  # curated feeds are not capped
