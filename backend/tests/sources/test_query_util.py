"""Search-term construction and shared helpers."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.core.http import UpstreamError
from app.sources.base import RawSignal
from app.sources.base import CompanyRef
from app.sources.query import (
    Mentions,
    clean_name,
    crowd_symbol_ambiguous,
    etf_theme,
    is_word_ticker,
    issuer_symbols,
    or_group,
    search_terms,
    us_symbol,
)
from app.sources.util import (
    DailyBudget,
    cap_per_author,
    clean_plain,
    clean_text,
    dedupe,
    from_epoch,
    gather_partial,
    is_listing_page,
    is_recent,
    parse_iso,
    parse_rfc822,
    parse_xml,
    sanitized_error,
)
from tests.sources.conftest import company


# --------------------------------------------------------------------------- query
@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Apple Inc.", "Apple"),
        ("The Boeing Company", "Boeing"),
        ("Alphabet Inc. Class A", "Alphabet"),
        ("Coca-Cola Company (The)", "Coca-Cola"),
        ("Tesla, Inc.", "Tesla"),
        ("Meta Platforms, Inc.", "Meta Platforms"),
        ("SoFi Technologies, Inc.", "SoFi Technologies"),
    ],
)
def test_clean_name(raw, clean):
    assert clean_name(raw) == clean


def test_equity_terms_use_brand_and_searchable_ticker():
    terms = search_terms(company("NVDA"))
    assert terms.names == ("Nvidia",)
    assert terms.symbol_searchable and not terms.ambiguous and terms.asset == "equity"
    assert "stock" in terms.context


def test_aliases_extend_names():
    assert search_terms(company("GOOGL")).names == ("Alphabet", "Google")


def test_ticker_equal_to_name_is_not_searched_separately():
    # Search engines are case-insensitive: SOFI == SoFi, so it adds noise (SoFi Stadium) not recall.
    assert search_terms(company("SOFI")).symbol_searchable is False


def test_everyday_word_names_are_ambiguous():
    terms = search_terms(company("TGT"))
    assert terms.ambiguous and terms.needs_context


def test_etf_searches_the_underlying_theme():
    spy = search_terms(company("SPY"))
    assert spy.names == ("S&P 500",) and spy.asset == "etf"
    assert spy.symbol_searchable is False  # "spy" is a word
    assert search_terms(company("QQQ")).names == ("Nasdaq 100", "Nasdaq")


def test_unknown_etf_theme_is_derived_from_fund_name():
    xlk = company("XLK")
    assert etf_theme(xlk) == ("tech stocks",)  # from the table
    xlk.ticker = "XLKX"  # not in the table -> derived from "Technology Select Sector SPDR Fund"
    assert etf_theme(xlk) == ("Technology stocks",)


def test_crypto_terms_strip_quote_currency():
    terms = search_terms(company("BTC-USD"))
    assert terms.names == ("Bitcoin",) and terms.symbol == "BTC" and terms.asset == "crypto"


def test_index_is_treated_as_a_theme_and_caret_symbol_never_searched():
    terms = search_terms(company("^GSPC"))
    assert terms.asset == "etf" and terms.names == ("S&P 500",) and not terms.symbol_searchable


@pytest.mark.parametrize(
    ("symbol", "word"), [("NOW", True), ("F", True), ("SPY", True), ("TRUMP", True), ("APE", True), ("NVDA", False)]
)
def test_word_tickers(symbol, word):
    assert is_word_ticker(symbol) is word


def test_crypto_table_names_and_ambiguity():
    trump = search_terms(company("TRUMP-USD"))  # Yahoo calls it just "TRUMP"
    assert trump.names[0] == "Trump memecoin" and not trump.symbol_searchable
    assert all(n in trump.self_evident for n in trump.names)
    avax = search_terms(CompanyRef(ticker="AVAX-USD", name="Avalanche USD", short_name="Avalanche", quote_type="CRYPTOCURRENCY"))
    assert avax.ambiguous and avax.names == ("Avalanche",) and not avax.self_evident  # needs crypto words
    unknown = search_terms(CompanyRef(ticker="KASX-USD", name="Kasx.dev", short_name="Kasx.dev", quote_type="CRYPTOCURRENCY"))
    assert unknown.names == ("Kasx",) and unknown.ambiguous  # unlisted coins: precision first


def test_themes_for_sector_funds_futures_and_mutual_funds():
    assert search_terms(company("XLV")).names[:2] == ("health care stocks", "healthcare stocks")
    gold = CompanyRef(ticker="GC=F", name="Gold Dec 26", short_name="Gold Dec 26", quote_type="FUTURE")
    terms = search_terms(gold)
    assert terms.asset == "etf" and terms.primary == "gold price" and "futures" in terms.context
    lumber = CompanyRef(ticker="LBS=F", name="Lumber Futures,Nov-2026", short_name="Lumber", quote_type="FUTURE")
    assert etf_theme(lumber) == ("Lumber prices",)
    vfiax = CompanyRef(ticker="VFIAX", name="Vanguard 500 Index Admiral", short_name="500 Admiral", quote_type="MUTUALFUND")
    assert search_terms(vfiax).primary == "S&P 500"
    reit = CompanyRef(ticker="IYRX", name="iShares U.S. Real Estate ETF", short_name="x", quote_type="ETF")
    assert etf_theme(reit) == ("U.S. Real Estate stocks",)  # derived themes always carry a finance noun
    assert search_terms(CompanyRef(ticker="^VIX", name="CBOE Volatility Index", short_name="VIX", quote_type="INDEX")).names[0] == "VIX"


def test_issuer_symbols_include_share_class_siblings():
    assert {"GOOG", "GOOGL"} <= issuer_symbols(company("GOOGL"))
    assert {"BRK-A", "BRK.A", "BRK-B", "BRK.B", "BRK/B"} <= issuer_symbols(company("BRK-B"))
    assert issuer_symbols(company("NVDA")) == {"NVDA"}


@pytest.mark.parametrize(
    ("symbol", "ambiguous"),
    [("YOU", True), ("ES", True), ("DTE", True), ("CD", True), ("HYSA", True), ("AI", True), ("ALL", True),
     ("MU", False), ("GM", False), ("KO", False), ("V", False), ("SPY", False), ("ARM", False), ("NVDA", False)],
)  # fmt: skip
def test_crowd_symbol_gate_is_about_words_not_length(symbol, ambiguous):
    assert crowd_symbol_ambiguous(symbol) is ambiguous


# Real texts from the 2026-10-04 live review (which the old filters let through) plus true positives.
@pytest.mark.parametrize(
    ("ticker", "text", "social", "expected"),
    [
        ("SOFI", "SoFi stadium is Star Trek and Allegiant is Star Wars", True, False),
        ("SOFI", "We lost this week… #SoFi #Rams", True, False),
        ("SOFI", "Purple Hat- SOFI TUKKER", True, False),
        ("SOFI", "Use my link to get up to $450 in cash bonuses with SoFi", True, False),
        ("SOFI", "SoFi Stock: Why the Opportunity May Be Too Good to Ignore", True, True),
        ("SOFI", "loading up on $sofi", True, True),
        ("AAPL", "your set up looks like a stock apple product", True, False),
        ("AAPL", "Big Apple marathon stock photo", True, False),
        ("AAPL", "Apple shares rise after iPhone launch", True, True),
        ("AAPL", "Back in Stock! Apple Desktop Bus ADB II Optical Mouse Retro Board #vintagecomputing", True, False),
        ("TGT", "Sell Gold Below 149300 SL ABOVE 150500 TGT 147800 - Axis Securities", False, False),
        ("TGT", "UP TGT 2026: Exam Pattern", False, False),
        ("TGT", "HSBC upgrades Target to Buy, raises PT to $190 (TGT:NYSE)", False, True),
        ("TGT", "Should You Be Adding Target (NYSE:TGT) To Your Watchlist Today?", False, True),
        ("ICE", "ICE shares about local detentions", False, False),
        ("ICE", "ICE stock rises after record futures volumes", False, True),
        ("TRUMP-USD", "New Poll Shows Trump's Approval at 27%", True, False),
        ("TRUMP-USD", "Best buds, #XI and #Trump", True, False),
        ("TRUMP-USD", "Trump memecoin holders get another dinner", True, True),
        ("GLD", "panning for gold in the creek", True, False),
        ("GLD", "Gold Tone Coffee Filter", True, False),
        ("GLD", "gold prices hit a record as the dollar slips", True, True),
        ("BTC-USD", "bitcoin is digital gold", True, True),
        ("TSLA", "Nikola Tesla – The Laboratory of Lightning", False, False),
        ("AAPL", "This Maine Apple Orchard Took The Top Spot In America", False, False),
        ("TGT", "Target market share gains for the retailer, says analyst", False, True),
        ("UPS", "the ups and downs of trading", False, False),
        ("UPS", "UPS shares fall after guidance cut", False, True),
    ],
)
def test_mentions_rules(ticker, text, social, expected):
    assert Mentions(search_terms(company(ticker))).about(text, social=social) is expected


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [("NVDA", "NVDA"), ("BRK-B", "BRK.B"), ("SPY", "SPY"), ("SHOP.TO", None), ("BTC-USD", None), ("^GSPC", None)],
)
def test_us_symbol(ticker, expected):
    assert us_symbol(company(ticker)) == expected


def test_or_group_quotes_phrases():
    assert or_group(["Nvidia"]) == "Nvidia"
    assert or_group(["Nvidia", "S&P 500"]) == '(Nvidia OR "S&P 500")'


# --------------------------------------------------------------------------- util
def test_clean_text_strips_html_entities_and_zero_width():
    raw = "<p>That&#x27;s&nbsp;<b>great</b>\u200b</p><p>x &lt; y</p>"
    assert clean_text(raw) == "That's great x < y"
    assert clean_text("&amp;#39;quoted&amp;#39;") == "'quoted'"  # double-escaped feeds
    assert clean_text("word " * 50, limit=20).endswith("…")


def test_date_parsers_return_utc():
    assert parse_rfc822("Sun, 04 Oct 2026 17:16:51 -0400") == datetime(2026, 10, 4, 21, 16, 51, tzinfo=UTC)
    assert parse_iso("2026-10-04T21:40:51Z") == datetime(2026, 10, 4, 21, 40, 51, tzinfo=UTC)
    assert parse_iso("2026-09-28T11:00:37.614626123Z").tzinfo == UTC
    assert from_epoch(1791150051).tzinfo == UTC
    assert parse_rfc822("garbage") is None and parse_iso(None) is None and from_epoch("x") is None


def test_is_recent():
    now = datetime(2026, 10, 4, tzinfo=UTC)
    assert is_recent(None, now=now)
    assert is_recent(now - timedelta(days=13), now=now)
    assert not is_recent(now - timedelta(days=15), now=now)


@pytest.mark.parametrize(
    ("title", "junk"),
    [
        # Real listing/converter pages seen in live results (2026-10-04).
        ("TGT Oct 2026 85.000 put (TGT261009P00085000) stock price, news, quote and history", True),
        ("SoFi Technologies, Inc. (SOFI) Stock Price, News, Quote & History", True),
        ("TARGET CORP (TGT) Stock Chart", True),
        ("Target (TGT) Earnings History & Trends", True),
        ("META 261016 650.00P (META261016P650000) Stock Options Chain | Quotes & News", True),
        ("META Stock Price Today (NASDAQ: META) — Live Chart & Levels", True),
        ("6600.0 CALL OPTION EXPIRING 18-MAR-2027 (ASX:XJOMC9)", True),
        ("Arsalan Official(@ArsalanOfficial)'s insights", True),
        ("GME Stock Price, Quote & Chart", True),
        ("GameStop (GME) Stock Price, Quote & Analysis", True),
        ("Gamestop Stock Price Forecast. Should You Buy GME?", True),
        ("GameStop Tokenized Stock (Robinhood) Price (GME/USD) Today | Live Price, Market Cap & Chart", True),
        ("Nvidia (NVDA): Company Profile, Stock Price, News, Rankings", True),
        ("GameStop Corporation (GME) Stock Forecasts", True),
        ("Convert 500 EUR (EUR) to APE (APE)", True),
        ("ApeCoin Price Today | Live APE Price, Chart & Market Data", True),
        ("symbol__ Stock Quote Price and Forecast", True),
        ("Greenland Energy Company Actuals & Estimates (NASDAQ:GLND)", True),
        # Real headlines that must survive.
        ("SoFi Technologies (NASDAQ:SOFI) Stock Price Down 1.2% - Should You Sell?", False),
        ("Morgan Stanley lowers Apple stock price target on limited upside", False),
        ("Nvidia stock chart shows a breakout pattern", False),
        ("Tesla stock price news today: shares surge after record deliveries", False),
        ("Nvidia Stock Price, News Recap: Shares hit record", False),
        ("Nvidia options chain shows heavy call buying ahead of earnings", False),
    ],
)
def test_listing_pages_are_detected(title, junk):
    assert is_listing_page(title) is junk


def test_clean_plain_keeps_angle_brackets_that_are_text():
    assert clean_plain("Shares <AAPL> rose; &amp; a<b and c>d") == "Shares <AAPL> rose; & a<b and c>d"
    assert clean_text("Shares <AAPL> rose <p>more</p>") == "Shares <AAPL> rose more"  # only real tags stripped


def test_parse_xml_refuses_entity_tricks_as_upstream_error():
    with pytest.raises(UpstreamError, match="forbidden"):
        parse_xml(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><x>&a;</x>', "x")


def test_daily_budget_resets_each_utc_day(monkeypatch):
    from app.sources import util

    day = [datetime(2026, 10, 4, 23, 0, tzinfo=UTC)]
    monkeypatch.setattr(util, "utc_now", lambda: day[0])
    budget = DailyBudget(2)
    assert budget.take() and budget.take() and not budget.take() and budget.remaining == 0
    day[0] += timedelta(hours=2)
    assert budget.remaining == 2 and budget.take()


def test_cap_per_author_collapses_templated_bot_posts():
    t0 = datetime(2026, 10, 4, tzinfo=UTC)
    bot = [RawSignal(title=f"ETH price update ${2400 + i}.5 (+{i}%)", author="bot", timestamp=t0 + timedelta(hours=i)) for i in range(6)]
    chatty = [RawSignal(title=f"take {c}", author="chatty", timestamp=t0 + timedelta(minutes=i)) for i, c in enumerate("abcd")]
    kept = cap_per_author([*bot, *chatty], cap=3)
    assert [s.author for s in kept].count("bot") == 1  # templated repeats collapse to the newest
    assert [s.author for s in kept].count("chatty") == 3


def test_dedupe_same_item_but_keep_syndication():
    a = RawSignal(title="Nvidia rises", url="u1", publisher="Reuters")
    same_url = RawSignal(title="Other", url="u1", publisher="X")
    same_pub = RawSignal(title="NVIDIA   rises!", url="u2", publisher="reuters")
    syndicated = RawSignal(title="Nvidia rises", url="u3", publisher="Yahoo Finance")
    assert dedupe([a, same_url, same_pub, syndicated]) == [a, syndicated]


def test_parse_xml_rejects_html_and_empty():
    with pytest.raises(UpstreamError):
        parse_xml(b"", "x")
    with pytest.raises(UpstreamError):
        parse_xml(b"<html><body>blocked", "x")


async def test_gather_partial_tolerates_some_failures_but_not_all():
    async def ok():
        return [1]

    async def boom():
        raise UpstreamError("down")

    assert await gather_partial(ok(), boom()) == [[1], None]
    with pytest.raises(UpstreamError):
        await gather_partial(boom(), boom())


def test_sanitized_error_never_echoes_the_url():
    request = httpx.Request("GET", "https://finnhub.io/api/v1/company-news?token=SECRET")
    response = httpx.Response(401, request=request)
    exc = httpx.HTTPStatusError("401 for url https://finnhub.io/...token=SECRET", request=request, response=response)
    err = sanitized_error(exc, "finnhub")
    assert "SECRET" not in str(err) and "401" in str(err) and "invalid API key" in str(err)
