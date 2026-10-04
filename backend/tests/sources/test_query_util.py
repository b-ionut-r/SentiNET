"""Search-term construction and shared helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.core.http import UpstreamError
from app.sources.base import RawSignal
from app.sources.query import (
    clean_name,
    etf_theme,
    is_word_ticker,
    or_group,
    search_terms,
    us_symbol,
)
from app.sources.util import (
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


@pytest.mark.parametrize(("symbol", "word"), [("NOW", True), ("F", True), ("SPY", True), ("NVDA", False)])
def test_word_tickers(symbol, word):
    assert is_word_ticker(symbol) is word


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
    assert parse_rfc822("Sun, 04 Oct 2026 17:16:51 -0400") == datetime(2026, 10, 4, 21, 16, 51, tzinfo=timezone.utc)
    assert parse_iso("2026-10-04T21:40:51Z") == datetime(2026, 10, 4, 21, 40, 51, tzinfo=timezone.utc)
    assert parse_iso("2026-09-28T11:00:37.614626123Z").tzinfo == timezone.utc
    assert from_epoch(1791150051).tzinfo == timezone.utc
    assert parse_rfc822("garbage") is None and parse_iso(None) is None and from_epoch("x") is None


def test_is_recent():
    now = datetime(2026, 10, 4, tzinfo=timezone.utc)
    assert is_recent(None, now=now)
    assert is_recent(now - timedelta(days=13), now=now)
    assert not is_recent(now - timedelta(days=15), now=now)


@pytest.mark.parametrize(
    ("title", "junk"),
    [
        ("TGT Oct 2026 85.000 put (TGT261009P00085000) stock price, news, quote and history", True),
        ("SoFi Technologies, Inc. (SOFI) Stock Price, News, Quote & History", True),
        ("TARGET CORP (TGT) Stock Chart", True),
        ("Target (TGT) Earnings History & Trends", True),
        ("SoFi Technologies (NASDAQ:SOFI) Stock Price Down 1.2% - Should You Sell?", False),
        ("Morgan Stanley lowers Apple stock price target on limited upside", False),
        ("Nvidia stock chart shows a breakout pattern", False),
    ],
)
def test_listing_pages_are_detected(title, junk):
    assert is_listing_page(title) is junk


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
