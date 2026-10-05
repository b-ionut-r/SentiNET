"""Ticker normalization, company resolution and symbol search."""
from __future__ import annotations

import pytest

from app.core.http import UpstreamError
from app.intel import market_data, sec
from app.resolve import symbols
from app.resolve.symbols import (
    build_company_ref,
    logo_url_for,
    matches_from_sec,
    matches_from_yahoo,
    normalize_ticker,
    resolve_company,
)
from tests.intel.helpers import info, load_json


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("aapl", "AAPL"),
        ("  $nvda ", "NVDA"),
        ("#tsla", "TSLA"),
        ("brk.b", "BRK-B"),
        ("BF/B", "BF-B"),
        ("brk b", "BRK-B"),
        ("BRK-B", "BRK-B"),
        ("NASDAQ:AAPL", "AAPL"),
        ("nyse: brk.a", "BRK-A"),
        ("btc", "BTC-USD"),
        ("BTC.X", "BTC-USD"),
        ("btcusd", "BTC-USD"),
        ("eth/usd", "ETH-USD"),
        ("SOL-USD", "SOL-USD"),
        ("sol", "SOL-USD"),
        ("LTC", "LTC"),  # LTC Properties owns the bare symbol
        ("ltcusd", "LTC-USD"),
        ("LINK", "LINK"),
        ("^vix", "^VIX"),
        ("gc=f", "GC=F"),
        ("shop.to", "SHOP.TO"),
        ("bp.l", "BP.L"),  # .L is an exchange suffix, not a share class
        ("7203.T", "7203.T"),
        ("", None),
        ("   ", None),
        ("AAPL; DROP TABLE", None),
        ("<script>", None),
        ("1234", None),  # must contain a letter
        ("A" * 20, None),
    ],
)
def test_normalize_ticker(raw: str, expected: str | None) -> None:
    assert normalize_ticker(raw) == expected


def test_logo_urls() -> None:
    assert logo_url_for("NVDA") == "https://logos.stocktwits-cdn.com/NVDA.png"
    assert logo_url_for("BRK-B") == "https://logos.stocktwits-cdn.com/BRK.B.png"
    assert logo_url_for("BTC-USD") == "https://logos.stocktwits-cdn.com/BTC.X.png"
    assert logo_url_for("SHOP.TO", "https://www.shopify.com") == "https://www.google.com/s2/favicons?domain=shopify.com&sz=128"
    assert logo_url_for("^VIX") is None
    assert logo_url_for("") is None


def test_build_company_ref_from_real_payloads() -> None:
    nvda = build_company_ref("NVDA", info("NVDA"), ("0001045810", "NVIDIA CORP"))
    assert (nvda.name, nvda.short_name, nvda.exchange, nvda.cik) == ("NVIDIA Corporation", "Nvidia", "NASDAQ", "0001045810")
    assert nvda.sector == "Technology" and nvda.website == "https://www.nvidia.com"

    aapl = build_company_ref("AAPL", info("AAPL"), ("0000320193", "Apple Inc."))
    assert aapl.short_name == "Apple" and aapl.quote_type == "EQUITY"

    sofi = build_company_ref("SOFI", info("SOFI"), None)
    assert sofi.short_name == "SoFi" and "SoFi Technologies" in sofi.aliases

    spy = build_company_ref("SPY", info("SPY"), ("0000884394", "SPDR S&P 500 ETF TRUST"))
    assert spy.quote_type == "ETF" and spy.short_name == "S&P 500" and spy.exchange == "NYSE Arca"

    btc = build_company_ref("BTC-USD", info("BTC-USD"), None)
    assert btc.is_crypto and btc.short_name == "Bitcoin" and btc.name == "Bitcoin" and btc.cik is None
    assert btc.base_symbol == "BTC" and btc.cashtag == "$BTC"


def test_build_company_ref_without_yahoo_uses_sec_title() -> None:
    ref = build_company_ref("NVDA", None, ("0001045810", "NVIDIA CORP"))
    assert ref.short_name == "Nvidia" and ref.name == "Nvidia Corp" and ref.cik == "0001045810"
    bare = build_company_ref("ZZZZ", None, None)
    assert (bare.name, bare.short_name, bare.quote_type) == ("ZZZZ", "ZZZZ", "EQUITY")
    crypto = build_company_ref("ETH-USD", None, None)
    assert crypto.quote_type == "CRYPTOCURRENCY" and crypto.short_name == "Ethereum"


def test_yahoo_search_ranking() -> None:
    apple = matches_from_yahoo(load_json("yahoo/search_apple.json"), "apple", 8)
    assert apple[0].symbol == "AAPL"
    assert all(m.type in {"EQUITY", "ETF", "CRYPTOCURRENCY", "INDEX", "MUTUALFUND"} for m in apple)  # no futures
    assert not any(m.symbol.endswith("=F") for m in apple)
    us_first = [("." in m.symbol) for m in apple]
    assert us_first == sorted(us_first)  # foreign listings after US ones

    btc = matches_from_yahoo(load_json("yahoo/search_btc.json"), "btc", 8)
    assert btc[0].symbol == "BTC-USD"  # "btc" normalizes to bitcoin


def test_sec_search_fallback() -> None:
    cik_map = sec.cik_map_from_json(load_json("sec/company_tickers_sample.json"))
    assert matches_from_sec(cik_map, "nvda", 5)[0].symbol == "NVDA"
    by_name = matches_from_sec(cik_map, "berkshire", 5)
    assert by_name and by_name[0].symbol.startswith("BRK")
    assert matches_from_sec(cik_map, "", 5) == []


async def test_resolve_company_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_info(t: str):
        return info("NVDA")

    async def fake_map():
        return {"NVDA": ("0001045810", "NVIDIA CORP")}

    monkeypatch.setattr(market_data, "get_info", fake_info)
    monkeypatch.setattr(sec, "get_cik_map", fake_map)
    ref = await resolve_company("$nvda")
    assert ref.ticker == "NVDA" and ref.short_name == "Nvidia" and ref.cik == "0001045810"


async def test_resolve_company_degrades_when_yahoo_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_info(t: str):
        raise UpstreamError("Yahoo rate limit (quote)")

    async def fake_map():
        return {"AAPL": ("0000320193", "Apple Inc.")}

    monkeypatch.setattr(market_data, "get_info", failing_info)
    monkeypatch.setattr(sec, "get_cik_map", fake_map)
    ref = await resolve_company("AAPL")
    assert ref.short_name == "Apple" and ref.cik == "0000320193"


async def test_resolve_company_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing(*_a):
        raise UpstreamError("down")

    monkeypatch.setattr(market_data, "get_info", failing)
    monkeypatch.setattr(sec, "get_cik_map", failing)
    ref = await resolve_company("QWERT")
    assert ref.ticker == "QWERT" and ref.short_name == "QWERT"


async def test_search_falls_back_to_sec(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(q: str, limit: int):
        raise RuntimeError("yahoo down")

    async def fake_map():
        return sec.cik_map_from_json(load_json("sec/company_tickers_sample.json"))

    monkeypatch.setattr(symbols, "_yahoo_search", boom)
    monkeypatch.setattr(sec, "get_cik_map", fake_map)
    res = await symbols.search_symbols("NVDA", 5)
    assert res and res[0].symbol == "NVDA"
    assert await symbols.search_symbols("   ") == []


def test_funds_carry_their_theme_as_alias() -> None:
    """Headlines say "S&P 500" / "regional banks", never the wrapper's name."""
    kre = build_company_ref("KRE", {"quoteType": "ETF", "longName": "SPDR S&P Regional Banking ETF"}, None)
    assert "regional banks" in kre.aliases
    spy = build_company_ref("SPY", {"quoteType": "ETF", "longName": "SPDR S&P 500 ETF Trust"}, ("0000884394", "SPDR"))
    assert spy.short_name == "S&P 500" and [a for a in spy.aliases if a.lower() == "s&p 500"] == []
    vix = build_company_ref("^VIX", {"quoteType": "INDEX", "longName": "CBOE Volatility Index"}, None)
    assert vix.short_name == "VIX" and vix.cik is None


def test_search_drops_collision_numbered_crypto_tokens() -> None:
    quotes = [
        {"symbol": "ETH-USD", "shortname": "Ethereum USD", "quoteType": "CRYPTOCURRENCY", "exchange": "CCC"},
        {"symbol": "USDE29470-USD", "shortname": "Ethena USDe USD", "quoteType": "CRYPTOCURRENCY", "exchange": "CCC"},
    ]
    assert [m.symbol for m in matches_from_yahoo(quotes, "eth", 8)] == ["ETH-USD"]
