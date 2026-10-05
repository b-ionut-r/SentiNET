"""Fetch/caching plumbing of app.intel.market_data (yfinance replaced by fixtures)."""
from __future__ import annotations

from datetime import datetime, UTC

import pytest

from app.core.http import UpstreamError
from app.intel import market_data as md
from app.intel import transforms as tx
from app.intel import sec
from app.schemas import InsiderView
from app.sources.base import CompanyRef
from tests.intel import helpers as fx

NOW = datetime(2026, 10, 4, 22, 0, tzinfo=UTC)


class YFRateLimitError(Exception):
    """Stand-in with yfinance's class name (classification is by name)."""


@pytest.fixture(autouse=True)
def _fixtures(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls: dict[str, int] = {}

    def count(name: str) -> None:
        calls[name] = calls.get(name, 0) + 1

    def fetch_info(sym: str):
        count("info")
        try:
            return fx.info(sym)
        except FileNotFoundError:
            raise RuntimeError("HTTP Error 404: Quote not found for symbol") from None

    def fetch_history(sym: str, period: str, interval: str, start):
        count(f"history:{period}:{interval}")
        kind = "daily" if interval == "1d" else "5m"
        return fx.bars(sym, kind), "USD"

    def fetch_analysts(sym: str):
        count("analysts")
        return fx.recommendations(sym), fx.upgrades(sym)

    def fetch_calendar(sym: str):
        count("calendar")
        return fx.calendar(sym)

    def fetch_earnings(sym: str):
        count("earnings")
        return fx.earnings_dates(sym), None

    def fetch_insiders(sym: str):
        count("insiders")
        return fx.insiders(sym)

    def fetch_indices(symbols: list[str]):
        count("indices")
        return fx.indices()

    monkeypatch.setattr(md, "_fetch_info", fetch_info)
    monkeypatch.setattr(md, "_fetch_history", fetch_history)
    monkeypatch.setattr(md, "_fetch_analyst_frames", fetch_analysts)
    monkeypatch.setattr(md, "_fetch_calendar", fetch_calendar)
    monkeypatch.setattr(md, "_fetch_earnings_frames", fetch_earnings)
    monkeypatch.setattr(md, "_fetch_insiders", fetch_insiders)
    monkeypatch.setattr(md, "_fetch_indices", fetch_indices)
    monkeypatch.setattr(md, "_now", lambda: NOW)
    return calls


async def test_quote_and_profile(_fixtures: dict[str, int]) -> None:
    q = await md.get_quote("NVDA")
    assert q and q.price == 233.95
    company = CompanyRef(ticker="NVDA", name="NVIDIA Corporation", short_name="Nvidia", cik="0001045810",
                         exchange="NASDAQ", website="https://www.nvidia.com")
    profile = await md.get_profile(company)
    assert profile and profile.short_name == "Nvidia" and profile.logo_url and profile.cik == "0001045810"
    assert _fixtures["info"] == 1  # one cached `info` payload serves quote + profile


async def test_quote_falls_back_to_bars_when_info_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def limited(sym: str):
        raise YFRateLimitError("Too Many Requests. Rate limited. Try after a while.")

    monkeypatch.setattr(md, "_fetch_info", limited)
    q = await md.get_quote("NVDA")
    assert q and q.price == pytest.approx(fx.bars("NVDA")["Close"].iloc[-1])


async def test_rate_limit_is_an_upstream_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def limited(sym: str):
        raise YFRateLimitError("Too Many Requests")

    monkeypatch.setattr(md, "_fetch_info", limited)
    with pytest.raises(UpstreamError, match="rate limit"):
        await md.get_info("AAPL")


async def test_unknown_symbol_is_none_not_error() -> None:
    assert await md.get_info("NOPE") is None


async def test_non_equities_skip_equity_intel(_fixtures: dict[str, int]) -> None:
    assert await md.get_analysts("BTC-USD", 1.0) is None
    assert await md.get_earnings("BTC-USD") is None
    assert await md.get_insiders("BTC-USD") is None
    assert await md.get_calendar_catalysts("BTC-USD") == []
    assert "analysts" not in _fixtures and "info" not in _fixtures  # crypto never hits Yahoo for these
    assert await md.get_analysts("SPY", 700.0) is None  # ETF: info says quoteType ETF
    assert await md.get_earnings("SPY") is None
    assert "analysts" not in _fixtures and "earnings" not in _fixtures


async def test_equity_intel(_fixtures: dict[str, int]) -> None:
    analysts = await md.get_analysts("NVDA", 233.95)
    assert analysts and analysts.consensus == "buy" and analysts.upside_pct == pytest.approx(40.07, abs=0.01)
    earnings = await md.get_earnings("NVDA")
    assert earnings and earnings.days_until == 44
    jpm_div = await md.get_calendar_catalysts("JPM")
    assert jpm_div and jpm_div[0].upcoming
    insiders = await md.get_insiders("SOFI")
    assert insiders and insiders.buys >= 1
    technicals = await md.get_technicals("NVDA")
    assert technicals and technicals.rsi_14 is not None
    closes = await md.get_daily_closes("NVDA", days=30)
    assert 15 <= len(closes) <= 23
    assert _fixtures["history:2y:1d"] == 1  # technicals + closes share one daily payload


async def test_insiders_fall_back_to_sec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(md, "_fetch_insiders", lambda sym: None)

    async def cik_map():
        return {"NVDA": ("0001045810", "NVIDIA CORP")}

    async def form4(cik: str, window_days: int = 180, max_filings: int = 15):
        assert cik == "0001045810"
        return InsiderView(buys=1, buy_value=1000.0, net_value=1000.0, ratio=1.0)

    monkeypatch.setattr(sec, "get_cik_map", cik_map)
    monkeypatch.setattr(sec, "get_form4_insiders", form4)
    view = await md.get_insiders("NVDA")
    assert view and view.buys == 1


async def test_insiders_error_surfaces_when_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def limited(sym: str):
        raise YFRateLimitError("Too Many Requests")

    async def cik_map():
        return {}

    monkeypatch.setattr(md, "_fetch_insiders", limited)
    monkeypatch.setattr(sec, "get_cik_map", cik_map)
    with pytest.raises(UpstreamError):
        await md.get_insiders("NVDA")


async def test_price_history_ranges(_fixtures: dict[str, int]) -> None:
    one_m = await md.get_price_history("NVDA", "1m")
    assert one_m.available and one_m.range == "1M" and one_m.interval == "1d" and 18 <= len(one_m.candles) <= 23
    one_y = await md.get_price_history("NVDA", "1Y")
    assert one_y.available and 240 <= len(one_y.candles) <= 254
    intraday = await md.get_price_history("NVDA", "1D")
    assert intraday.available and intraday.interval == "5m" and intraday.currency == "USD"
    bad = await md.get_price_history("NVDA", "10Y")
    assert not bad.available and bad.error and "unsupported" in bad.error
    assert _fixtures["history:2y:1d"] == 1  # 1M and 1Y are slices of one cached payload


async def test_price_history_error_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_a):
        raise YFRateLimitError("Too Many Requests")

    monkeypatch.setattr(md, "_fetch_history", broken)
    resp = await md.get_price_history("NVDA", "5D")
    assert not resp.available and resp.error and "rate limit" in resp.error


async def test_indices() -> None:
    quotes = await md.get_indices()
    assert {q.symbol for q in quotes} == set(md.INDEX_SYMBOLS)
    assert all(q.price and q.spark for q in quotes)


async def test_indices_use_the_24h_quote_when_a_crypto_bar_is_missing(
    _fixtures: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = fx.indices()
    holed = frame.drop(frame[("Close", "BTC-USD")].dropna().index[-2])
    monkeypatch.setattr(md, "_fetch_indices", lambda symbols: holed)
    btc = next(q for q in await md.get_indices() if q.symbol == "BTC-USD")
    expected = tx.quote_from_info(fx.info("BTC-USD"))
    assert expected is not None and btc.change_pct == expected.change_pct


async def test_crypto_one_day_return_survives_a_missing_bar(
    _fixtures: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    df = fx.bars("BTC-USD")
    monkeypatch.setattr(md, "_fetch_history", lambda sym, period, interval, start: (df.drop(df.index[-2]), "USD"))
    tech = await md.get_technicals("BTC-USD")
    expected = tx.quote_from_info(fx.info("BTC-USD"))
    assert tech is not None and expected is not None and tech.return_1d == round(expected.change_pct, 2)
