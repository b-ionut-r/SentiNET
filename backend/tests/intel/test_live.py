"""Live smoke tests against the real providers (deselected by default).

    cd backend && pytest -m live tests/intel/test_live.py -q

GDELT and Wikipedia aggressively rate-limit shared cloud IPs; those tests skip
(with the provider's refusal as the reason) instead of failing.
"""
from __future__ import annotations

import pytest

from app.core.http import UpstreamError
from app.intel import attention, gdelt, market, market_data, sec
from app.resolve.symbols import resolve_company, search_symbols

pytestmark = pytest.mark.live

TICKERS = ["NVDA", "AAPL", "SOFI", "SPY", "BTC-USD"]


@pytest.mark.parametrize("ticker", TICKERS)
async def test_resolve_quote_technicals(ticker: str) -> None:
    company = await resolve_company(ticker)
    assert company.short_name and company.short_name != ticker
    quote = await market_data.get_quote(ticker)
    assert quote and quote.price and quote.price > 0
    technicals = await market_data.get_technicals(ticker)
    assert technicals and technicals.return_1d is not None and technicals.rsi_14 is not None
    for rng in ("1D", "1M", "5Y"):
        price = await market_data.get_price_history(ticker, rng)
        assert price.available and price.candles, (ticker, rng, price.error)


@pytest.mark.parametrize("ticker", ["NVDA", "AAPL", "SOFI"])
async def test_equity_smart_money(ticker: str) -> None:
    company = await resolve_company(ticker)
    quote = await market_data.get_quote(ticker)
    analysts = await market_data.get_analysts(ticker, quote.price if quote else None)
    assert analysts and analysts.total > 0 and analysts.target_mean
    earnings = await market_data.get_earnings(ticker)
    assert earnings and earnings.history
    insiders = await market_data.get_insiders(ticker)
    assert insiders and insiders.transactions
    filings = await sec.get_filings(company)
    assert filings and any(f.form.startswith(("10-", "8-K")) for f in filings)


async def test_non_equities_have_no_equity_intel() -> None:
    assert await market_data.get_analysts("BTC-USD", 1.0) is None
    assert await market_data.get_earnings("SPY") is None
    assert await sec.get_filings(await resolve_company("SPY")) == []


async def test_search() -> None:
    assert (await search_symbols("nvidia"))[0].symbol == "NVDA"
    assert (await search_symbols("$aapl"))[0].symbol == "AAPL"


async def test_market_overview_inputs() -> None:
    indices = await market_data.get_indices()
    assert len(indices) >= 6 and all(i.spark for i in indices)
    fg = await market.get_cnn_fear_greed()
    assert fg and len(fg.components) == 7 and fg.history
    cfg = await market.get_crypto_fear_greed()
    assert cfg and cfg.history
    trending = await market.get_trending()
    assert trending
    headlines = await market.get_market_headlines()
    assert len(headlines) >= 10


async def test_gdelt_tone() -> None:
    company = await resolve_company("NVDA")
    try:
        trend = await gdelt.get_tone_trend(company)
    except UpstreamError as exc:
        pytest.skip(f"GDELT refused from this IP: {exc}")
    assert trend and trend.series and trend.tone_30d is not None


async def test_wikipedia_pageviews() -> None:
    company = await resolve_company("AAPL")
    try:
        views = await attention.get_wiki_pageviews(company, 90)
    except UpstreamError as exc:
        pytest.skip(f"Wikipedia refused from this IP: {exc}")
    assert views and len(views) >= 55


@pytest.mark.parametrize(
    ("ticker", "title"),
    [("AAPL", "Apple Inc."), ("TGT", "Target Corporation"), ("META", "Meta Platforms"), ("XYZ", "Block, Inc."),
     ("SNAP", "Snap Inc."), ("SOFI", "SoFi"), ("NVDA", "Nvidia"), ("BTC-USD", "Bitcoin"), ("SPY", "S&P 500")],
)
async def test_wikipedia_article_choice_for_homonyms(ticker: str, title: str) -> None:
    company = await resolve_company(ticker)
    try:
        page = await attention.find_article(company)
    except UpstreamError as exc:
        pytest.skip(f"Wikipedia refused from this IP: {exc}")
    assert page is not None and page["title"] == title
