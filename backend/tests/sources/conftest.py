"""Shared helpers for source tests.

Fixture provenance (tests/fixtures/sources/):
* REAL payloads captured live on 2026-10-04 and trimmed to ~10 items: google_news_*,
  bing_news_*, seeking_alpha_*, nasdaq_*, yahoo_search_news_nvda, hackernews_*,
  stocktwits_*, apewisdom_*, tradestie_* (two dates, evidencing the frozen sentiment),
  bluesky_nvda_cashtag / bluesky_sofi_name, alphavantage_news_aapl_demo (public
  "demo" key payload) and alphavantage_demo_key_refusal.
* HANDMADE (no key available), shaped per provider docs with obviously synthetic
  "Example: ..." text: *_handmade.json (Finnhub, Marketaux, Reddit, yfinance
  get_news shape, Bluesky session).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.config import settings
from app.core.http import close_client
from app.core.ratelimit import limiter
from app.sources.base import CompanyRef

# Fixed "now" close to the capture time so recency filters are deterministic.
CAPTURE_NOW = datetime(2026, 10, 4, 22, 30, tzinfo=UTC)


def company(ticker: str) -> CompanyRef:
    refs = {
        "NVDA": CompanyRef(ticker="NVDA", name="NVIDIA Corporation", short_name="Nvidia"),
        "AAPL": CompanyRef(ticker="AAPL", name="Apple Inc.", short_name="Apple"),
        "TSLA": CompanyRef(ticker="TSLA", name="Tesla, Inc.", short_name="Tesla"),
        "SOFI": CompanyRef(ticker="SOFI", name="SoFi Technologies, Inc.", short_name="SoFi"),
        "TGT": CompanyRef(ticker="TGT", name="Target Corporation", short_name="Target"),
        "GOOGL": CompanyRef(ticker="GOOGL", name="Alphabet Inc. Class A", short_name="Alphabet", aliases=["Google"]),
        "BRK-B": CompanyRef(ticker="BRK-B", name="Berkshire Hathaway Inc.", short_name="Berkshire Hathaway"),
        "SPY": CompanyRef(ticker="SPY", name="SPDR S&P 500 ETF Trust", short_name="SPDR S&P 500", quote_type="ETF"),
        "QQQ": CompanyRef(ticker="QQQ", name="Invesco QQQ Trust, Series 1", short_name="Invesco QQQ", quote_type="ETF"),
        "XLK": CompanyRef(
            ticker="XLK", name="Technology Select Sector SPDR Fund", short_name="Tech SPDR", quote_type="ETF"
        ),
        "BTC-USD": CompanyRef(ticker="BTC-USD", name="Bitcoin USD", short_name="Bitcoin", quote_type="CRYPTOCURRENCY"),
        "SHOP.TO": CompanyRef(ticker="SHOP.TO", name="Shopify Inc.", short_name="Shopify"),
        # As `resolve_company` returned them live on 2026-10-04:
        "ICE": CompanyRef(ticker="ICE", name="Intercontinental Exchange, Inc.", short_name="Intercontinental Exchange"),
        "MAR": CompanyRef(ticker="MAR", name="Marriott International, Inc.", short_name="Marriott"),
        "T": CompanyRef(ticker="T", name="AT&T Inc.", short_name="AT&T"),
        "UPS": CompanyRef(ticker="UPS", name="United Parcel Service, Inc.", short_name="UPS"),
        "MU": CompanyRef(ticker="MU", name="Micron Technology, Inc.", short_name="Micron", aliases=["Micron Technology"]),
        "GLD": CompanyRef(
            ticker="GLD", name="SPDR Gold Shares", short_name="Gold", aliases=["gold prices"], quote_type="ETF"
        ),
        "XLV": CompanyRef(
            ticker="XLV", name="State Street Health Care Select Sector SPDR ETF", short_name="Health Care", quote_type="ETF"
        ),
        "TRUMP-USD": CompanyRef(ticker="TRUMP-USD", name="TRUMP", short_name="TRUMP", quote_type="CRYPTOCURRENCY"),
        "^GSPC": CompanyRef(ticker="^GSPC", name="S&P 500", short_name="S&P 500", quote_type="INDEX"),
    }
    return refs[ticker]


@pytest.fixture(autouse=True)
async def _fresh_http_client(request, monkeypatch):
    """Each test gets its own shared httpx client (it binds to the creating loop).

    Offline tests also drop per-host politeness spacing, which only slows mocks down.
    """
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setattr(limiter, "_spacing", {})
    await close_client()
    yield
    await close_client()


@pytest.fixture
def no_keys(monkeypatch):
    for name in (
        "finnhub_api_key",
        "alphavantage_api_key",
        "marketaux_api_key",
        "reddit_client_id",
        "reddit_client_secret",
        "bluesky_handle",
        "bluesky_app_password",
        "disabled_sources",
    ):
        monkeypatch.setattr(settings, name, "")


@pytest.fixture
def frozen_now(monkeypatch):
    """Pin `utc_now()` (used by recency filters and query windows) to the capture time."""
    from app.sources import util

    monkeypatch.setattr(util, "utc_now", lambda: CAPTURE_NOW)
    for module in ("bluesky", "finnhub", "marketaux", "alphavantage", "tradestie"):
        monkeypatch.setattr(f"app.sources.{module}.utc_now", lambda: CAPTURE_NOW, raising=False)
    return CAPTURE_NOW
