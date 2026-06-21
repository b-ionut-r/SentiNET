"""Ticker -> company name resolution.

A small static map covers the most common tickers instantly (and offline);
anything else falls back to a cached yfinance lookup.
"""
from __future__ import annotations

from typing import Optional

from cachetools import TTLCache

STATIC_MAP: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "GOOG": "Alphabet", "AMZN": "Amazon", "META": "Meta Platforms",
    "TSLA": "Tesla", "NVDA": "NVIDIA", "AMD": "AMD", "NFLX": "Netflix",
    "INTC": "Intel", "GME": "GameStop", "AMC": "AMC Entertainment",
    "PLTR": "Palantir", "COIN": "Coinbase", "HOOD": "Robinhood",
    "DIS": "Disney", "BA": "Boeing", "JPM": "JPMorgan Chase",
    "BAC": "Bank of America", "WMT": "Walmart", "NKE": "Nike",
    "PYPL": "PayPal", "SOFI": "SoFi", "F": "Ford", "T": "AT&T",
    "BABA": "Alibaba", "UBER": "Uber", "SHOP": "Shopify",
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF",
}

_resolved: TTLCache = TTLCache(maxsize=512, ttl=24 * 3600)


async def get_company(ticker: str) -> Optional[str]:
    key = ticker.upper()
    if key in STATIC_MAP:
        return STATIC_MAP[key]
    if key in _resolved:
        return _resolved[key]
    # Lazy import to avoid pulling yfinance into light code paths.
    from app.sources.yahoo import resolve_company

    name = await resolve_company(key)
    if name:
        _resolved[key] = name
    return name
