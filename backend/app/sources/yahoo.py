"""Yahoo Finance via yfinance — company news headlines + price history.

yfinance is synchronous and network-bound, so every call is offloaded to a
thread to keep the async event loop responsive. This module also backs the
/api/price endpoint and ticker -> company-name resolution.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from app.sources.base import RawSignal

_RANGE_MAP = {
    "1D": ("1d", "5m"),
    "5D": ("5d", "30m"),
    "1M": ("1mo", "1d"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
}


class YahooNewsSource:
    name = "yahoo"
    label = "Yahoo Finance"
    kind = "news"
    weight = 1.1

    async def fetch(self, ticker: str, company: Optional[str]) -> list[RawSignal]:
        return await asyncio.to_thread(self._fetch_sync, ticker)

    def _fetch_sync(self, ticker: str) -> list[RawSignal]:
        news = yf.Ticker(ticker).news or []
        signals: list[RawSignal] = []
        for item in news:
            content = item.get("content", item)  # yfinance schema varies by version
            title = content.get("title") or item.get("title") or ""
            if not title:
                continue
            summary = content.get("summary") or content.get("description") or ""
            provider = ""
            prov = content.get("provider")
            if isinstance(prov, dict):
                provider = prov.get("displayName", "")
            provider = provider or item.get("publisher", "") or "Yahoo Finance"

            ts: Optional[datetime] = None
            pub = content.get("pubDate") or content.get("displayTime")
            if isinstance(pub, str):
                try:
                    ts = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                except ValueError:
                    ts = None
            elif item.get("providerPublishTime"):
                ts = datetime.fromtimestamp(
                    item["providerPublishTime"], tz=timezone.utc
                )

            link = None
            click = content.get("clickThroughUrl") or content.get("canonicalUrl")
            if isinstance(click, dict):
                link = click.get("url")
            link = link or item.get("link")

            signals.append(
                RawSignal(
                    text=title,
                    url=link,
                    author=provider,
                    timestamp=ts,
                    extra={"summary": summary},
                )
            )
        return signals


async def resolve_company(ticker: str) -> Optional[str]:
    return await asyncio.to_thread(_resolve_company_sync, ticker)


def _resolve_company_sync(ticker: str) -> Optional[str]:
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("shortName") or info.get("longName")
    except Exception:  # noqa: BLE001
        return None


async def fetch_price(ticker: str, rng: str = "1M") -> dict:
    return await asyncio.to_thread(_fetch_price_sync, ticker, rng)


def _fetch_price_sync(ticker: str, rng: str) -> dict:
    period, interval = _RANGE_MAP.get(rng.upper(), _RANGE_MAP["1M"])
    tk = yf.Ticker(ticker)
    hist = tk.history(period=period, interval=interval)
    points = []
    for idx, row in hist.iterrows():
        ts = idx.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        close = row.get("Close")
        if close is None or close != close:  # skip NaN
            continue
        points.append({"t": ts, "close": round(float(close), 4)})

    current = points[-1]["close"] if points else None
    fast = getattr(tk, "fast_info", {}) or {}
    prev_close = None
    try:
        prev_close = fast.get("previous_close") or fast.get("previousClose")
    except Exception:  # noqa: BLE001
        prev_close = None
    if prev_close is None and len(points) >= 2:
        prev_close = points[0]["close"]

    change = change_pct = None
    if current is not None and prev_close:
        change = round(current - prev_close, 4)
        change_pct = round((change / prev_close) * 100, 2)

    currency = None
    try:
        currency = fast.get("currency")
    except Exception:  # noqa: BLE001
        currency = None

    return {
        "current_price": current,
        "previous_close": round(float(prev_close), 4) if prev_close else None,
        "change": change,
        "change_pct": change_pct,
        "currency": currency,
        "points": points,
    }
