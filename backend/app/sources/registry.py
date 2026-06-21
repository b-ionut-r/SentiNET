"""Source registry: enable/disable + concurrent fan-out with graceful degradation.

This is the heart of ingestion. Every enabled source is fetched concurrently
with a per-source timeout; a slow, dead or blocked source yields an error
status instead of failing the whole request.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.sources.apewisdom import ApeWisdomSource
from app.sources.base import RawSignal, Source
from app.sources.google_news import GoogleNewsSource
from app.sources.hackernews import HackerNewsSource
from app.sources.reddit import RedditSource
from app.sources.stocktwits import StockTwitsSource
from app.sources.tradestie import TradestieSource
from app.sources.yahoo import YahooNewsSource

logger = logging.getLogger(__name__)

# All registered sources. Order is the default display order.
ALL_SOURCES: list[Source] = [
    YahooNewsSource(),
    GoogleNewsSource(),
    RedditSource(),
    TradestieSource(),
    ApeWisdomSource(),
    StockTwitsSource(),
    HackerNewsSource(),
]


def enabled_sources() -> list[Source]:
    disabled = settings.disabled_source_set
    return [s for s in ALL_SOURCES if s.name not in disabled]


@dataclass
class SourceResult:
    source: Source
    signals: list[RawSignal]
    status: str  # "ok" | "empty" | "error" | "disabled"
    error: Optional[str] = None


async def _fetch_one(source: Source, ticker: str, company: Optional[str]) -> SourceResult:
    try:
        signals = await asyncio.wait_for(
            source.fetch(ticker, company), timeout=settings.source_timeout
        )
        status = "ok" if signals else "empty"
        return SourceResult(source=source, signals=signals, status=status)
    except asyncio.TimeoutError:
        logger.warning("Source %s timed out", source.name)
        return SourceResult(source, [], "error", "timeout")
    except Exception as exc:  # noqa: BLE001 - one source must never break the rest
        logger.warning("Source %s failed: %s", source.name, exc)
        return SourceResult(source, [], "error", str(exc)[:200])


async def gather_signals(ticker: str, company: Optional[str]) -> list[SourceResult]:
    sources = enabled_sources()
    results = await asyncio.gather(
        *(_fetch_one(s, ticker, company) for s in sources)
    )
    return list(results)
