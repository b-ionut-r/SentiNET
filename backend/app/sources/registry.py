"""Registry of every text/crowd source and its per-company availability.

Order matters only for display: news first (most trusted), then social, then
metrics-only crowd gauges. Dropped after live evaluation on 2026-10-04:
* Lemmy — "SoFi" matched "Sofia" posts, "Nvidia" returned Linux-driver threads,
  "$NVDA" one 207-day-old post; the few finance hits were reposted headlines
  already covered by the news sources.
* Mastodon tag timelines — #nvda is the NVDA screen reader, #spy is spy fiction,
  #aapl/#sofi had nothing newer than 6-11 days; no usable finance signal.
"""
from __future__ import annotations

from typing import Literal

from app.config import settings
from app.sources.alphavantage import AlphaVantageSource
from app.sources.apewisdom import ApeWisdomSource
from app.sources.base import CompanyRef, Source
from app.sources.bing_news import BingNewsSource
from app.sources.bluesky import BlueskySource
from app.sources.finnhub import FinnhubSource
from app.sources.google_news import GoogleNewsSource
from app.sources.hackernews import HackerNewsSource
from app.sources.marketaux import MarketauxSource
from app.sources.nasdaq import NasdaqSource
from app.sources.reddit import RedditSource
from app.sources.seeking_alpha import SeekingAlphaSource
from app.sources.stocktwits import StockTwitsSource
from app.sources.tradestie import TradestieSource
from app.sources.yahoo_news import YahooNewsSource

Availability = Literal["enabled", "disabled", "unconfigured", "unsupported"]

ALL_SOURCES: list[Source] = [
    # News
    GoogleNewsSource(),
    BingNewsSource(),
    YahooNewsSource(),
    SeekingAlphaSource(),
    NasdaqSource(),
    FinnhubSource(),
    MarketauxSource(),
    AlphaVantageSource(),
    # Social text
    StockTwitsSource(),
    RedditSource(),
    BlueskySource(),
    HackerNewsSource(),
    # Crowd metrics only
    ApeWisdomSource(),
    TradestieSource(),
]

_BY_KEY: dict[str, Source] = {s.key: s for s in ALL_SOURCES}


def all_sources() -> list[Source]:
    return list(ALL_SOURCES)


def get_source(key: str) -> Source | None:
    return _BY_KEY.get((key or "").strip().lower())


def availability(source: Source, company: CompanyRef | None = None) -> Availability:
    """Why a source will or won't run: user-disabled > missing key > asset not covered."""
    if source.key in settings.disabled_source_set:
        return "disabled"
    if not source.configured():
        return "unconfigured"
    if company is not None and not source.supports(company):
        return "unsupported"
    return "enabled"


def enabled_sources(company: CompanyRef) -> list[tuple[Source, str]]:
    """Every registered source with its status for this company (run only the "enabled" ones)."""
    return [(source, availability(source, company)) for source in ALL_SOURCES]
