"""Marketaux news (free key) — entity-tagged financial news from 5,000+ sources.

Why trusted: Marketaux runs entity recognition over each article and returns,
per matched symbol, where it was found (title vs. body) and its own entity
sentiment. An item is `ticker_specific` when the symbol is highlighted in the
*title*. The provider's entity sentiment is kept in `extra["provider_score"]`
for reference only (the pipeline scores text itself).
Limits: free plan = 100 requests/day and 3 articles per request (we read 2
pages), so results are cached per symbol for 1 h and pages are counted against a
local daily budget; when it is spent the source says so instead of failing calls.
Docs: https://www.marketaux.com/documentation (key: https://www.marketaux.com/register)
"""
from __future__ import annotations

import copy
from datetime import timedelta
from typing import Any

import httpx

from app.config import settings
from app.core import http
from app.core.cache import cached
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import us_symbol
from app.sources.util import (
    DailyBudget,
    clean_text,
    dedupe,
    domain_of,
    gather_partial,
    is_listing_page,
    newest_first,
    parse_iso,
    sanitized_error,
    utc_now,
)

URL = "https://api.marketaux.com/v1/news/all"
PAGES = 2
LOOKBACK = timedelta(days=7)
CACHE_TTL = 3600
BUDGET = DailyBudget(100)


def parse_items(payload: Any, symbol: str) -> list[RawSignal]:
    rows = payload.get("data") if isinstance(payload, dict) else None
    out: list[RawSignal] = []
    for row in rows or []:
        title = clean_text(row.get("title"))
        if not title or is_listing_page(title):
            continue
        entity: dict[str, Any] = next(
            (e for e in row.get("entities") or [] if str(e.get("symbol", "")).upper() == symbol), {}
        )
        in_title = any(h.get("highlighted_in") == "title" for h in entity.get("highlights") or [])
        extra: dict[str, Any] = {}
        if entity and isinstance(entity.get("sentiment_score"), (int, float)):
            extra["provider_score"] = float(entity["sentiment_score"])
        source = clean_text(row.get("source")) or domain_of(row.get("url"))
        out.append(
            RawSignal(
                title=title,
                body=clean_text(row.get("description") or row.get("snippet"), limit=600) or None,
                url=row.get("url") or None,
                publisher=source or None,
                timestamp=parse_iso(row.get("published_at")),
                ticker_specific=in_title,
                extra=extra,
            )
        )
    return out


async def _page(symbol: str, page: int) -> Any:
    if not BUDGET.take():
        raise http.UpstreamError(f"marketaux: daily free quota ({BUDGET.limit} requests) used up; resets 00:00 UTC")
    params = {
        "symbols": symbol,
        "filter_entities": "true",
        "language": "en",
        "published_after": (utc_now() - LOOKBACK).strftime("%Y-%m-%dT%H:%M"),
        "page": page,
        "api_token": settings.marketaux_api_key,
    }
    try:
        return await http.fetch_json(URL, params=params)
    except (httpx.HTTPError, http.UpstreamError) as exc:
        raise sanitized_error(exc, "marketaux") from None


@cached(ttl=CACHE_TTL, none_ttl=60, maxsize=256)
async def load_news(symbol: str) -> SourceBatch:
    pages = await gather_partial(*(_page(symbol, n) for n in range(1, PAGES + 1)))
    signals = [s for payload in pages if payload for s in parse_items(payload, symbol)]
    return SourceBatch(signals=newest_first(dedupe(signals)))


class MarketauxSource:
    key = "marketaux"
    label = "Marketaux"
    kind: SignalKind = "news"
    weight = 1.1
    requires_key = True
    description = "Marketaux entity-tagged financial news from thousands of outlets (free API key)."
    docs_url: str | None = "https://www.marketaux.com/documentation"

    def configured(self) -> bool:
        return bool(settings.marketaux_api_key)

    def supports(self, company: CompanyRef) -> bool:
        return us_symbol(company) is not None

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        symbol = us_symbol(company)
        if symbol is None or not self.configured():
            return SourceBatch()
        return copy.deepcopy(await load_news(symbol))  # cached object: hand out a private copy
