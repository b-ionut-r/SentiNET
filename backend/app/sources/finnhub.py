"""Finnhub company news (free key) — ticker-keyed headlines with summaries.

Why trusted: Finnhub maps articles to companies itself (`related`), so items are
ticker-specific by construction, with a clean publisher and summary.
Limits: free tier = 60 calls/min, North-American equities; very liquid names
return hundreds of items per week (we keep the newest 80).
Docs: https://finnhub.io/docs/api/company-news (key: https://finnhub.io/register)
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import httpx

from app.config import settings
from app.core import http
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import us_symbol
from app.sources.util import (
    clean_text,
    dedupe,
    from_epoch,
    is_listing_page,
    newest_first,
    sanitized_error,
    utc_now,
)

URL = "https://finnhub.io/api/v1/company-news"
LOOKBACK = timedelta(days=7)
MAX_ITEMS = 80


def parse_items(rows: Any) -> list[RawSignal]:
    out: list[RawSignal] = []
    for row in rows if isinstance(rows, list) else []:
        title = clean_text(row.get("headline"))
        if not title or is_listing_page(title):
            continue
        out.append(
            RawSignal(
                title=title,
                body=clean_text(row.get("summary"), limit=600) or None,
                url=row.get("url") or None,
                publisher=clean_text(row.get("source")) or None,
                timestamp=from_epoch(row.get("datetime")),
                ticker_specific=True,
                extra={"category": row.get("category")} if row.get("category") else {},
            )
        )
    return out


class FinnhubSource:
    key = "finnhub"
    label = "Finnhub"
    kind: SignalKind = "news"
    weight = 1.15
    requires_key = True
    description = "Finnhub company news: ticker-mapped headlines with summaries (free API key)."
    docs_url: str | None = "https://finnhub.io/docs/api/company-news"

    def configured(self) -> bool:
        return bool(settings.finnhub_api_key)

    def supports(self, company: CompanyRef) -> bool:
        return company.quote_type == "EQUITY" and us_symbol(company) is not None

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        symbol = us_symbol(company)
        if symbol is None or not self.configured():
            return SourceBatch()
        today = utc_now().date()
        params = {
            "symbol": symbol,
            "from": (today - LOOKBACK).isoformat(),
            "to": today.isoformat(),
            "token": settings.finnhub_api_key,
        }
        try:
            rows = await http.fetch_json(URL, params=params)
        except (httpx.HTTPError, http.UpstreamError) as exc:
            raise sanitized_error(exc, "finnhub") from None
        if isinstance(rows, dict) and rows.get("error"):
            raise http.UpstreamError(f"finnhub: {str(rows['error'])[:120]}")
        return SourceBatch(signals=newest_first(dedupe(parse_items(rows)))[:MAX_ITEMS])
