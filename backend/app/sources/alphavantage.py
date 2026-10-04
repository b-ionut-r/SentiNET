"""Alpha Vantage NEWS_SENTIMENT (free key) — news plus the provider's ticker-level sentiment.

Why trusted: each article carries Alpha Vantage's per-ticker relevance and
sentiment. The article text goes through our own pipeline like any headline;
the provider's view is exposed separately as metric `av_sentiment` (relevance-
weighted mean of its per-ticker scores, -1..1) and `av_articles`, a useful
second opinion. Relevance below ~0.6 is mostly incidental mentions (13F filings,
"stocks to buy" lists), so only relevance >= 0.5 feeds the metric and only
>= 0.9 marks an item `ticker_specific`. Timestamps are UTC (cross-checked
against Google News pubDates on 2026-10-04).
Limits: free tier = 25 requests/day; throttling is signalled with HTTP 200 and
an "Information"/"Note" message, which we surface as an error.
Docs: https://www.alphavantage.co/documentation/#news-sentiment
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
    domain_of,
    is_listing_page,
    newest_first,
    sanitized_error,
    utc_now,
)

URL = "https://www.alphavantage.co/query"
LOOKBACK = timedelta(days=7)
METRIC_MIN_RELEVANCE = 0.5
SPECIFIC_MIN_RELEVANCE = 0.9


def av_symbol(company: CompanyRef) -> str | None:
    if company.is_crypto:
        return f"CRYPTO:{company.base_symbol.upper()}"
    return us_symbol(company)


def parse_time(value: str | None) -> datetime | None:
    """'20261004T151215' (UTC) -> datetime."""
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(value or "", fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_payload(payload: Any, symbol: str) -> SourceBatch:
    if not isinstance(payload, dict):
        raise http.UpstreamError("alphavantage: unexpected payload")
    if "feed" not in payload:
        message = str(payload.get("Information") or payload.get("Note") or payload.get("Error Message") or "")
        if message.lower().startswith("invalid inputs"):
            return SourceBatch()  # symbol not covered
        raise http.UpstreamError(f"alphavantage: {message[:140] or 'no feed in response'}")
    signals: list[RawSignal] = []
    weighted, weights = 0.0, 0.0
    for row in payload.get("feed") or []:
        title = clean_text(row.get("title"))
        if not title or is_listing_page(title):
            continue
        mine: dict[str, Any] = next(
            (t for t in row.get("ticker_sentiment") or [] if str(t.get("ticker", "")).upper() == symbol), {}
        )
        relevance, score = _float(mine.get("relevance_score")), _float(mine.get("ticker_sentiment_score"))
        if relevance is not None and score is not None and relevance >= METRIC_MIN_RELEVANCE:
            weighted += relevance * score
            weights += relevance
        extra: dict[str, Any] = {}
        if score is not None:
            extra["provider_score"] = score
        if relevance is not None:
            extra["provider_relevance"] = relevance
        signals.append(
            RawSignal(
                title=title,
                body=clean_text(row.get("summary"), limit=600) or None,
                url=row.get("url") or None,
                author=", ".join(row.get("authors") or []) or None,
                publisher=clean_text(row.get("source")) or domain_of(row.get("url")),
                timestamp=parse_time(row.get("time_published")),
                ticker_specific=(relevance or 0.0) >= SPECIFIC_MIN_RELEVANCE,
                extra=extra,
            )
        )
    metrics: dict[str, Any] = {}
    if weights > 0:
        metrics["av_sentiment"] = round(max(-1.0, min(1.0, weighted / weights)), 4)
        metrics["av_articles"] = sum(
            1 for s in signals if (s.extra.get("provider_relevance") or 0) >= METRIC_MIN_RELEVANCE
        )
    return SourceBatch(signals=newest_first(dedupe(signals)), metrics=metrics)


class AlphaVantageSource:
    key = "alphavantage"
    label = "Alpha Vantage"
    kind: SignalKind = "news"
    weight = 1.1
    requires_key = True
    description = "Alpha Vantage news with the provider's own ticker sentiment as a second opinion (free API key)."
    docs_url: str | None = "https://www.alphavantage.co/documentation/#news-sentiment"

    def configured(self) -> bool:
        return bool(settings.alphavantage_api_key)

    def supports(self, company: CompanyRef) -> bool:
        return av_symbol(company) is not None

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        symbol = av_symbol(company)
        if symbol is None or not self.configured():
            return SourceBatch()
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "sort": "LATEST",
            "limit": 50,
            "time_from": (utc_now() - LOOKBACK).strftime("%Y%m%dT%H%M"),
            "apikey": settings.alphavantage_api_key,
        }
        try:
            payload = await http.fetch_json(URL, params=params)
        except (httpx.HTTPError, http.UpstreamError) as exc:
            raise sanitized_error(exc, "alphavantage") from None
        return parse_payload(payload, symbol)
