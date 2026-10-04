"""Yahoo Finance news via yfinance — ticker stream + precise name search.

Why trusted: Yahoo's newsroom index (Reuters, Bloomberg, Barron's, IBD,
Motley Fool, Simply Wall St…), every item with a publisher and `relatedTickers`.
Live findings (2026-10-04): `Ticker.get_news()` returned [] for every symbol
(kept only as a fallback); `yf.Search("NVDA").news` is a *ticker stream* — the
last ~10 h, but only 10/50 headlines named Nvidia (Motley Fool disclosure
boilerplate tags NVDA on dividend pieces); `yf.Search("Nvidia stock").news`
switches to text search — 50/50 on-topic over ~11 days. So equities run both
(fresh + precise); crypto searches the coin name ("Bitcoin": 50/50 on-topic vs
8 stale items for "BTC-USD"); ETFs search ticker + theme. Equity items whose
Yahoo tags exclude the symbol are dropped; nothing is marked `ticker_specific`.
Raw Yahoo HTTP is throttled from datacenter IPs, hence yfinance's
browser-impersonating session via `run_yahoo`.
Docs: https://github.com/ranaroussi/yfinance (unofficial Yahoo API).
"""
from __future__ import annotations

from typing import Any

from app.core.sync import run_yahoo
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import search_terms
from app.sources.util import (
    clean_text,
    dedupe,
    domain_of,
    from_epoch,
    gather_partial,
    is_listing_page,
    is_recent,
    newest_first,
    parse_iso,
)

SEARCH_NEWS = 50


def _search_news(query: str) -> list[dict[str, Any]]:
    import yfinance as yf

    search = yf.Search(query, max_results=1, news_count=SEARCH_NEWS, lists_count=0, recommended=0, timeout=8)
    return list(search.news or [])


def search_queries(company: CompanyRef) -> list[str]:
    terms = search_terms(company)
    if terms.asset == "crypto":
        return [terms.primary]
    if terms.asset == "etf":
        return [company.ticker.upper(), terms.primary]
    return [company.ticker.upper(), f"{terms.primary} stock"]


def _ticker_news(symbol: str) -> list[dict[str, Any]]:
    import yfinance as yf

    return list(yf.Ticker(symbol).get_news(count=30) or [])


def parse_item(raw: dict[str, Any], symbol: str, require_tag: bool = True) -> RawSignal | None:
    """Accepts both the Search shape (flat) and the get_news shape ({"content": {...}}).

    `require_tag`: drop items Yahoo tagged with other symbols only (equities; theme
    searches for ETFs/crypto are tagged with the index/coin, not the fund).
    """
    content = raw.get("content") if isinstance(raw.get("content"), dict) else None
    if content is not None:
        title = clean_text(content.get("title"))
        url = ((content.get("clickThroughUrl") or {}).get("url") or (content.get("canonicalUrl") or {}).get("url"))
        publisher = (content.get("provider") or {}).get("displayName")
        timestamp = parse_iso(content.get("pubDate") or content.get("displayTime"))
        body = clean_text(content.get("summary") or content.get("description"), limit=600) or None
        related = [t.get("symbol", "") for t in ((content.get("finance") or {}).get("stockTickers") or [])]
    else:
        title = clean_text(raw.get("title"))
        url = raw.get("link")
        publisher = raw.get("publisher")
        timestamp = from_epoch(raw.get("providerPublishTime"))
        body = None
        related = list(raw.get("relatedTickers") or [])
    if not title or is_listing_page(title):
        return None
    related_set = {str(t).upper() for t in related if t}
    if require_tag and related_set and symbol.upper() not in related_set:
        return None  # Yahoo's own tags say it's about something else
    return RawSignal(
        title=title,
        body=body,
        url=url,
        publisher=clean_text(publisher) or domain_of(url),
        timestamp=timestamp,
        extra={"symbols": len(related_set)} if related_set else {},
    )


class YahooNewsSource:
    key = "yahoo_news"
    label = "Yahoo Finance"
    kind: SignalKind = "news"
    weight = 1.1
    requires_key = False
    description = "Yahoo Finance news: the ticker's live stream plus a precise name search (via yfinance)."
    docs_url: str | None = "https://finance.yahoo.com"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return True  # Yahoo symbols are the app's canonical symbols

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        symbol = company.ticker.upper()
        search_error: Exception | None = None
        raw: list[dict[str, Any]] = []
        try:
            results = await gather_partial(*(run_yahoo(_search_news, q) for q in search_queries(company)))
            raw = [item for batch in results for item in (batch or [])]
        except Exception as exc:  # noqa: BLE001 - yfinance raises assorted types; try the fallback first
            search_error = exc
        if not raw:
            try:
                raw = await run_yahoo(_ticker_news, symbol)
            except Exception:
                if search_error is not None:
                    raise search_error from None
                raise
        require_tag = search_terms(company).asset in ("equity", "other")
        parsed = (parse_item(r, symbol, require_tag) for r in raw if isinstance(r, dict))
        signals = [s for s in parsed if s is not None and is_recent(s.timestamp)]
        return SourceBatch(signals=newest_first(dedupe(signals)))
