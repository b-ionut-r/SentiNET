"""Yahoo Finance news via yfinance — ticker stream + precise name search.

Why trusted: Yahoo's newsroom index (Reuters, Bloomberg, Barron's, IBD,
Motley Fool, Simply Wall St…), every item with a publisher and `relatedTickers`.
Live findings (2026-10-04): `Ticker.get_news()` returned [] for every symbol
(kept only as a fallback); `yf.Search("NVDA").news` is a *ticker stream* — the
last ~10 h, but only 10/50 headlines named Nvidia (Motley Fool disclosure
boilerplate tags NVDA on dividend pieces); `yf.Search("Nvidia stock").news`
switches to text search — 50/50 on-topic over ~11 days. So equities run both
(fresh + precise); crypto searches the coin name ("Bitcoin": 50/50 on-topic vs
8 stale items for "BTC-USD"); funds/indices/futures search ticker + theme.
Equity items whose Yahoo tags name neither the symbol nor a share-class sibling
are dropped (Yahoo tags all Alphabet news GOOG, never GOOGL); if that would drop
> 80% of the name search, the tags evidently use another symbol for this
issuer and the filter is skipped for that query. Equity items must also name
the company (`Mentions`): the ticker stream carries tag-only noise (NVDA 60/100
on-topic, SOFI 29/54 live). Nothing is `ticker_specific`.
Raw Yahoo HTTP is throttled from datacenter IPs, hence yfinance's
browser-impersonating session via `run_yahoo`.
Docs: https://github.com/ranaroussi/yfinance (unofficial Yahoo API).
"""
from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any

from app.core.sync import run_yahoo
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import Mentions, issuer_symbols, search_terms
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
logger = logging.getLogger(__name__)


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


def parse_item(raw: dict[str, Any], symbols: Collection[str] | str, require_tag: bool = True) -> RawSignal | None:
    """Accepts both the Search shape (flat) and the get_news shape ({"content": {...}}).

    `symbols`: the issuer's symbols (`issuer_symbols`). `require_tag`: drop items Yahoo
    tagged with other symbols only (equities; theme searches for funds/coins are tagged
    with the index/coin, not the fund).
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
    own = {symbols} if isinstance(symbols, str) else set(symbols)
    if require_tag and related_set and not related_set & own:
        return None  # Yahoo's own tags say it's about something else
    return RawSignal(
        title=title,
        body=body,
        url=url,
        publisher=clean_text(publisher) or domain_of(url),
        timestamp=timestamp,
        extra={"symbols": len(related_set)} if related_set else {},
    )


def parse_results(
    batches: list[list[dict[str, Any]] | None],
    symbols: Collection[str],
    require_tag: bool,
    mentions: Mentions | None = None,
) -> list[RawSignal]:
    """Parse each query's rows; the safety valve skips the tag filter where it would gut a name search.

    `mentions` (equities): drop items whose headline/summary never names the company —
    Motley Fool disclosure boilerplate tags NVDA on Nike-earnings pieces.
    """
    out: list[RawSignal] = []
    for index, batch in enumerate(batches):
        rows = [r for r in batch or [] if isinstance(r, dict)]
        kept = [s for s in (parse_item(r, symbols, require_tag) for r in rows) if s]
        if require_tag and index > 0 and len(rows) >= 5 and len(kept) < 0.2 * len(rows):
            logger.info("yahoo_news: tag filter kept %d/%d for a name search; using untagged results", len(kept), len(rows))
            kept = [s for s in (parse_item(r, symbols, False) for r in rows) if s]
        out.extend(s for s in kept if mentions is None or mentions.about(f"{s.title} {s.body or ''}"))
    return out


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
        search_error: Exception | None = None
        batches: list[list[dict[str, Any]] | None] = []
        try:
            batches = await gather_partial(*(run_yahoo(_search_news, q) for q in search_queries(company)))
        except Exception as exc:  # noqa: BLE001 - yfinance raises assorted types; try the fallback first
            search_error = exc
        if not any(batches):
            try:
                batches = [await run_yahoo(_ticker_news, company.ticker.upper())]
            except Exception:
                if search_error is not None:
                    raise search_error from None
                raise
        terms = search_terms(company)
        equity = terms.asset in ("equity", "other")
        parsed = parse_results(batches, issuer_symbols(company), equity, Mentions(terms) if equity else None)
        signals = [s for s in parsed if is_recent(s.timestamp)]
        return SourceBatch(signals=newest_first(dedupe(signals)))
