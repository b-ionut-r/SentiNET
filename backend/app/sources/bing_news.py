"""Bing News RSS search — date-sorted headlines *with snippets*.

Why trusted: Microsoft's news index (incl. MSN syndication of Barron's, Motley
Fool, TheStreet…) complements Google with different ranking and a one-line
summary per item. Limits (observed live 2026-10): ~10-12 items per page; a
single-word query returns *generic trending news* instead of matches, so every
query is multi-word; bursts sometimes get an empty 200 body (retried once).
Links are `bing.com/news/apiclick.aspx?...&url=<real>` wrappers — unwrapped.
Timestamps are labelled GMT but are US-Pacific wall time (corrected, see `parse_pubdate`).
Docs: https://www.bing.com/news (RSS via `format=rss`, unofficial).
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

from app.core import http
from app.core.http import UpstreamError
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import SearchTerms, search_terms
from app.sources.util import (
    clean_text,
    dedupe,
    domain_of,
    gather_partial,
    is_listing_page,
    newest_first,
    parse_rfc822,
    parse_xml,
)

URL = "https://www.bing.com/news/search"
_FINANCE_NOUN = re.compile(r"\b(stocks?|prices?|yields|market|treasuries|etf|index)\b", re.IGNORECASE)


def build_queries(terms: SearchTerms) -> list[str]:
    """Two complementary multi-word queries per asset class."""
    name = f'"{terms.primary}"'
    if terms.asset == "crypto":
        return [f"{name} price", f"{name} crypto"]
    if terms.asset == "etf":
        if _FINANCE_NOUN.search(terms.primary):  # theme is already a market phrase ("chip stocks")
            return [name, f"{name} ETF"]
        return [f"{name} stocks", f"{name} market"]
    if terms.ambiguous and terms.symbol.lower() != terms.primary.lower():
        return [f"{name} {terms.symbol} stock", f"{name} {terms.symbol} shares"]
    return [f"{name} stock", f"{name} shares"]


def unwrap_link(link: str | None) -> str | None:
    """`http://www.bing.com/news/apiclick.aspx?...&url=https%3a%2f%2f...` -> the publisher URL."""
    if not link:
        return None
    parts = urlsplit(link)
    if parts.hostname and parts.hostname.endswith("bing.com"):
        real = parse_qs(parts.query).get("url")
        if real:
            return real[0]
    return link


def _nth_sunday(year: int, month: int, n: int) -> datetime:
    first = datetime(year, month, 1)  # noqa: DTZ001 - deliberately naive wall-clock arithmetic
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def pacific_to_utc(wall: datetime) -> datetime:
    """Interpret a naive US-Pacific wall-clock time (DST: 2nd Sun Mar - 1st Sun Nov, 02:00)."""
    dst_start = _nth_sunday(wall.year, 3, 2).replace(hour=2)
    dst_end = _nth_sunday(wall.year, 11, 1).replace(hour=2)
    offset = 7 if dst_start <= wall < dst_end else 8
    return (wall + timedelta(hours=offset)).replace(tzinfo=timezone.utc)


def parse_pubdate(value: str | None) -> datetime | None:
    """Bing labels pubDate "GMT" but it is US-Pacific wall time for mkt=en-US.

    Verified live 2026-10-04: 20+ identical articles were exactly 7.0 h behind
    their Google News / Yahoo timestamps (PDT = UTC-7).
    """
    parsed = parse_rfc822(value)
    return pacific_to_utc(parsed.replace(tzinfo=None)) if parsed else None


def parse_feed(content: bytes) -> list[RawSignal]:
    root = parse_xml(content, "bing_news")
    out: list[RawSignal] = []
    for item in root.iterfind("./channel/item"):
        title = clean_text(item.findtext("title"))
        if not title or is_listing_page(title):
            continue
        url = unwrap_link(item.findtext("link"))
        # The News: namespace URI embeds the query string, so match on the local name.
        source = next((clean_text(ch.text) for ch in item if ch.tag.endswith("}Source") and ch.text), None)
        publisher = source.removesuffix(" on MSN").strip() if source else domain_of(url)
        body = clean_text(item.findtext("description"), limit=600) or None
        out.append(
            RawSignal(
                title=title,
                body=body,
                url=url,
                publisher=publisher,
                timestamp=parse_pubdate(item.findtext("pubDate")),
                extra={"domain": domain_of(url)} if url else {},
            )
        )
    return out


class BingNewsSource:
    key = "bing_news"
    label = "Bing News"
    kind: SignalKind = "news"
    weight = 1.1
    requires_key = False
    description = "Date-sorted Bing News headlines with snippets (incl. MSN-syndicated Barron's, TheStreet, Motley Fool)."
    docs_url: str | None = "https://www.bing.com/news"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return company.quote_type in {"EQUITY", "ETF", "CRYPTOCURRENCY", "MUTUALFUND", "INDEX"}

    async def _search(self, query: str) -> list[RawSignal]:
        params = {"q": query, "format": "rss", "qft": 'sortbydate="1"', "mkt": "en-US"}
        for attempt in range(2):
            resp = await http.fetch(URL, params=params)
            try:
                return parse_feed(resp.content)
            except UpstreamError:
                if attempt or resp.content.strip():
                    raise
                await asyncio.sleep(1.5)  # empty 200 body = soft throttle; one polite retry
        return []

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        terms = search_terms(company)
        results = await gather_partial(*(self._search(q) for q in build_queries(terms)))
        mention = re.compile("|".join(re.escape(t) for t in (*terms.names, terms.symbol)), re.IGNORECASE)
        signals = dedupe(
            sig
            for batch in results
            # A page where nothing names the asset is Bing's generic "trending" fallback, not results.
            if batch and any(mention.search(f"{s.title} {s.body or ''}") for s in batch)
            for sig in batch
        )
        return SourceBatch(signals=newest_first(signals))
