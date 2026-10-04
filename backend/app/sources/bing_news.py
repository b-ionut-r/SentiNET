"""Bing News RSS search — date-sorted headlines *with snippets*.

Why trusted: Microsoft's news index (incl. MSN syndication of Barron's, Motley
Fool, TheStreet…) complements Google with different ranking and a one-line
summary per item. Limits (observed live 2026-10): ~10-12 items per page; a
single-word query returns *generic trending news* instead of matches, so every
query is multi-word; bursts sometimes get an empty 200 body (retried once).
Links are `bing.com/news/apiclick.aspx?...&url=<real>` wrappers — unwrapped.
Timestamps are labelled GMT but are US-Pacific wall time (corrected, see `parse_pubdate`).
Bing half-ignores some queries ('"AT&T" shares' returned celebrity "shares her…"
stories), so every item must itself name the asset (`Mentions`, title + snippet)
and be at most 14 days old.
Docs: https://www.bing.com/news (RSS via `format=rss`, unofficial).
"""
from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

from app.core import http
from app.core.http import UpstreamError
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import THEME_TYPES, Mentions, SearchTerms, search_terms
from app.sources.util import (
    clean_text,
    dedupe,
    domain_of,
    gather_partial,
    is_listing_page,
    is_recent,
    newest_first,
    parse_rfc822,
    parse_xml,
    utc_now,
)

URL = "https://www.bing.com/news/search"
_FINANCE_NOUN = re.compile(r"\b(stocks?|prices?|yields|markets?|treasuries|etf|index|futures|bonds|reits)\b", re.IGNORECASE)


def build_queries(terms: SearchTerms) -> list[str]:
    """Two complementary multi-word queries per asset class."""
    name = f'"{terms.primary}"'
    if terms.asset == "crypto":
        return [f"{name} price", f"{name} crypto"]
    if terms.asset == "etf":  # funds/indices/futures/FX: context words fit the instrument
        if " " in terms.primary and _FINANCE_NOUN.search(terms.primary):  # already a phrase ("chip stocks")
            return [name, f"{name} {terms.context[-1]}"]
        return [f"{name} {terms.context[0]}", f"{name} {terms.context[2]}"]
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
    return (wall + timedelta(hours=offset)).replace(tzinfo=UTC)


def parse_pubdate(value: str | None) -> datetime | None:
    """Bing labels pubDate "GMT" but it is US-Pacific wall time for mkt=en-US.

    Verified live 2026-10-04 (and by an independent review across five markets):
    identical articles were exactly 7.0 h behind their Google News / Yahoo
    timestamps (PDT = UTC-7). Only the mislabelled "GMT" form is shifted; a real
    numeric offset is trusted as-is.
    """
    parsed = parse_rfc822(value)
    if parsed is None or not (value or "").strip().upper().endswith("GMT"):
        return parsed
    return pacific_to_utc(parsed.replace(tzinfo=None))


def parse_feed(content: bytes, now: datetime | None = None) -> list[RawSignal]:
    """Items with unwrapped links, publishers and Pacific-corrected times.

    Self-check: if any corrected time lands > 10 min in the future, Bing has
    started sending true UTC, so every item keeps its raw timestamp instead.
    """
    root = parse_xml(content, "bing_news")
    out: list[RawSignal] = []
    raw_times: list[datetime | None] = []
    for item in root.iterfind("./channel/item"):
        title = clean_text(item.findtext("title"))
        if not title or is_listing_page(title):
            continue
        url = unwrap_link(item.findtext("link"))
        # The News: namespace URI embeds the query string, so match on the local name.
        source = next((clean_text(ch.text) for ch in item if ch.tag.endswith("}Source") and ch.text), None)
        publisher = source.removesuffix(" on MSN").strip() if source else domain_of(url)
        body = clean_text(item.findtext("description"), limit=600) or None
        pub_date = item.findtext("pubDate")
        raw_times.append(parse_rfc822(pub_date))
        out.append(
            RawSignal(
                title=title,
                body=body,
                url=url,
                publisher=publisher,
                timestamp=parse_pubdate(pub_date),
                extra={"domain": domain_of(url)} if url else {},
            )
        )
    limit = (now or utc_now()) + timedelta(minutes=10)
    if any(s.timestamp and s.timestamp > limit for s in out):
        for signal, raw in zip(out, raw_times, strict=True):
            signal.timestamp = raw
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
        return company.quote_type in {"EQUITY", "CRYPTOCURRENCY", *THEME_TYPES}

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
        about = Mentions(terms).about
        # Per item, not per page: Bing's generic "trending" fallback and half-matched pages
        # mix on-topic and unrelated stories.
        signals = dedupe(
            s
            for batch in results
            for s in batch or []
            if is_recent(s.timestamp) and about(f"{s.title} {s.body or ''}")
        )
        return SourceBatch(signals=newest_first(signals))
