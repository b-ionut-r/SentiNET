"""Google News RSS search — the broadest free headline firehose.

Why trusted: aggregates thousands of outlets (wires, majors, trade press) with
a real publisher on every item. Limits: max 100 items per query, ranked by
*relevance* (a 7-day query misses most of the last 24 h), no snippets, links
are Google redirect URLs. So we run two queries — a 24-hour one for freshness
and a 7-day one for context — and merge them.

Query design (see `app.sources.query`): the asset's name(s) must appear in the
headline (`intitle:`), anchored by finance context words in the article text;
everyday-word names ("Target") are anchored on the ticker/legal name instead
(avoids "price target" floods); ETFs search their underlying theme ("S&P 500").
Docs: https://news.google.com (RSS search endpoint, unofficial but stable).
"""
from __future__ import annotations

from app.core import http
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import SearchTerms, clean_name, or_group, search_terms
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

URL = "https://news.google.com/rss/search"
MAX_ITEMS = 100


def _intitle(term: str) -> str:
    return f'intitle:"{term}"' if not term.isalnum() else f"intitle:{term}"


def build_query(terms: SearchTerms, company: CompanyRef) -> str:
    """The window-free part of the query (callers append `when:1d` / `when:7d`)."""
    heads = [_intitle(n) for n in terms.names]
    if terms.symbol_searchable:
        heads.append(_intitle(terms.symbol))
    subject = heads[0] if len(heads) == 1 else "(" + " OR ".join(heads) + ")"
    if terms.asset == "equity" and terms.ambiguous:
        anchors = [terms.symbol] if terms.symbol.lower() != terms.primary.lower() else []
        legal = clean_name(company.name)
        if legal.lower() != terms.primary.lower():
            anchors.append(legal)
        elif company.name.strip(" .").lower() != terms.primary.lower():
            anchors.append(company.name.strip(" ."))  # "Snap Inc." -> "Snap Inc"
        return f"{subject} {or_group(anchors)}" if anchors else f"{subject} {or_group(terms.context)}"
    if terms.asset == "crypto" and not terms.ambiguous:
        return subject
    return f"{subject} {or_group(terms.context)}"


def parse_feed(content: bytes) -> list[RawSignal]:
    root = parse_xml(content, "google_news")
    out: list[RawSignal] = []
    for item in root.iterfind("./channel/item"):
        title = clean_text(item.findtext("title"))
        source = item.find("source")
        publisher = clean_text(source.text) if source is not None and source.text else None
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3].rstrip()
        elif " - " in title and publisher is None:
            title, publisher = (part.strip() for part in title.rsplit(" - ", 1))
        if not title or is_listing_page(title):
            continue
        domain = domain_of(source.get("url")) if source is not None else None
        out.append(
            RawSignal(
                title=title,
                url=item.findtext("link"),
                publisher=publisher,
                timestamp=parse_rfc822(item.findtext("pubDate")),
                extra={"domain": domain} if domain else {},
            )
        )
    return out


class GoogleNewsSource:
    key = "google_news"
    label = "Google News"
    kind: SignalKind = "news"
    weight = 1.2
    requires_key = False
    description = "Headline search across thousands of outlets (24 h + 7 d windows, headline must name the asset)."
    docs_url: str | None = "https://news.google.com"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return company.quote_type in {"EQUITY", "ETF", "CRYPTOCURRENCY", "MUTUALFUND", "INDEX"}

    async def _search(self, query: str) -> list[RawSignal]:
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        resp = await http.fetch(URL, params=params)
        return parse_feed(resp.content)

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        base = build_query(search_terms(company), company)
        fresh, week = await gather_partial(self._search(f"{base} when:1d"), self._search(f"{base} when:7d"))
        signals = dedupe([*(fresh or []), *(week or [])])
        return SourceBatch(signals=newest_first(signals)[:MAX_ITEMS])
