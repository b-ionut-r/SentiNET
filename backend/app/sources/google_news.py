"""Google News RSS search — the broadest free headline firehose.

Why trusted: aggregates thousands of outlets (wires, majors, trade press) with
a real publisher on every item. Limits: max 100 items per query, ranked by
*relevance* (a 7-day query misses most of the last 24 h), no snippets, links
are Google redirect URLs. So we run two queries — a 24-hour one for freshness
and a 7-day one for context — and merge them.

Query design (see `app.sources.query`): the asset's name(s) must appear in the
headline (`intitle:`), anchored by finance context words in the article text;
everyday-word names ("Target") are anchored on the ticker/legal name instead
(avoids "price target" floods); funds/indices search their underlying theme
("S&P 500"); word-like coins ("Avalanche") need crypto words. The ticker is a
*separate*, name-anchored query (`intitle:TGT Target`): OR-ing `intitle:TGT`
into the name query let in "TGT 147800" commodity calls and teacher-exam posts.
Known quote/option-chain page phrases are excluded server-side, so they don't
eat the 100-result cap (GME: ~60 of 100 raw results were option pages).
Docs: https://news.google.com (RSS search endpoint, unofficial but stable).
"""
from __future__ import annotations

import re

from app.core import http
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import THEME_TYPES, Mentions, SearchTerms, clean_name, or_group, search_terms
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
SEARCHABLE_TYPES = {"EQUITY", "CRYPTOCURRENCY", *THEME_TYPES}


# Server-side exclusion of quote/listing pages (the client-side filter catches the rest).
EXCLUDE = '-"quote & history" -"quote and history" -"interactive stock chart" -"historical prices" -"options chain"'


def _intitle(term: str) -> str:
    return f'intitle:"{term}"' if not term.isalnum() else f"intitle:{term}"


def _quoted(term: str) -> str:
    return term if term.isalnum() else f'"{term}"'


def headline_names(terms: SearchTerms) -> list[str]:
    """Names usable in `intitle:`. Google returns nothing for an OR of near-identical phrases
    ("Nasdaq 100" OR "Nasdaq-100": 0 results vs 100) and little with long fund names in the
    group (7 words: 68 -> 7), so hyphen/plural variants collapse and names over 4 words are
    left to the other sources."""
    seen: set[str] = set()
    out: list[str] = []
    for name in terms.names:
        key = " ".join(word.rstrip("s") for word in re.split(r"[\s-]+", name.lower()))
        if key not in seen and len(name.split()) <= 4:
            seen.add(key)
            out.append(name)
    return out or [terms.primary]


def name_query(terms: SearchTerms, company: CompanyRef) -> str:
    """Headline names the asset + context; window-free (callers add `when:`)."""
    names = headline_names(terms)
    heads = [_intitle(n) for n in names]
    subject = heads[0] if len(heads) == 1 else "(" + " OR ".join(heads) + ")"
    if terms.asset == "equity" and terms.ambiguous:
        anchors = [terms.symbol] if terms.symbol.lower() != terms.primary.lower() else []
        legal = clean_name(company.name)
        if legal.lower() != terms.primary.lower():
            anchors.append(legal)
        elif company.name.strip(" .").lower() != terms.primary.lower():
            anchors.append(company.name.strip(" ."))  # "Snap Inc." -> "Snap Inc"
        anchors += [f"{terms.primary} stock", f"{terms.primary} shares"]  # exact phrases: TGT 48 -> 71 on-topic
        return f"{subject} {or_group(anchors)}"
    if terms.asset == "crypto" and not terms.ambiguous:
        return subject
    if terms.asset == "etf" and set(names) <= set(terms.self_evident):  # "S&P 500", "REITs": 68 -> 100 kept
        return subject
    return f"{subject} {or_group(terms.context)}"


def symbol_query(terms: SearchTerms) -> str | None:
    """`intitle:NVDA Nvidia`: ticker in the headline, name in the text (no nesting: Google drops it)."""
    if not terms.symbol_searchable:
        return None
    return f"intitle:{terms.symbol} {_quoted(terms.primary)}"


def build_queries(terms: SearchTerms, company: CompanyRef) -> list[str]:
    """Fresh (24 h) + weekly name queries, plus the weekly ticker query when the ticker is safe."""
    base = name_query(terms, company)
    queries = [f"{base} when:1d", f"{base} when:7d"]
    symbol = symbol_query(terms)
    if symbol:
        queries.append(f"{symbol} when:7d")
    return [f"{q} {EXCLUDE}" for q in queries]


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
    description = "Headline search across thousands of outlets (24 h + 7 d windows; headline must name the asset or ticker)."
    docs_url: str | None = "https://news.google.com"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return company.quote_type in SEARCHABLE_TYPES

    async def _search(self, query: str) -> list[RawSignal]:
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        resp = await http.fetch(URL, params=params)
        return parse_feed(resp.content)

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        terms = search_terms(company)
        results = await gather_partial(*(self._search(q) for q in build_queries(terms, company)))
        # Google matches `intitle:Target` case-insensitively: drop "target stock price of 73,000 won".
        homonym = Mentions(terms).homonym_only
        signals = dedupe(s for batch in results for s in (batch or []) if not homonym(s.title))
        return SourceBatch(signals=newest_first(signals)[:MAX_ITEMS])
