"""Hacker News via the Algolia search API — the tech crowd's take.

Why trusted: a high-signal, moderated engineering/startup audience; stories
carry points + comment counts (real engagement), and its reaction often leads
on tech names (chips, AI, cloud, devices). Limits: tiny coverage outside tech;
comments have no score in the API.

Precision (live-checked 2026-10-04): Algolia's typo tolerance, prefix matching
and plural folding turn "Nvidia" into "avidiax", "SoFi" into "Sofia" and MAR
into "Mars" — all disabled. Bare-symbol queries are not run (ICE, COP, CVS
matched immigration, policing and version control 30/30), everyday-word names
("Target") are not searched at all, and every hit must name the asset with the
right casing (`Mentions`); coins also need crypto vocabulary.

Queries (last 14 days): stories whose *title* names the company (brand and one
distinct alias, e.g. Alphabet + Google), and comments naming it next to
"stock" ("price" for coins).
Docs: https://hn.algolia.com/api
"""
from __future__ import annotations

import time
from typing import Any

from app.core import http
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import Mentions, SearchTerms, search_terms
from app.sources.util import (
    clean_plain,
    clean_text,
    domain_of,
    from_epoch,
    gather_partial,
    int_or_zero,
    newest_first,
)

URL = "https://hn.algolia.com/api/v1/search_by_date"
ITEM_URL = "https://news.ycombinator.com/item?id={id}"
WINDOW_SECONDS = 14 * 86400
MAX_ITEMS = 60
COMMENT_CHARS = 500
STRICT = {"typoTolerance": "false", "queryType": "prefixNone", "advancedSyntax": "true", "ignorePlurals": "false"}


def _phrase(term: str) -> str:
    return f'"{term}"' if " " in term else term


def build_requests(terms: SearchTerms) -> list[dict[str, str]]:
    """Algolia parameter sets (without the time filter); [] when HN text can't be about the asset."""
    if terms.ambiguous or terms.asset not in ("equity", "crypto"):
        return []
    primary = terms.primary
    titles = [primary, *(n for n in terms.names[1:] if primary.lower() not in n.lower())][:2]
    reqs = [
        {"query": _phrase(n), "tags": "story", "restrictSearchableAttributes": "title", "hitsPerPage": "40"}
        for n in titles
    ]
    if "&" not in primary:  # Algolia tokenises "AT&T" away; "AT&T stock" matches any "stock" comment
        qualifier = "price" if terms.asset == "crypto" else "stock"
        reqs.append({"query": f"{_phrase(primary)} {qualifier}", "tags": "comment", "hitsPerPage": "30"})
    return reqs


def excerpt(text: str, needle: str | None, limit: int) -> str:
    """Up to `limit` chars, shifted so the first mention of `needle` stays in view (relevance reads it)."""
    at = text.lower().find(needle.lower()) if needle else -1
    if len(text) <= limit or at < 0 or at + len(needle or "") <= limit - 40:
        return clean_plain(text, limit=limit)
    start = max(0, at - limit // 2)
    sentence = text.rfind(". ", 0, at) + 2  # begin at the mention's sentence when it fits…
    start = sentence if sentence >= start else text.find(" ", start) + 1  # …else at a word boundary
    return "…" + clean_plain(text[start:], limit=limit)


def parse_hit(hit: dict[str, Any], mentions: Mentions | None = None, social: bool = False) -> RawSignal | None:
    """A story or comment; with `mentions`, None unless the full text names the asset."""
    tags = hit.get("_tags") or []
    object_id = hit.get("objectID")
    url = ITEM_URL.format(id=object_id) if object_id else None
    if "story" in tags and hit.get("title"):
        title, body = clean_plain(hit.get("title")), clean_text(hit.get("story_text"), limit=600) or None
        if mentions is not None and not mentions.about(f"{title} {body or ''}", social=social):
            return None
        link = hit.get("url")
        extra: dict[str, Any] = {"type": "story"}
        if domain_of(link):
            extra["domain"] = domain_of(link)
        return RawSignal(
            title=title,
            body=body,
            url=url,
            author=hit.get("author"),
            publisher="Hacker News",
            timestamp=from_epoch(hit.get("created_at_i")),
            engagement=int_or_zero(hit.get("points")) + int_or_zero(hit.get("num_comments")),
            extra=extra,
        )
    full = clean_text(hit.get("comment_text"))
    if "comment" not in tags or not full:
        return None
    if mentions is not None and not mentions.about(full, social=social):
        return None
    return RawSignal(
        title=excerpt(full, mentions.named(full) if mentions else None, COMMENT_CHARS),
        url=url,
        author=hit.get("author"),
        publisher="Hacker News",
        timestamp=from_epoch(hit.get("created_at_i")),
        extra={"type": "comment", "story": clean_text(hit.get("story_title"), limit=120)},
    )


class HackerNewsSource:
    key = "hackernews"
    label = "Hacker News"
    kind: SignalKind = "social"
    weight = 0.7
    requires_key = False
    description = "Hacker News stories and comments naming the company (tech-crowd reaction, last 14 days)."
    docs_url: str | None = "https://hn.algolia.com/api"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return company.quote_type in {"EQUITY", "CRYPTOCURRENCY"} and bool(build_requests(search_terms(company)))

    async def _search(self, params: dict[str, str], since: int) -> list[dict[str, Any]]:
        data = await http.fetch_json(URL, params={**STRICT, **params, "numericFilters": f"created_at_i>{since}"})
        return list(data.get("hits") or []) if isinstance(data, dict) else []

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        terms = search_terms(company)
        requests = build_requests(terms)
        if not requests:
            return SourceBatch()
        since = int(time.time()) - WINDOW_SECONDS
        results = await gather_partial(*(self._search(p, since) for p in requests))
        mentions, social = Mentions(terms), terms.asset == "crypto"
        seen: set[str] = set()
        signals: list[RawSignal] = []
        for hit in (h for batch in results for h in (batch or [])):
            object_id = str(hit.get("objectID") or "")
            if not object_id or object_id in seen:
                continue
            seen.add(object_id)
            signal = parse_hit(hit, mentions, social)
            if signal is not None:
                signals.append(signal)
        # Stories (headlines with points) outrank stray comments when trimming.
        stories = [s for s in signals if s.extra.get("type") == "story"]
        comments = newest_first([s for s in signals if s.extra.get("type") != "story"])
        return SourceBatch(signals=newest_first(stories + comments[: max(0, MAX_ITEMS - len(stories))]))
