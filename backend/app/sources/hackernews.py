"""Hacker News via the Algolia search API — the tech crowd's take.

Why trusted: a high-signal, moderated engineering/startup audience; stories
carry points + comment counts (real engagement), and its reaction often leads
on tech names (chips, AI, cloud, devices). Limits: tiny coverage outside tech;
comments have no score in the API. Algolia's default typo tolerance and prefix
matching turn "Nvidia" into "avidiax" and "SoFi" into "Sofia" — both disabled.

Queries (last 14 days): stories whose *title* names the company; comments that
mention the company together with "stock"; and, when the symbol is not a
plain word, anything citing the ticker (e.g. "pushing up TSLA").
Docs: https://hn.algolia.com/api
"""
from __future__ import annotations

import time
from typing import Any

from app.core import http
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import SearchTerms, search_terms
from app.sources.util import (
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
STRICT = {"typoTolerance": "false", "queryType": "prefixNone", "advancedSyntax": "true"}


def _phrase(term: str) -> str:
    return f'"{term}"' if " " in term else term


def build_requests(terms: SearchTerms) -> list[dict[str, str]]:
    """Algolia parameter sets (without the time filter)."""
    name = _phrase(terms.primary)
    reqs: list[dict[str, str]] = []
    if not terms.ambiguous:
        reqs.append({"query": name, "tags": "story", "restrictSearchableAttributes": "title", "hitsPerPage": "40"})
    qualifier = "price" if terms.asset == "crypto" else "stock"
    reqs.append({"query": f"{name} {qualifier}", "tags": "comment", "hitsPerPage": "30"})
    if terms.symbol_searchable:
        reqs.append({"query": terms.symbol, "tags": "(story,comment)", "hitsPerPage": "30"})
    return reqs


def parse_hit(hit: dict[str, Any]) -> RawSignal | None:
    tags = hit.get("_tags") or []
    object_id = hit.get("objectID")
    url = ITEM_URL.format(id=object_id) if object_id else None
    if "story" in tags and hit.get("title"):
        link = hit.get("url")
        extra: dict[str, Any] = {"type": "story"}
        if domain_of(link):
            extra["domain"] = domain_of(link)
        return RawSignal(
            title=clean_text(hit.get("title")),
            body=clean_text(hit.get("story_text"), limit=600) or None,
            url=url,
            author=hit.get("author"),
            publisher="Hacker News",
            timestamp=from_epoch(hit.get("created_at_i")),
            engagement=int_or_zero(hit.get("points")) + int_or_zero(hit.get("num_comments")),
            extra=extra,
        )
    text = clean_text(hit.get("comment_text"), limit=500)
    if "comment" in tags and text:
        return RawSignal(
            title=text,
            url=url,
            author=hit.get("author"),
            publisher="Hacker News",
            timestamp=from_epoch(hit.get("created_at_i")),
            extra={"type": "comment", "story": clean_text(hit.get("story_title"), limit=120)},
        )
    return None


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
        return company.quote_type in {"EQUITY", "CRYPTOCURRENCY"}

    async def _search(self, params: dict[str, str], since: int) -> list[dict[str, Any]]:
        data = await http.fetch_json(URL, params={**STRICT, **params, "numericFilters": f"created_at_i>{since}"})
        return list(data.get("hits") or []) if isinstance(data, dict) else []

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        since = int(time.time()) - WINDOW_SECONDS
        results = await gather_partial(*(self._search(p, since) for p in build_requests(search_terms(company))))
        seen: set[str] = set()
        signals: list[RawSignal] = []
        for hit in (h for batch in results for h in (batch or [])):
            object_id = str(hit.get("objectID") or "")
            if not object_id or object_id in seen:
                continue
            seen.add(object_id)
            signal = parse_hit(hit)
            if signal is not None:
                signals.append(signal)
        # Stories (headlines with points) outrank stray comments when trimming.
        stories = [s for s in signals if s.extra.get("type") == "story"]
        comments = newest_first([s for s in signals if s.extra.get("type") != "story"])
        return SourceBatch(signals=newest_first(stories + comments[: max(0, MAX_ITEMS - len(stories))]))
