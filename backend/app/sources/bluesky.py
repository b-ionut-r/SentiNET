"""Bluesky post search — the open "FinSky" crowd (cashtags, market bots, commentators).

Why used: public, real-time, and a growing finance community that writes in
cashtags. Limits (live-verified 2026-10-04): the unauthenticated AppView serves
only the first page (up to 100 posts; the cursor page returns 403), and search
ignores the "$" — "$SPY" matches spy-novel posts and "$SOFI" matches "Sofi
Tukker". So every post goes through `Mentions` in social mode: `$SYM` counts,
a name counts only with the right casing, outside venue phrases ("SoFi
Stadium") and next to market vocabulary (themes like "S&P 500" excepted),
`#SYM` needs strong market words. Hourly price bots dominate some tickers (top-3
authors were 29-46% of posts), so templated repeats collapse and each author
contributes at most 3 posts.
If BLUESKY_HANDLE/BLUESKY_APP_PASSWORD are set, requests use an authenticated
session (token cached ~90 min); bad credentials fall back to the public AppView.

Metrics: bluesky_posts (kept posts after bot control), bluesky_authors (unique),
bluesky_saturated (a query hit the 100-post cap, so it covers less than 7 days),
bluesky_posts_per_day (kept posts per day from the busiest query over the span it
fully covers — the comparable rate; raw counts saturate) and bluesky_span_hours
(that span).
Engagement = likes + reposts + replies.
Docs: https://docs.bsky.app/docs/api/app-bsky-feed-search-posts
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import httpx

from app.config import settings
from app.core import http
from app.core.cache import cached
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import Mentions, SearchTerms, search_terms
from app.sources.util import (
    cap_per_author,
    clean_plain,
    clean_text,
    gather_partial,
    int_or_zero,
    parse_iso,
    utc_now,
)

PUBLIC_URL = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
AUTH_HOST = "https://bsky.social"
LIMIT = 100
MAX_ITEMS = 100
PER_AUTHOR = 3
WINDOW = timedelta(days=7)


@cached(ttl=90 * 60, none_ttl=120, maxsize=2)
async def session_token(handle: str) -> str | None:
    """App-password session (accessJwt lives ~2 h). Password comes from settings, never the cache key."""
    data = await http.fetch_json(
        f"{AUTH_HOST}/xrpc/com.atproto.server.createSession",
        method="POST",
        json={"identifier": handle, "password": settings.bluesky_app_password},
    )
    return data.get("accessJwt") if isinstance(data, dict) else None


def build_queries(terms: SearchTerms) -> list[str]:
    second = terms.primary if terms.asset in ("crypto", "etf") else f"{terms.primary} stock"
    return [f"${terms.symbol.replace('-', '.')}", second]  # share classes are written $BRK.B


def post_time(post: dict[str, Any]) -> datetime | None:
    """createdAt is client-supplied; trust the AppView's indexedAt when createdAt is in the future."""
    created, indexed = parse_iso((post.get("record") or {}).get("createdAt")), parse_iso(post.get("indexedAt"))
    if created is None or (indexed is not None and created > indexed + timedelta(minutes=10)):
        return indexed
    return created


def parse_post(post: dict[str, Any], mentions: Mentions) -> RawSignal | None:
    record = post.get("record") or {}
    text = clean_plain(record.get("text"), limit=600)
    facet_tags = [
        f.get("tag", "") for facet in record.get("facets") or [] for f in facet.get("features") or [] if f.get("tag")
    ]
    if not text or not mentions.about(" ".join([text, *(f"#{t}" for t in facet_tags)]), social=True):
        return None
    author = post.get("author") or {}
    handle = author.get("handle")
    rkey = str(post.get("uri") or "").rsplit("/", 1)[-1]
    external = ((post.get("embed") or {}).get("external")) or {}
    card = clean_text(" — ".join(x for x in (external.get("title"), external.get("description")) if x), limit=300)
    return RawSignal(
        title=text,
        body=card or None,
        url=f"https://bsky.app/profile/{handle}/post/{rkey}" if handle and rkey else None,
        author=handle,
        publisher="Bluesky",
        timestamp=post_time(post),
        engagement=sum(int_or_zero(post.get(k)) for k in ("likeCount", "repostCount", "replyCount")),
    )


def query_rate(batch: list[dict[str, Any]], kept_uris: set[str], now: datetime) -> tuple[float, float, bool]:
    """(kept posts/day, hours covered, saturated) for one query's raw results.

    A query that hit the 100-post cap only covers back to its oldest post; others cover the window.
    """
    saturated = len(batch) >= LIMIT
    times = [t for t in (post_time(p) for p in batch) if t is not None]
    start = max(now - WINDOW, min(times)) if saturated and times else now - WINDOW
    hours = max((now - start).total_seconds() / 3600, 1.0)
    hits = sum(1 for p in batch if str(p.get("uri") or "") in kept_uris and (post_time(p) or now) >= start)
    return hits * 24 / hours, hours, saturated


def crowd_metrics(
    kept: list[RawSignal], batches: list[list[dict[str, Any]] | None], kept_uris: set[str], now: datetime
) -> dict[str, Any]:
    """The rate comes from the busiest query (a lower bound for the union): one junk-saturated
    query ("$KO" = knockouts, 100 posts in 2 h) must not zero out what the name query found."""
    rates = [query_rate(batch, kept_uris, now) for batch in batches if batch]
    per_day, hours, _ = max(rates, default=(0.0, WINDOW.total_seconds() / 3600, False))
    return {
        "bluesky_posts": len(kept),
        "bluesky_authors": len({s.author for s in kept if s.author}),
        "bluesky_saturated": any(saturated for _, _, saturated in rates),
        "bluesky_span_hours": round(hours, 1),
        "bluesky_posts_per_day": round(per_day, 1),
    }


class BlueskySource:
    key = "bluesky"
    label = "Bluesky"
    kind: SignalKind = "social"
    weight = 0.6
    requires_key = False
    description = "Bluesky posts carrying the cashtag or naming the asset in market talk (FinSky crowd, last 7 days)."
    docs_url: str | None = "https://docs.bsky.app/docs/api/app-bsky-feed-search-posts"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return company.quote_type in {"EQUITY", "ETF", "CRYPTOCURRENCY"}

    async def _search(self, query: str, token: str | None) -> list[dict[str, Any]]:
        since = (utc_now() - WINDOW).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {"q": query, "sort": "latest", "limit": LIMIT, "lang": "en", "since": since}
        if token:
            url = f"{AUTH_HOST}/xrpc/app.bsky.feed.searchPosts"
            data = await http.fetch_json(url, params=params, headers={"Authorization": f"Bearer {token}"})
        else:
            data = await http.fetch_json(PUBLIC_URL, params=params)
        return list(data.get("posts") or []) if isinstance(data, dict) else []

    async def _token(self) -> str | None:
        if not (settings.bluesky_handle and settings.bluesky_app_password):
            return None
        try:
            return await session_token(settings.bluesky_handle)
        except (httpx.HTTPError, http.UpstreamError):  # bad credentials / PDS down: use the public AppView
            return None

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        terms = search_terms(company)
        token = await self._token()
        results = await gather_partial(*(self._search(q, token) for q in build_queries(terms)))
        mentions = Mentions(terms)
        uri_of: dict[int, str] = {}
        seen: set[str] = set()
        signals: list[RawSignal] = []
        for post in (p for batch in results for p in (batch or [])):
            uri = str(post.get("uri") or "")
            if uri in seen:
                continue
            seen.add(uri)
            signal = parse_post(post, mentions)
            if signal is not None:
                signals.append(signal)
                uri_of[id(signal)] = uri
        kept = cap_per_author(signals, PER_AUTHOR)
        if not kept:
            return SourceBatch()
        kept_uris = {uri_of[id(s)] for s in kept}
        metrics = crowd_metrics(kept, list(results), kept_uris, utc_now())
        return SourceBatch(signals=kept[:MAX_ITEMS], metrics=metrics)
