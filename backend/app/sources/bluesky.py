"""Bluesky post search — the open "FinSky" crowd (cashtags, market bots, commentators).

Why used: public, real-time, and a growing finance community that writes in
cashtags. Limits (live-verified 2026-10-04): the unauthenticated AppView serves
only the first page (up to 100 posts; the cursor page returns 403), and search
ignores the "$" — "$SPY" matches spy-novel posts and "$SOFI" matches "Sofi
Tukker". So results are filtered client-side (`PostFilter`): a post is kept
only if it carries the cashtag/hashtag, or names the asset in a market context
(finance vocabulary; a bare "$450 bonus" referral does not count).
If BLUESKY_HANDLE/BLUESKY_APP_PASSWORD are set, requests use an authenticated
session (token cached ~90 min); bad credentials fall back to the public AppView.

Metric: bluesky_posts — matching posts in the sample (≤ 100 per query, 7 days).
Engagement = likes + reposts + replies.
Docs: https://docs.bsky.app/docs/api/app-bsky-feed-search-posts
"""
from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import httpx

from app.config import settings
from app.core import http
from app.core.cache import cached
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import SearchTerms, search_terms
from app.sources.util import (
    clean_text,
    gather_partial,
    int_or_zero,
    newest_first,
    parse_iso,
    utc_now,
)

PUBLIC_URL = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
AUTH_HOST = "https://bsky.social"
LIMIT = 100
MAX_ITEMS = 100  # newest kept as text; the metric counts every match
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


# Market vocabulary that separates "SoFi stock slides" from "SoFi Stadium" / "Sofi Tukker".
_FINANCE_INTENT = re.compile(
    r"(?i)\b(stocks?|shares|earnings|revenue|guidance|analysts?|price target|upgraded?|downgraded?|investors?|"
    r"valuation|market cap|bullish|bearish|short sellers?|rally|ipo|dividend|buyback|nasdaq|nyse|eps|"
    r"all-time high|record high|52-week)\b"
)


class PostFilter:
    """Keep a post if it carries the cashtag/hashtag, or names the asset in a market context.

    Coins and ETF themes ("Bitcoin", "S&P 500") are market talk by themselves;
    company names need finance vocabulary nearby. Everyday-word names match
    case-sensitively.
    """

    def __init__(self, terms: SearchTerms) -> None:
        symbol = re.escape(terms.symbol).replace(r"\-", "[-./]")  # $BRK.B / $BRK-B / $BRK/B
        self.tags = re.compile(rf"(?i)(?<![\w$#])[$#]{symbol}\b")
        names = "|".join(rf"\b{re.escape(n)}\b" for n in terms.names)
        self.names = re.compile(names if terms.ambiguous else f"(?i:{names})")
        self.needs_intent = terms.asset in ("equity", "other")

    def __call__(self, text: str) -> bool:
        if self.tags.search(text):
            return True
        if not self.names.search(text):
            return False
        return not self.needs_intent or bool(_FINANCE_INTENT.search(text))


def parse_post(post: dict[str, Any], keep: PostFilter) -> RawSignal | None:
    record = post.get("record") or {}
    text = clean_text(record.get("text"), limit=600)
    facet_tags = [
        f.get("tag", "") for facet in record.get("facets") or [] for f in facet.get("features") or [] if f.get("tag")
    ]
    if not text or not keep(" ".join([text, *(f"#{t}" for t in facet_tags)])):
        return None
    created, indexed = parse_iso(record.get("createdAt")), parse_iso(post.get("indexedAt"))
    if created is None or (indexed is not None and created > indexed + timedelta(minutes=10)):
        created = indexed  # createdAt is client-supplied; trust the AppView's clock when it's in the future
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
        timestamp=created,
        engagement=sum(int_or_zero(post.get(k)) for k in ("likeCount", "repostCount", "replyCount")),
    )


class BlueskySource:
    key = "bluesky"
    label = "Bluesky"
    kind: SignalKind = "social"
    weight = 0.6
    requires_key = False
    description = "Bluesky posts carrying the cashtag or naming the asset (open FinSky crowd, last 7 days)."
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
        keep = PostFilter(terms)
        seen: set[str] = set()
        signals: list[RawSignal] = []
        for post in (p for batch in results for p in (batch or [])):
            uri = str(post.get("uri") or "")
            if uri in seen:
                continue
            seen.add(uri)
            signal = parse_post(post, keep)
            if signal is not None:
                signals.append(signal)
        if not signals:
            return SourceBatch()
        return SourceBatch(signals=newest_first(signals)[:MAX_ITEMS], metrics={"bluesky_posts": len(signals)})
