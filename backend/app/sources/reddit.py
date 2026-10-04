"""Reddit search via app-only OAuth (needs REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET).

Why used: the investing subreddits (r/wallstreetbets, r/stocks, r/investing,
r/StockMarket, r/options; crypto subs for coins) are the deepest retail
discussion pool, with score + comment counts as engagement. Keyless `.json`
endpoints are blocked since May 2026, so this source only runs with an
approved app's credentials (client_credentials grant, token cached).
Docs: https://www.reddit.com/dev/api/#GET_search
"""
from __future__ import annotations

import base64
from typing import Any

import httpx

from app.config import settings
from app.core import http
from app.core.cache import cached
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import Mentions, SearchTerms, search_terms
from app.sources.util import (
    clean_plain,
    from_epoch,
    int_or_zero,
    newest_first,
    sanitized_error,
)

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
SEARCH_URL = "https://oauth.reddit.com/r/{subs}/search"
STOCK_SUBS = "wallstreetbets+stocks+investing+StockMarket+options"
CRYPTO_SUBS = "CryptoCurrency+Bitcoin+CryptoMarkets+ethereum"


@cached(ttl=50 * 60, none_ttl=60, maxsize=2)
async def access_token(client_id: str) -> str | None:
    """client_credentials token (valid 24 h; refreshed hourly). The secret is read from settings."""
    resp = await http.fetch(
        TOKEN_URL,
        method="POST",
        data={"grant_type": "client_credentials"},
        headers={"Authorization": _basic(client_id, settings.reddit_client_secret)},
        api_ua=True,
    )
    return resp.json().get("access_token")


def _basic(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def build_query(terms: SearchTerms) -> str:
    """Cashtag, plain symbol (if not a word) and quoted names, OR-ed."""
    parts = [f'"${terms.symbol.replace("-", ".")}"']
    if terms.symbol_searchable:
        parts.append(terms.symbol)
    parts += [f'"{n}"' for n in terms.names if not terms.ambiguous]
    return " OR ".join(parts)


def parse_listing(payload: Any, mentions: Mentions | None = None) -> list[RawSignal]:
    """Posts (no NSFW/stickied); with `mentions`, only those that name the asset ("price target" != Target)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    children = (data or {}).get("children") or []
    out: list[RawSignal] = []
    for child in children:
        post = child.get("data") or {}
        title = clean_plain(post.get("title"))
        if not title or post.get("over_18") or post.get("stickied"):
            continue
        body = clean_plain(post.get("selftext"), limit=600) or None
        if mentions is not None and not mentions.about(f"{title} {body or ''}"):
            continue
        permalink = post.get("permalink")
        out.append(
            RawSignal(
                title=title,
                body=body,
                url=f"https://www.reddit.com{permalink}" if permalink else post.get("url"),
                author=post.get("author"),
                publisher=f"r/{post.get('subreddit')}" if post.get("subreddit") else "Reddit",
                timestamp=from_epoch(post.get("created_utc")),
                engagement=int_or_zero(post.get("score")) + int_or_zero(post.get("num_comments")),
                extra={"flair": post["link_flair_text"]} if post.get("link_flair_text") else {},
            )
        )
    return out


class RedditSource:
    key = "reddit"
    label = "Reddit"
    kind: SignalKind = "social"
    weight = 0.7
    requires_key = True
    description = "Posts from r/wallstreetbets, r/stocks, r/investing, r/options (crypto subs for coins); needs a Reddit app."
    docs_url: str | None = "https://www.reddit.com/dev/api/#GET_search"

    def configured(self) -> bool:
        return bool(settings.reddit_client_id and settings.reddit_client_secret)

    def supports(self, company: CompanyRef) -> bool:
        return company.quote_type in {"EQUITY", "ETF", "CRYPTOCURRENCY"}

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        if not self.configured():
            return SourceBatch()
        subs = CRYPTO_SUBS if company.is_crypto else STOCK_SUBS
        terms = search_terms(company)
        params = {
            "q": build_query(terms),
            "restrict_sr": "1",
            "sort": "new",
            "t": "week",
            "limit": "50",
            "raw_json": "1",
        }
        try:
            token = await access_token(settings.reddit_client_id)
            if not token:
                raise http.UpstreamError("reddit: no access token returned")
            headers = {"Authorization": f"Bearer {token}"}
            payload = await http.fetch_json(SEARCH_URL.format(subs=subs), params=params, headers=headers, api_ua=True)
        except (httpx.HTTPError, http.UpstreamError) as exc:
            raise sanitized_error(exc, "reddit") from None
        return SourceBatch(signals=newest_first(parse_listing(payload, Mentions(terms))))
