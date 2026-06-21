"""Reddit source via the public search JSON endpoint (no auth required).

We search finance subreddits for the ticker and company name. Works keyless;
if Reddit rate-limits or blocks, the adapter degrades to an empty list.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.sources.base import RawSignal
from app.utils.http import get_client

SUBREDDITS = "wallstreetbets+stocks+investing+StockMarket+options"


class RedditSource:
    name = "reddit"
    label = "Reddit"
    kind = "social"
    weight = 0.7

    async def fetch(self, ticker: str, company: Optional[str]) -> list[RawSignal]:
        query = f"${ticker} OR {ticker}"
        if company:
            query = f'{query} OR "{company}"'
        url = f"https://www.reddit.com/r/{SUBREDDITS}/search.json"
        params = {
            "q": query,
            "restrict_sr": "on",
            "sort": "new",
            "limit": 40,
            "t": "week",
        }
        client = get_client()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        signals: list[RawSignal] = []
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            title = post.get("title", "") or ""
            selftext = post.get("selftext", "") or ""
            text = f"{title}. {selftext}".strip()
            created = post.get("created_utc")
            ts = (
                datetime.fromtimestamp(created, tz=timezone.utc)
                if created
                else None
            )
            permalink = post.get("permalink")
            signals.append(
                RawSignal(
                    text=text,
                    url=f"https://www.reddit.com{permalink}" if permalink else None,
                    author=post.get("author"),
                    timestamp=ts,
                    engagement=int(post.get("score", 0))
                    + int(post.get("num_comments", 0)),
                )
            )
        return signals
