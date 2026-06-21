"""Google News RSS source (free, no key).

We query Google News for the company (preferred) or ticker and parse the RSS
feed with feedparser. Headlines are high-signal news content, weighted above
anonymous social chatter.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import mktime
from typing import Optional
from urllib.parse import quote_plus

import feedparser

from app.sources.base import RawSignal
from app.utils.http import get_client


class GoogleNewsSource:
    name = "google_news"
    label = "Google News"
    kind = "news"
    weight = 1.2

    async def fetch(self, ticker: str, company: Optional[str]) -> list[RawSignal]:
        term = company or ticker
        query = quote_plus(f"{term} stock")
        url = (
            f"https://news.google.com/rss/search?q={query}"
            "&hl=en-US&gl=US&ceid=US:en"
        )
        client = get_client()
        resp = await client.get(url)
        resp.raise_for_status()
        # feedparser is sync/CPU-bound; keep the event loop responsive.
        feed = await asyncio.to_thread(feedparser.parse, resp.content)

        signals: list[RawSignal] = []
        for entry in feed.entries[:30]:
            title = entry.get("title", "") or ""
            summary = entry.get("summary", "") or ""
            source_name = ""
            if entry.get("source") and entry.source.get("title"):
                source_name = entry.source.title
            text = title if not source_name else f"{title} ({source_name})"
            ts: Optional[datetime] = None
            if entry.get("published_parsed"):
                ts = datetime.fromtimestamp(
                    mktime(entry.published_parsed), tz=timezone.utc
                )
            signals.append(
                RawSignal(
                    text=text,
                    url=entry.get("link"),
                    author=source_name or "Google News",
                    timestamp=ts,
                    extra={"summary": summary},
                )
            )
        return signals
