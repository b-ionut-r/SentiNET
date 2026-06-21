"""Hacker News source via the Algolia search API (free, no key).

Good for tech-company sentiment and product discussion. We search the company
name (preferred) or ticker over recent stories/comments.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.sources.base import RawSignal
from app.utils.http import get_client

URL = "https://hn.algolia.com/api/v1/search_by_date"


class HackerNewsSource:
    name = "hackernews"
    label = "Hacker News"
    kind = "news"
    weight = 0.9

    async def fetch(self, ticker: str, company: Optional[str]) -> list[RawSignal]:
        term = company or ticker
        params = {
            "query": term,
            "tags": "(story,comment)",
            "hitsPerPage": 30,
        }
        client = get_client()
        resp = await client.get(URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        signals: list[RawSignal] = []
        for hit in data.get("hits", []):
            text = hit.get("title") or hit.get("comment_text") or hit.get("story_text") or ""
            if not text:
                continue
            created = hit.get("created_at_i")
            ts = (
                datetime.fromtimestamp(created, tz=timezone.utc) if created else None
            )
            object_id = hit.get("objectID")
            signals.append(
                RawSignal(
                    text=text,
                    url=f"https://news.ycombinator.com/item?id={object_id}"
                    if object_id
                    else hit.get("url"),
                    author=hit.get("author"),
                    timestamp=ts,
                    engagement=int(hit.get("points") or 0)
                    + int(hit.get("num_comments") or 0),
                )
            )
        return signals
