"""StockTwits public symbol stream (best-effort).

StockTwits froze new-developer API registration, but the public read endpoint
may still respond. We treat it as best-effort: any failure degrades to empty.
Messages can carry a user-tagged Bullish/Bearish label, which we pass through
as a prelabel hint.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.sources.base import RawSignal
from app.utils.http import get_client


class StockTwitsSource:
    name = "stocktwits"
    label = "StockTwits"
    kind = "social"
    weight = 0.8

    async def fetch(self, ticker: str, company: Optional[str]) -> list[RawSignal]:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.upper()}.json"
        client = get_client()
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

        signals: list[RawSignal] = []
        for msg in data.get("messages", []):
            text = msg.get("body", "") or ""
            created = msg.get("created_at")
            ts: Optional[datetime] = None
            if created:
                try:
                    ts = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except ValueError:
                    ts = None
            prelabeled = None
            entities = msg.get("entities") or {}
            sentiment = (entities.get("sentiment") or {})
            basic = sentiment.get("basic") if isinstance(sentiment, dict) else None
            if basic == "Bullish":
                prelabeled = 0.6
            elif basic == "Bearish":
                prelabeled = -0.6
            user = msg.get("user", {}) or {}
            signals.append(
                RawSignal(
                    text=text,
                    url=f"https://stocktwits.com/symbol/{ticker.upper()}",
                    author=user.get("username"),
                    timestamp=ts,
                    engagement=int(user.get("followers", 0) or 0),
                    prelabeled_score=prelabeled,
                )
            )
        return signals
