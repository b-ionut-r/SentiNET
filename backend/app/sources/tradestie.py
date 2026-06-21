"""Tradestie WallStreetBets sentiment API (free, no key).

Endpoint returns the day's most-discussed WSB tickers with a comment count and
a bullish/bearish sentiment label + score. We surface the row matching our
ticker as a single, pre-labeled aggregate signal.
Docs: https://tradestie.com/apps/reddit/api/
"""
from __future__ import annotations

from typing import Optional

from app.sources.base import RawSignal
from app.utils.http import get_client

URL = "https://tradestie.com/api/v1/apps/reddit"


class TradestieSource:
    name = "tradestie"
    label = "WSB (Tradestie)"
    kind = "social"
    weight = 0.8

    async def fetch(self, ticker: str, company: Optional[str]) -> list[RawSignal]:
        client = get_client()
        resp = await client.get(URL)
        resp.raise_for_status()
        rows = resp.json()

        ticker_u = ticker.upper()
        for row in rows:
            if str(row.get("ticker", "")).upper() != ticker_u:
                continue
            comments = int(row.get("no_of_comments", 0))
            sentiment = str(row.get("sentiment", "")).lower()
            sscore = row.get("sentiment_score")
            prelabeled = None
            if isinstance(sscore, (int, float)):
                prelabeled = float(sscore)
            elif sentiment == "bullish":
                prelabeled = 0.5
            elif sentiment == "bearish":
                prelabeled = -0.5
            text = (
                f"WallStreetBets ranks {ticker_u} among today's most-discussed "
                f"tickers with {comments} comments — community sentiment: {sentiment}."
            )
            return [
                RawSignal(
                    text=text,
                    url="https://tradestie.com/apps/reddit/",
                    author="r/wallstreetbets",
                    engagement=comments,
                    prelabeled_score=prelabeled,
                )
            ]
        return []
