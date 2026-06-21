"""ApeWisdom mentions API (free, no key).

Returns trending tickers across Reddit with mention counts and 24h change.
ApeWisdom does not provide sentiment, only social *volume*, so we emit a
volume-context signal with no prelabel — the engine scores its text neutrally,
and the mention count feeds the "buzz" stats.
Docs: https://apewisdom.io/api/
"""
from __future__ import annotations

from typing import Optional

from app.sources.base import RawSignal
from app.utils.http import get_client

URL = "https://apewisdom.io/api/v1.0/filter/all-stocks"


class ApeWisdomSource:
    name = "apewisdom"
    label = "ApeWisdom"
    kind = "social"
    weight = 0.5

    async def fetch(self, ticker: str, company: Optional[str]) -> list[RawSignal]:
        client = get_client()
        signals: list[RawSignal] = []
        ticker_u = ticker.upper()
        # Scan the first few pages of trending tickers for our symbol.
        for page in (1, 2, 3):
            resp = await client.get(f"{URL}/page/{page}")
            resp.raise_for_status()
            data = resp.json()
            for row in data.get("results", []):
                if str(row.get("ticker", "")).upper() != ticker_u:
                    continue
                mentions = int(row.get("mentions", 0))
                upvotes = int(row.get("upvotes", 0))
                rank = row.get("rank")
                prev = row.get("mentions_24h_ago")
                trend = ""
                if isinstance(prev, int) and prev > 0:
                    delta = mentions - prev
                    direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
                    trend = f" Mentions are {direction} vs. 24h ago ({prev} -> {mentions})."
                text = (
                    f"{ticker_u} is trending #{rank} on social platforms with "
                    f"{mentions} mentions and {upvotes} upvotes in the last 24h.{trend}"
                )
                signals.append(
                    RawSignal(
                        text=text,
                        url="https://apewisdom.io/",
                        author="ApeWisdom",
                        engagement=mentions,
                        extra={"mentions": mentions, "mentions_24h_ago": prev},
                    )
                )
                return signals
        return signals
