"""Polite per-host rate limiting for free APIs.

Free endpoints ban aggressive clients. Each host gets a minimum spacing between
request *starts*; callers simply `await limiter("api.gdeltproject.org")` before
sending. Spacing is enforced across concurrent coroutines.
"""
from __future__ import annotations

import asyncio
import time

# Minimum seconds between request starts, per host. Unlisted hosts: no limit.
HOST_SPACING: dict[str, float] = {
    "api.gdeltproject.org": 5.5,  # documented: 1 request / 5 s
    "data.sec.gov": 0.15,  # SEC fair access: ≤ 10 req/s
    "www.sec.gov": 0.15,
    "wikimedia.org": 0.2,
    "api.stocktwits.com": 0.5,
    "apewisdom.io": 0.3,
    "hn.algolia.com": 0.1,
    "news.google.com": 0.3,
    "www.bing.com": 0.3,
    "api.bsky.app": 0.3,
    "finnhub.io": 1.0,  # 60/min free tier
    "www.alphavantage.co": 1.2,
}


class HostLimiter:
    def __init__(self, spacing: dict[str, float]) -> None:
        self._spacing = spacing
        self._next_at: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def spacing_for(self, host: str) -> float:
        return self._spacing.get(host, 0.0)

    async def __call__(self, host: str) -> None:
        gap = self.spacing_for(host)
        if gap <= 0:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            wait = self._next_at.get(host, 0.0) - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_at[host] = time.monotonic() + gap

    def reset(self) -> None:
        self._next_at.clear()
        self._locks.clear()


limiter = HostLimiter(HOST_SPACING)
