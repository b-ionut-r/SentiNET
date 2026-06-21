"""A single shared async HTTP client, reused by every source adapter.

Keeping one client (connection pool) is both faster and politer to the free
endpoints we depend on. A sensible default User-Agent reduces the chance of
being blocked.
"""
from __future__ import annotations

from typing import Optional

import httpx

from app.config import settings

_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            headers={
                "User-Agent": settings.user_agent,
                "Accept": "application/json, text/html, application/xml;q=0.9, */*;q=0.8",
            },
            timeout=httpx.Timeout(settings.source_timeout),
            follow_redirects=True,
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
