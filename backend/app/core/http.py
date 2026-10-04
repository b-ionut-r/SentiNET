"""One shared async HTTP client with polite defaults.

* Connection pooling (faster, kinder to free endpoints).
* Per-host rate limiting (`app.core.ratelimit`).
* A browser-like UA by default; `api_ua=True` sends the descriptive,
  contact-bearing UA that SEC EDGAR and Wikimedia require.
* One retry with backoff on 429/5xx/transport errors.

Usage:
    resp = await fetch("https://...", params={...})          # raises on HTTP error
    data = await fetch_json("https://...", api_ua=True)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from app.config import settings
from app.core.ratelimit import limiter

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None

RETRY_STATUSES = {429, 500, 502, 503, 504}


class UpstreamError(RuntimeError):
    """A data provider answered, but not usefully (rate limit, bad payload…)."""


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            headers={
                "User-Agent": settings.browser_user_agent,
                "Accept": "application/json, application/rss+xml, application/xml;q=0.9, text/html;q=0.8, */*;q=0.5",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=httpx.Timeout(settings.source_timeout, connect=6.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=60, max_keepalive_connections=20),
        )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


async def fetch(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    api_ua: bool = False,
    method: str = "GET",
    json: Any = None,
    data: Any = None,
    timeout: Optional[float] = None,
    retries: int = 1,
) -> httpx.Response:
    """Rate-limited request; raises httpx.HTTPStatusError on non-2xx after retries."""
    host = urlsplit(url).hostname or ""
    hdrs = dict(headers or {})
    if api_ua:
        hdrs.setdefault("User-Agent", settings.api_user_agent)
    client = get_client()
    attempt = 0
    while True:
        await limiter(host)
        try:
            resp = await client.request(
                method, url, params=params, headers=hdrs, json=json, data=data,
                timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
            )
            if resp.status_code in RETRY_STATUSES and attempt < retries:
                attempt += 1
                delay = _retry_delay(resp, attempt)
                logger.info("HTTP %s from %s; retrying in %.1fs", resp.status_code, host, delay)
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.TransportError,) as exc:
            if attempt < retries:
                attempt += 1
                await asyncio.sleep(0.8 * attempt)
                continue
            raise UpstreamError(f"{host}: {type(exc).__name__}") from exc


async def fetch_json(url: str, **kwargs: Any) -> Any:
    resp = await fetch(url, **kwargs)
    try:
        return resp.json()
    except ValueError as exc:
        snippet = resp.text[:160].replace("\n", " ")
        raise UpstreamError(f"non-JSON response from {urlsplit(url).hostname}: {snippet}") from exc


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            return min(8.0, float(ra))
        except ValueError:
            pass
    return min(6.0, 1.5 * attempt)
