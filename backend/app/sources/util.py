"""Small, dependency-free helpers shared by the source adapters.

Sources must not depend on the NLP layer (it is optional at import time and
owned elsewhere), so the bits of text/date/XML hygiene they need live here.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from collections.abc import Awaitable, Iterable
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar
from urllib.parse import urlsplit
from xml.etree.ElementTree import Element, ParseError

import httpx
from defusedxml import ElementTree as SafeET

from app.core.http import UpstreamError
from app.sources.base import RawSignal

T = TypeVar("T")
logger = logging.getLogger(__name__)

# Items older than this are not "current sentiment" for feeds that have no date filter.
MAX_AGE = timedelta(days=14)

_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")  # requires a letter: "x < y" is not a tag
_BLOCK_TAG_RE = re.compile(r"</?(p|br|div|li|ul|ol|blockquote|pre|h\d)\b[^<>]*>", re.IGNORECASE)
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_WS_RE = re.compile(r"\s+")

# Search engines index evergreen quote/listing pages next to real articles. Live
# 2026-10-04: 20 of 94 Google results for TGT were Yahoo option-contract pages such as
# "TGT Oct 2026 85.000 put (TGT261009P00085000) stock price, news, quote and history".
_NON_ARTICLE_RE = re.compile(
    r"\b[A-Z]{1,6}\d{6}[CP]\d{8}\b"  # OCC option-contract symbol
    r"|(?i:(?:stock|share) price,? (?:news|quote)\b"
    r"|interactive stock chart"
    r"|historical prices (?:&|and) data"
    r"|\bstock chart(?:, market cap (?:&|and) news today)?\s*$"
    r"|earnings history (?:&|and) trends"
    r"|stock forecast (?:&|and) analyst predictions"
    r"|research (?:&|and) ratings"
    r"|\bstock quote\s*$)"
)


def is_listing_page(title: str) -> bool:
    """True for quote/chart/option-chain pages that carry no news or opinion."""
    return bool(_NON_ARTICLE_RE.search(title))


def clean_text(value: str | None, limit: int | None = None) -> str:
    """Strip HTML tags/entities, zero-width chars and redundant whitespace."""
    if not value:
        return ""
    text = value
    for _ in range(2):  # two passes: some feeds double-escape ("&amp;#39;", "&lt;p&gt;")
        text = _BLOCK_TAG_RE.sub(" ", text)
        text = _TAG_RE.sub("", text)
        text = html.unescape(text)
    text = _ZERO_WIDTH_RE.sub("", text).replace("\xa0", " ")
    text = _WS_RE.sub(" ", text).strip()
    if limit and len(text) > limit:
        cut = text[:limit].rsplit(" ", 1)[0]
        text = (cut or text[:limit]).rstrip(" ,;:-") + "…"
    return text


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    """Normalize to tz-aware UTC (naive values are assumed to be UTC)."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_rfc822(value: str | None) -> datetime | None:
    """RSS `pubDate` ("Sun, 04 Oct 2026 21:45:10 GMT" / "-0400") -> UTC datetime."""
    if not value:
        return None
    try:
        return as_utc(parsedate_to_datetime(value.strip()))
    except (TypeError, ValueError, IndexError):
        return None


def parse_iso(value: str | None) -> datetime | None:
    """ISO-8601 ("2026-10-04T21:22:51.245Z", "2026-10-04T20:58:27+00:00") -> UTC datetime."""
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    # Python 3.11 accepts most ISO forms; trim >6 fractional digits (Lemmy/Postgres style).
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    try:
        return as_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def from_epoch(value: Any) -> datetime | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def is_recent(ts: datetime | None, max_age: timedelta = MAX_AGE, now: datetime | None = None) -> bool:
    """Undated items are kept (the pipeline decides); dated ones must be within `max_age`."""
    if ts is None:
        return True
    return ts >= (now or utc_now()) - max_age


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else (host or None)


def parse_xml(content: bytes, source: str) -> Element:
    """Parse untrusted XML safely; raise `UpstreamError` on garbage (HTML error pages…)."""
    if not content or not content.strip():
        raise UpstreamError(f"{source}: empty response")
    try:
        return SafeET.fromstring(content)
    except ParseError as exc:
        snippet = content[:80].decode("utf-8", "replace").replace("\n", " ")
        raise UpstreamError(f"{source}: unparseable feed ({snippet!r})") from exc


def int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def dedupe(signals: Iterable[RawSignal]) -> list[RawSignal]:
    """Drop repeats of the *same item*: same URL, or same headline from the same publisher.

    The same headline from different publishers is syndication — a coverage
    signal the pipeline counts — so it is deliberately kept.
    """
    seen: set[tuple[str, str]] = set()
    out: list[RawSignal] = []
    for sig in signals:
        title = re.sub(r"\W+", " ", sig.title.lower()).strip()
        keys = {("title", f"{title}|{(sig.publisher or '').lower()}")}
        if sig.url:
            keys.add(("url", sig.url))
        if keys & seen:
            continue
        seen |= keys
        out.append(sig)
    return out


def newest_first(signals: list[RawSignal]) -> list[RawSignal]:
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(signals, key=lambda s: s.timestamp or epoch, reverse=True)


async def gather_partial(*aws: Awaitable[T]) -> list[T | None]:
    """Run sub-requests concurrently and tolerate partial failure.

    Returns results in call order, with `None` in place of failed calls. If
    *every* call failed, re-raises the first error so the orchestrator reports
    the source as "error" rather than "empty".
    """
    results = await asyncio.gather(*aws, return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors and len(errors) == len(results):
        raise errors[0]
    for err in errors:  # partial degradation stays visible in logs
        logger.info("source sub-request failed: %s: %s", type(err).__name__, str(err)[:160])
    return [None if isinstance(r, BaseException) else r for r in results]


def sanitized_error(exc: Exception, provider: str) -> UpstreamError:
    """Turn an HTTP error into a message that never echoes the request URL.

    Keyed providers put API tokens in query strings; httpx's default message
    includes the full URL, which would leak the key into source reports/logs.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        hint = {401: "invalid API key", 403: "access denied", 429: "rate limited"}.get(status, "request failed")
        return UpstreamError(f"{provider}: HTTP {status} ({hint})")
    if isinstance(exc, UpstreamError):
        return exc
    return UpstreamError(f"{provider}: {type(exc).__name__}")
