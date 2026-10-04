"""StockTwits symbol stream — the retail trader crowd, with *author-declared* stance.

Why trusted: every message is posted to the symbol's stream by a trader, and
many carry an explicit Bullish/Bearish tag chosen by the author — a direct,
unmodelled sentiment label. Limits: 30 messages per page via the `max` cursor;
a structural bullish skew (analytics baselines it); spam that cashtags many
symbols; on quiet names two pages reach back 2+ weeks, on hot names they cover
~30 minutes (so we page up to 4 times while the sample spans < 24 h).

Attribution rule (honest labels): a message's tag/subject counts for this
symbol only when it is the message's sole symbol, or its *first* cashtag in a
message naming at most 3 symbols ("$MU And $AAPL wants Chinese memory?" tagged
Bullish is a $MU call, not an $AAPL one). Others are kept as text but marked
not ticker-specific and unlabeled.

Metrics — stance tallies only count tags from the last 72 h, so a quiet name's
week-old calls never pose as the current crowd:
* stocktwits_bullish / stocktwits_bearish — attributable tagged messages;
* stocktwits_bull_authors / stocktwits_bear_authors — one vote per author (their
  latest tag): the preferred ratio input, since one prolific poster supplied 9
  of BRK-B's 14 bearish tags in a live check;
* stocktwits_window_hours — hours those tallies actually cover (≤ 72);
* stocktwits_messages (fetched), stocktwits_watchers (`symbol.watchlist_count`),
  stocktwits_span_hours (time covered by the fetched sample: a velocity gauge).
As text, only messages from the last 14 days are returned (quiet names' pages
reach back months: ICE's two pages spanned 74 days live).
Engagement = likes + reshares + replies.
Docs: https://api.stocktwits.com/developers/docs (public read endpoint).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Literal

import httpx

from app.core import http
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import us_symbol
from app.sources.util import clean_plain, int_or_zero, is_recent, newest_first, parse_iso, utc_now

URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
MIN_PAGES, MAX_PAGES = 2, 4
DEEP_SPAN_HOURS = 24.0  # keep paging while the sample covers less than this
TAG_WINDOW = timedelta(hours=72)
_CASHTAG = re.compile(r"\$([A-Za-z][A-Za-z0-9.\-]{0,11})")


def stream_symbol(company: CompanyRef) -> str | None:
    """NVDA -> NVDA, BRK-B -> BRK.B, BTC-USD -> BTC.X; None when StockTwits has no stream."""
    if company.is_crypto:
        return f"{company.base_symbol.upper()}.X"
    return us_symbol(company)


def attributable(message: dict[str, Any], symbol: str) -> bool:
    """`symbol` is the stream symbol ("AAPL", "BRK.B", "BTC.X")."""
    symbols = [str(s.get("symbol", "")).upper() for s in message.get("symbols") or []]
    if symbols == [symbol]:
        return True
    first = _CASHTAG.search(message.get("body") or "")
    if first is None or len(symbols) > 3:
        return False
    return first.group(1).rstrip(".-").upper() in {symbol, symbol.removesuffix(".X")}


def parse_message(message: dict[str, Any], symbol: str) -> RawSignal | None:
    text = clean_plain(message.get("body"), limit=600)
    if not text:
        return None
    own = attributable(message, symbol)
    basic = ((message.get("entities") or {}).get("sentiment") or {}).get("basic")
    label: Literal["bullish", "bearish"] | None = None
    if own and basic in ("Bullish", "Bearish"):
        label = "bullish" if basic == "Bullish" else "bearish"
    user = message.get("user") or {}
    username = user.get("username")
    engagement = (
        int_or_zero((message.get("likes") or {}).get("total"))
        + int_or_zero((message.get("reshares") or {}).get("reshared_count"))
        + int_or_zero((message.get("conversation") or {}).get("replies"))
    )
    msg_id = message.get("id")
    return RawSignal(
        title=text,
        url=f"https://stocktwits.com/{username}/message/{msg_id}" if username and msg_id else None,
        author=username,
        publisher="StockTwits",
        timestamp=parse_iso(message.get("created_at")),
        engagement=engagement,
        user_label=label,
        ticker_specific=own,
        extra={"symbols": len(message.get("symbols") or []), "followers": int_or_zero(user.get("followers"))},
    )


def span_hours(signals: list[RawSignal]) -> float | None:
    times = [s.timestamp for s in signals if s.timestamp]
    return (max(times) - min(times)).total_seconds() / 3600 if len(times) >= 2 else None


def stance_metrics(signals: list[RawSignal], now: datetime) -> dict[str, Any]:
    """Tag tallies within `TAG_WINDOW`: per message and one-vote-per-author (latest tag wins)."""
    since = now - TAG_WINDOW
    recent = [s for s in newest_first(signals) if s.user_label and s.timestamp and s.timestamp >= since]
    latest: dict[str, str] = {}
    for s in recent:
        latest.setdefault(s.author or f"anon-{id(s)}", s.user_label or "")
    oldest = min((s.timestamp for s in signals if s.timestamp), default=now)
    return {
        "stocktwits_bullish": sum(1 for s in recent if s.user_label == "bullish"),
        "stocktwits_bearish": sum(1 for s in recent if s.user_label == "bearish"),
        "stocktwits_bull_authors": sum(1 for label in latest.values() if label == "bullish"),
        "stocktwits_bear_authors": sum(1 for label in latest.values() if label == "bearish"),
        "stocktwits_window_hours": round((now - max(oldest, since)).total_seconds() / 3600, 2),
    }


def build_batch(pages: list[dict[str, Any]], symbol: str, now: datetime | None = None) -> SourceBatch:
    messages: dict[Any, dict[str, Any]] = {}
    for page in pages:
        for msg in page.get("messages") or []:
            messages.setdefault(msg.get("id"), msg)
    signals = [s for s in (parse_message(m, symbol) for m in messages.values()) if s is not None]
    if not signals:
        return SourceBatch()
    now = now or utc_now()
    watchers = ((pages[0].get("symbol") or {}).get("watchlist_count")) if pages else None
    metrics: dict[str, Any] = {"stocktwits_messages": len(signals), **stance_metrics(signals, now)}
    if watchers is not None:
        metrics["stocktwits_watchers"] = int_or_zero(watchers)
    span = span_hours(signals)
    if span is not None:
        metrics["stocktwits_span_hours"] = round(span, 2)
    # Metrics describe the whole sample; as text, quiet names' months-old posts are not current sentiment.
    recent = [s for s in signals if is_recent(s.timestamp, now=now)]
    return SourceBatch(signals=newest_first(recent), metrics=metrics)


def _needs_more(pages: list[dict[str, Any]], symbol: str) -> bool:
    if len(pages) < MIN_PAGES:
        return True
    if len(pages) >= MAX_PAGES:
        return False
    signals = [s for s in (parse_message(m, symbol) for p in pages for m in p.get("messages") or []) if s]
    span = span_hours(signals)
    return span is not None and span < DEEP_SPAN_HOURS


class StockTwitsSource:
    key = "stocktwits"
    label = "StockTwits"
    kind: SignalKind = "social"
    weight = 0.8
    requires_key = False
    description = "Retail trader stream with author-tagged Bullish/Bearish stance, watcher count and chatter velocity."
    docs_url: str | None = "https://api.stocktwits.com/developers/docs"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return stream_symbol(company) is not None

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        symbol = stream_symbol(company)
        if symbol is None:
            return SourceBatch()
        pages: list[dict[str, Any]] = []
        params: dict[str, Any] = {}
        while _needs_more(pages, symbol):
            try:
                page = await http.fetch_json(URL.format(symbol=symbol), params=params or None)
            except (httpx.HTTPStatusError, http.UpstreamError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
                    return SourceBatch()  # "Symbol not found"
                if pages:  # keep page 1 if a later page fails
                    break
                raise
            pages.append(page)
            cursor = page.get("cursor") or {}
            if not cursor.get("more") or not cursor.get("max"):
                break
            params = {"max": cursor["max"]}
        return build_batch(pages, symbol)
