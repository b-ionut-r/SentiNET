"""StockTwits symbol stream — the retail trader crowd, with *author-declared* stance.

Why trusted: every message is posted to the symbol's stream by a trader, and
many carry an explicit Bullish/Bearish tag chosen by the author — a direct,
unmodelled sentiment label. Limits: 30 messages per page (we read 2 pages via
the `max` cursor), a structural bullish skew (analytics baselines it), and spam
that cashtags many symbols at once.

Attribution rule (honest labels): a message's tag/subject counts for this
symbol only when it is the message's sole symbol, or its *first* cashtag in a
message naming at most 3 symbols ("$MU And $AAPL wants Chinese memory?" tagged
Bullish is a $MU call, not an $AAPL one). Others are kept as text but marked
not ticker-specific and unlabeled.

Metrics: stocktwits_bullish / stocktwits_bearish (attributable tags),
stocktwits_messages (fetched), stocktwits_watchers (`symbol.watchlist_count`),
stocktwits_span_hours (time covered by the fetched messages; messages/hour is
a chatter-velocity gauge). Engagement = likes + reshares + replies.
Docs: https://api.stocktwits.com/developers/docs (public read endpoint).
"""
from __future__ import annotations

import re
from typing import Any, Literal

import httpx

from app.core import http
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import us_symbol
from app.sources.util import clean_text, int_or_zero, newest_first, parse_iso

URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
PAGES = 2
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
    text = clean_text(message.get("body"), limit=600)
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


def build_batch(pages: list[dict[str, Any]], symbol: str) -> SourceBatch:
    messages: dict[Any, dict[str, Any]] = {}
    for page in pages:
        for msg in page.get("messages") or []:
            messages.setdefault(msg.get("id"), msg)
    signals = [s for s in (parse_message(m, symbol) for m in messages.values()) if s is not None]
    if not signals:
        return SourceBatch()
    times = [s.timestamp for s in signals if s.timestamp]
    watchers = ((pages[0].get("symbol") or {}).get("watchlist_count")) if pages else None
    metrics: dict[str, Any] = {
        "stocktwits_bullish": sum(1 for s in signals if s.user_label == "bullish"),
        "stocktwits_bearish": sum(1 for s in signals if s.user_label == "bearish"),
        "stocktwits_messages": len(signals),
    }
    if watchers is not None:
        metrics["stocktwits_watchers"] = int_or_zero(watchers)
    if len(times) >= 2:
        metrics["stocktwits_span_hours"] = round((max(times) - min(times)).total_seconds() / 3600, 2)
    return SourceBatch(signals=newest_first(signals), metrics=metrics)


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
        for _ in range(PAGES):
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
