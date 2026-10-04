"""Seeking Alpha symbol feed — SA News flashes plus contributor analysis.

Why trusted: SA News is a fast, editor-written market wire, and contributor
articles carry explicit bull/bear theses. Limits: the symbol feed includes
market-wide items that merely *tag* the symbol (an S&P earnings preview tags 14
tickers), item links point at the symbol page (we rebuild the real article URL
from the GUID), ~30 items reaching back weeks for quiet names.

`ticker_specific` is honest: True only when SA tags *this symbol alone*
(ignoring its own Canadian cross-listing aliases like "NVDA:CA"); crypto feeds
tag BTC-USD on general crypto items, so crypto is never marked specific.
Docs: https://seekingalpha.com/api/sa/combined/{SYMBOL}.xml (public RSS).
"""
from __future__ import annotations

import httpx

from app.core import http
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import us_symbol
from app.sources.util import (
    clean_text,
    is_recent,
    newest_first,
    parse_rfc822,
    parse_xml,
)

URL = "https://seekingalpha.com/api/sa/combined/{symbol}.xml"
NS = {"sa": "https://seekingalpha.com/api/1.0"}


def feed_symbol(company: CompanyRef) -> str | None:
    return company.ticker.upper() if company.is_crypto else us_symbol(company)


def article_url(guid: str | None, fallback: str | None) -> str | None:
    """'https://seekingalpha.com/MarketCurrent:4650034' -> 'https://seekingalpha.com/news/4650034'."""
    tail = (guid or "").rsplit("/", 1)[-1]
    kind, _, ident = tail.partition(":")
    if ident.isdigit():
        if kind == "MarketCurrent":
            return f"https://seekingalpha.com/news/{ident}"
        if kind == "Article":
            return f"https://seekingalpha.com/article/{ident}"
    return fallback


def parse_feed(content: bytes, symbol: str, crypto: bool = False) -> list[RawSignal]:
    root = parse_xml(content, "seeking_alpha")
    out: list[RawSignal] = []
    for item in root.iterfind("./channel/item"):
        title = clean_text(item.findtext("title"))
        if not title:
            continue
        tagged = [s.findtext("sa:symbol", default="", namespaces=NS).upper() for s in item.iterfind("sa:stock", NS)]
        primary_tags = {t for t in tagged if t and ":" not in t}  # drop "NVDA:CA"-style aliases
        guid = item.findtext("guid")
        out.append(
            RawSignal(
                title=title,
                url=article_url(guid, item.findtext("link")),
                author=clean_text(item.findtext("sa:author_name", namespaces=NS)) or None,
                publisher="Seeking Alpha",
                timestamp=parse_rfc822(item.findtext("pubDate")),
                ticker_specific=not crypto and primary_tags == {symbol},
                extra={
                    "symbols": len(primary_tags),
                    "type": "analysis" if "Article:" in (guid or "") else "news",
                },
            )
        )
    return out


class SeekingAlphaSource:
    key = "seeking_alpha"
    label = "Seeking Alpha"
    kind: SignalKind = "news"
    weight = 1.0
    requires_key = False
    description = "SA News flashes and contributor bull/bear analysis tagged with the symbol."
    docs_url: str | None = "https://seekingalpha.com"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return feed_symbol(company) is not None

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        symbol = feed_symbol(company)
        if symbol is None:
            return SourceBatch()
        try:
            resp = await http.fetch(URL.format(symbol=symbol))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:  # SA doesn't cover this symbol
                return SourceBatch()
            raise
        signals = [s for s in parse_feed(resp.content, symbol, company.is_crypto) if is_recent(s.timestamp)]
        return SourceBatch(signals=newest_first(signals))
