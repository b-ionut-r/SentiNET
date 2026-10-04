"""Nasdaq.com symbol RSS — syndicated Motley Fool, Zacks, Barchart, MarketBeat… coverage.

Why trusted: Nasdaq's editorial desk tags each article with the tickers it
covers (`nasdaq:tickers`, primary symbol first), and items carry a summary.
Limits: 15 items; heavy on listicles and market wraps that tag 30+ tickers
(left to the relevance filter); unknown symbols silently fall back to the
site-wide "Latest Article Feed", which we detect and discard.

`ticker_specific` is True when the article is tagged with this issuer alone
(symbol or share-class sibling), or when Nasdaq's primary-symbol convention
(first tag repeated: "NVDA,NVDA,MU") names the issuer *and* the headline names
the company. The convention alone is not enough: "NVDA,NVDA,IONQ" was a
two-stock quantum-computing listicle that never mentions Nvidia.
Docs: https://www.nasdaq.com/feed/rssoutbound?symbol={SYMBOL}
"""
from __future__ import annotations

from collections.abc import Collection

from app.core import http
from app.schemas import SignalKind
from app.sources.base import CompanyRef, RawSignal, SourceBatch
from app.sources.query import Mentions, issuer_symbols, search_terms, us_symbol
from app.sources.util import (
    clean_text,
    is_recent,
    newest_first,
    parse_rfc822,
    parse_xml,
)

URL = "https://www.nasdaq.com/feed/rssoutbound"
NS = {"nasdaq": "http://nasdaq.com/reference/feeds/1.0", "dc": "http://purl.org/dc/elements/1.1/"}


def is_specific(tickers: list[str], issuer: set[str], title_names_it: bool) -> bool:
    tagged = {t for t in tickers if t}
    if tagged and tagged <= issuer:
        return True
    primary = len(tickers) >= 2 and tickers[0] == tickers[1] and tickers[0] in issuer
    return primary and title_names_it


def parse_feed(
    content: bytes, symbol: str, issuer: Collection[str] | None = None, mentions: Mentions | None = None
) -> list[RawSignal]:
    """`issuer`: the company's symbols (default: `symbol`); `mentions`: headline check for the primary-tag rule."""
    own = set(issuer or ()) | {symbol.upper()}
    root = parse_xml(content, "nasdaq")
    channel_title = (root.findtext("./channel/title") or "").upper()
    if symbol.upper() not in channel_title.split():  # generic fallback feed, not this symbol's
        return []
    out: list[RawSignal] = []
    for item in root.iterfind("./channel/item"):
        title = clean_text(item.findtext("title"))
        if not title:
            continue
        tickers = [t.strip().upper() for t in (item.findtext("nasdaq:tickers", namespaces=NS) or "").split(",")]
        tagged = {t for t in tickers if t}
        named = mentions is not None and mentions.about(title)
        out.append(
            RawSignal(
                title=title,
                body=clean_text(item.findtext("description"), limit=600) or None,
                url=item.findtext("link"),
                publisher=clean_text(item.findtext("dc:creator", namespaces=NS)) or "Nasdaq",
                timestamp=parse_rfc822(item.findtext("pubDate")),
                ticker_specific=is_specific(tickers, own, named),
                extra={"symbols": len(tagged)},
            )
        )
    return out


class NasdaqSource:
    key = "nasdaq"
    label = "Nasdaq"
    kind: SignalKind = "news"
    weight = 1.0
    requires_key = False
    description = "Nasdaq.com symbol feed (Motley Fool, Zacks, Barchart…) with editor ticker tags and summaries."
    docs_url: str | None = "https://www.nasdaq.com/nasdaq-RSS-Feeds"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return us_symbol(company) is not None

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        symbol = us_symbol(company)
        if symbol is None:
            return SourceBatch()
        resp = await http.fetch(URL, params={"symbol": symbol})
        parsed = parse_feed(resp.content, symbol, issuer_symbols(company), Mentions(search_terms(company)))
        signals = [s for s in parsed if is_recent(s.timestamp)]
        return SourceBatch(signals=newest_first(signals))
