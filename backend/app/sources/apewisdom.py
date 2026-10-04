"""ApeWisdom — Reddit (WSB, r/stocks, r/investing, r/CryptoCurrency…) mention leaderboard.

Why trusted: counts cashtag/ticker mentions across the big finance subreddits
every few minutes and publishes the 24 h-ago snapshot alongside, so mention
*momentum* and rank changes are measurable. It provides no text and no
sentiment — this source emits metrics only (never synthetic text).
Limits: a global leaderboard (~700 stocks over 7 pages, ~140 coins), so the
whole board is fetched once and cached for 10 minutes; tickers absent from the
board had too few mentions to rank and yield an empty batch (status "empty").

Metrics: reddit_mentions, reddit_mentions_prev (24 h ago), reddit_rank,
reddit_rank_prev, reddit_upvotes, reddit_tracked (board size, for "rank 26 of 692").
Docs: https://apewisdom.io/api/
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.core import http
from app.core.cache import cached
from app.schemas import SignalKind
from app.sources.base import CompanyRef, SourceBatch
from app.sources.query import us_symbol

URL = "https://apewisdom.io/api/v1.0/filter/{board}/page/{page}"
MAX_PAGES = 10


@cached(ttl=600, none_ttl=60, maxsize=4)
async def load_board(board: str) -> dict[str, dict[str, Any]]:
    """Every ranked row of a leaderboard ("all-stocks" | "all-crypto"), keyed by upper-case ticker."""
    first = await http.fetch_json(URL.format(board=board, page=1))
    pages = max(1, min(int(first.get("pages") or 1), MAX_PAGES))
    rest = await asyncio.gather(*(http.fetch_json(URL.format(board=board, page=n)) for n in range(2, pages + 1)))
    rows: dict[str, dict[str, Any]] = {}
    for page in (first, *rest):
        for row in page.get("results") or []:
            ticker = str(row.get("ticker") or "").upper()
            if ticker:
                rows.setdefault(ticker, row)
    return rows


def board_and_symbol(company: CompanyRef) -> tuple[str, str] | None:
    if company.is_crypto:
        return "all-crypto", f"{company.base_symbol.upper()}.X"
    symbol = us_symbol(company)
    return ("all-stocks", symbol) if symbol else None


def metrics_for(row: dict[str, Any], tracked: int) -> dict[str, Any]:
    mapping = {
        "reddit_mentions": "mentions",
        "reddit_mentions_prev": "mentions_24h_ago",
        "reddit_rank": "rank",
        "reddit_rank_prev": "rank_24h_ago",
        "reddit_upvotes": "upvotes",
    }
    out: dict[str, Any] = {}
    for key, field in mapping.items():
        value = row.get(field)
        if value is not None:
            try:
                out[key] = int(value)
            except (TypeError, ValueError):
                continue
    if out:
        out["reddit_tracked"] = tracked
    return out


class ApeWisdomSource:
    key = "apewisdom"
    label = "Reddit buzz (ApeWisdom)"
    kind: SignalKind = "social"
    weight = 0.5
    requires_key = False
    description = "Reddit mention rank and 24h momentum across WSB, r/stocks, r/investing and crypto subs (metrics only)."
    docs_url: str | None = "https://apewisdom.io/api/"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return board_and_symbol(company) is not None

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        target = board_and_symbol(company)
        if target is None:
            return SourceBatch()
        board, symbol = target
        rows = await load_board(board)
        row = rows.get(symbol)
        return SourceBatch(metrics=metrics_for(row, len(rows))) if row else SourceBatch()
