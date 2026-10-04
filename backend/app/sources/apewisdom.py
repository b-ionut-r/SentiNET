"""ApeWisdom — Reddit (WSB, r/stocks, r/investing, r/CryptoCurrency…) mention leaderboard.

Why trusted: counts cashtag/ticker mentions across the big finance subreddits
every few minutes and publishes the 24 h-ago snapshot alongside, so mention
*momentum* and rank changes are measurable. It provides no text and no
sentiment — this source emits metrics only (never synthetic text).
Limits: a global leaderboard (~700 stocks over 7 pages, ~140 coins), so the
whole board is fetched once and cached for 10 minutes; tickers absent from the
board had too few mentions to rank and yield an empty batch (status "empty") —
unless some pages failed to load, in which case absence proves nothing and the
source reports an error. The board counts upper-case tokens, so ticker-words
(YOU, ALL, ES, DTE, CD: rank 9 "CLEAR Secure" was the word "YOU") are
unsupported on the stock board (`crowd_symbol_ambiguous`).

Metrics: reddit_mentions, reddit_mentions_prev (24 h ago), reddit_rank,
reddit_rank_prev, reddit_upvotes, reddit_tracked (board size, for "rank 26 of 692").
Docs: https://apewisdom.io/api/
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.core import http
from app.core.cache import cached
from app.core.http import UpstreamError
from app.schemas import SignalKind
from app.sources.base import CompanyRef, SourceBatch
from app.sources.query import crowd_symbol_ambiguous, us_symbol

URL = "https://apewisdom.io/api/v1.0/filter/{board}/page/{page}"
MAX_PAGES = 10
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Board:
    rows: dict[str, dict[str, Any]]  # upper-case ticker -> row
    complete: bool  # every page loaded


@cached(ttl=600, none_ttl=60, maxsize=4)
async def load_board(board: str) -> Board:
    """Every ranked row of a leaderboard ("all-stocks" | "all-crypto"); later pages may fail individually."""
    first = await http.fetch_json(URL.format(board=board, page=1))
    pages = max(1, min(int(first.get("pages") or 1), MAX_PAGES))
    rest = await asyncio.gather(
        *(http.fetch_json(URL.format(board=board, page=n)) for n in range(2, pages + 1)), return_exceptions=True
    )
    failed = [r for r in rest if isinstance(r, BaseException)]
    if failed:
        logger.info("apewisdom %s: %d of %d pages failed (%s)", board, len(failed), pages, type(failed[0]).__name__)
    rows: dict[str, dict[str, Any]] = {}
    for page in (first, *(r for r in rest if isinstance(r, dict))):
        for row in page.get("results") or []:
            ticker = str(row.get("ticker") or "").upper()
            if ticker:
                rows.setdefault(ticker, row)
    return Board(rows, complete=not failed)


def board_and_symbol(company: CompanyRef) -> tuple[str, str] | None:
    """Which board to read and the symbol as ApeWisdom writes it; None when unsupported."""
    if company.is_crypto:  # crypto subs: "SOL", "LINK", "ONE" mean the coins there
        return "all-crypto", f"{company.base_symbol.upper()}.X"
    symbol = us_symbol(company)
    if symbol is None or crowd_symbol_ambiguous(symbol):
        return None
    return "all-stocks", symbol


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
        try:
            value = int(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if key.startswith("reddit_rank") and value <= 0:  # rank_24h_ago 0 = was not ranked
            continue
        out[key] = value
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
        board_key, symbol = target
        board = await load_board(board_key)
        row = board.rows.get(symbol)
        if row is not None:
            return SourceBatch(metrics=metrics_for(row, len(board.rows)))
        if not board.complete:
            raise UpstreamError(f"apewisdom: leaderboard partially unavailable; {symbol} not found on the loaded pages")
        return SourceBatch()
