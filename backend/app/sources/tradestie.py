"""Tradestie — r/wallstreetbets top-50 most-discussed tickers (metrics only).

Why used: a cheap daily read of WSB discussion volume (`no_of_comments`) and
rank. Data-quality gate (important): live checks on 2026-10-04 showed the
feed's per-ticker `sentiment_score`/`sentiment` are FROZEN — NVDA 0.011
"Bullish", TSLA -0.292, SPY -0.165 on every date from 2026-01-05 to 2026-10-04
while comment counts changed daily. Publishing that as "WSB sentiment" would be
fabricated signal, so wsb_sentiment/wsb_label are emitted only when a
comparison against the list from 28 days earlier proves the scores are being
recomputed (self-healing if the provider fixes it). Ticker-words and WSB lingo
("AI", "UK", "CD", "HYSA", "CAPE" — all on the live board) are unsupported:
their counts measure the word, not the stock (`crowd_symbol_ambiguous`; MU, GM,
KO are real tickers there). The list is rebuilt through the day and is thin on
weekends (Sunday's top-50 summed to 100 comments vs 746 on a Thursday), so a
rank is only reported from a board with >= 300 comments and >= 5 for the ticker.

Metrics: wsb_comments, wsb_board_total (comments across the top-50, for share
and context), wsb_rank (1-50, gated as above) and — only when the sentiment
gate passes — wsb_sentiment (-1..1) and wsb_label ("bullish"/"bearish").
Docs: https://tradestie.com/apps/reddit/api/
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import httpx

from app.core import http
from app.core.cache import cached
from app.schemas import SignalKind
from app.sources.base import CompanyRef, SourceBatch
from app.sources.query import crowd_symbol_ambiguous, us_symbol
from app.sources.util import utc_now

URL = "https://tradestie.com/api/v1/apps/reddit"
MIN_BOARD_COMMENTS = 300  # below this the day's list is too partial to rank against
MIN_TICKER_COMMENTS = 5


def _index(rows: Any) -> dict[str, tuple[int, dict[str, Any]]]:
    if not isinstance(rows, list):
        return {}
    return {str(r.get("ticker", "")).upper(): (i + 1, r) for i, r in enumerate(rows) if isinstance(r, dict)}


@cached(ttl=900, none_ttl=60, maxsize=2)
async def load_today() -> dict[str, tuple[int, dict[str, Any]]]:
    return _index(await http.fetch_json(URL))


def scores_are_live(today: dict[str, tuple[int, dict]], past: dict[str, tuple[int, dict]]) -> bool:
    """False when most tickers carry the *identical* score four weeks apart (a frozen feed)."""
    common = [t for t in today if t in past]
    if len(common) < 5:
        return False
    same = sum(1 for t in common if today[t][1].get("sentiment_score") == past[t][1].get("sentiment_score"))
    return same / len(common) < 0.5


@cached(ttl=6 * 3600, none_ttl=300, maxsize=2)
async def sentiment_gate() -> bool | None:
    try:
        past_date = (utc_now() - timedelta(days=28)).strftime("%m-%d-%Y")
        past = _index(await http.fetch_json(URL, params={"date": past_date}))
        return scores_are_live(await load_today(), past)
    except (httpx.HTTPError, http.UpstreamError, ValueError):  # gate unavailable: fail closed, retry in 5 min
        return None


def _comments(row: dict[str, Any]) -> int:
    try:
        return max(0, int(row.get("no_of_comments") or 0))
    except (TypeError, ValueError):
        return 0


def board_total(board: dict[str, tuple[int, dict[str, Any]]]) -> int:
    return sum(_comments(row) for _, row in board.values())


def metrics_for(rank: int, row: dict[str, Any], sentiment_live: bool, total: int) -> dict[str, Any]:
    comments = _comments(row)
    out: dict[str, Any] = {"wsb_comments": comments, "wsb_board_total": total}
    if total >= MIN_BOARD_COMMENTS and comments >= MIN_TICKER_COMMENTS:
        out["wsb_rank"] = rank
    score, label = row.get("sentiment_score"), str(row.get("sentiment") or "").lower()
    if sentiment_live and isinstance(score, (int, float)) and label in ("bullish", "bearish"):
        out["wsb_sentiment"] = max(-1.0, min(1.0, float(score)))
        out["wsb_label"] = label
    return out


class TradestieSource:
    key = "tradestie"
    label = "WSB buzz (Tradestie)"
    kind: SignalKind = "social"
    weight = 0.4
    requires_key = False
    description = "r/wallstreetbets top-50 comment volume and rank (metrics only; frozen sentiment gated out)."
    docs_url: str | None = "https://tradestie.com/apps/reddit/api/"

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        symbol = us_symbol(company)
        return symbol is not None and not crowd_symbol_ambiguous(symbol)

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        if not self.supports(company):
            return SourceBatch()
        board = await load_today()
        hit = board.get(us_symbol(company) or "")
        if hit is None:
            return SourceBatch()
        rank, row = hit
        return SourceBatch(metrics=metrics_for(rank, row, bool(await sentiment_gate()), board_total(board)))
