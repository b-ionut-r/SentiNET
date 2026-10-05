"""Tone ↔ price history for one ticker ("does news tone lead this stock?").

Gathers, concurrently and time-boxed: GDELT daily tone/volume, daily closes,
stored SentiNET snapshots and Wikipedia pageviews; the pure analytics layer
(`build_history`) aligns them and computes the lead/lag correlations.

The tone and Wikipedia calls use the analyzer's own arguments, so both share
the provider caches. When history obtains data that the ticker's cached
analysis went without (GDELT refused or timed out during that run), the
analysis is superseded (`analyzer.intel_landed`) and the next load recomputes
with it, instead of a verdict saying "tone not loaded" being served next to a
drawn tone pane for the rest of its cache TTL.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import settings
from app.core.sync import run_cpu
from app.schemas import HistoryResponse, ToneTrend
from app.services import analyzer
from app.services.analyzer import TONE_DAYS, resolve_or_bare
from app.services.errors import Unavailable
from app.services.tasks import Outcome, deferred, describe_error, has_data, run_bounded, status_of
from app.storage import db

logger = logging.getLogger(__name__)

SNAPSHOT_LIMIT = 5000


def _landed(symbol: str, key: str, expect: type, task: asyncio.Task[Any]) -> None:
    """A kept-alive call outlived the request: if it brought data, tell the analyzer anyway."""
    if task.cancelled() or task.exception() is not None:
        return
    value = task.result()
    if isinstance(value, expect) and has_data(value):
        analyzer.intel_landed(symbol, key)


def _share(symbol: str, key: str, out: Outcome[Any], status: str, expect: type) -> None:
    """Report data fetched with the analyzer's own arguments (now in the provider cache)."""
    if status == "ok":
        analyzer.intel_landed(symbol, key)
    elif out.pending is not None:
        out.pending.add_done_callback(functools.partial(_landed, symbol, key, expect))


async def get_history(symbol: str, days: int) -> HistoryResponse:
    """History for a canonical symbol over `days` (provider failures degrade, never raise)."""
    company = await resolve_or_bare(symbol)
    since = datetime.now(UTC) - timedelta(days=days)
    timeout = settings.intel_timeout
    # Same arguments as the analyzer's calls, so both hit the same provider caches.
    tone, closes, snaps, wiki = await asyncio.gather(
        run_bounded(deferred("app.intel.gdelt", "get_tone_trend", company, TONE_DAYS),
                    timeout * 1.5, keep_alive=True, name="tone"),
        run_bounded(deferred("app.intel.market_data", "get_daily_closes", symbol),
                    timeout, keep_alive=True, name="closes"),
        run_bounded(lambda: db.list_snapshots(symbol, SNAPSHOT_LIMIT, since), 5.0, name="snapshots"),
        run_bounded(deferred("app.intel.attention", "get_wiki_pageviews", company, max(days, TONE_DAYS)),
                    timeout, keep_alive=True, name="wiki"),
    )
    status = {
        "tone": status_of(tone, ToneTrend),
        "price": status_of(closes, list),
        "snapshots": status_of(snaps),
        "wiki": status_of(wiki, list),
    }
    _share(symbol, "tone", tone, status["tone"], ToneTrend)
    if max(days, TONE_DAYS) == TONE_DAYS:  # a longer window is another cache entry than the analyzer's
        _share(symbol, "wiki", wiki, status["wiki"], list)

    def value(out: Outcome[Any], key: str) -> Any:
        return out.value if status[key] == "ok" else None

    try:
        from app.analytics.stats import build_history

        return await run_cpu(
            build_history,
            symbol,
            days,
            value(tone, "tone"),
            list(value(closes, "price") or []),
            list(reversed(value(snaps, "snapshots") or [])),  # oldest first
            value(wiki, "wiki"),
            status,
        )
    except Exception as exc:
        logger.exception("history build failed for %s", symbol)
        raise Unavailable(f"History for {symbol} could not be computed ({describe_error(exc)}).") from exc
