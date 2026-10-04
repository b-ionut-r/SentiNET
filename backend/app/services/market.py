"""Market overview: fear & greed, indices, Reddit/StockTwits trending, market narratives.

Five providers are queried concurrently (each time-boxed), then the pure
analytics layer derives the regime and clusters the headlines. The result is
cached for a few minutes and computed once for concurrent callers.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.core.cache import cached
from app.core.sync import run_cpu
from app.schemas import FearGreed, MarketOverview
from app.services.errors import Unavailable
from app.services.tasks import Outcome, deferred, describe_error, run_bounded, status_of

logger = logging.getLogger(__name__)

MARKET_TTL = 180.0




@cached(ttl=MARKET_TTL, none_ttl=0.0, maxsize=4)
async def get_market_overview() -> MarketOverview:
    """The market-wide picture (raises `Unavailable` only if synthesis itself fails)."""
    timeout = settings.intel_timeout

    def task(module: str, fn: str) -> Any:
        return run_bounded(deferred(f"app.intel.{module}", fn), timeout, keep_alive=True, name=fn)

    fg, crypto, indices, trending, headlines = await asyncio.gather(
        task("market", "get_cnn_fear_greed"),
        task("market", "get_crypto_fear_greed"),
        task("market_data", "get_indices"),
        task("market", "get_trending"),
        task("market", "get_market_headlines"),
    )
    status = {
        "fear_greed": status_of(fg, FearGreed),
        "crypto_fear_greed": status_of(crypto, FearGreed),
        "indices": status_of(indices, list),
        "trending": status_of(trending, list),
        "headlines": status_of(headlines, list),
    }

    def value(out: Outcome[Any], key: str) -> Any:
        return out.value if status[key] == "ok" else None

    try:
        from app.analytics.market import build_market_overview

        return await run_cpu(
            build_market_overview,
            value(fg, "fear_greed"),
            value(crypto, "crypto_fear_greed"),
            list(value(indices, "indices") or []),
            list(value(trending, "trending") or []),
            list(value(headlines, "headlines") or []),
            status,
            datetime.now(UTC),
        )
    except Exception as exc:
        logger.exception("market overview build failed")
        raise Unavailable(f"Market overview could not be computed ({describe_error(exc)}).") from exc
