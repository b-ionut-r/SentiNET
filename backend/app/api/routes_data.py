"""Market data endpoints: price candles, tone/price history, market overview, symbol search."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import Symbol
from app.config import settings
from app.schemas import HistoryResponse, MarketOverview, PriceResponse, SymbolMatch
from app.services import history, market
from app.services.errors import InvalidInput
from app.services.tasks import deferred, run_bounded

logger = logging.getLogger(__name__)

router = APIRouter(tags=["market data"])

# Range → candle interval (documented contract with the intel layer).
RANGE_INTERVALS: dict[str, str] = {
    "1D": "5m", "5D": "30m", "1M": "1d", "3M": "1d", "6M": "1d", "1Y": "1d", "5Y": "1wk",
}
SEARCH_TIMEOUT = 8.0


@router.get("/price/{ticker}", response_model=PriceResponse)
async def get_price(
    symbol: Symbol,
    range_: Annotated[str, Query(alias="range", description="1D | 5D | 1M | 3M | 6M | 1Y | 5Y")] = "1M",
) -> PriceResponse:
    """OHLCV candles. Provider failures return `available=false` with the reason (never a 5xx)."""
    rng = range_.strip().upper()
    if rng not in RANGE_INTERVALS:
        raise InvalidInput(f"Unknown range '{range_}'. Use one of: {', '.join(RANGE_INTERVALS)}.")
    out = await run_bounded(deferred("app.intel.market_data", "get_price_history", symbol, rng),
                            settings.intel_timeout, keep_alive=True, name="price")
    if out.ok and isinstance(out.value, PriceResponse):
        return out.value
    error = out.error or ("no price data" if out.value is None else "unexpected price payload")
    return PriceResponse(ticker=symbol, range=rng, interval=RANGE_INTERVALS[rng], available=False, error=error)


@router.get("/history/{ticker}", response_model=HistoryResponse)
async def get_history(
    symbol: Symbol,
    days: Annotated[int, Query(ge=7, le=365, description="Window in days (GDELT tone covers ≤ 90)")] = 90,
) -> HistoryResponse:
    """Daily global news tone vs. price with lead/lag correlation ("does tone lead this stock?")."""
    return await history.get_history(symbol, days)


@router.get("/market", response_model=MarketOverview)
async def get_market() -> MarketOverview:
    """Market regime, fear & greed, indices, trending tickers and market-wide narratives (cached ~3 min)."""
    return await market.get_market_overview()


@router.get("/search", response_model=list[SymbolMatch])
async def search(
    q: Annotated[str, Query(min_length=1, max_length=48, description="Ticker or company name")],
    limit: Annotated[int, Query(ge=1, le=20)] = 8,
) -> list[SymbolMatch]:
    """Symbol autocomplete. Never fails: provider problems yield an empty list."""
    query = q.strip().lstrip("$").strip()
    if not query:
        return []
    out = await run_bounded(deferred("app.resolve.symbols", "search_symbols", query, limit),
                            SEARCH_TIMEOUT, name="search")
    if not out.ok:
        logger.info("symbol search failed for %r: %s", query, out.error)
        return []
    return [m for m in (out.value or []) if isinstance(m, SymbolMatch)][:limit]
