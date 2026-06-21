"""GET /api/price/{ticker} — price history, cached independently of sentiment."""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from app.cache import store
from app.cache.ticker_map import get_company
from app.schemas import PricePoint, PriceResponse
from app.sources.yahoo import fetch_price

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")
_VALID_RANGES = {"1D", "5D", "1M", "6M", "1Y"}


@router.get("/price/{ticker}", response_model=PriceResponse)
async def price(ticker: str, range: str = "1M") -> PriceResponse:
    ticker = ticker.strip().upper()
    rng = range.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")
    if rng not in _VALID_RANGES:
        rng = "1M"

    cached = store.get_price(ticker, rng)
    if cached is not None:
        return cached

    company = await get_company(ticker)
    try:
        data = await fetch_price(ticker, rng)
        resp = PriceResponse(
            ticker=ticker,
            company=company,
            currency=data["currency"],
            current_price=data["current_price"],
            previous_close=data["previous_close"],
            change=data["change"],
            change_pct=data["change_pct"],
            range=rng,
            points=[PricePoint(t=p["t"], close=p["close"]) for p in data["points"]],
            available=bool(data["points"]),
        )
    except Exception as exc:  # noqa: BLE001 - price is best-effort
        resp = PriceResponse(
            ticker=ticker, company=company, range=rng,
            available=False, error=str(exc)[:200],
        )

    store.set_price(ticker, rng, resp)
    return resp
