"""GET /api/analyze/{ticker} — the core sentiment endpoint."""
from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from app.cache import store
from app.cache.ticker_map import get_company
from app.pipeline.aggregate import build_response
from app.pipeline.normalize import normalize_and_score
from app.schemas import AnalyzeResponse
from app.sources.registry import gather_signals

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")


@router.get("/analyze/{ticker}", response_model=AnalyzeResponse)
async def analyze(ticker: str, refresh: bool = False) -> AnalyzeResponse:
    ticker = ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    if not refresh:
        cached = store.get_analysis(ticker)
        if cached is not None:
            cached.cached = True
            return cached

    company = await get_company(ticker)
    results = await gather_signals(ticker, company)
    signals = normalize_and_score(results)
    response = build_response(ticker, company, results, signals)

    store.set_analysis(ticker, response)
    return response
