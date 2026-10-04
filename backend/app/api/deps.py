"""Shared request dependencies."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Path

from app.services.analyzer import normalize


def ticker_param(
    ticker: Annotated[str, Path(description="Ticker symbol, e.g. NVDA, BRK-B, SHOP.TO, BTC-USD, $aapl")],
) -> str:
    """Canonical symbol from the path; invalid input → 400 with a helpful message."""
    return normalize(ticker)


Symbol = Annotated[str, Depends(ticker_param)]
