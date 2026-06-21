"""Probe each source for a ticker and report what the runtime network allows.

Usage:
    python -m scripts.check_sources AAPL

Run this from the `backend/` directory with the venv active. It calls every
registered source directly (bypassing cache) and prints a one-line status per
source, so you can see which free endpoints are reachable from the current
environment's network policy.
"""
from __future__ import annotations

import asyncio
import sys

from app.cache.ticker_map import get_company
from app.sources.registry import gather_signals
from app.utils.http import close_client


async def main(ticker: str) -> None:
    company = await get_company(ticker)
    print(f"Ticker {ticker} -> company: {company}\n")
    results = await gather_signals(ticker, company)
    width = max(len(r.source.name) for r in results)
    for r in results:
        line = f"  {r.source.name:<{width}}  {r.status:<7}  signals={len(r.signals):<3}"
        if r.error:
            line += f"  error={r.error}"
        print(line)
    ok = sum(1 for r in results if r.status == "ok")
    print(f"\n{ok}/{len(results)} sources returned data.")
    await close_client()


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(main(symbol.upper()))
