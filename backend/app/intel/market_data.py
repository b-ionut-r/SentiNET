"""Market data & structured "smart money" intel from Yahoo Finance (yfinance).

Raw Yahoo HTTP endpoints rate-limit this deployment's IPs, but yfinance's
curl_cffi browser impersonation works, so every call goes through yfinance on
the dedicated `run_yahoo` pool. Yahoo throttles bursts: each upstream payload
is fetched once and cached (`info` 2 min, daily bars 5 min, analyst/earnings/
insider tables `intel_cache_ttl`), and every public function is derived from
those few payloads by the pure functions in `app.intel.transforms`.

Contract: functions return None/[] for "no data" (crypto has no analysts, an
ETF has no earnings) and raise `UpstreamError` only for hard upstream failures
so the orchestrator can report an honest "error" status.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta, UTC
from typing import Any, TypeVar

import pandas as pd

from app.config import settings
from app.core.cache import cached
from app.core.http import UpstreamError
from app.core.sync import run_yahoo
from app.intel import transforms as tx
from app.resolve.symbols import is_crypto_symbol, logo_url_for
from app.schemas import (
    AnalystView,
    Catalyst,
    EarningsView,
    IndexQuote,
    InsiderView,
    PriceResponse,
    Profile,
    Quote,
    Technicals,
)
from app.sources.base import CompanyRef

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Range -> (yfinance period, interval). Daily ranges are sliced from one cached
# 2-year frame instead of separate requests.
RANGES: dict[str, tuple[str, str]] = {
    "1D": ("1d", "5m"),
    "5D": ("5d", "30m"),
    "1M": ("1mo", "1d"),
    "3M": ("3mo", "1d"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
}
_DAILY_SLICE = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}  # months

CRYPTO_QUOTE_TIMEOUT = 4.0  # s: the tape never waits longer than this on a crypto benchmark's quote
INDEX_SYMBOLS: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow 30",
    "IWM": "Russell 2000",
    "^VIX": "VIX",
    "^TNX": "10Y yield",
    "GC=F": "Gold",
    "BTC-USD": "Bitcoin",
}


def _now() -> datetime:
    return datetime.now(UTC)


def _is_not_found(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "not found" in text or "404" in text or "no data found" in text or "delisted" in text


async def _yahoo(fn: Callable[..., T], *args: Any, what: str) -> T | None:
    """Run a yfinance call; None for "not found", UpstreamError for real failures."""
    try:
        return await run_yahoo(fn, *args)
    except Exception as exc:
        if _is_not_found(exc):
            return None
        name = type(exc).__name__
        if "RateLimit" in name or "too many requests" in str(exc).lower():
            raise UpstreamError(f"Yahoo rate limit ({what})") from exc
        raise UpstreamError(f"Yahoo {what}: {name}: {str(exc)[:120]}") from exc


def _ticker(symbol: str):
    import yfinance as yf

    return yf.Ticker(symbol)


def _frame(value: Any) -> pd.DataFrame | None:
    return value if isinstance(value, pd.DataFrame) and not value.empty else None


# --------------------------------------------------------------------------- #
# Raw payloads (cached)
# --------------------------------------------------------------------------- #
def _fetch_info(symbol: str) -> dict[str, Any] | None:
    info = _ticker(symbol).info
    # yfinance returns a stub like {"trailingPegRatio": None} for unknown symbols.
    if not info or not (info.get("quoteType") or info.get("regularMarketPrice") or info.get("longName")):
        return None
    return dict(info)


@cached(ttl=settings.price_cache_ttl, none_ttl=30)
async def get_info(ticker: str) -> dict[str, Any] | None:
    """Yahoo `info` dict (names, quote, targets, dividend dates); None if unknown."""
    return await _yahoo(_fetch_info, ticker, what="quote")


def _fetch_history(
    symbol: str, period: str, interval: str, start: datetime | None
) -> tuple[pd.DataFrame | None, str | None]:
    """(bars, currency); (None, None) when Yahoo answers with no bars. Transport failures raise.

    By default `history()` swallows every request failure (connection refused, timeout, an HTML
    error page) and returns the same empty frame as a delisted symbol, so an outage would read
    as "no price data". `raise_errors=True` is yfinance 1.7's per-call switch (its replacement,
    `yf.config.debug.hide_exceptions`, is process-wide and would change other modules' yfinance
    calls): failures propagate to `_yahoo` (-> UpstreamError) while Yahoo's own "no data"
    answers arrive as `YFTickerMissingError`, mapped back to None here."""
    from yfinance.exceptions import YFTickerMissingError

    t = _ticker(symbol)
    kwargs: dict[str, Any] = {"interval": interval, "auto_adjust": False, "actions": False, "raise_errors": True}
    if start is not None:
        kwargs["start"] = start
    else:
        kwargs["period"] = period
    try:
        df = t.history(**kwargs)
    except YFTickerMissingError:  # YFPricesMissingError / YFTzMissingError: Yahoo answered "nothing"
        return None, None
    currency = None
    try:
        currency = (t.history_metadata or {}).get("currency")
    except Exception:  # noqa: BLE001 - metadata is optional
        currency = None
    return _frame(df), currency


@cached(ttl=300, none_ttl=30)
async def _daily_bars(ticker: str) -> tuple[pd.DataFrame, str | None] | None:
    """~2 years of daily OHLCV (feeds technicals, closes and 1M-1Y charts)."""
    res = await _yahoo(_fetch_history, ticker, "2y", "1d", None, what="daily history")
    if not res or res[0] is None:
        return None
    return res  # type: ignore[return-value]


@cached(ttl=settings.price_cache_ttl, none_ttl=30)
async def _bars(
    ticker: str, period: str, interval: str, crypto: bool
) -> tuple[pd.DataFrame, str | None] | None:
    start = None
    if crypto and period in {"1d", "5d"}:  # 24/7 markets: a rolling window, not "since midnight"
        start = _now() - timedelta(days=1 if period == "1d" else 5)
    res = await _yahoo(_fetch_history, ticker, period, interval, start, what=f"{period} history")
    if not res or res[0] is None:
        return None
    return res  # type: ignore[return-value]


def _fetch_analyst_frames(symbol: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    t = _ticker(symbol)
    return _frame(t.recommendations), _frame(t.upgrades_downgrades)


@cached(ttl=settings.intel_cache_ttl, none_ttl=60)
async def _analyst_frames(ticker: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None] | None:
    return await _yahoo(_fetch_analyst_frames, ticker, what="analyst ratings")


def _fetch_calendar(symbol: str) -> dict[str, Any] | None:
    cal = _ticker(symbol).calendar
    return dict(cal) if isinstance(cal, dict) and cal else None


@cached(ttl=settings.intel_cache_ttl, none_ttl=60)
async def _calendar(ticker: str) -> dict[str, Any] | None:
    return await _yahoo(_fetch_calendar, ticker, what="calendar")


def _fetch_earnings_frames(symbol: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    t = _ticker(symbol)
    dates = None
    try:
        dates = _frame(t.get_earnings_dates(limit=16))
    except Exception as exc:  # noqa: BLE001 - scraped page; fall back to the API table
        logger.info("earnings_dates failed for %s: %s", symbol, exc)
    history = None if dates is not None else _frame(t.earnings_history)
    return dates, history


@cached(ttl=settings.intel_cache_ttl, none_ttl=60)
async def _earnings_frames(ticker: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None] | None:
    return await _yahoo(_fetch_earnings_frames, ticker, what="earnings")


def _fetch_insiders(symbol: str) -> pd.DataFrame | None:
    return _frame(_ticker(symbol).insider_transactions)


@cached(ttl=settings.intel_cache_ttl, none_ttl=60)
async def _insider_frame(ticker: str) -> pd.DataFrame | None:
    return await _yahoo(_fetch_insiders, ticker, what="insider transactions")


async def _quote_type(ticker: str) -> str:
    """EQUITY / ETF / CRYPTOCURRENCY… from cached info (cheap), inferred otherwise."""
    if is_crypto_symbol(ticker):
        return "CRYPTOCURRENCY"
    if ticker.startswith("^"):
        return "INDEX"
    try:
        info = await get_info(ticker)
    except UpstreamError:
        info = None
    return str((info or {}).get("quoteType") or "EQUITY").upper()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
async def get_profile(company: CompanyRef) -> Profile | None:
    """Company profile: identity from resolution + sector/summary/employees from Yahoo."""
    try:
        info = await get_info(company.ticker)
    except UpstreamError as exc:
        logger.info("profile info unavailable for %s: %s", company.ticker, exc)
        info = None
    return tx.profile_from_info(
        info,
        symbol=company.ticker,
        name=company.name,
        short_name=company.short_name,
        quote_type=company.quote_type,
        exchange=company.exchange,
        cik=company.cik,
        logo_url=logo_url_for(company.ticker, company.website or (info or {}).get("website")),
    )


async def get_quote(ticker: str) -> Quote | None:
    """Latest regular-session quote; falls back to daily bars if `info` fails."""
    try:
        quote = tx.quote_from_info(await get_info(ticker))
    except UpstreamError:
        quote = None
    if quote is not None:
        return quote
    bars = await _daily_bars(ticker)
    return tx.quote_from_history(bars[0], bars[1]) if bars else None


async def get_price_history(ticker: str, range: str) -> PriceResponse:
    """OHLCV candles for a chart range (1D 5m, 5D 30m, 1M-1Y daily, 5Y weekly)."""
    rng = (range or "").upper()
    if rng not in RANGES:
        return PriceResponse(ticker=ticker, range=rng, interval="", available=False,
                             error=f"unsupported range; use one of {', '.join(RANGES)}")
    period, interval = RANGES[rng]
    try:
        if rng in _DAILY_SLICE:
            bars = await _daily_bars(ticker)
            if bars:
                df, currency = bars
                cutoff = pd.Timestamp(df.index[-1]) - pd.DateOffset(months=_DAILY_SLICE[rng])
                bars = (df[df.index > cutoff], currency)
        else:
            bars = await _bars(ticker, period, interval, is_crypto_symbol(ticker))
    except UpstreamError as exc:
        return PriceResponse(ticker=ticker, range=rng, interval=interval, available=False, error=str(exc))
    candles = tx.candles_from_history(bars[0]) if bars else []
    return PriceResponse(
        ticker=ticker, range=rng, interval=interval, currency=bars[1] if bars else None,
        candles=candles, available=bool(candles), error=None if candles else "no price data",
    )


async def get_daily_closes(ticker: str, days: int = 400) -> list[tuple[date, float]]:
    """[(session date, close)] for the last `days` calendar days, oldest first."""
    bars = await _daily_bars(ticker)
    if not bars:
        return []
    since = (_now() - timedelta(days=days)).date()
    return [(d, c) for d, c in tx.closes_from_history(bars[0]) if d >= since]


async def get_technicals(ticker: str) -> Technicals | None:
    """Returns, moving averages, RSI, volatility and trend from daily bars."""
    bars = await _daily_bars(ticker)
    if not bars:
        return None
    crypto = is_crypto_symbol(ticker)
    tech = tx.technicals_from_history(bars[0], now=_now(), is_crypto=crypto)
    if tech is not None and crypto and tech.return_1d is None:
        # Yahoo skipped yesterday's daily bar: the quote's rolling 24 h change is the 1-day move.
        try:
            quote = await get_quote(ticker)
        except UpstreamError:
            quote = None
        if quote is not None and quote.change_pct is not None:
            tech = tech.model_copy(update={"return_1d": round(quote.change_pct, 2)})
    return tech


async def get_analysts(ticker: str, price: float | None) -> AnalystView | None:
    """Wall Street consensus, price targets, rating trend and recent actions."""
    if await _quote_type(ticker) not in {"EQUITY"}:
        return None
    frames = await _analyst_frames(ticker)
    try:
        info = await get_info(ticker)
    except UpstreamError:
        info = None
    recs, upgrades = frames if frames else (None, None)
    return tx.analysts_from_frames(info, recs, upgrades, price=price, now=_now())


async def get_earnings(ticker: str) -> EarningsView | None:
    """Next report date with EPS/revenue consensus; surprise history and beat rate."""
    if await _quote_type(ticker) != "EQUITY":
        return None
    calendar = await _calendar(ticker)
    frames = await _earnings_frames(ticker)
    try:
        info = await get_info(ticker)
    except UpstreamError:
        info = None
    dates, history = frames if frames else (None, None)
    return tx.earnings_from_frames(calendar, dates, info, today=_now().date(), earnings_history=history)


async def get_calendar_catalysts(ticker: str) -> list[Catalyst]:
    """Upcoming ex-dividend and dividend payment dates (`upcoming=True`)."""
    qtype = await _quote_type(ticker)
    if qtype not in {"EQUITY", "ETF"}:
        return []
    # Yahoo has no calendar module for funds (404); their dividend dates live in `info`.
    calendar = await _calendar(ticker) if qtype == "EQUITY" else None
    try:
        info = await get_info(ticker)
    except UpstreamError:
        info = None
    return tx.dividend_catalysts(calendar, info, today=_now().date())


async def get_insiders(ticker: str) -> InsiderView | None:
    """Insider transactions (Yahoo); SEC Form 4 parsing when Yahoo has none."""
    if await _quote_type(ticker) != "EQUITY":
        return None
    yahoo_error: UpstreamError | None = None
    try:
        view = tx.insiders_from_frame(await _insider_frame(ticker), now=_now())
    except UpstreamError as exc:
        view, yahoo_error = None, exc
    if view is not None:
        return view
    from app.intel.sec import get_cik_map, get_form4_insiders

    try:
        entry = (await get_cik_map()).get(ticker)
        if entry:
            view = await get_form4_insiders(entry[0])
    except UpstreamError as exc:
        logger.info("SEC Form 4 fallback failed for %s: %s", ticker, exc)
    if view is None and yahoo_error is not None:
        raise yahoo_error
    return view


def _fetch_indices(symbols: list[str]) -> pd.DataFrame | None:
    import yfinance as yf

    df = yf.download(symbols, period="3mo", interval="1d", auto_adjust=False, progress=False,
                     group_by="column", threads=True, multi_level_index=True)
    return _frame(df)


async def _live_crypto_quote(symbol: str) -> Quote | None:
    """The quote's rolling 24 h view of a crypto benchmark, or None when `info` is slow or failing
    (no daily-bar fallback: the tape already has those). A slow lookup keeps running in the
    background (shielded), so the next tape gets it from the cache."""
    try:
        return tx.quote_from_info(await asyncio.wait_for(asyncio.shield(get_info(symbol)), CRYPTO_QUOTE_TIMEOUT))
    except (UpstreamError, TimeoutError):
        return None


@cached(ttl=settings.price_cache_ttl, none_ttl=30)
async def get_indices() -> list[IndexQuote]:
    """Benchmarks strip: SPY QQQ DIA IWM ^VIX ^TNX GC=F BTC-USD with ~1M sparklines."""
    # Crypto trades 24/7: the tape shows the same rolling 24 h change as the quote (and the Intel
    # page), not the move since the 00:00 UTC daily bar. Looked up alongside the download.
    live = {s: asyncio.ensure_future(_live_crypto_quote(s)) for s in INDEX_SYMBOLS if is_crypto_symbol(s)}
    try:
        df = await _yahoo(_fetch_indices, list(INDEX_SYMBOLS), what="indices")
        quotes = [q for q in tx.indices_from_download(df, INDEX_SYMBOLS) if q.price is not None]
        if not quotes:  # yf.download swallows per-symbol errors and returns an empty frame
            raise UpstreamError("Yahoo indices: no prices returned")
        for i, q in enumerate(quotes):
            quote = await live[q.symbol] if q.symbol in live else None
            if quote is not None and quote.change_pct is not None:
                quotes[i] = q.model_copy(update={"change_pct": quote.change_pct, "price": quote.price})
        return quotes
    finally:
        for task in live.values():
            task.cancel()
