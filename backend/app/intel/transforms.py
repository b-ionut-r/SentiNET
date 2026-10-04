"""Pure transforms from Yahoo Finance (yfinance) payloads into API models.

Everything here is synchronous, network-free and deterministic given its inputs
(including an explicit `now`/`today`), so it is tested against recorded
fixtures. `app.intel.market_data` does the fetching and caching.

yfinance shapes handled (v1.7):
* `Ticker.info` -> dict (prices, names, analyst targets, dividend dates…)
* `Ticker.history(...)` -> DataFrame[Open, High, Low, Close, Adj Close, Volume, …]
* `Ticker.recommendations` -> DataFrame[period, strongBuy, buy, hold, sell, strongSell]
* `Ticker.upgrades_downgrades` -> DataFrame indexed by GradeDate
  [Firm, ToGrade, FromGrade, Action, priceTargetAction, currentPriceTarget, priorPriceTarget]
* `Ticker.calendar` -> dict ("Earnings Date": [date…], "Ex-Dividend Date": date, …)
* `Ticker.earnings_dates` -> DataFrame indexed by tz-aware "Earnings Date"
  [EPS Estimate, Reported EPS, Surprise(%)]
* `Ticker.insider_transactions` -> DataFrame
  [Shares, Value, URL, Text, Insider, Position, Transaction, Start Date, Ownership]
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta, timezone
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.schemas import (
    AnalystAction,
    AnalystView,
    Candle,
    Catalyst,
    EarningsEvent,
    EarningsView,
    IndexQuote,
    InsiderTxn,
    InsiderView,
    Profile,
    Quote,
    RatingCounts,
    Technicals,
)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Small coercion helpers
# --------------------------------------------------------------------------- #
def num(value: Any) -> float | None:
    """Finite float or None (NaN, None, '', 'Infinity' -> None)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def pos(value: Any) -> float | None:
    """Positive finite float or None (Yahoo uses 0 for 'no value' in price fields)."""
    f = num(value)
    return f if f is not None and f > 0 else None


def to_date(value: Any) -> date | None:
    """date from date/datetime/Timestamp/epoch seconds/ISO string; None otherwise."""
    if value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(float(value), UTC).date()
    if isinstance(value, str) and not value.strip():
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(ts) else ts.date()


def to_utc(value: Any) -> datetime | None:
    """tz-aware UTC datetime; naive inputs are assumed to be UTC."""
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    return ts.to_pydatetime()


def date_noon_utc(d: date) -> datetime:
    """A date-only event as 12:00 UTC: renders on the same calendar day in UTC-11..UTC+11."""
    return datetime.combine(d, time(12, 0), tzinfo=UTC)


def pct(a: float | None, b: float | None) -> float | None:
    """Percent change from b to a, rounded to 2 decimals."""
    if a is None or b is None or b == 0:
        return None
    return round((a / b - 1.0) * 100.0, 2)


def _r(value: float | None, nd: int = 2) -> float | None:
    return None if value is None else round(value, nd)


# --------------------------------------------------------------------------- #
# Quote & profile
# --------------------------------------------------------------------------- #
def quote_from_info(info: dict[str, Any] | None) -> Quote | None:
    """`Ticker.info` -> Quote (regular-session fields; change_pct is in percent)."""
    if not info:
        return None
    price = pos(info.get("regularMarketPrice")) or pos(info.get("currentPrice"))
    prev = pos(info.get("regularMarketPreviousClose")) or pos(info.get("previousClose"))
    if price is None:
        return None
    change = num(info.get("regularMarketChange"))
    if change is None and prev is not None:
        change = price - prev
    change_pct = num(info.get("regularMarketChangePercent"))
    if change_pct is None and prev:
        change_pct = (price / prev - 1.0) * 100.0
    as_of = info.get("regularMarketTime")
    return Quote(
        price=price,
        change=_r(change, 4),
        change_pct=_r(change_pct, 3),
        currency=info.get("currency") or None,
        previous_close=prev,
        open=pos(info.get("regularMarketOpen")) or pos(info.get("open")),
        day_high=pos(info.get("regularMarketDayHigh")) or pos(info.get("dayHigh")),
        day_low=pos(info.get("regularMarketDayLow")) or pos(info.get("dayLow")),
        year_high=pos(info.get("fiftyTwoWeekHigh")),
        year_low=pos(info.get("fiftyTwoWeekLow")),
        volume=pos(info.get("regularMarketVolume")) or pos(info.get("volume")),
        avg_volume=pos(info.get("averageVolume")) or pos(info.get("averageDailyVolume3Month")),
        market_cap=pos(info.get("marketCap")),
        fifty_day_avg=pos(info.get("fiftyDayAverage")),
        two_hundred_day_avg=pos(info.get("twoHundredDayAverage")),
        as_of=to_utc(datetime.fromtimestamp(as_of, UTC)) if isinstance(as_of, (int, float)) and as_of > 0 else None,
    )


def quote_from_history(df: pd.DataFrame | None, currency: str | None = None) -> Quote | None:
    """Fallback quote from daily bars when `info` is unavailable."""
    if df is None or df.empty or "Close" not in df:
        return None
    bars = df.dropna(subset=["Close"])
    if bars.empty:
        return None
    last = bars.iloc[-1]
    prev = num(bars["Close"].iloc[-2]) if len(bars) > 1 else None
    price = num(last["Close"])
    return Quote(
        price=price,
        change=_r(price - prev, 4) if price is not None and prev is not None else None,
        change_pct=pct(price, prev),
        currency=currency,
        previous_close=prev,
        open=pos(last.get("Open")),
        day_high=pos(last.get("High")),
        day_low=pos(last.get("Low")),
        volume=pos(last.get("Volume")),
        as_of=to_utc(bars.index[-1]),
    )


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def trim_summary(text: str | None, limit: int = 900) -> str | None:
    """Trim a business summary at a sentence boundary (keeps it readable)."""
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    out = ""
    for sentence in _SENTENCE_END.split(text):
        if len(out) + len(sentence) + 1 > limit:
            break
        out = f"{out} {sentence}".strip()
    return out or text[:limit].rsplit(" ", 1)[0] + "…"


def profile_from_info(
    info: dict[str, Any] | None,
    *,
    symbol: str,
    name: str,
    short_name: str | None,
    quote_type: str | None,
    exchange: str | None,
    cik: str | None,
    logo_url: str | None,
) -> Profile:
    """Merge `info` with the resolved identity into a Profile."""
    info = info or {}
    employees = num(info.get("fullTimeEmployees"))
    return Profile(
        symbol=symbol,
        name=name,
        short_name=short_name,
        quote_type=quote_type or info.get("quoteType"),
        exchange=exchange,
        sector=info.get("sector") or info.get("category") or None,
        industry=info.get("industry") or info.get("fundFamily") or None,
        country=info.get("country") or None,
        website=info.get("website") or None,
        summary=trim_summary(info.get("longBusinessSummary") or info.get("description")),
        employees=int(employees) if employees else None,
        logo_url=logo_url,
        cik=cik,
    )


# --------------------------------------------------------------------------- #
# Price history
# --------------------------------------------------------------------------- #
def candles_from_history(df: pd.DataFrame | None) -> list[Candle]:
    """OHLCV DataFrame -> candles (UTC timestamps, incomplete rows dropped)."""
    if df is None or df.empty:
        return []
    cols = ["Open", "High", "Low", "Close"]
    if any(c not in df for c in cols):
        return []
    bars = df.dropna(subset=cols)
    out: list[Candle] = []
    for ts, row in bars.iterrows():
        t = to_utc(ts)
        if t is None:
            continue
        vol = num(row.get("Volume"))
        out.append(Candle(t=t, o=float(row["Open"]), h=float(row["High"]), l=float(row["Low"]),
                          c=float(row["Close"]), v=vol if vol and vol > 0 else None))
    return out


def _local_dates(index: pd.Index) -> list[date]:
    """Exchange-local session dates of a (possibly tz-aware) DatetimeIndex."""
    return [ts.date() for ts in pd.DatetimeIndex(index)]


def closes_from_history(df: pd.DataFrame | None) -> list[tuple[date, float]]:
    """Daily bars -> [(session date, close)] oldest first."""
    if df is None or df.empty or "Close" not in df:
        return []
    closes = df["Close"].dropna()
    return [(d, float(c)) for d, c in zip(_local_dates(closes.index), closes.to_numpy())]


def wilder_rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder's RSI (the standard 14-period definition)."""
    if len(closes) <= period:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _close_on_or_before(dates: list[date], closes: list[float], target: date) -> float | None:
    for d, c in zip(reversed(dates), reversed(closes)):
        if d <= target:
            return c
    return None


def _session_in_progress(last: date, now: datetime, is_crypto: bool) -> bool:
    """True when the last daily bar is a still-running session (partial volume)."""
    if is_crypto:
        return last >= now.astimezone(UTC).date()
    local = now.astimezone(NY)
    return last >= local.date() and local.weekday() < 5 and local.time() < time(16, 0)


def technicals_from_history(
    df: pd.DataFrame | None, *, now: datetime, is_crypto: bool = False
) -> Technicals | None:
    """~1-2y of daily bars -> price-implied sentiment (returns in percent)."""
    if df is None or df.empty or "Close" not in df:
        return None
    bars = df.dropna(subset=["Close"])
    if len(bars) < 2:
        return None
    closes = [float(c) for c in bars["Close"].to_numpy()]
    dates = _local_dates(bars.index)
    last, last_date = closes[-1], dates[-1]

    def back(days: int = 0, months: int = 0) -> float | None:
        target = (pd.Timestamp(last_date) - pd.DateOffset(days=days, months=months)).date()
        return _close_on_or_before(dates[:-1], closes[:-1], target) if dates[0] <= target else None

    sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
    sma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None

    vol30 = None
    if len(closes) >= 21:
        window = closes[-31:]
        rets = [math.log(b / a) for a, b in pairwise(window) if a > 0 and b > 0]
        if len(rets) >= 20:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            vol30 = math.sqrt(var) * math.sqrt(365 if is_crypto else 252) * 100

    year_ago = (pd.Timestamp(last_date) - pd.DateOffset(years=1)).date()
    in_year = [i for i, d in enumerate(dates) if d > year_ago]
    highs = bars["High"].to_numpy() if "High" in bars else bars["Close"].to_numpy()
    high_52w = max((float(highs[i]) for i in in_year if num(highs[i]) is not None), default=None)

    volume_ratio = None
    if "Volume" in bars:
        vols = [num(v) or 0.0 for v in bars["Volume"].to_numpy()]
        end = len(vols) - 1
        if _session_in_progress(last_date, now, is_crypto):
            end -= 1  # compare the last *complete* session, not a partial one
        if end >= 21:
            base = [v for v in vols[max(0, end - 63):end] if v > 0]
            if base and vols[end] > 0:
                volume_ratio = round(vols[end] / (sum(base) / len(base)), 2)

    prev_year_end = _close_on_or_before(dates, closes, date(last_date.year - 1, 12, 31))
    trend = None
    if sma50 is not None:
        # Classic alignment: price vs 50-DMA and 50 vs 200-DMA (20 vs 50 when < 200 bars).
        sma20 = sum(closes[-20:]) / 20
        stacked_up = sma50 > sma200 if sma200 is not None else sma20 > sma50
        if last > sma50 and stacked_up:
            trend = "uptrend"
        elif last < sma50 and not stacked_up:
            trend = "downtrend"
        else:
            trend = "sideways"

    rsi = wilder_rsi(closes)
    return Technicals(
        return_1d=pct(last, closes[-2]),
        return_5d=pct(last, closes[-6]) if len(closes) >= 6 else None,
        return_1m=pct(last, back(months=1)),
        return_3m=pct(last, back(months=3)),
        return_ytd=pct(last, prev_year_end) if dates[0].year < last_date.year else None,
        vs_50dma_pct=pct(last, sma50),
        vs_200dma_pct=pct(last, sma200),
        rsi_14=_r(rsi, 1),
        volatility_30d=_r(vol30, 1),
        pct_from_52w_high=pct(last, high_52w),
        volume_ratio=volume_ratio,
        trend=trend,
    )


# --------------------------------------------------------------------------- #
# Analysts
# --------------------------------------------------------------------------- #
_ACTIONS = {"up": "up", "down": "down", "init": "init", "main": "main", "reit": "reit"}
_CONSENSUS_FROM_KEY = {
    "strong_buy": "strong_buy", "buy": "buy", "outperform": "buy", "hold": "hold",
    "underperform": "sell", "sell": "strong_sell", "strong_sell": "strong_sell",
}


def consensus_for(mean_rating: float | None) -> str | None:
    """Yahoo's 1..5 scale -> consensus bucket (1 strong buy … 5 strong sell)."""
    if mean_rating is None:
        return None
    for limit, label in ((1.5, "strong_buy"), (2.5, "buy"), (3.5, "hold"), (4.5, "sell")):
        if mean_rating <= limit:
            return label
    return "strong_sell"


def rating_counts(df: pd.DataFrame | None) -> list[RatingCounts]:
    """`Ticker.recommendations` -> RatingCounts per period, current month first."""
    if df is None or df.empty or "period" not in df:
        return []
    out = []
    for _, row in df.iterrows():
        out.append(RatingCounts(
            period=str(row["period"]),
            strong_buy=int(num(row.get("strongBuy")) or 0),
            buy=int(num(row.get("buy")) or 0),
            hold=int(num(row.get("hold")) or 0),
            sell=int(num(row.get("sell")) or 0),
            strong_sell=int(num(row.get("strongSell")) or 0),
        ))

    def order(rc: RatingCounts) -> int:
        m = re.fullmatch(r"(-?\d+)m", rc.period)
        return -int(m.group(1)) if m else 99

    return sorted(out, key=order)


def _counts_total(rc: RatingCounts) -> int:
    return rc.strong_buy + rc.buy + rc.hold + rc.sell + rc.strong_sell


def _mean_from_counts(rc: RatingCounts) -> float | None:
    n = _counts_total(rc)
    if not n:
        return None
    return (rc.strong_buy + 2 * rc.buy + 3 * rc.hold + 4 * rc.sell + 5 * rc.strong_sell) / n


def analyst_actions(df: pd.DataFrame | None) -> list[AnalystAction]:
    """`Ticker.upgrades_downgrades` -> de-duplicated actions, most recent first."""
    if df is None or df.empty:
        return []
    frame = df.reset_index()
    date_col = "GradeDate" if "GradeDate" in frame else frame.columns[0]
    out: list[AnalystAction] = []
    seen: set[tuple] = set()
    for _, row in frame.iterrows():
        when = to_utc(row[date_col])
        firm = str(row.get("Firm") or "").strip()
        if when is None or not firm:
            continue
        action = _ACTIONS.get(str(row.get("Action") or "").strip().lower(), "other")
        target = pos(row.get("currentPriceTarget"))
        prior = pos(row.get("priorPriceTarget"))
        key = (when.date(), firm.lower(), action, target)
        if key in seen:
            continue
        seen.add(key)
        out.append(AnalystAction(
            date=when, firm=firm, action=action,  # type: ignore[arg-type]
            from_grade=str(row.get("FromGrade") or "").strip() or None,
            to_grade=str(row.get("ToGrade") or "").strip() or None,
            price_target=target, prior_target=prior if action != "init" else None,
        ))
    out.sort(key=lambda a: a.date, reverse=True)
    return out


def _pt_direction(row_action: str | None, target: float | None, prior: float | None) -> int:
    """+1 raise, -1 cut, 0 otherwise (Yahoo's label first, then the numbers)."""
    label = (row_action or "").strip().lower()
    if label.startswith("raise"):
        return 1
    if label.startswith("lower"):
        return -1
    if target and prior and abs(target - prior) / prior > 0.001:
        return 1 if target > prior else -1
    return 0


def analysts_from_frames(
    info: dict[str, Any] | None,
    recommendations: pd.DataFrame | None,
    upgrades: pd.DataFrame | None,
    *,
    price: float | None,
    now: datetime,
) -> AnalystView | None:
    """Consensus, targets, recent actions and revision momentum."""
    info = info or {}
    trend = rating_counts(recommendations)
    current = next((rc for rc in trend if rc.period == "0m"), trend[0] if trend else None)
    actions = analyst_actions(upgrades)

    mean_rating = num(info.get("recommendationMean"))
    if mean_rating is None and current is not None:
        mean_rating = _mean_from_counts(current)
    consensus = consensus_for(mean_rating) or _CONSENSUS_FROM_KEY.get(str(info.get("recommendationKey") or ""))
    total = _counts_total(current) if current else 0
    total = total or int(num(info.get("numberOfAnalystOpinions")) or 0)

    targets = {k: pos(info.get(f"target{k}Price")) for k in ("Mean", "Median", "High", "Low")}
    price = price or pos(info.get("currentPrice")) or pos(info.get("regularMarketPrice"))
    if not (total or targets["Mean"] or actions):
        return None

    up90 = down90 = raises30 = cuts30 = 0
    raw = upgrades.reset_index() if upgrades is not None and not upgrades.empty else None
    pt_labels: dict[tuple, str | None] = {}
    if raw is not None:
        date_col = "GradeDate" if "GradeDate" in raw else raw.columns[0]
        for _, row in raw.iterrows():
            when = to_utc(row[date_col])
            if when is not None:
                pt_labels[(when.date(), str(row.get("Firm") or "").strip().lower())] = row.get("priceTargetAction")
    for a in actions:
        age = now - a.date
        if age <= timedelta(days=90):
            up90 += a.action == "up"
            down90 += a.action == "down"
        if age <= timedelta(days=30) and a.action != "init":
            d = _pt_direction(pt_labels.get((a.date.date(), a.firm.lower())), a.price_target, a.prior_target)
            raises30 += d > 0
            cuts30 += d < 0

    return AnalystView(
        consensus=consensus,
        mean_rating=_r(mean_rating, 2),
        counts=current,
        total=total,
        target_mean=targets["Mean"],
        target_median=targets["Median"],
        target_high=targets["High"],
        target_low=targets["Low"],
        upside_pct=pct(targets["Mean"], price),
        actions=actions[:25],
        upgrades_90d=up90,
        downgrades_90d=down90,
        pt_raises_30d=raises30,
        pt_cuts_30d=cuts30,
        trend=trend[:4],
    )


# --------------------------------------------------------------------------- #
# Earnings & dividends
# --------------------------------------------------------------------------- #
def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def earnings_from_frames(
    calendar: dict[str, Any] | None,
    earnings_dates: pd.DataFrame | None,
    info: dict[str, Any] | None,
    *,
    today: date,
    earnings_history: pd.DataFrame | None = None,
    max_history: int = 8,
) -> EarningsView | None:
    """Next report date + consensus, last reported quarters and beat rate."""
    calendar = calendar or {}
    info = info or {}
    upcoming: list[date] = [d for d in (to_date(v) for v in _as_list(calendar.get("Earnings Date"))) if d]
    history: list[EarningsEvent] = []

    if earnings_dates is not None and not earnings_dates.empty:
        for ts, row in earnings_dates.iterrows():
            d = to_date(pd.Timestamp(ts).tz_convert(NY) if pd.Timestamp(ts).tzinfo else ts)
            if d is None:
                continue
            est, act = num(row.get("EPS Estimate")), num(row.get("Reported EPS"))
            if act is None:
                if d >= today:
                    upcoming.append(d)
                continue
            surprise = num(row.get("Surprise(%)"))
            if surprise is None and est not in (None, 0):
                surprise = (act - est) / abs(est) * 100  # type: ignore[operator]
            history.append(EarningsEvent(date=d, eps_estimate=est, eps_actual=act, surprise_pct=_r(surprise, 2)))
    elif earnings_history is not None and not earnings_history.empty:
        for ts, row in earnings_history.iterrows():
            d = to_date(ts)
            est, act = num(row.get("epsEstimate")), num(row.get("epsActual"))
            if d is None or act is None:
                continue
            sp = num(row.get("surprisePercent"))
            history.append(EarningsEvent(date=d, eps_estimate=_r(est, 4), eps_actual=act,
                                         surprise_pct=_r(sp * 100, 2) if sp is not None else None))
    for key in ("earningsTimestampStart", "earningsTimestamp"):
        d = to_date(info.get(key))
        if d:
            upcoming.append(d)

    future = sorted(d for d in upcoming if d >= today)
    next_date = future[0] if future else None
    history.sort(key=lambda e: e.date, reverse=True)
    history = history[:max_history]
    scored = [e for e in history if e.eps_estimate is not None and e.eps_actual is not None]
    beat_rate = (
        round(sum(e.eps_actual > e.eps_estimate for e in scored) / len(scored), 3)  # type: ignore[operator]
        if len(scored) >= 2 else None
    )
    has_estimates = next_date is not None
    view = EarningsView(
        next_date=next_date,
        days_until=(next_date - today).days if next_date else None,
        eps_estimate=num(calendar.get("Earnings Average")) if has_estimates else None,
        eps_low=num(calendar.get("Earnings Low")) if has_estimates else None,
        eps_high=num(calendar.get("Earnings High")) if has_estimates else None,
        revenue_estimate=pos(calendar.get("Revenue Average")) if has_estimates else None,
        history=history,
        beat_rate=beat_rate,
    )
    return view if (view.next_date or view.history) else None


def dividend_catalysts(
    calendar: dict[str, Any] | None, info: dict[str, Any] | None, *, today: date
) -> list[Catalyst]:
    """Upcoming ex-dividend / payment dates (only future-dated, never stale)."""
    calendar = calendar or {}
    info = info or {}
    per_payment = pos(info.get("lastDividendValue"))
    annual = pos(info.get("dividendRate"))
    yld = pos(info.get("dividendYield"))  # already in percent (0.43 == 0.43%)
    parts = []
    if per_payment:
        parts.append(f"${per_payment:.4g}/share")
    if annual:
        parts.append(f"${annual:.4g}/yr")
    if yld:
        parts.append(f"yield {yld:.2f}%")
    detail = " · ".join(parts) or None

    out: list[Catalyst] = []
    ex = to_date(calendar.get("Ex-Dividend Date")) or to_date(info.get("exDividendDate"))
    if ex is None or ex < today:
        alt = to_date(info.get("exDividendDate"))
        ex = alt if alt and alt >= today else None
    pay = to_date(calendar.get("Dividend Date")) or to_date(info.get("dividendDate"))
    if pay is None or pay < today:
        alt = to_date(info.get("dividendDate"))
        pay = alt if alt and alt >= today else None
    if ex is not None:
        days = (ex - today).days
        out.append(Catalyst(
            date=date_noon_utc(ex), kind="dividend", upcoming=True,
            title="Ex-dividend date" + (" (today)" if days == 0 else f" in {days} day{'s' if days != 1 else ''}"),
            detail=(detail + " · buy before the ex-date to receive it") if detail else
            "Buy before the ex-date to receive the dividend",
        ))
    if pay is not None and (ex is None or pay >= ex):
        out.append(Catalyst(date=date_noon_utc(pay), kind="dividend", upcoming=True,
                            title="Dividend payment", detail=detail))
    return out


# --------------------------------------------------------------------------- #
# Insiders
# --------------------------------------------------------------------------- #
_KIND_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*purchase", re.IGNORECASE), "buy"),
    (re.compile(r"^\s*sale", re.IGNORECASE), "sell"),
    (re.compile(r"stock\s+(?:award|grant)|\bgrant\b|\baward\b", re.IGNORECASE), "award"),
    (re.compile(r"exercise|conversion", re.IGNORECASE), "exercise"),
    (re.compile(r"\bgift\b", re.IGNORECASE), "gift"),
)
_ENTITY = re.compile(r"\b(?:inc|corp|llc|lp|l\.p|ltd|trust|fund|partners|holdings|capital|group|foundation|co)\b\.?", re.IGNORECASE)
_SUFFIXES = {"JR", "SR", "II", "III", "IV", "JR.", "SR."}


def classify_insider(text: str | None, transaction: str | None = None) -> str:
    """Yahoo transaction text -> buy/sell/award/exercise/gift/other.

    Only "Purchase …"/"Sale …" are open-market trades. An empty text is usually a
    tax-withholding disposition on vesting (Form 4 code F) — not a signal.
    """
    for source in (text, transaction):
        if source:
            for pattern, kind in _KIND_RULES:
                if pattern.search(source):
                    return kind
    return "other"


def pretty_insider_name(raw: str | None) -> str:
    """SEC "LAST FIRST MIDDLE" -> "First Middle Last" when unambiguous; entities kept."""
    name = re.sub(r"\s+", " ", (raw or "").strip())
    if not name:
        return "Unknown"
    if not name.isupper() or _ENTITY.search(name):
        return name if not name.isupper() else name.title()
    tokens = name.split(" ")
    suffix = [t for t in tokens if t.upper() in _SUFFIXES]
    core = [t for t in tokens if t.upper() not in _SUFFIXES]

    def word(t: str) -> str:
        t = t.title()
        t = re.sub(r"^(Mc|Mac)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t) if len(t) > 4 else t
        return f"{t}." if len(t) == 1 else t

    if 2 <= len(core) <= 3:
        ordered = core[1:] + core[:1]
    else:
        ordered = core
    out = " ".join(word(t) for t in ordered)
    return f"{out} {' '.join(s.title().rstrip('.') + '.' for s in suffix)}".strip()


def insiders_from_frame(df: pd.DataFrame | None, *, now: datetime, window_days: int = 180) -> InsiderView | None:
    """Insider transactions -> open-market flow over `window_days` + notable rows."""
    if df is None or df.empty:
        return None
    since = (now - timedelta(days=window_days)).date()
    rows: list[InsiderTxn] = []
    for _, row in df.iterrows():
        d = to_date(row.get("Start Date"))
        if d is None:
            continue
        text = str(row.get("Text") or "").strip()
        kind = classify_insider(text, str(row.get("Transaction") or ""))
        shares = num(row.get("Shares"))
        value = num(row.get("Value"))
        if kind in {"award", "gift"} and value == 0:
            value = None  # $0 "value" on grants/gifts is not a trade value
        ownership = str(row.get("Ownership") or "").strip().upper()
        note = text or None
        if note and ownership == "I":
            note = f"{note} (indirect)"
        rows.append(InsiderTxn(
            date=d, insider=pretty_insider_name(row.get("Insider")),
            position=str(row.get("Position") or "").strip() or None,
            kind=kind,  # type: ignore[arg-type]
            shares=abs(shares) if shares is not None else None,
            value=abs(value) if value is not None else None, text=note,
        ))
    return insider_view(rows, since=since, window_days=window_days)


def insider_view(rows: list[InsiderTxn], *, since: date, window_days: int) -> InsiderView | None:
    """Aggregate transactions (shared by the Yahoo path and the SEC Form 4 fallback)."""
    if not rows:
        return None
    recent = [t for t in rows if t.date >= since]
    buys = [t for t in recent if t.kind == "buy"]
    sells = [t for t in recent if t.kind == "sell"]
    buy_value = sum(t.value or 0.0 for t in buys)
    sell_value = sum(t.value or 0.0 for t in sells)
    gross = buy_value + sell_value
    # Keep the informative rows: every open-market trade in the window first,
    # then the most recent other activity; present most recent first.
    trades = sorted(buys + sells, key=lambda t: t.date, reverse=True)
    trade_ids = {id(t) for t in trades}
    others = sorted((t for t in rows if id(t) not in trade_ids), key=lambda t: t.date, reverse=True)
    shown = sorted((trades + others)[:25], key=lambda t: t.date, reverse=True)
    return InsiderView(
        window_days=window_days,
        buys=len(buys),
        sells=len(sells),
        buy_value=round(buy_value, 2),
        sell_value=round(sell_value, 2),
        net_value=round(buy_value - sell_value, 2),
        ratio=round((buy_value - sell_value) / gross, 3) if gross > 0 else None,
        transactions=shown,
    )


# --------------------------------------------------------------------------- #
# Market indices
# --------------------------------------------------------------------------- #
def indices_from_download(
    df: pd.DataFrame | None, names: dict[str, str], *, spark_days: int = 22
) -> list[IndexQuote]:
    """Batched `yf.download` (column MultiIndex: field × symbol) -> IndexQuotes."""
    out: list[IndexQuote] = []
    for symbol, label in names.items():
        closes: Iterable[Any] = []
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                if ("Close", symbol) in df.columns:
                    closes = df[("Close", symbol)].dropna().to_numpy()
            elif "Close" in df.columns and len(names) == 1:
                closes = df["Close"].dropna().to_numpy()
        series = [float(c) for c in closes if num(c) is not None]
        price = series[-1] if series else None
        prev = series[-2] if len(series) > 1 else None
        out.append(IndexQuote(
            symbol=symbol, name=label, price=_r(price, 4),
            change_pct=pct(price, prev), spark=[round(v, 4) for v in series[-spark_days:]],
        ))
    return out
