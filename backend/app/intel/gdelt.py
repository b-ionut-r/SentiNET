"""Global news tone & volume from the GDELT DOC 2.0 API (thousands of outlets).

GDELT scores the tone of every article it monitors (roughly -10..+10; most
company coverage sits in -3..+3), which gives SentiNET a 90-day baseline that
our own snapshot history cannot: "is today's coverage unusually negative *for
this company*?".

Two hard parts live here:

* **Query precision.** GDELT matches full article text case-insensitively, so
  `"Apple"` drags in orchards and pies. Common-word brands get curated context
  (`"Apple" (iPhone OR iPad OR …)`), and any other common-word name is anchored
  to company phrases (`"X Inc" OR "X shares" …`). Verified with artlist samples.
* **Politeness.** GDELT allows one request per 5 s per IP (shared limiter in
  `app.core.ratelimit`) and answers violations with a plain-text "Please limit
  requests…" body (HTTP 429 or even 200). We retry once, cache for hours, and
  degrade to tone-only when the volume call is refused.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from datetime import date, datetime, timedelta, UTC
from typing import Any

import httpx

from app.config import settings
from app.core.cache import cached
from app.core.http import UpstreamError, fetch
from app.resolve.names import is_common_word_name
from app.schemas import TonePoint, ToneTrend
from app.sources.base import CompanyRef

logger = logging.getLogger(__name__)

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
LANG = "sourcelang:english"
RATE_LIMIT_WAIT = 6.0

# Curated queries where the brand alone is ambiguous or not what the news says.
# Each must be valid GDELT syntax: quoted phrases, one level of (… OR …) groups —
# parentheses around a single term are rejected ("may only be used around OR'd statements").
CURATED: dict[str, str] = {
    "AAPL": '"Apple" (iPhone OR iPad OR Mac OR iOS OR "Apple Inc" OR AAPL OR "Tim Cook" OR Cupertino)',
    "META": '"Meta" (Zuckerberg OR Instagram OR WhatsApp OR Facebook OR "Meta Platforms" OR "Meta AI")',
    "TGT": '("Target Corp" OR "Target Corporation" OR "Target stores" OR "Target store" OR "Target shares" '
           'OR "Target stock" OR "Target CEO" OR "at Target")',
    "XYZ": '("Block Inc" OR "Cash App" OR "Jack Dorsey" OR Afterpay OR "Square payments" OR "Block shares")',
    "SQ": '("Block Inc" OR "Cash App" OR "Jack Dorsey" OR Afterpay OR "Square payments" OR "Block shares")',
    "SNAP": '("Snap Inc" OR Snapchat OR "Snap shares" OR "Snap stock")',
    "V": '("Visa Inc" OR "Visa and Mastercard" OR "Visa card" OR "Visa shares" OR "Visa stock" OR "Visa CEO")',
    "SHEL": '("Shell plc" OR "Royal Dutch Shell" OR "Shell shares" OR "Shell stock" OR "Shell CEO" OR "Shell oil")',
    "ORCL": '("Oracle Corp" OR "Oracle Corporation" OR "Oracle Cloud" OR "Larry Ellison" OR "Oracle shares" '
            'OR "Oracle stock")',
    "F": '("Ford Motor" OR "Ford F-150" OR "Ford shares" OR "Ford stock" OR "Ford CEO" OR "Ford EV")',
    "AMZN": '"Amazon" (AWS OR Prime OR Bezos OR Jassy OR "Amazon.com" OR retailer OR ecommerce OR AMZN)',
    "GOOGL": '("Alphabet" OR "Google")',
    "GOOG": '("Alphabet" OR "Google")',
    "ZM": '("Zoom Video" OR "Zoom Communications" OR "Zoom shares" OR "Zoom stock")',
    "U": '("Unity Software" OR "Unity Technologies")',
    "LCID": '("Lucid Group" OR "Lucid Motors" OR "Lucid Air" OR "Lucid Gravity")',
    "SOFI": '("SoFi Technologies" OR "SoFi Bank" OR "SoFi shares" OR "SoFi stock" OR "SoFi Invest" OR "SoFi CEO")',
    "GAP": '("Gap Inc" OR "Old Navy" OR "Banana Republic" OR "Gap shares")',
    "GPS": '("Gap Inc" OR "Old Navy" OR "Banana Republic" OR "Gap shares")',
    "CCL": '("Carnival Corp" OR "Carnival Cruise" OR "Carnival shares")',
    "DAL": '("Delta Air Lines" OR "Delta Airlines")',
    "UAL": '"United Airlines"',
    "AAL": '"American Airlines"',
    "LUV": '"Southwest Airlines"',
    "BP": '("BP plc" OR "BP shares" OR "BP oil" OR "BP CEO")',
    "GE": '("GE Aerospace" OR "General Electric")',
    # GDELT tokenizes "AT&T" to "at t" (matches "S & T", "at the"…): require telecom context.
    "T": '"AT&T" (wireless OR telecom OR carrier OR broadband OR fiber OR Verizon OR "T-Mobile")',
    "C": '("Citigroup" OR "Citi bank" OR "Citibank")',
    "HOOD": '("Robinhood Markets" OR "Robinhood app" OR "Robinhood shares" OR "Robinhood stock" OR "Robinhood CEO")',
    "SPY": '"S&P 500"',
    "VOO": '"S&P 500"',
    "IVV": '"S&P 500"',
    "QQQ": '("Nasdaq 100" OR "Nasdaq-100" OR "Nasdaq Composite")',
    "QQQM": '("Nasdaq 100" OR "Nasdaq-100" OR "Nasdaq Composite")',
    "TQQQ": '("Nasdaq 100" OR "Nasdaq-100" OR "Nasdaq Composite")',
    "DIA": '("Dow Jones Industrial Average" OR "Dow Jones index")',
    "IWM": '"Russell 2000"',
    "GLD": '("gold prices" OR "gold price" OR "price of gold")',
    "IAU": '("gold prices" OR "gold price" OR "price of gold")',
    "SLV": '("silver prices" OR "silver price" OR "price of silver")',
    "USO": '("oil prices" OR "crude oil" OR "Brent crude")',
    "TLT": '("Treasury yields" OR "bond market" OR "Treasury bonds")',
    "ETH-USD": '"Ethereum"',
    "XRP-USD": '("XRP" OR "Ripple Labs")',
    "SOL-USD": '"Solana"',
}


def _phrase(term: str) -> str:
    return f'"{term.replace(chr(34), "")}"'


_ACRONYM_CONTEXT = "(shares OR stock OR investors OR CEO OR earnings OR company)"


def build_query(company: CompanyRef) -> str:
    """GDELT query for a company (language-filtered), precise for ambiguous brands.

    * curated query when we have one (common-word brands, funds, majors);
    * short acronyms ("IBM", "AMD") must co-occur with corporate context words;
    * other common-word names are anchored to company phrases ("X Inc", "X shares"…);
    * distinctive names are searched as-is, OR'd with their aliases.
    """
    if company.ticker in CURATED:
        return f"{CURATED[company.ticker]} {LANG}"
    short = (company.short_name or company.ticker).strip()
    names = [short] + [a for a in company.aliases if len(a) >= 4 and short.lower() not in a.lower()]
    names = [n for n in dict.fromkeys(names) if n]
    if company.quote_type == "EQUITY":
        # Short acronyms and "X&Y" initials ("H&R", "M&T") tokenize to near-stopwords.
        if len(short.replace("&", "")) <= 3 or re.search(r"\b\w{1,2}&\w{1,2}\b", short):
            return f"{_phrase(short)} {_ACRONYM_CONTEXT} {LANG}"
        if is_common_word_name(short):
            anchors = [f"{short} Inc", f"{short} Corp", f"{short} shares", f"{short} stock", f"{short} CEO"]
            anchors += [a for a in company.aliases if " " in a]
            return f"({' OR '.join(_phrase(t) for t in dict.fromkeys(anchors))}) {LANG}"
    if len(names) == 1:
        return f"{_phrase(names[0])} {LANG}"
    return f"({' OR '.join(_phrase(n) for n in names[:5])}) {LANG}"


def fallback_query(company: CompanyRef) -> str:
    """Simplest valid query, used when GDELT rejects the precise one."""
    return f"{_phrase(company.short_name or company.ticker)} {LANG}"


# --------------------------------------------------------------------------- #
# Parsing & statistics (pure)
# --------------------------------------------------------------------------- #
def parse_timeline(payload: Any) -> dict[date, tuple[float, float | None]]:
    """GDELT timeline JSON -> {day: (value, norm)} (first series only)."""
    timeline = payload.get("timeline") if isinstance(payload, dict) else None
    if not timeline:
        return {}
    out: dict[date, tuple[float, float | None]] = {}
    for point in timeline[0].get("data") or []:
        raw = str(point.get("date") or "")
        try:
            day = date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
            value = float(point.get("value"))
        except (ValueError, TypeError):
            continue
        if not math.isfinite(value):
            continue
        norm = point.get("norm")
        out[day] = (value, float(norm) if isinstance(norm, (int, float)) else None)
    return out


def _incomplete_tail(volume: dict[date, tuple[float, float | None]], days: list[date]) -> set[date]:
    """Trailing days GDELT has not finished ingesting.

    `norm` is the total number of articles GDELT monitored that day; while a day
    is still being processed it is far below normal (e.g. 47k vs ~150k), and its
    raw count would read as a fake collapse in attention.
    """
    incomplete: set[date] = set()
    for i in range(len(days) - 1, -1, -1):
        day = days[i]
        norm = volume.get(day, (0.0, None))[1]
        history = [volume[d][1] for d in days[max(0, i - 14):i] if d in volume and volume[d][1]]
        if norm is None or len(history) < 5:
            break
        baseline = sorted(history)[len(history) // 2]
        if norm >= 0.6 * baseline:
            break
        incomplete.add(day)
    return incomplete


def merge_series(
    tone: dict[date, tuple[float, float | None]],
    volume: dict[date, tuple[float, float | None]],
    *,
    today: date,
) -> list[TonePoint]:
    """Union of complete days, oldest first.

    Today's UTC day and any trailing day GDELT is still ingesting are dropped:
    their volume is incomplete and would read as a fake slump in attention.
    """
    days = sorted(d for d in set(tone) | set(volume) if d < today)
    skip = _incomplete_tail(volume, days)
    return [
        TonePoint(
            date=d,
            tone=round(tone[d][0], 4) if d in tone else None,
            volume=volume[d][0] if d in volume else None,
        )
        for d in days
        if d not in skip
    ]


def _window_mean(points: list[TonePoint], end: date, days: int) -> float | None:
    """Volume-weighted mean tone over (end - days, end]; simple mean without volume."""
    start = end - timedelta(days=days)
    rows = [p for p in points if start < p.date <= end and p.tone is not None]
    if not rows:
        return None
    weights = [p.volume for p in rows]
    if all(w is not None and w > 0 for w in weights):
        total = sum(weights)  # type: ignore[arg-type]
        return sum(p.tone * w for p, w in zip(rows, weights, strict=True)) / total  # type: ignore[operator]
    return sum(p.tone for p in rows) / len(rows)  # type: ignore[misc]


def tone_stats(points: list[TonePoint]) -> dict[str, float | None]:
    """tone_7d/30d/90d, 7d-vs-30d change and where the 7d tone sits in 90 days."""
    toned = [p for p in points if p.tone is not None]
    if not toned:
        return {"tone_7d": None, "tone_30d": None, "tone_90d": None,
                "change_7d_vs_30d": None, "percentile_7d": None}
    end = toned[-1].date
    t7, t30, t90 = (_window_mean(points, end, n) for n in (7, 30, 90))
    # Percentile of the latest rolling 7-day tone among all rolling 7-day tones
    # in the window (mid-rank, so ties don't bias it high).
    rolling: list[float] = []
    for p in toned:
        if p.date <= end - timedelta(days=90):
            continue
        n_obs = sum(1 for q in toned if p.date - timedelta(days=7) < q.date <= p.date)
        if n_obs >= 4:
            value = _window_mean(points, p.date, 7)
            if value is not None:
                rolling.append(value)
    percentile = None
    if t7 is not None and len(rolling) >= 10:
        below = sum(1 for v in rolling if v < t7)
        equal = sum(1 for v in rolling if v == t7)
        percentile = round((below + 0.5 * equal) / len(rolling), 3)
    return {
        "tone_7d": round(t7, 3) if t7 is not None else None,
        "tone_30d": round(t30, 3) if t30 is not None else None,
        "tone_90d": round(t90, 3) if t90 is not None else None,
        "change_7d_vs_30d": round(t7 - t30, 3) if t7 is not None and t30 is not None else None,
        "percentile_7d": percentile,
    }


def build_trend(query: str, tone_payload: Any, volume_payload: Any, *, today: date) -> ToneTrend | None:
    """Pure: two GDELT payloads -> ToneTrend (None when both are empty)."""
    series = merge_series(parse_timeline(tone_payload), parse_timeline(volume_payload), today=today)
    if not series:
        return None
    return ToneTrend(query=query, series=series, **tone_stats(series))


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
class GdeltRateLimited(UpstreamError):
    """GDELT refused the request (1 request / 5 s per IP)."""


class GdeltQueryRejected(UpstreamError):
    """GDELT answered with a plain-text error about the query itself."""


def _is_rate_limit_text(text: str) -> bool:
    return "limit requests" in text.lower()


async def _gdelt(query: str, mode: str, days: int, *, retry: bool = True) -> Any:
    """One GDELT timeline call; retries once after a rate-limit answer if `retry`."""
    params = {"query": query, "mode": mode, "timespan": f"{days}d", "format": "json"}
    attempts = 2 if retry else 1
    for attempt in range(attempts):
        try:
            resp = await fetch(API_URL, params=params, api_ua=True, retries=0, timeout=20.0)
        except httpx.HTTPStatusError as exc:
            resp = exc.response
            if resp.status_code != 429:
                raise UpstreamError(f"GDELT HTTP {resp.status_code}") from exc
        body = resp.text.lstrip()
        if body.startswith("{"):
            try:
                return resp.json()
            except ValueError as exc:
                raise UpstreamError("GDELT returned malformed JSON") from exc
        if not body and resp.status_code == 200:
            return {}
        if resp.status_code == 429 or _is_rate_limit_text(body):
            if attempt + 1 < attempts:
                await asyncio.sleep(RATE_LIMIT_WAIT)
                continue
            raise GdeltRateLimited("GDELT rate limit (1 request / 5 s per IP)")
        raise GdeltQueryRejected(f"GDELT: {body[:120]}")
    raise GdeltRateLimited("GDELT rate limit (1 request / 5 s per IP)")


@cached(ttl=settings.history_cache_ttl, none_ttl=900)
async def _tone_trend(query: str, fallback: str, days: int) -> ToneTrend | None:
    try:
        tone = await _gdelt(query, "timelinetone", days)
    except GdeltQueryRejected as exc:
        logger.info("GDELT rejected %r (%s); retrying with %r", query, exc, fallback)
        query = fallback
        tone = await _gdelt(query, "timelinetone", days)
    try:
        # Single attempt: a second rate-limit wait would push the task past the
        # orchestrator's intel timeout and lose the tone we already have.
        volume = await _gdelt(query, "timelinevolraw", days, retry=False)
    except UpstreamError as exc:  # tone without volume is still the core signal
        logger.info("GDELT volume unavailable for %r: %s", query, exc)
        volume = {}
    return build_trend(query, tone, volume, today=datetime.now(UTC).date())


# Last good result per query: GDELT refuses often from shared IPs, and a trend
# fetched a few hours ago is far more useful than none (its dates say how old it is).
_LAST_GOOD: dict[tuple[str, int], tuple[float, ToneTrend]] = {}
STALE_MAX_SECONDS = 24 * 3600


async def get_tone_trend(company: CompanyRef, days: int = 90) -> ToneTrend | None:
    """Daily GDELT tone + article volume for the company over the last `days` (<= 90).

    On a GDELT refusal, serves the last good trend for the same query if it is
    less than a day old; otherwise raises `UpstreamError`.
    """
    span = max(7, min(int(days), 90))
    query = build_query(company)
    key = (query, span)
    try:
        trend = await _tone_trend(query, fallback_query(company), span)
    except UpstreamError:
        stale = _LAST_GOOD.get(key)
        if stale and time.monotonic() - stale[0] < STALE_MAX_SECONDS:
            logger.info("GDELT unavailable; serving last good trend for %r", query)
            return stale[1]
        raise
    if trend is not None:
        if len(_LAST_GOOD) > 512:
            _LAST_GOOD.pop(next(iter(_LAST_GOOD)))
        _LAST_GOOD[key] = (time.monotonic(), trend)
    return trend
