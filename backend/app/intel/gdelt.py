"""Global news tone & volume from the GDELT DOC 2.0 API (thousands of outlets).

GDELT scores the tone of every article it monitors (roughly -10..+10; most
company coverage sits in -3..+3), which gives SentiNET a 90-day baseline that
our own snapshot history cannot: "is today's coverage unusually negative *for
this company*?".

Two hard parts live here:

* **Query precision.** GDELT matches full article text case-insensitively, so
  `"Apple"` drags in orchards and pies (~15% of a 3-day artlist sample was about
  the company; the curated query below: ~75%). Common-word brands get curated
  context (`"Apple" (iPhone OR iPad OR …)`), any other common-word name is
  anchored to company phrases (`"X Inc" OR "X shares" …`), funds search their
  theme ("regional banks"), and short words are never quoted (GDELT rejects
  quoted phrases under 5 characters).
* **Politeness.** GDELT allows one request per 5 s per IP and answers
  violations with a plain-text "Please limit requests…" body (HTTP 429 or even
  200) after a 10-15 s wait. Requests are serialized and spaced, a refusal
  pauses all requests (fail fast, clear message), and cached payloads are
  served stale while one background job refreshes them (see "Fetching").
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
import weakref
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from app.config import settings
from app.core.cache import TTLStore
from app.core.http import UpstreamError, fetch
from app.resolve.names import is_common_word_name
from app.resolve.words import COMMON_WORDS
from app.schemas import TonePoint, ToneTrend
from app.sources.base import CompanyRef

logger = logging.getLogger(__name__)

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
LANG = "sourcelang:english"
RATE_LIMIT_WAIT = 6.0

# Curated queries where the brand alone is ambiguous or not what the news says.
# Rules (see the traps below): every term >= 5 characters; no social/app-store
# names that appear in page chrome ("Follow us on Facebook", "Get it on Google
# Play" — a Meta query built on them matched 1 article in 75); no market words
# after an everyday-word brand. Valid GDELT syntax: quoted phrases, one level of
# (… OR …) groups (parentheses around a single term are rejected).
CURATED: dict[str, str] = {
    "AAPL": '"Apple" (iPhone OR "Apple Inc" OR "Tim Cook" OR Cupertino OR MacBook OR "Apple Watch" OR "Vision Pro")',
    "META": '("Meta Platforms" OR Zuckerberg OR "Meta CEO" OR "Meta Superintelligence")',
    "GOOGL": '("Alphabet Inc" OR "Sundar Pichai" OR "Google parent" OR "Google CEO" OR "Google antitrust" '
             'OR "Google Gemini")',
    "GOOG": '("Alphabet Inc" OR "Sundar Pichai" OR "Google parent" OR "Google CEO" OR "Google antitrust" '
            'OR "Google Gemini")',
    "AMZN": '("Amazon.com" OR "Andy Jassy" OR "Amazon Web Services" OR "Amazon CEO" OR Bezos)',
    "TGT": '("Target Corp" OR "Target Corporation" OR "retailer Target" OR "Target CEO")',
    # Not bare "Cash App": event listings say "pay via Cash App" (most of a 3-day sample).
    "XYZ": '("Block Inc" OR "Jack Dorsey" OR Afterpay OR "Square payments" OR "Cash App owner" OR "Block CEO")',
    "SQ": '("Block Inc" OR "Jack Dorsey" OR Afterpay OR "Square payments" OR "Cash App owner" OR "Block CEO")',
    "SNAP": '("Snap Inc" OR Snapchat OR "Evan Spiegel")',
    "V": '("Visa Inc" OR "Visa and Mastercard" OR "Visa CEO" OR "Visa card")',
    "SHEL": '("Shell plc" OR "Royal Dutch Shell" OR "Shell CEO" OR "Shell oil")',
    "ORCL": '("Oracle Corp" OR "Oracle Corporation" OR "Oracle Cloud" OR "Larry Ellison" OR "Oracle CEO")',
    "F": '("Ford Motor" OR "Ford CEO" OR "Jim Farley" OR "Ford Mustang")',
    "MSTR": '("MicroStrategy" OR "Michael Saylor" OR "Strategy Inc")',
    "ZM": '("Zoom Video" OR "Zoom Communications" OR "Zoom CEO" OR "Eric Yuan")',
    "U": '("Unity Software" OR "Unity Technologies")',
    "LCID": '("Lucid Group" OR "Lucid Motors" OR "Lucid Gravity")',
    "SOFI": '("SoFi Technologies" OR "SoFi Bank" OR "SoFi Invest" OR "SoFi CEO" OR "Anthony Noto")',
    "GAP": '("Gap Inc" OR "Old Navy" OR "Banana Republic")',
    "GPS": '("Gap Inc" OR "Old Navy" OR "Banana Republic")',
    "CCL": '("Carnival Corp" OR "Carnival Corporation" OR "Carnival Cruise")',
    "DAL": '("Delta Air Lines" OR "Delta Airlines")',
    "UAL": '"United Airlines"',
    "AAL": '"American Airlines"',
    "LUV": '"Southwest Airlines"',
    "BP": '("BP plc" OR "British Petroleum" OR "BP CEO")',
    "GE": '("GE Aerospace" OR "General Electric")',
    # "AT&T" tokenizes to "at t" (matches "S & T", "at the"…): require telecom context.
    "T": '"AT&T" (wireless OR telecom OR carrier OR broadband OR Verizon OR "T-Mobile")',
    "C": '("Citigroup" OR "Citi bank" OR "Citibank")',
    "HOOD": '("Robinhood Markets" OR "Robinhood app" OR "Robinhood CEO" OR "Vlad Tenev")',
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
    "XRP-USD": '("Ripple Labs" OR "XRP price" OR "XRP Ledger" OR "XRP token")',
    "SOL-USD": '"Solana"',
}


# GDELT's full-text index has two traps (both verified 2026-10-04):
# * a quoted phrase under 5 characters is rejected ("The specified phrase is too
#   short." for "Meta"/"IBM") and a bare short keyword silently matches nothing
#   (`Meta`, `IBM (shares OR stock)` -> {}), so a short name is only searchable
#   inside a longer phrase ("IBM shares", "Meta Platforms");
# * phrases ignore stopwords, so "at Target" == "Target" and "Target shares"
#   matches "price target on shares of …": an everyday-word brand is never
#   paired with market vocabulary, only with its legal form, CEO, products.
MIN_PHRASE = 5


def searchable(term: str) -> bool:
    """True when GDELT can match `term` at all (see the traps above)."""
    return len(term.strip()) >= MIN_PHRASE


def _phrase(term: str) -> str:
    """Quote a (searchable) term for GDELT."""
    return f'"{term.replace(chr(34), "").strip()}"'


def _or(terms: list[str]) -> str | None:
    """`term` or `(a OR b …)` over the searchable, de-duplicated terms (max 6); None if none."""
    unique: dict[str, str] = {}
    for term in terms:
        term = (term or "").strip()
        if term and searchable(term):
            unique.setdefault(term.lower(), term)
    picked = list(unique.values())[:6]
    if not picked:
        return None
    return _phrase(picked[0]) if len(picked) == 1 else f"({' OR '.join(_phrase(t) for t in picked)})"


def build_query(company: CompanyRef) -> str | None:
    """GDELT query for a company (language-filtered), or None when nothing is searchable.

    * curated query when we have one (homonym brands, funds, majors);
    * funds, indices and futures search their theme ("regional banks", "gold prices");
    * short names ("IBM", "Nike") only inside phrases: "IBM shares", "Nike CEO", full names;
    * everyday-word names ("Chewy") only in their legal form / with "CEO" / via aliases;
    * distinctive names are searched as-is, OR'd with their aliases.
    """
    if company.ticker in CURATED:
        return f"{CURATED[company.ticker]} {LANG}"
    short = (company.short_name or company.ticker).strip()
    if company.quote_type in {"ETF", "MUTUALFUND", "INDEX", "FUTURE"}:
        themed = _or(_fund_themes(company))
        if themed:
            return f"{themed} {LANG}"
    phrases = [a for a in company.aliases if " " in a.strip()]
    if company.quote_type == "EQUITY" and is_common_word_name(short):
        terms = [f"{short} Inc", f"{short} Corp", f"{short} CEO", *phrases]
        if short.lower() not in COMMON_WORDS:  # acronym-like ("IBM"): market words are safe to pair
            terms += [f"{short} shares", f"{short} stock"]
    else:
        terms = [short, *(a for a in company.aliases if short.lower() not in a.lower())]
        if not searchable(short) and company.quote_type == "EQUITY":
            terms += [f"{short} shares", f"{short} stock", f"{short} CEO"]
    query = _or(terms)
    return f"{query} {LANG}" if query else None


# Single words that appear in nearly every market article (exchange listings, …).
_TOO_BROAD = {"nasdaq", "nyse", "dow", "market", "stocks"}


def _fund_themes(company: CompanyRef) -> list[str]:
    """What a fund, index or future is about, as GDELT search terms ([] = generic path).

    Funds use the search sources' theme table ("regional banks" for KRE); indices
    their name plus aliases; futures "<name> prices/futures". A lone everyday or
    ubiquitous word ("Gold", "Nasdaq") is never searched by itself.
    """
    short = (company.short_name or "").strip()
    terms: list[str] = []
    if company.quote_type in {"ETF", "MUTUALFUND"}:
        try:
            from app.sources.query import etf_theme

            terms = list(etf_theme(company))
        except ImportError:  # sources package mid-edit: fall through to aliases
            terms = []
    if not terms:
        terms = [short, *(a for a in company.aliases if " " in a)]
        if company.quote_type == "FUTURE" and short:
            terms += [f"{short} prices", f"{short} futures"]

    def precise(term: str) -> bool:
        if " " in term.strip():
            return True
        word = term.strip().lower()
        return word not in COMMON_WORDS and word not in _TOO_BROAD

    return [t for t in terms if t and precise(t)]


def fallback_query(company: CompanyRef) -> str | None:
    """Simplest valid query (one searchable name), used when GDELT rejects the precise one."""
    names = [n.strip() for n in [company.short_name, *company.aliases, company.name] if n and searchable(n)]
    return f"{_phrase(names[0])} {LANG}" if names else None


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
# GDELT answers slowly (10-15 s, refusals included) and allows one request per
# 5 s per IP, so the fetch layer is built to never make things worse:
#
# * one request in flight at a time, the next one >= `SPACING` s after the
#   previous *finished* (the shared start-spacing limiter alone lets slow
#   requests overlap, which GDELT counts against us);
# * after a refusal, a cooldown (30 s doubling to 5 min) during which no
#   request is sent and callers fail fast with a clear message;
# * tone and volume payloads are cached separately: fresh for
#   `settings.history_cache_ttl`, servable (stale) for 24 h while a single
#   background job refreshes them;
# * a cold caller waits only for the tone; volume follows in the background.
SPACING = 5.5
COOLDOWN_BASE = 30.0
COOLDOWN_MAX = 300.0
STALE_MAX_SECONDS = 24 * 3600
REQUEST_TIMEOUT = 25.0
REFRESH_GAP = 120.0  # min seconds between background refreshes of one query (failing volume…)
TONE, VOLUME = "timelinetone", "timelinevolraw"


class GdeltRateLimited(UpstreamError):
    """GDELT refused the request (1 request / 5 s per IP)."""


class GdeltQueryRejected(UpstreamError):
    """GDELT answered with a plain-text error about the query itself."""


def _is_rate_limit_text(text: str) -> bool:
    return "limit requests" in text.lower()


@dataclass
class _Breaker:
    """Cooldown after refusals: consecutive strikes double the pause."""

    until: float = 0.0
    strikes: int = 0

    def remaining(self) -> float:
        return max(0.0, self.until - time.monotonic())

    def trip(self) -> None:
        self.strikes += 1
        pause = min(COOLDOWN_MAX, COOLDOWN_BASE * 2 ** (self.strikes - 1))
        self.until = time.monotonic() + pause
        logger.info("GDELT refused us; pausing requests for %.0f s", pause)

    def reset(self) -> None:
        self.until, self.strikes = 0.0, 0


_breaker = _Breaker()


@dataclass
class _Gate:
    """Serializes GDELT requests within one event loop."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_ok: float = 0.0


_gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Gate] = weakref.WeakKeyDictionary()


def _gate() -> _Gate:
    loop = asyncio.get_running_loop()
    gate = _gates.get(loop)
    if gate is None:
        gate = _gates[loop] = _Gate()
    return gate


def cooling_down() -> float:
    """Seconds until GDELT may be asked again (0 when it may be asked now)."""
    return _breaker.remaining()


def _refused_message() -> str:
    wait = _breaker.remaining()
    when = f"next try in {wait:.0f} s" if wait > 0 else "retry shortly"
    return f"GDELT is rate-limiting this server's IP (1 request / 5 s); {when}"


async def _send(params: dict[str, str]) -> httpx.Response:
    """One serialized, spaced GDELT request (raises GdeltRateLimited while cooling down)."""
    gate = _gate()
    async with gate.lock:
        if _breaker.remaining() > 0:
            raise GdeltRateLimited(_refused_message())
        wait = gate.next_ok - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            return await fetch(API_URL, params=params, api_ua=True, retries=0, timeout=REQUEST_TIMEOUT)
        except httpx.HTTPStatusError as exc:
            return exc.response
        except UpstreamError as exc:
            raise UpstreamError(f"GDELT unreachable ({exc})") from exc
        finally:
            gate.next_ok = time.monotonic() + SPACING


async def _gdelt(query: str, mode: str, days: int, *, retry: bool = True) -> Any:
    """One GDELT timeline call; on a refusal waits `RATE_LIMIT_WAIT` and retries once if `retry`."""
    params = {"query": query, "mode": mode, "timespan": f"{days}d", "format": "json"}
    for attempt in range(2 if retry else 1):
        if attempt:
            await asyncio.sleep(RATE_LIMIT_WAIT)
        resp = await _send(params)
        body = resp.text.lstrip()
        if resp.status_code == 200 and body.startswith("{"):
            _breaker.reset()
            try:
                return resp.json()
            except ValueError as exc:
                raise UpstreamError("GDELT returned malformed JSON") from exc
        if resp.status_code == 200 and not body:
            _breaker.reset()
            return {}  # no matching articles
        if resp.status_code == 429 or _is_rate_limit_text(body):
            continue
        if resp.status_code == 200:
            raise GdeltQueryRejected(f"GDELT rejected the query: {body[:100].strip()}")
        raise UpstreamError(f"GDELT HTTP {resp.status_code}")
    _breaker.trip()
    raise GdeltRateLimited(_refused_message())


# ---- payload cache ------------------------------------------------------------ #
@dataclass(frozen=True)
class _Payload:
    data: Any
    query: str  # the query that produced it (the fallback, if GDELT rejected the precise one)
    fetched: float  # time.monotonic()

    def fresh(self) -> bool:
        return time.monotonic() - self.fetched < settings.history_cache_ttl


_payloads = TTLStore(ttl=STALE_MAX_SECONDS, maxsize=1024)  # (query, mode, span) -> _Payload
_jobs: dict[tuple[str, int], tuple[asyncio.Task[None], asyncio.Future[None]]] = {}
_last_start: dict[tuple[str, int], float] = {}


def _cached_payload(query: str, mode: str, span: int) -> _Payload | None:
    return _payloads.get((query, mode, span))


async def _refresh(query: str, fallback: str | None, span: int, tone_ready: asyncio.Future[None]) -> None:
    """Fetch whatever is missing or stale: tone first (signals `tone_ready`), then volume."""
    tone = _cached_payload(query, TONE, span)
    try:
        if tone is None or not tone.fresh():
            try:
                data, used = await _gdelt(query, TONE, span), query
            except GdeltQueryRejected as exc:
                if not fallback or fallback == query:
                    raise
                logger.info("GDELT rejected %r (%s); using %r", query, exc, fallback)
                data, used = await _gdelt(fallback, TONE, span), fallback
            tone = _Payload(data, used, time.monotonic())
            _payloads.set((query, TONE, span), tone)
    except asyncio.CancelledError:
        tone_ready.cancel()  # shutdown: never leave a waiter hanging
        raise
    except Exception as exc:  # noqa: BLE001 - handed to the waiting caller
        tone_ready.set_exception(exc)
        tone_ready.exception()  # mark retrieved: a background refresh may have no waiter
        return
    tone_ready.set_result(None)
    volume = _cached_payload(query, VOLUME, span)
    if volume is not None and volume.fresh() and volume.query == tone.query:
        return
    try:
        # One attempt: tone is the core signal, volume can wait for the next refresh.
        data = await _gdelt(tone.query, VOLUME, span, retry=False)
        _payloads.set((query, VOLUME, span), _Payload(data, tone.query, time.monotonic()))
    except UpstreamError as exc:
        logger.info("GDELT volume unavailable for %r: %s", tone.query, exc)


def _start_refresh(query: str, fallback: str | None, span: int) -> asyncio.Future[None]:
    """Single-flight background refresh per (query, span); returns its tone-ready future."""
    key = (query, span)
    job = _jobs.get(key)
    loop = asyncio.get_running_loop()
    if job is not None and not job[0].done() and job[0].get_loop() is loop:
        return job[1]
    tone_ready: asyncio.Future[None] = loop.create_future()
    task = loop.create_task(_refresh(query, fallback, span, tone_ready), name=f"gdelt:{query[:40]}")
    _jobs[key] = (task, tone_ready)
    _last_start[key] = time.monotonic()

    def _forget(done: asyncio.Task[None]) -> None:
        if key in _jobs and _jobs[key][0] is done:
            del _jobs[key]

    task.add_done_callback(_forget)
    return tone_ready


def _trend_from_cache(query: str, span: int) -> ToneTrend | None:
    tone = _cached_payload(query, TONE, span)
    if tone is None:
        return None
    volume = _cached_payload(query, VOLUME, span)
    vol_data = volume.data if volume is not None and volume.query == tone.query else {}
    return build_trend(tone.query, tone.data, vol_data, today=datetime.now(UTC).date())


async def get_tone_trend(company: CompanyRef, days: int = 90) -> ToneTrend | None:
    """Daily GDELT tone + article volume for the company over the last `days` (<= 90).

    Serves cached payloads (refreshing stale ones in the background); a cold call
    waits for the tone only. Raises `GdeltRateLimited` (clear message, no request
    sent while cooling down) or `UpstreamError` when nothing usable is cached.
    Returns None when GDELT has no coverage, or no searchable name exists.
    """
    span = max(7, min(int(days), 90))
    query = build_query(company)
    if query is None:
        logger.info("GDELT cannot search %s (no name of %d+ characters)", company.ticker, MIN_PHRASE)
        return None
    tone = _cached_payload(query, TONE, span)
    volume = _cached_payload(query, VOLUME, span)
    complete = tone is not None and tone.fresh() and volume is not None and volume.fresh()
    if tone is not None:
        recently = time.monotonic() - _last_start.get((query, span), -REFRESH_GAP) < REFRESH_GAP
        if not complete and not recently and _breaker.remaining() == 0:
            _start_refresh(query, fallback_query(company), span)
        return _trend_from_cache(query, span)
    if _breaker.remaining() > 0:
        raise GdeltRateLimited(_refused_message())
    await asyncio.shield(_start_refresh(query, fallback_query(company), span))
    return _trend_from_cache(query, span)


async def drain() -> None:
    """Wait for background refreshes to finish (tests, graceful shutdown)."""
    tasks = [task for task, _ in list(_jobs.values()) if task.get_loop() is asyncio.get_running_loop()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def reset_state() -> None:
    """Test helper: forget payloads, jobs, refresh history and the cooldown."""
    _payloads.clear()
    _jobs.clear()
    _last_start.clear()
    _breaker.reset()
