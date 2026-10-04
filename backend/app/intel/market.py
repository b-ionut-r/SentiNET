"""Market-wide context: Fear & Greed gauges, what the crowd is trending, headlines.

* CNN Fear & Greed (7 components + 1y history). CNN's data endpoint only answers
  browser-like requests from cnn.com (UA + Referer/Origin).
* Crypto Fear & Greed from alternative.me (daily since 2018).
* Trending: ApeWisdom's Reddit mention leaderboard (with 24h change) and
  StockTwits' trending symbols.
* Headlines: CNBC + MarketWatch section feeds and Google/Bing "stock market"
  searches, merged newest-first and de-duplicated (raw; analytics scores them).

Every parser is pure and fixture-tested; fetchers degrade per feed.
"""
from __future__ import annotations

import asyncio
import calendar
import html
import logging
import math
import re
from datetime import datetime, timedelta, UTC
from typing import Any
from urllib.parse import parse_qs, urlsplit

import feedparser

from app.config import settings
from app.core.cache import cached
from app.core.http import UpstreamError, fetch, fetch_json
from app.core.sync import run_cpu
from app.schemas import FearGreed, GaugeComponent, HistoryValue, TrendingTicker
from app.sources.base import RawSignal

logger = logging.getLogger(__name__)

CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CRYPTO_FNG_URL = "https://api.alternative.me/fng/"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"

UTC = UTC


def _now() -> datetime:
    return datetime.now(UTC)

# --------------------------------------------------------------------------- #
# Fear & Greed
# --------------------------------------------------------------------------- #
CNN_COMPONENTS: dict[str, str] = {
    "market_momentum_sp500": "S&P 500 momentum",
    "stock_price_strength": "52-week highs vs lows",
    "stock_price_breadth": "Market breadth",
    "put_call_options": "Put/call ratio",
    "market_volatility_vix": "Volatility (VIX)",
    "junk_bond_demand": "Junk bond demand",
    "safe_haven_demand": "Safe-haven demand",
}


def _rating(text: Any) -> str:
    return " ".join(w.capitalize() for w in str(text or "").split()) or "Unknown"


def _f(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_cnn(payload: Any) -> FearGreed | None:
    """Pure: CNN graphdata JSON -> FearGreed (score, 7 components, ~1y daily history)."""
    head = (payload or {}).get("fear_and_greed") if isinstance(payload, dict) else None
    score = _f((head or {}).get("score"))
    if score is None:
        return None
    components = []
    for key, label in CNN_COMPONENTS.items():
        comp = payload.get(key) or {}
        components.append(GaugeComponent(key=key, label=label, score=_r1(_f(comp.get("score"))),
                                         rating=_rating(comp.get("rating")) if comp.get("rating") else None))
    history = []
    for point in ((payload.get("fear_and_greed_historical") or {}).get("data") or []):
        t, v = _f(point.get("x")), _f(point.get("y"))
        if t is not None and v is not None:
            history.append(HistoryValue(t=datetime.fromtimestamp(t / 1000, UTC), v=round(v, 2)))
    history.sort(key=lambda h: h.t)
    return FearGreed(
        score=round(score, 1),
        rating=_rating(head.get("rating")),
        previous_close=_r1(_f(head.get("previous_close"))),
        week_ago=_r1(_f(head.get("previous_1_week"))),
        month_ago=_r1(_f(head.get("previous_1_month"))),
        year_ago=_r1(_f(head.get("previous_1_year"))),
        components=components,
        history=history,
    )


def _r1(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def parse_crypto_fng(payload: Any) -> FearGreed | None:
    """Pure: alternative.me /fng JSON -> FearGreed (history oldest first)."""
    items = payload.get("data") if isinstance(payload, dict) else None
    rows = []
    for item in items or []:
        v, ts = _f(item.get("value")), _f(item.get("timestamp"))
        if v is not None and ts is not None:
            rows.append((datetime.fromtimestamp(ts, UTC), v, str(item.get("value_classification") or "")))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    latest_t, latest_v, latest_label = rows[-1]

    def ago(days: int) -> float | None:
        target = latest_t - timedelta(days=days)
        best = min(rows, key=lambda r: abs((r[0] - target).total_seconds()))
        return best[1] if abs((best[0] - target).total_seconds()) <= 2 * 86400 else None

    return FearGreed(
        score=latest_v,
        rating=_rating(latest_label),
        previous_close=rows[-2][1] if len(rows) > 1 else None,
        week_ago=ago(7),
        month_ago=ago(30),
        year_ago=ago(365),
        components=[],
        history=[HistoryValue(t=t, v=v) for t, v, _ in rows[-366:]],
    )


@cached(ttl=900, none_ttl=120)
async def get_cnn_fear_greed() -> FearGreed | None:
    """CNN Fear & Greed index with its 7 components and ~1 year of history."""
    headers = {
        "User-Agent": settings.browser_user_agent,
        "Referer": "https://www.cnn.com/",
        "Origin": "https://www.cnn.com",
        "Accept": "application/json, text/plain, */*",
    }
    return parse_cnn(await fetch_json(CNN_URL, headers=headers, timeout=12.0))


@cached(ttl=1800, none_ttl=120)
async def get_crypto_fear_greed() -> FearGreed | None:
    """Crypto Fear & Greed (alternative.me) with ~1 year of daily history."""
    return parse_crypto_fng(await fetch_json(CRYPTO_FNG_URL, params={"limit": 400}, timeout=12.0))


# --------------------------------------------------------------------------- #
# Trending
# --------------------------------------------------------------------------- #
def parse_apewisdom(payload: Any, limit: int = 25) -> list[TrendingTicker]:
    """Pure: ApeWisdom leaderboard -> top Reddit tickers with 24h mention change."""
    out: list[TrendingTicker] = []
    for item in ((payload or {}).get("results") or [])[:limit] if isinstance(payload, dict) else []:
        sym = str(item.get("ticker") or "").upper().replace(".", "-")
        if not sym:
            continue
        mentions = _int(item.get("mentions"))
        prev = _int(item.get("mentions_24h_ago"))
        change = round((mentions - prev) / prev * 100, 1) if mentions is not None and prev else None
        out.append(TrendingTicker(
            symbol=sym, name=html.unescape(str(item.get("name") or "")) or None, source="reddit",
            rank=_int(item.get("rank")), rank_prev=_int(item.get("rank_24h_ago")),
            mentions=mentions, mentions_prev=prev, change_pct=change,
        ))
    return out


def stocktwits_symbol(sym: str) -> str:
    """StockTwits symbology -> ours: BTC.X -> BTC-USD, T.TSX -> T.TO."""
    sym = sym.upper()
    if sym.endswith(".X"):
        return f"{sym[:-2]}-USD"
    if sym.endswith(".TSX"):
        return f"{sym[:-4]}.TO"
    return sym


def parse_stocktwits_trending(payload: Any, limit: int = 30) -> list[TrendingTicker]:
    """Pure: StockTwits trending symbols -> TrendingTicker (rank order)."""
    out: list[TrendingTicker] = []
    for i, item in enumerate(((payload or {}).get("symbols") or [])[:limit] if isinstance(payload, dict) else []):
        sym = str(item.get("symbol") or "")
        if not sym:
            continue
        out.append(TrendingTicker(symbol=stocktwits_symbol(sym), name=item.get("title") or None,
                                  source="stocktwits", rank=_int(item.get("rank")) or i + 1))
    return out


def _int(value: Any) -> int | None:
    f = _f(value)
    return int(f) if f is not None else None


@cached(ttl=600, none_ttl=60)
async def get_trending() -> list[TrendingTicker]:
    """Reddit (ApeWisdom) top ~25 by mentions + StockTwits trending symbols."""
    ape, st = await asyncio.gather(
        fetch_json(APEWISDOM_URL, timeout=10.0),
        fetch_json(STOCKTWITS_TRENDING_URL, timeout=10.0),
        return_exceptions=True,
    )
    out: list[TrendingTicker] = []
    if not isinstance(ape, BaseException):
        out += parse_apewisdom(ape)
    else:
        logger.info("ApeWisdom trending failed: %s", ape)
    if not isinstance(st, BaseException):
        out += parse_stocktwits_trending(st)
    else:
        logger.info("StockTwits trending failed: %s", st)
    if not out and isinstance(ape, BaseException) and isinstance(st, BaseException):
        raise UpstreamError("trending: ApeWisdom and StockTwits unavailable")
    return out


# --------------------------------------------------------------------------- #
# Headlines
# --------------------------------------------------------------------------- #
CNBC_URL = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id={id}"
# Not "Wall Street": it matches every Wall Street Journal story.
GOOGLE_NEWS_URL = ("https://news.google.com/rss/search?q=%22stock+market%22+OR+%22S%26P+500%22+OR+%22Dow+Jones%22"
                   "+when:1d&hl=en-US&gl=US&ceid=US:en")
FEEDS: list[tuple[str, str, str | None, bool]] = [
    # (key, url, publisher, needs_market_filter) — most trusted first (dedupe keeps the first copy)
    ("cnbc_top", CNBC_URL.format(id=100003114), "CNBC", True),
    ("cnbc_finance", CNBC_URL.format(id=10000664), "CNBC", True),
    ("cnbc_economy", CNBC_URL.format(id=20910258), "CNBC", True),
    ("marketwatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "MarketWatch", True),
    ("google_news", GOOGLE_NEWS_URL, None, True),
    ("bing_news", "https://www.bing.com/news/search?q=%22stock+market%22&format=rss", None, True),
    ("bing_wallstreet", "https://www.bing.com/news/search?q=Wall+Street+stocks&format=rss", None, True),
]
MAX_HEADLINES = 120
PER_FEED_CAP = 40  # keep one aggregator from drowning out the others

# General news feeds carry politics/lifestyle too; keep what can move markets.
_MARKET_TERMS = re.compile(
    r"\b(stocks?|shares?|market|markets|wall street|s&p|nasdaq|dow|fed|federal reserve|powell|rates?|"
    r"yields?|treasur(?:y|ies)|bonds?|inflation|cpi|ppi|jobs|payrolls|unemployment|gdp|recession|"
    r"economy|economic|earnings|revenue|profit|guidance|ipo|merger|acquisition|deal|tariffs?|trade war|"
    r"oil|crude|opec|gold|bitcoin|crypto|dollar|investors?|traders?|futures|rally|sell-?off|"
    r"bank|banks|tech|ai|chip|chips|semiconductors?|layoffs?|bankruptcy|sec|antitrust|ceo)\b",
    re.IGNORECASE,
)
# Single-country market stories (Lagos, Dhaka, Seoul…) crowd out what moves US
# markets; kept only when they also mention Wall Street / US benchmarks.
_FOREIGN = re.compile(
    r"\b(bangladesh\w*|dhaka|nigeria\w*|lagos|ghana\w*|kenya\w*|nairobi|pakistan\w*|karachi|psx|sri lanka\w*|"
    r"india\w*|sensex|nifty|bse|nse|korea\w*|seoul|kospi|philippine\w*|psei|vietnam\w*|thai\w*|indonesia\w*|"
    r"malaysia\w*|bursa|egypt\w*|egx|saudi|tadawul|turk\w*|borsa|french|cac 40|german\w*|dax|ftse|uk stocks|"
    r"nikkei|hang seng|shanghai|shenzhen|asx|tsx|jse|zimbabwe\w*|uganda\w*|zambia\w*)\b",
    re.IGNORECASE,
)
_US_MARKET = re.compile(r"\b(wall street|s&p|nasdaq|dow|fed|federal reserve|treasur\w+|u\.?s\.?|american|nyse)\b",
                        re.IGNORECASE)
# First-person advice columns ("I'm 71 and still working…?") are not market news.
_ADVICE = re.compile(r"^[‘'\"“]?(?:I|I’m|I'm|I’ve|I've|My|We|We’re|We're|Our|Should I|Can I|How do I)\b")
_TAG = re.compile(r"<[^>]+>")
_SUFFIX_SPLIT = re.compile(r"\s+[-–|]\s+(?=[^-–|]+$)")


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", text or ""))).strip()


def _unwrap_bing(url: str) -> str:
    if "bing.com/news/apiclick" in url:
        target = parse_qs(urlsplit(url).query).get("url")
        if target:
            return target[0]
    return url


def _publisher_from_url(url: str) -> str | None:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else (host or None)


def parse_feed(xml_text: str, key: str, publisher: str | None, market_filter: bool) -> list[RawSignal]:
    """Pure: one RSS feed -> RawSignals (publisher split off, sponsored items dropped)."""
    feed = feedparser.parse(xml_text)
    out: list[RawSignal] = []
    for entry in feed.entries:
        if str(entry.get("metadata_sponsored") or "").lower() == "true":
            continue
        title = _clean(entry.get("title"))
        url = _unwrap_bing(str(entry.get("link") or "")) or None
        pub = publisher
        if key == "google_news":
            source = entry.get("source") or {}
            pub = source.get("title") if isinstance(source, dict) else None
            parts = _SUFFIX_SPLIT.split(title)
            if len(parts) == 2 and (not pub or parts[1].strip().lower() == pub.lower()):
                title, pub = parts[0].strip(), pub or parts[1].strip()
        elif key.startswith("bing"):
            pub = re.sub(r"\s+on MSN$", "", str(entry.get("news_source") or "")).strip() or None
        pub = pub or (_publisher_from_url(url) if url else None)
        if not title or len(title) < 12:
            continue
        body = None if key == "google_news" else (_clean(entry.get("summary")) or None)
        if market_filter and (_ADVICE.match(title) or not _MARKET_TERMS.search(f"{title} {body or ''}")):
            continue
        if _FOREIGN.search(title) and not _US_MARKET.search(title):
            continue
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        ts = datetime.fromtimestamp(calendar.timegm(parsed), UTC) if parsed else None
        out.append(RawSignal(title=title, body=body if body and body != title else None, url=url,
                             publisher=pub, timestamp=ts, extra={"feed": key}))
    return out


def _dedupe_key(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", title.lower()).strip()[:90]


def merge_headlines(batches: list[list[RawSignal]], *, now: datetime, limit: int = MAX_HEADLINES) -> list[RawSignal]:
    """Pure: merge feeds (first copy wins), keep fresh items, newest first.

    Fresh = last 48 h; on quiet days (weekends) the window widens to 96 h so the
    page still shows the latest market narrative instead of nothing.
    """
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    merged: list[RawSignal] = []
    for batch in batches:
        newest_first = sorted(batch, key=lambda s: s.timestamp or datetime.min.replace(tzinfo=UTC), reverse=True)
        for sig in newest_first[:PER_FEED_CAP]:
            key = _dedupe_key(sig.title)
            if key in seen_titles or (sig.url and sig.url in seen_urls):
                continue
            seen_titles.add(key)
            if sig.url:
                seen_urls.add(sig.url)
            merged.append(sig)
    for hours in (48, 96):
        fresh = [s for s in merged if s.timestamp is None or now - s.timestamp <= timedelta(hours=hours)]
        if len(fresh) >= 25:
            break
    fresh = [s for s in fresh if s.timestamp is None or s.timestamp <= now + timedelta(minutes=10)]
    fresh.sort(key=lambda s: s.timestamp or datetime.min.replace(tzinfo=UTC), reverse=True)
    return fresh[:limit]


@cached(ttl=600, none_ttl=60)
async def get_market_headlines() -> list[RawSignal]:
    """Up to 120 fresh, de-duplicated market headlines from CNBC, MarketWatch, Google and Bing."""
    async def one(key: str, url: str, publisher: str | None, filt: bool) -> list[RawSignal]:
        resp = await fetch(url, timeout=10.0)
        return await run_cpu(parse_feed, resp.text, key, publisher, filt)

    results = await asyncio.gather(*(one(*f) for f in FEEDS), return_exceptions=True)
    batches: list[list[RawSignal]] = []
    for (key, *_), res in zip(FEEDS, results, strict=True):
        if isinstance(res, BaseException):
            logger.info("headline feed %s failed: %s", key, res)
        else:
            batches.append(res)
    if not batches:
        raise UpstreamError("all market headline feeds failed")
    return merge_headlines(batches, now=_now())
