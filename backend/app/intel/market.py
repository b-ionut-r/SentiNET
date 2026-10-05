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
from collections.abc import Callable
from datetime import datetime, timedelta, UTC
from typing import Any, Literal
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
# Below this many mentions a day ago, a % change is noise (META 1 -> 12 read "+1100%").
MIN_CHANGE_BASE = 5
APEWISDOM_NAME_MAX = 70  # ApeWisdom cuts names here
_CUSTODIAN = re.compile(r"\b(?:trust company|n\.a\.|ishares trust|spdr series trust)\b", re.IGNORECASE)
_CUSTODIAN_CODE = re.compile(r"^[A-Z]{2,4}\s+(?=iShares|SPDR|Vanguard|Invesco|Schwab)")


def clean_board_name(raw: Any) -> str | None:
    """ApeWisdom's registry-style names -> display names.

    "BlackRock Institutional Trust Company N.A. - BTC iShares 20+ Year Trea" (the custodian,
    then a cut-off fund name) -> "iShares 20+ Year Trea…"; ordinary names pass unchanged.
    """
    name = html.unescape(str(raw or "")).strip()
    if not name:
        return None
    truncated = len(name) >= APEWISDOM_NAME_MAX
    head, sep, tail = name.partition(" - ")
    if sep and _CUSTODIAN.search(head) and tail.strip():
        name = _CUSTODIAN_CODE.sub("", tail.strip())
    return f"{name}…" if truncated else name


def parse_apewisdom(
    payload: Any, limit: int = 25, skip: Callable[[str], bool] | None = None
) -> list[TrendingTicker]:
    """Pure: ApeWisdom leaderboard -> top Reddit tickers with 24h mention change.

    `skip` drops symbols whose Reddit count measures a word, not the stock ("DTE" is
    days-to-expiry on options subs, not DTE Energy). The change is left out when the
    day-ago base is under `MIN_CHANGE_BASE` mentions.
    """
    out: list[TrendingTicker] = []
    for item in ((payload or {}).get("results") or []) if isinstance(payload, dict) else []:
        if len(out) >= limit:
            break
        sym = str(item.get("ticker") or "").upper().replace(".", "-")
        if not sym or (skip is not None and skip(sym)):
            continue
        mentions = _int(item.get("mentions"))
        prev = _int(item.get("mentions_24h_ago"))
        change = (round((mentions - prev) / prev * 100, 1)
                  if mentions is not None and prev is not None and prev >= MIN_CHANGE_BASE else None)
        out.append(TrendingTicker(
            symbol=sym, name=clean_board_name(item.get("name")), source="reddit",
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


def _crowd_word() -> Callable[[str], bool] | None:
    """The search sources' check for ticker-words on crowd boards (shared so both agree)."""
    try:
        from app.sources.query import crowd_symbol_ambiguous
    except ImportError:  # sources package mid-edit: show the board unfiltered
        return None
    return crowd_symbol_ambiguous


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
        out += parse_apewisdom(ape, skip=_crowd_word())
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
# Market-relevance filter per feed: "title" = the headline itself must be market talk
# (general top-news feeds: politics, lifestyle); "any" = headline or summary.
FilterMode = Literal["title", "any"] | None
FEEDS: list[tuple[str, str, str | None, FilterMode]] = [
    # (key, url, publisher, filter) — most trusted first (dedupe keeps the first copy)
    ("cnbc_top", CNBC_URL.format(id=100003114), "CNBC", "title"),
    ("cnbc_finance", CNBC_URL.format(id=10000664), "CNBC", "any"),
    ("cnbc_economy", CNBC_URL.format(id=20910258), "CNBC", "any"),
    ("marketwatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "MarketWatch", "any"),
    ("google_news", GOOGLE_NEWS_URL, None, "any"),
    ("bing_news", "https://www.bing.com/news/search?q=%22stock+market%22&format=rss", None, "any"),
    ("bing_wallstreet", "https://www.bing.com/news/search?q=Wall+Street+stocks&format=rss", None, "any"),
]
MAX_HEADLINES = 120
PER_FEED_CAP = 40  # keep one aggregator from drowning out the others
PER_PUBLISHER_CAP = 4  # per outlet within search feeds (8 Motley Fool listicles in one live sample)

# General news feeds carry politics/lifestyle too; keep what can move markets.
_MARKET_TERMS = re.compile(
    r"\b(stocks?|shares?|market|markets|wall street|s&p|nasdaq|dow|fed|federal reserve|powell|rates?|"
    r"yields?|treasur(?:y|ies)|bonds?|inflation|cpi|ppi|jobs|payrolls|unemployment|gdp|recession|"
    r"economy|economic|earnings|revenue|profit|guidance|ipo|merger|acquisition|deal|tariffs?|trade war|"
    r"oil|crude (?:oil|prices?|futures)|opec|gold|bitcoin|crypto|dollar|investors?|traders?|futures|"
    r"(?<!campaign )rall(?:y|ies|ied)(?! speech)|sell-?off|"
    r"bank|banks|tech|ai|chip|chips|semiconductors?|layoffs?|bankruptcy|sec|antitrust|ceo|"
    r"index|indexes|indices|benchmark|nikkei|hang seng|ftse|dax|stoxx|vix|volatility|yen|euro|currenc(?:y|ies))\b",
    re.IGNORECASE,
)
# Search feeds (Google/Bing "stock market") also return other countries' local market
# reports ("Taiwan Stock Market Surges Over 1,000 Points", Lagos' "Stock Market Drops by
# N813bn"). On those feeds a headline about a local index or bourse is dropped unless it
# also names a US benchmark or a globally traded asset (oil, tariffs, crypto…); curated
# feeds (CNBC, MarketWatch) are never filtered this way: their editors already chose.
_FOREIGN = re.compile(
    r"\b(bangladesh\w*|dhaka|nigeria\w*|lagos|ghana\w*|kenya\w*|nairobi|pakistan\w*|karachi|psx|sri lanka\w*|"
    r"india(?:n|ns|'s)?|sensex|nifty|bse|nse|korea\w*|seoul|kospi|philippine\w*|psei|vietnam\w*|thai(?:land)?|"
    r"indonesia\w*|malaysia\w*|bursa|egypt\w*|egx|saudi|tadawul|turk(?:ey|ish|iye)|borsa|french|cac 40|"
    r"german\w*|dax|ftse|uk stocks|nikkei|japan\w*|hang seng|shanghai|shenzhen|taiwan\w*|taiex|asx|tsx|jse|"
    r"zimbabwe\w*|uganda\w*|zambia\w*)\b",
    re.IGNORECASE,
)
_LOCAL_MARKET = re.compile(r"\b(stock market|stocks?|shares|equities|index|indices|bourse|exchange|points|"
                           r"benchmark|market cap\w*|trading|investors?|sensex|nifty|kospi|nikkei|hang seng|taiex|"
                           r"psei|jse|asx|tsx|ftse|dax)\b", re.IGNORECASE)
_GLOBAL_ASSET = re.compile(r"\b(oil|crude|brent|opec\+?|lng|gold|bitcoin|crypto\w*|tariffs?|trade (?:war|deal)|"
                           r"sanctions?|dollar|fed|treasur\w+|chips?|semiconductor\w*|rare earths?)\b", re.IGNORECASE)
_US_MARKET = re.compile(r"\b(wall street|s&p|nasdaq|dow|fed|federal reserve|treasur\w+|u\.?s\.?|american|nyse)\b",
                        re.IGNORECASE)
SEARCH_FEEDS = frozenset({"google_news", "bing_news", "bing_wallstreet"})
# A search-feed headline with no US anchor needs an outlet this trusted (`app.nlp.publishers`:
# majors >= 1.0, Motley Fool/Benzinga 0.9, unknown local sites 0.8).
SEARCH_MIN_TRUST = 0.9
# Advice columns ("I'm 71 and still working. Am I doing the right thing…?"): first-person
# questions, or any first-person headline on MarketWatch (home of the Moneyist).
_ADVICE = re.compile(r"^[‘'\"“]?(?:I|I’m|I'm|I’ve|I've|My|We|We’re|We're|Our|Should I|Can I|How do I)\b")
# Evergreen listicles from search feeds ("3 Stocks to Buy and Hold Forever"): not today's market.
_EVERGREEN = re.compile(
    r"\b(?:stocks?|etfs?|shares) to (?:buy|own|hold|sell|avoid)\b|\bbuy and hold\b|\bforever\b|\bmillionaire|"
    r"\bshould you (?:buy|sell)\b|\bpassive income\b|\bbest (?:\w+ ){0,3}(?:stocks?|etfs?) (?:for|to)\b|"
    r"\bno-brainer\b|\bmonster (?:stocks?|growth)\b|\bscreaming buy\b|\bI'?d buy\b|\bI'?m (?:still )?buying\b|"
    r"\byou'?d need\b|\bmonthly (?:dividends?|income)\b|\bhistory says\b|\bforget the\b|\bsmart buys?\b|"
    r"\$\d[\d,.]* (?:million|thousand) portfolio|\bzero-fee\b",
    re.IGNORECASE,
)
# SEO ticker-page farms and outlets that cover their own (non-US) market under generic
# "stock market" headlines ("Stock market dips 0.52%" is the Nigerian Exchange).
_BLOCKED_PUBLISHERS = re.compile(
    r"stock ?traders ?daily|punchng|businessday\.ng|nairametrics|the ?daily ?star|dawn\.com|tribune\.com\.pk|"
    r"philstar|business ?mirror|inquirer\.net|bworldonline|manila ?times|thestar\.com\.my|the ?edge ?malaysia|"
    r"vnexpress|bangkok ?post|money ?control|economic ?times|live ?mint|business[- ]standard|financial ?express|"
    r"ndtv ?profit|the collegian",
    re.IGNORECASE,
)
_NON_LATIN = re.compile(r"[^\x00-\u024f\u2000-\u206f\u20ac]")  # 매일경제, 日経: non-English outlets


def _blocked(publisher: str | None) -> bool:
    return bool(publisher) and bool(_BLOCKED_PUBLISHERS.search(publisher) or _NON_LATIN.search(publisher))


def _trust(publisher: str | None) -> float:
    """Outlet trust from the shared publisher table (unknown outlets: 0.8)."""
    try:
        from app.nlp.publishers import publisher_trust
    except ImportError:  # nlp package mid-edit: treat every outlet as unknown
        return 0.8
    return publisher_trust(publisher)


def foreign_local_market(title: str) -> bool:
    """A headline about another country's local market ("Taiwan stocks surge 1,000 points")."""
    return bool(_FOREIGN.search(title) and _LOCAL_MARKET.search(title)
                and not _GLOBAL_ASSET.search(title) and not _US_MARKET.search(title))


def _keep_search_hit(title: str, publisher: str | None) -> bool:
    """Search-feed gate: not a foreign local-market report, not an evergreen listicle, and
    either US-anchored or from a trusted outlet (unknown sites mostly cover their own market)."""
    if foreign_local_market(title) or _EVERGREEN.search(title) or title.endswith(("...", "…")):
        return False  # truncated titles ("…outperforms S&P 500 in 2026 with risin...") say too little
    return bool(_US_MARKET.search(title)) or _trust(publisher) >= SEARCH_MIN_TRUST


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


def parse_feed(xml_text: str, key: str, publisher: str | None, market_filter: FilterMode | bool) -> list[RawSignal]:
    """Pure: one RSS feed -> RawSignals (publisher split off; sponsored, off-topic and junk dropped)."""
    mode: FilterMode = "any" if market_filter is True else (market_filter or None)
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
        scope = title if mode == "title" else f"{title} {body or ''}"
        advice = _ADVICE.match(title) and (title.rstrip().endswith("?") or key == "marketwatch")
        if mode and (advice or not _MARKET_TERMS.search(scope)):
            continue
        if _blocked(pub) or (key in SEARCH_FEEDS and not _keep_search_hit(title, pub)):
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
    per_outlet: dict[str, int] = {}
    merged: list[RawSignal] = []
    for batch in batches:
        newest_first = sorted(batch, key=lambda s: s.timestamp or datetime.min.replace(tzinfo=UTC), reverse=True)
        for sig in newest_first[:PER_FEED_CAP]:
            key = _dedupe_key(sig.title)
            if key in seen_titles or (sig.url and sig.url in seen_urls):
                continue
            if sig.extra.get("feed") in SEARCH_FEEDS:
                outlet = (sig.publisher or "").lower()
                if per_outlet.get(outlet, 0) >= PER_PUBLISHER_CAP:
                    continue
                per_outlet[outlet] = per_outlet.get(outlet, 0) + 1
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
