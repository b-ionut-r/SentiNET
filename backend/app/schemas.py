"""SentiNET API contract.

Every response the API returns is defined here. The frontend mirrors these
models in `frontend/src/api/types.ts` — keep the two in sync.

Conventions
-----------
* Sentiment scores are floats in [-1, 1] (negative = bearish).
* "SentiNET scores" / component scores are ints/floats in [0, 100] (50 = neutral).
* Timestamps are timezone-aware UTC datetimes (serialized ISO-8601).
* Optional (`None`) means "not available" — never a fabricated placeholder.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Shared literals
# --------------------------------------------------------------------------- #
SentimentLabel = Literal["bullish", "bearish", "neutral"]
Polarity = Literal["bull", "bear", "neutral"]
SignalKind = Literal["news", "social", "analysis", "filing"]
SourceStatus = Literal["ok", "empty", "error", "disabled", "unconfigured"]
Confidence = Literal["low", "medium", "high"]


# --------------------------------------------------------------------------- #
# Signals (individual pieces of text content)
# --------------------------------------------------------------------------- #
class Driver(BaseModel):
    """A term/phrase that moved an item's sentiment (explainability)."""

    term: str
    impact: float  # signed contribution, roughly [-1, 1]


class Signal(BaseModel):
    id: str  # stable hash of source + url/title
    source: str  # source key, e.g. "google_news"
    source_label: str
    kind: SignalKind
    title: str  # headline or post text (cleaned)
    body: Optional[str] = None  # snippet / summary, if any
    url: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None  # outlet / domain, e.g. "Reuters"
    timestamp: Optional[datetime] = None
    engagement: int = 0  # likes+comments, points, followers… (source-specific)

    score: float = 0.0  # -1..1
    label: SentimentLabel = "neutral"
    confidence: float = 0.0  # 0..1
    relevance: float = 1.0  # 0..1, how clearly the item is about this ticker
    weight: float = 0.0  # final aggregation weight (trust × recency × engagement × relevance)
    themes: list[str] = []  # theme keys (see ThemeStat)
    events: list[str] = []  # detected event keys, e.g. "pt_raise", "earnings_beat"
    drivers: list[Driver] = []
    user_label: Optional[SentimentLabel] = None  # author-declared (e.g. StockTwits tag)
    narrative_id: Optional[str] = None
    duplicates: int = 0  # syndicated near-copies collapsed into this item


class SentimentStat(BaseModel):
    """A reusable aggregate over a set of signals."""

    score: float = 0.0  # weighted mean, -1..1
    label: SentimentLabel = "neutral"
    n: int = 0
    bullish: int = 0
    bearish: int = 0
    neutral: int = 0
    confidence: float = 0.0  # 0..1


class SourceReport(BaseModel):
    key: str
    label: str
    kind: SignalKind
    status: SourceStatus
    fetched: int = 0  # raw items returned
    kept: int = 0  # items kept after relevance + dedupe
    score: Optional[float] = None  # weighted mean of kept items
    bullish_pct: float = 0.0
    bearish_pct: float = 0.0
    weight: float = 0.0  # source trust
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    requires_key: bool = False
    has_metrics: bool = False  # contributed structured metrics (crowd/buzz)


# --------------------------------------------------------------------------- #
# Synthesized intelligence
# --------------------------------------------------------------------------- #
class Narrative(BaseModel):
    """A story cluster: many headlines/posts about the same development."""

    id: str
    headline: str  # most representative title
    count: int  # items in the cluster (incl. syndicated copies)
    publishers: list[str] = []  # distinct outlets
    score: float = 0.0  # weighted mean sentiment of the cluster
    label: SentimentLabel = "neutral"
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    velocity_24h: int = 0  # items in the last 24h
    themes: list[str] = []
    events: list[str] = []
    impact: float = 0.0  # 0..1 ranking metric: coverage × |sentiment| × recency
    url: Optional[str] = None  # link of the representative item
    signal_ids: list[str] = []
    is_new: bool = False  # not present in the previous snapshot


class ThemeStat(BaseModel):
    theme: str  # key, e.g. "earnings"
    label: str  # display label, e.g. "Earnings"
    count: int
    score: float  # weighted mean sentiment of items tagged with the theme
    share: float  # 0..1 share of all signals


class Keyword(BaseModel):
    term: str
    count: int
    score: float  # mean sentiment of items containing it


class TimelineBucket(BaseModel):
    t: datetime  # bucket start (UTC)
    score: float
    count: int
    news: int = 0
    social: int = 0
    bullish: int = 0
    bearish: int = 0


class Reason(BaseModel):
    """One line of 'why' behind the verdict."""

    text: str
    polarity: Polarity
    weight: float  # 0..1 importance
    ref: Optional[str] = None  # narrative id / signal id / component key


class Component(BaseModel):
    """One input to the SentiNET composite score."""

    key: Literal["news", "social", "analysts", "insiders", "momentum", "technicals"]
    label: str
    score: Optional[float] = None  # 0..100, None if unavailable
    weight: float  # nominal weight (renormalized over available components)
    available: bool
    detail: str  # short human explanation, e.g. "+0.31 across 54 articles"
    confidence: float = 0.0  # 0..1


class Verdict(BaseModel):
    score: int  # SentiNET composite 0..100
    label: str  # "Strongly Bullish" | "Bullish" | "Leaning Bullish" | "Neutral" | …
    stance: SentimentLabel
    confidence: Confidence
    confidence_value: float  # 0..1
    headline: str  # one-line synthesized takeaway
    reasons: list[Reason] = []  # top drivers of the verdict, strongest first
    components: list[Component] = []


class Insight(BaseModel):
    """A noteworthy, actionable observation (divergence, spike, crowding…)."""

    kind: Literal[
        "divergence", "attention", "reversal", "crowding", "catalyst",
        "smart_money", "risk", "momentum", "quality",
    ]
    severity: Literal["info", "watch", "alert"]
    polarity: Polarity
    title: str
    detail: str


class Brief(BaseModel):
    """Deterministic, evidence-backed analyst brief."""

    summary: str  # 2-4 sentence paragraph
    bull_points: list[str] = []
    bear_points: list[str] = []
    watch: list[str] = []  # what to watch next (catalysts, risks)


class DeltaView(BaseModel):
    """What changed since the previous stored snapshot of this ticker."""

    previous_at: Optional[datetime] = None
    score_change: Optional[float] = None  # change in overall sentiment (-2..2)
    sentinel_change: Optional[int] = None  # change in SentiNET score
    price_change_pct: Optional[float] = None
    new_narratives: list[str] = []  # headlines of narratives not seen before
    note: Optional[str] = None  # human summary, e.g. "Sentiment improved sharply (+12)"


# --------------------------------------------------------------------------- #
# Market data & structured intel
# --------------------------------------------------------------------------- #
class Profile(BaseModel):
    symbol: str
    name: str
    short_name: Optional[str] = None  # cleaned brand name, e.g. "Apple"
    quote_type: Optional[str] = None  # EQUITY | ETF | CRYPTOCURRENCY | INDEX …
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    website: Optional[str] = None
    summary: Optional[str] = None
    employees: Optional[int] = None
    logo_url: Optional[str] = None
    cik: Optional[str] = None


class Quote(BaseModel):
    price: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    currency: Optional[str] = None
    previous_close: Optional[float] = None
    open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    year_high: Optional[float] = None
    year_low: Optional[float] = None
    volume: Optional[float] = None
    avg_volume: Optional[float] = None  # 3-month average
    market_cap: Optional[float] = None
    fifty_day_avg: Optional[float] = None
    two_hundred_day_avg: Optional[float] = None
    as_of: Optional[datetime] = None


class Technicals(BaseModel):
    """Price-implied sentiment, computed from ~1y of daily closes."""

    return_1d: Optional[float] = None  # percent
    return_5d: Optional[float] = None
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    return_ytd: Optional[float] = None
    vs_50dma_pct: Optional[float] = None
    vs_200dma_pct: Optional[float] = None
    rsi_14: Optional[float] = None
    volatility_30d: Optional[float] = None  # annualized, percent
    pct_from_52w_high: Optional[float] = None
    volume_ratio: Optional[float] = None  # last volume / 3m avg volume
    trend: Optional[Literal["uptrend", "downtrend", "sideways"]] = None


class AnalystAction(BaseModel):
    date: datetime
    firm: str
    action: Literal["up", "down", "init", "main", "reit", "other"]
    from_grade: Optional[str] = None
    to_grade: Optional[str] = None
    price_target: Optional[float] = None
    prior_target: Optional[float] = None


class RatingCounts(BaseModel):
    period: str  # "0m", "-1m", …
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0


class AnalystView(BaseModel):
    consensus: Optional[str] = None  # "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
    mean_rating: Optional[float] = None  # 1 (strong buy) .. 5 (strong sell)
    counts: Optional[RatingCounts] = None  # current month
    total: int = 0
    target_mean: Optional[float] = None
    target_median: Optional[float] = None
    target_high: Optional[float] = None
    target_low: Optional[float] = None
    upside_pct: Optional[float] = None  # target_mean vs current price
    actions: list[AnalystAction] = []  # most recent first (≤ 25)
    upgrades_90d: int = 0
    downgrades_90d: int = 0
    pt_raises_30d: int = 0
    pt_cuts_30d: int = 0
    trend: list[RatingCounts] = []  # last ~4 months, current first


class InsiderTxn(BaseModel):
    date: date
    insider: str
    position: Optional[str] = None
    kind: Literal["buy", "sell", "award", "exercise", "gift", "other"]
    shares: Optional[float] = None
    value: Optional[float] = None
    text: Optional[str] = None


class InsiderView(BaseModel):
    window_days: int = 180
    buys: int = 0  # open-market purchases
    sells: int = 0  # open-market sales
    buy_value: float = 0.0
    sell_value: float = 0.0
    net_value: float = 0.0
    ratio: Optional[float] = None  # (buy - sell) / (buy + sell) by value, -1..1
    transactions: list[InsiderTxn] = []  # most recent first (≤ 25)


class EarningsEvent(BaseModel):
    date: date
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    surprise_pct: Optional[float] = None


class EarningsView(BaseModel):
    next_date: Optional[date] = None
    days_until: Optional[int] = None
    eps_estimate: Optional[float] = None
    eps_low: Optional[float] = None
    eps_high: Optional[float] = None
    revenue_estimate: Optional[float] = None
    history: list[EarningsEvent] = []  # most recent first
    beat_rate: Optional[float] = None  # 0..1 share of reported quarters that beat


class Filing(BaseModel):
    form: str  # "8-K", "10-Q", "4", "SC 13G" …
    date: date
    title: str  # human-readable, 8-K items decoded
    items: list[str] = []  # 8-K item codes, e.g. ["2.02", "9.01"]
    url: Optional[str] = None
    importance: Literal["high", "medium", "low"] = "low"
    polarity: Polarity = "neutral"


class CrowdView(BaseModel):
    """Retail positioning & chatter (structured, not text-scored)."""

    stocktwits_bullish: Optional[int] = None  # author-tagged messages in sample
    stocktwits_bearish: Optional[int] = None
    stocktwits_bull_ratio: Optional[float] = None  # bullish / tagged, 0..1
    stocktwits_bull_authors: Optional[int] = None  # one vote per account (latest stance)
    stocktwits_bear_authors: Optional[int] = None
    stocktwits_messages: Optional[int] = None
    stocktwits_watchers: Optional[int] = None
    reddit_mentions: Optional[int] = None  # ApeWisdom, last 24h
    reddit_mentions_prev: Optional[int] = None  # 24h before that
    reddit_rank: Optional[int] = None
    reddit_rank_prev: Optional[int] = None
    reddit_upvotes: Optional[int] = None
    wsb_sentiment: Optional[float] = None  # Tradestie score
    wsb_label: Optional[SentimentLabel] = None
    wsb_comments: Optional[int] = None
    bluesky_posts: Optional[int] = None


class AttentionView(BaseModel):
    heat: int  # 0..100 composite attention
    label: Literal["Quiet", "Normal", "Elevated", "Spiking"]
    news_volume_z: Optional[float] = None  # latest GDELT volume vs 30d baseline
    wiki_views_7d: Optional[float] = None  # avg daily Wikipedia pageviews (7d)
    wiki_views_z: Optional[float] = None
    reddit_change_pct: Optional[float] = None
    signals_24h: int = 0  # signals timestamped in the last 24h


class Catalyst(BaseModel):
    """A dated event: upcoming (earnings, ex-div) or recent (rating change, 8-K…)."""

    date: datetime
    kind: Literal["earnings", "dividend", "analyst", "filing", "insider", "news"]
    title: str
    detail: Optional[str] = None
    polarity: Polarity = "neutral"
    upcoming: bool = False
    url: Optional[str] = None


class TonePoint(BaseModel):
    date: date
    tone: Optional[float] = None  # GDELT average tone (≈ -10..10, typically -3..3)
    volume: Optional[float] = None  # GDELT article volume (raw count) for the day


class ToneTrend(BaseModel):
    """Global news tone from GDELT (thousands of outlets), last ~90 days."""

    query: str
    tone_7d: Optional[float] = None
    tone_30d: Optional[float] = None
    tone_90d: Optional[float] = None
    change_7d_vs_30d: Optional[float] = None
    percentile_7d: Optional[float] = None  # where the 7d tone sits in its 90d range, 0..1
    series: list[TonePoint] = []


# --------------------------------------------------------------------------- #
# The full analysis
# --------------------------------------------------------------------------- #
class Analysis(BaseModel):
    ticker: str
    generated_at: datetime
    cached: bool = False
    elapsed_ms: int = 0
    engine: str = "sentinel"

    profile: Optional[Profile] = None
    quote: Optional[Quote] = None
    technicals: Optional[Technicals] = None

    verdict: Verdict
    brief: Brief
    delta: DeltaView = Field(default_factory=DeltaView)
    insights: list[Insight] = []

    sentiment: SentimentStat = Field(default_factory=SentimentStat)  # all signals
    news: SentimentStat = Field(default_factory=SentimentStat)
    social: SentimentStat = Field(default_factory=SentimentStat)

    narratives: list[Narrative] = []  # ranked by impact
    themes: list[ThemeStat] = []
    keywords: list[Keyword] = []
    timeline: list[TimelineBucket] = []
    tone: Optional[ToneTrend] = None

    analysts: Optional[AnalystView] = None
    insiders: Optional[InsiderView] = None
    earnings: Optional[EarningsView] = None
    filings: list[Filing] = []
    crowd: Optional[CrowdView] = None
    attention: Optional[AttentionView] = None
    catalysts: list[Catalyst] = []

    sources: list[SourceReport] = []
    signals: list[Signal] = []  # ranked by weight, ≤ 200


class ProgressEvent(BaseModel):
    """Streamed during `/api/analyze/{ticker}/stream` as `event: progress`."""

    stage: Literal["resolve", "source", "intel", "nlp", "analytics", "done"]
    key: str
    label: str
    status: Literal["running", "ok", "empty", "error", "skipped"]
    count: Optional[int] = None
    ms: Optional[int] = None
    detail: Optional[str] = None


# --------------------------------------------------------------------------- #
# Price & history
# --------------------------------------------------------------------------- #
class Candle(BaseModel):
    t: datetime
    o: float
    h: float
    l: float  # noqa: E741
    c: float
    v: Optional[float] = None


class PriceResponse(BaseModel):
    ticker: str
    range: str  # 1D | 5D | 1M | 3M | 6M | 1Y | 5Y
    interval: str
    currency: Optional[str] = None
    candles: list[Candle] = []
    available: bool = True
    error: Optional[str] = None


class LagStat(BaseModel):
    lag_days: int  # >0: tone leads returns by N days; 0: same day; <0: returns lead tone
    r: float  # Pearson correlation
    n: int
    p_value: float


class HistoryPoint(BaseModel):
    date: date
    tone: Optional[float] = None
    volume: Optional[float] = None
    close: Optional[float] = None
    ret_pct: Optional[float] = None  # daily return
    snapshot_score: Optional[int] = None  # stored SentiNET score that day
    wiki_views: Optional[float] = None


class HistoryResponse(BaseModel):
    ticker: str
    days: int
    points: list[HistoryPoint] = []
    lags: list[LagStat] = []  # lag -3..+3
    best_lag: Optional[LagStat] = None
    interpretation: str = ""  # plain-English read of tone ↔ price relationship
    status: dict[str, str] = {}  # per series: "ok" | "error: …" | "empty"


class Snapshot(BaseModel):
    ticker: str
    at: datetime
    sentinel_score: int
    score: float
    label: SentimentLabel
    n_signals: int
    price: Optional[float] = None
    news_score: Optional[float] = None
    social_score: Optional[float] = None
    narratives: list[str] = []  # narrative headlines at the time


# --------------------------------------------------------------------------- #
# Market overview
# --------------------------------------------------------------------------- #
class GaugeComponent(BaseModel):
    key: str
    label: str
    score: Optional[float] = None  # 0..100
    rating: Optional[str] = None


class HistoryValue(BaseModel):
    t: datetime
    v: float


class FearGreed(BaseModel):
    score: float
    rating: str
    previous_close: Optional[float] = None
    week_ago: Optional[float] = None
    month_ago: Optional[float] = None
    year_ago: Optional[float] = None
    components: list[GaugeComponent] = []
    history: list[HistoryValue] = []


class IndexQuote(BaseModel):
    symbol: str
    name: str
    price: Optional[float] = None
    change_pct: Optional[float] = None
    spark: list[float] = []  # ~1 month of daily closes


class TrendingTicker(BaseModel):
    symbol: str
    name: Optional[str] = None
    source: Literal["reddit", "stocktwits", "yahoo"]
    rank: Optional[int] = None
    rank_prev: Optional[int] = None
    mentions: Optional[int] = None
    mentions_prev: Optional[int] = None
    change_pct: Optional[float] = None  # mention growth vs 24h ago
    sentiment: Optional[float] = None  # if the source provides one


class MarketOverview(BaseModel):
    generated_at: datetime
    regime: str  # e.g. "Risk-off: Fear"
    regime_detail: str
    fear_greed: Optional[FearGreed] = None
    crypto_fear_greed: Optional[FearGreed] = None
    indices: list[IndexQuote] = []
    trending: list[TrendingTicker] = []
    headlines: SentimentStat = Field(default_factory=SentimentStat)
    narratives: list[Narrative] = []  # market-wide story clusters
    status: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# Search, lab, watchlist, alerts, meta
# --------------------------------------------------------------------------- #
class SymbolMatch(BaseModel):
    symbol: str
    name: str
    exchange: Optional[str] = None
    type: Optional[str] = None  # EQUITY | ETF | CRYPTOCURRENCY …
    logo_url: Optional[str] = None


class ScoreRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)
    ticker: Optional[str] = None  # optional: adds relevance scoring


class ScoredText(BaseModel):
    text: str
    score: float
    label: SentimentLabel
    confidence: float
    drivers: list[Driver] = []
    themes: list[str] = []
    events: list[str] = []
    relevance: Optional[float] = None


class ScoreResponse(BaseModel):
    engine: str
    results: list[ScoredText]
    summary: SentimentStat
    themes: list[ThemeStat] = []


class WatchItem(BaseModel):
    ticker: str
    added_at: datetime
    name: Optional[str] = None
    last: Optional[Snapshot] = None
    previous: Optional[Snapshot] = None
    spark: list[int] = []  # recent SentiNET scores, oldest first


class WatchAdd(BaseModel):
    ticker: str


AlertKind = Literal[
    "score_above", "score_below", "score_change", "attention_spike",
    "new_narrative", "analyst_action",
]


class AlertRuleIn(BaseModel):
    ticker: str
    kind: AlertKind
    threshold: Optional[float] = None


class AlertRule(AlertRuleIn):
    id: int
    enabled: bool = True
    created_at: datetime
    last_triggered_at: Optional[datetime] = None


class AlertEvent(BaseModel):
    id: int
    rule_id: Optional[int] = None
    ticker: str
    at: datetime
    title: str
    detail: str
    delivered: bool = False


class SourceInfo(BaseModel):
    key: str
    label: str
    kind: SignalKind
    enabled: bool
    requires_key: bool
    configured: bool
    description: str
    docs_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    engine: str
    sources: list[SourceInfo]
    features: dict[str, bool] = {}
