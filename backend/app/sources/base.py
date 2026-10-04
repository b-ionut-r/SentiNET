"""The contract every text/crowd source conforms to.

A source fetches content for one company and returns a `SourceBatch`:

* `signals` — raw text items (headlines, posts). The pipeline cleans, filters
  for relevance, de-duplicates, scores and aggregates them; sources never
  compute sentiment themselves.
* `metrics` — structured, *non-text* data the source provides (mention counts,
  author-tagged bull/bear tallies, a provider's own sentiment score…). These
  feed `CrowdView`/`AttentionView` directly. Sources must NOT synthesize fake
  text to smuggle metrics into the text pipeline.

Metric keys (documented contract, consumed by the analytics layer):
    stocktwits:  stocktwits_bullish, stocktwits_bearish, stocktwits_messages,
                 stocktwits_watchers
    apewisdom:   reddit_mentions, reddit_mentions_prev, reddit_rank,
                 reddit_rank_prev, reddit_upvotes
    tradestie:   wsb_sentiment (float -1..1), wsb_label ("bullish"/"bearish"),
                 wsb_comments
    bluesky:     bluesky_posts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from app.schemas import SignalKind


@dataclass
class CompanyRef:
    """Everything a source needs to know about the target."""

    ticker: str  # canonical upper-case symbol, e.g. "AAPL", "BRK-B", "BTC-USD"
    name: str  # official/long name, e.g. "Apple Inc."
    short_name: str  # cleaned brand name for search, e.g. "Apple"
    aliases: list[str] = field(default_factory=list)  # extra names that identify it
    quote_type: str = "EQUITY"  # EQUITY | ETF | CRYPTOCURRENCY | INDEX | MUTUALFUND
    cik: Optional[str] = None  # zero-padded 10-digit SEC CIK, if US-listed
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = None

    @property
    def is_crypto(self) -> bool:
        return self.quote_type == "CRYPTOCURRENCY"

    @property
    def base_symbol(self) -> str:
        """Symbol without exchange suffix / quote currency: BTC-USD -> BTC, SHOP.TO -> SHOP."""
        sym = self.ticker
        if self.is_crypto and "-" in sym:
            return sym.split("-")[0]
        return sym.split(".")[0]

    @property
    def cashtag(self) -> str:
        return f"${self.base_symbol}"


@dataclass
class RawSignal:
    title: str  # headline / post text
    body: Optional[str] = None  # summary / snippet / selftext
    url: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None  # outlet, e.g. "Reuters"
    timestamp: Optional[datetime] = None  # tz-aware UTC
    engagement: int = 0
    # Author-declared stance (StockTwits "Bullish"/"Bearish" tag).
    user_label: Optional[Literal["bullish", "bearish"]] = None
    # True when the source guarantees the item is about the ticker (symbol
    # streams, ticker-keyed news APIs). Broad keyword searches leave it False
    # so the relevance filter decides.
    ticker_specific: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceBatch:
    signals: list[RawSignal] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # "did we get anything?"
        return bool(self.signals or self.metrics)


@runtime_checkable
class Source(Protocol):
    key: str  # stable id, snake_case
    label: str  # display name
    kind: SignalKind  # "news" | "social" | "analysis"
    weight: float  # relative trust in aggregation (news ~1.0-1.3, social ~0.5-0.9)
    requires_key: bool
    description: str
    docs_url: Optional[str]

    def configured(self) -> bool:
        """False when a required key/credential is missing (source is skipped)."""
        ...

    def supports(self, company: CompanyRef) -> bool:
        """False when the source can't serve this asset type (e.g. crypto)."""
        ...

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        ...
