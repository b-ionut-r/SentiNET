"""Pydantic schemas shared across the API surface."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

SentimentLabel = Literal["bullish", "bearish", "neutral"]


class Signal(BaseModel):
    """A single normalized piece of content from any source."""

    source: str
    text: str
    url: Optional[str] = None
    author: Optional[str] = None
    timestamp: Optional[datetime] = None
    engagement: int = 0
    # Populated by the sentiment engine:
    score: float = 0.0  # -1 (bearish) .. +1 (bullish)
    label: SentimentLabel = "neutral"


class SourceBreakdown(BaseModel):
    source: str
    status: Literal["ok", "empty", "error", "disabled"]
    count: int = 0
    avg_score: float = 0.0
    bullish_pct: float = 0.0
    bearish_pct: float = 0.0
    neutral_pct: float = 0.0
    weight: float = 0.0
    error: Optional[str] = None


class TimelinePoint(BaseModel):
    bucket: datetime
    avg_score: float
    count: int


class TrendingKeyword(BaseModel):
    keyword: str
    count: int


class AnalyzeResponse(BaseModel):
    ticker: str
    company: Optional[str] = None
    generated_at: datetime
    cached: bool = False

    overall_score: float = 0.0  # weighted, -1 .. +1
    overall_label: SentimentLabel = "neutral"
    total_signals: int = 0
    active_sources: int = 0

    bullish_pct: float = 0.0
    bearish_pct: float = 0.0
    neutral_pct: float = 0.0

    sources: list[SourceBreakdown] = []
    timeline: list[TimelinePoint] = []
    trending: list[TrendingKeyword] = []
    signals: list[Signal] = []  # most recent / influential, for the news feed


class PricePoint(BaseModel):
    t: datetime
    close: float


class PriceResponse(BaseModel):
    ticker: str
    company: Optional[str] = None
    currency: Optional[str] = None
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    range: str
    points: list[PricePoint] = []
    available: bool = True
    error: Optional[str] = None


class SourceInfo(BaseModel):
    name: str
    label: str
    kind: Literal["social", "news", "filings"]
    enabled: bool


class HealthResponse(BaseModel):
    status: str
    sentiment_engine: str
    sources: list[SourceInfo]
