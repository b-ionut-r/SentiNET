"""The Source contract every adapter conforms to.

A source fetches raw content for a ticker and returns RawSignal items. The
pipeline normalizes, scores and aggregates them — sources never know about
sentiment. Each source also declares a `weight` (trust) and `kind` so the
aggregator can value news above anonymous social volume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Protocol

SourceKind = Literal["social", "news", "filings"]


@dataclass
class RawSignal:
    text: str
    url: Optional[str] = None
    author: Optional[str] = None
    timestamp: Optional[datetime] = None
    engagement: int = 0
    # Some sources (e.g. StockTwits, Tradestie) ship their own bull/bear label.
    # When present it's a hint in [-1, 1]; the engine score still takes priority.
    prelabeled_score: Optional[float] = None
    extra: dict = field(default_factory=dict)


class Source(Protocol):
    name: str
    label: str
    kind: SourceKind
    weight: float  # relative trust used in weighted aggregation

    async def fetch(self, ticker: str, company: Optional[str]) -> list[RawSignal]:
        ...
