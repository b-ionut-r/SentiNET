"""Interfaces between the NLP layer and the rest of the app.

The pipeline calls (all synchronous, CPU-only, deterministic):

    engine = get_engine()                                  # app.nlp.engine
    results = engine.score(texts, kinds)                   # -> list[TextAnalysis]
    rel = relevance(text, company)                         # app.nlp.relevance -> float 0..1
    clusters = cluster_narratives(items)                   # app.nlp.narratives -> list[Cluster]
    groups = find_duplicates(titles)                       # app.nlp.narratives -> list[list[int]]
    kws = extract_keywords(texts, company, scores)         # app.nlp.keywords
    THEMES                                                 # app.nlp.themes: key -> label
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Protocol

Label = Literal["bullish", "bearish", "neutral"]


@dataclass
class DetectedEvent:
    """A market-moving event recognized in text (headline-level)."""

    key: str  # e.g. "pt_raise", "analyst_downgrade", "earnings_beat", "lawsuit"
    polarity: Literal["bull", "bear", "neutral"]
    firm: Optional[str] = None  # e.g. "Morgan Stanley" for analyst actions
    value: Optional[float] = None  # e.g. new price target
    span: Optional[str] = None  # matched text


@dataclass
class TextAnalysis:
    score: float  # -1..1
    label: Label
    confidence: float  # 0..1
    drivers: list[tuple[str, float]] = field(default_factory=list)  # (term, signed impact)
    themes: list[str] = field(default_factory=list)  # theme keys
    events: list[DetectedEvent] = field(default_factory=list)


class SentimentEngine(Protocol):
    name: str

    def score(self, texts: list[str], kinds: Optional[list[str]] = None) -> list[TextAnalysis]:
        """Score a batch. `kinds[i]` is "news" | "social" | … and lets the engine
        adapt to register (headline vs. slang). Must return len(texts) results."""
        ...


@dataclass
class ClusterItem:
    id: str
    title: str
    timestamp: Optional[datetime] = None
    score: float = 0.0
    weight: float = 1.0
    publisher: Optional[str] = None


@dataclass
class Cluster:
    item_ids: list[str]  # members, representative first
    representative_id: str
    title: str  # representative headline
    terms: list[str] = field(default_factory=list)  # top distinguishing terms
