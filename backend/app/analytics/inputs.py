"""The contract between the orchestrator (services/analyzer.py) and the
analytics layer (analytics/build.py: `build_analysis(inputs) -> Analysis`).

The orchestrator gathers everything over the network; analytics is pure,
synchronous and deterministic given these inputs (easy to test, cheap to rerun).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from app.schemas import (
    AnalystView,
    Catalyst,
    EarningsView,
    Filing,
    InsiderView,
    Profile,
    Quote,
    Snapshot,
    SourceStatus,
    Technicals,
    ToneTrend,
)
from app.sources.base import CompanyRef, Source, SourceBatch


@dataclass
class SourceRun:
    """Outcome of running one source for this analysis."""

    source: Source
    status: SourceStatus  # ok | empty | error | disabled | unconfigured
    batch: Optional[SourceBatch] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None


@dataclass
class AnalysisInputs:
    company: CompanyRef
    now: datetime  # tz-aware UTC
    engine_name: str

    profile: Optional[Profile] = None
    quote: Optional[Quote] = None
    technicals: Optional[Technicals] = None
    analysts: Optional[AnalystView] = None
    insiders: Optional[InsiderView] = None
    earnings: Optional[EarningsView] = None
    filings: list[Filing] = field(default_factory=list)
    calendar_catalysts: list[Catalyst] = field(default_factory=list)  # upcoming ex-div etc.
    tone: Optional[ToneTrend] = None  # GDELT global news tone
    wiki_views: Optional[list[tuple[date, float]]] = None

    source_runs: list[SourceRun] = field(default_factory=list)
    previous: Optional[Snapshot] = None  # latest stored snapshot older than ~15 min
    intel_status: dict[str, str] = field(default_factory=dict)  # task key -> "ok"|"empty"|"error: …"
