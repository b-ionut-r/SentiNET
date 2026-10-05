"""Everything derived from one `AnalysisInputs`, shared by the synthesis modules.

`build.py` fills a `Facts` step by step; insights, catalysts, the brief and the
verdict read from it, so every sentence they write is backed by the same
numbers the API returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.analytics.aggregate import Summary
from app.analytics.composite import Composite
from app.analytics.crowd import Tally
from app.analytics.inputs import AnalysisInputs
from app.analytics.narratives import Story
from app.analytics.prepare import Prepared
from app.schemas import AttentionView, Catalyst, CrowdView, ThemeStat

SOFT_PENDING = "still loading"  # GDELT is slow: a temporary gap, not an outage


@dataclass
class Facts:
    inputs: AnalysisInputs
    prepared: Prepared
    overall: Summary
    news: Summary
    social: Summary
    news_recent: Summary  # news/analysis items <= 48 h old
    news_older: Summary  # news/analysis items 48 h .. 7 d old
    stories: list[Story]
    themes: list[ThemeStat]
    metrics: dict[str, Any]
    crowd: CrowdView | None
    stocktwits: Tally | None  # tag tallies behind crowd.stocktwits_* (per account when available)
    attention: AttentionView | None
    composite: Composite
    catalysts: list[Catalyst] = field(default_factory=list)

    @property
    def now(self) -> datetime:
        return self.inputs.now

    @property
    def name(self) -> str:
        c = self.inputs.company
        return c.short_name or c.name or c.ticker

    @property
    def is_equity(self) -> bool:
        return self.inputs.company.quote_type == "EQUITY"

    @property
    def market_cap(self) -> float | None:
        q = self.inputs.quote
        return q.market_cap if q is not None else None

    @property
    def sources_down(self) -> list[str]:
        """Labels of text/crowd sources that failed this run."""
        return [r.source.label for r in self.inputs.source_runs if r.status == "error"]

    @property
    def sources_ok(self) -> int:
        return sum(1 for r in self.inputs.source_runs if r.status == "ok")

    def intel_failed(self) -> dict[str, str]:
        """Intel tasks that hard-failed (soft "still loading" gaps excluded)."""
        return {k: v[len("error:"):].strip() for k, v in self.inputs.intel_status.items()
                if v.startswith("error") and SOFT_PENDING not in v}

    def intel_pending(self) -> list[str]:
        """Intel tasks that are only temporarily missing (ready on next refresh)."""
        return [k for k, v in self.inputs.intel_status.items() if v.startswith("error") and SOFT_PENDING in v]
