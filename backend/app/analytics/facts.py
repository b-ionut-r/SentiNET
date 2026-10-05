"""Everything derived from one `AnalysisInputs`, shared by the synthesis modules.

`build.py` fills a `Facts` step by step; insights, catalysts, the brief and the
verdict read from it, so every sentence they write is backed by the same
numbers the API returns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from app.analytics.aggregate import Summary
from app.analytics.composite import ComponentKey, Composite, Part
from app.analytics.crowd import Tally
from app.analytics.deals import Deal, DealInPlay
from app.analytics.inputs import AnalysisInputs
from app.analytics.narratives import Story
from app.analytics.prepare import Prepared
from app.analytics.util import INSIDER_CURRENCY, finite
from app.schemas import AttentionView, Catalyst, CrowdView, Profile, ThemeStat

SOFT_PENDING = "still loading"  # a feed that timed out but keeps loading in the background: a temporary gap
_ISO_CODE = re.compile(r"^[A-Z]{3}$")

# Intel task keys (AnalysisInputs.intel_status) as said to users.
FEED_NAMES = {"analysts": "analyst ratings", "insiders": "insider trades", "earnings": "earnings",
              "technicals": "price history", "quote": "the quote", "filings": "SEC filings", "tone": "GDELT tone",
              "wiki": "Wikipedia pageviews", "profile": "the profile", "calendar": "the dividend calendar"}
# The score component each structured feed supplies: without the feed it is n/a.
COMPONENT_FEEDS: dict[ComponentKey, str] = {"analysts": "analysts", "insiders": "insiders",
                                            "technicals": "technicals", "momentum": "tone"}
FeedState = Literal["pending", "failed"]


def feed_state(status: str | None) -> FeedState | None:
    """'pending' (timed out, still loading), 'failed' (errored) or None (answered, or not run)."""
    if not status or not status.startswith("error"):
        return None
    return "pending" if SOFT_PENDING in status else "failed"


def unloaded(part: Part, intel_status: dict[str, str]) -> Part:
    """An unavailable component whose feed did not load says so ('analyst ratings not loaded in time
    this run'), never what an empty answer would mean ('no analyst coverage' for NVDA's 61 analysts);
    `facts["unloaded"]` keeps the feed's state for the verdict and the rail."""
    feed = COMPONENT_FEEDS.get(part.key)
    if feed is None or part.available:
        return part
    state = feed_state(intel_status.get(feed))
    if state is not None:
        how = "not loaded in time" if state == "pending" else "could not be loaded"
        part.detail = f"{FEED_NAMES[feed]} {how} this run"
        part.facts["unloaded"] = state
    return part


def reporting_currency(profile: Profile | None, quote_currency: str) -> str | None:
    """See `Facts.reporting_currency`."""
    if quote_currency == "USD":
        return "USD"  # US lines (ADRs too: TSM's per-ADR estimates are USD although TSMC reports in TWD)
    code = (profile.financial_currency or "").strip().upper() if profile is not None else ""
    return code if _ISO_CODE.match(code) else None


def usd_rate(inputs: AnalysisInputs) -> float | None:
    """USD per unit of the quote's major currency (GBP for a GBp listing), when the caller provides one.

    Read defensively: `fx_usd` is a requested (optional) addition to `AnalysisInputs`;
    until it exists, USD amounts are compared only with USD market caps."""
    rate = finite(getattr(inputs, "fx_usd", None), 0.0)
    return rate if rate > 0 else None


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
    failed: list[str] = field(default_factory=list)  # parts that raised and were left out (data quality)
    deal: Deal | None = None  # a pending acquisition of the company (see deals.py)
    # Market cap in USD, to size USD amounts (insider trades, quoted deal values); None when it cannot
    # be had without guessing an FX rate (see build.market_cap_usd).
    market_cap_usd: float | None = None
    deal_in_play: DealInPlay | None = None  # fresh, corroborated M&A coverage involving the company (deals.py)

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
    def currency(self) -> str:
        """Currency of the quote, and of analyst targets and dividends quoted against it (USD when unknown)."""
        q = self.inputs.quote
        return (q.currency if q is not None and q.currency else None) or "USD"

    @property
    def reporting_currency(self) -> str | None:
        """Currency of EPS/revenue estimates: USD for a USD listing (ADRs' estimates are per ADR, in
        USD); otherwise the profile's reporting currency when the provider gives one (Shopify on the
        TSX reports in USD, Vodafone in EUR); else None (not known — shown without a symbol)."""
        return reporting_currency(self.inputs.profile, self.currency)

    @property
    def action_currency(self) -> str | None:
        """Currency of broker *action* targets: USD for a USD listing, else None — the action feed
        of a cross-listed company quotes its US line (SHOP.TO's Wedbush $176 vs a C$216 price), so
        only the % change is shown (consensus targets are in the quote currency, `currency`)."""
        return "USD" if self.currency == "USD" else None

    @property
    def insider_currency(self) -> str:
        """Insider trade values are USD for every listing (see util.INSIDER_CURRENCY)."""
        return INSIDER_CURRENCY

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
                if feed_state(v) == "failed"}

    def intel_pending(self) -> list[str]:
        """Intel tasks that timed out and are still loading in the background (in on a later load)."""
        return [k for k, v in self.inputs.intel_status.items() if feed_state(v) == "pending"]

    def unloaded_parts(self) -> list[Part]:
        """Score components left n/a because their feed did not load this run (see `unloaded`)."""
        return [p for p in self.composite.parts.values() if p.facts.get("unloaded")]
