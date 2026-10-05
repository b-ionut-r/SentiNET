"""Market overview: risk regime from fear & greed + VIX + S&P 500 trend, and market-wide
headline tone clustered into the stories moving markets.

Regime: a weighted risk-appetite score in [-1, 1] —

    CNN Fear & Greed  (w 0.45)  (score − 50) / 50
    VIX               (w 0.25)  (20 − VIX) / 10, clipped
    S&P 500 trend     (w 0.30)  tanh(1-month return / 3%)

>= +0.3 "Risk-on", <= −0.3 "Risk-off", else "Mixed" (with the gauges that make it
so: "Mixed: Fear & Greed 31 (Fear), VIX calm") — except when sentiment and prices
disagree: fearful sentiment with firm prices is a "Wall of worry", greedy
sentiment with slipping prices is "Complacent".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from app.analytics.aggregate import summarize
from app.analytics.inputs import SourceRun
from app.analytics.narratives import build_narratives
from app.analytics.prepare import prepare
from app.analytics.util import clamp, pct
from app.schemas import FearGreed, IndexQuote, MarketOverview, SignalKind, TrendingTicker
from app.sources.base import CompanyRef, RawSignal, SourceBatch

MAX_MARKET_NARRATIVES = 8
REGIME_THRESHOLD = 0.3


@dataclass
class _HeadlineFeed:
    """Stand-in `Source` for the merged market headline feeds."""

    key: str = "market_headlines"
    label: str = "Market headlines"
    kind: SignalKind = "news"
    weight: float = 1.0
    requires_key: bool = False
    description: str = "CNBC, MarketWatch, Google News and Bing market headlines"
    docs_url: str | None = None

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return True

    async def fetch(self, company: CompanyRef) -> SourceBatch:  # pragma: no cover - never fetched
        return SourceBatch()


def build_market_overview(fear_greed: FearGreed | None, crypto_fg: FearGreed | None, indices: list[IndexQuote],
                          trending: list[TrendingTicker], headline_raws: list[RawSignal], status: dict[str, str],
                          now: datetime) -> MarketOverview:
    regime, detail = market_regime(fear_greed, indices)
    run = SourceRun(source=_HeadlineFeed(), status="ok", batch=SourceBatch(signals=list(headline_raws)))
    prepared = prepare(None, [run], now)
    stories = build_narratives(prepared.items, None, now, None, limit=MAX_MARKET_NARRATIVES)
    return MarketOverview(
        generated_at=now, regime=regime, regime_detail=detail, fear_greed=fear_greed, crypto_fear_greed=crypto_fg,
        indices=list(indices), trending=list(trending), headlines=summarize(prepared.items).stat(),
        narratives=[s.narrative for s in stories], status=dict(status),
    )


def _vix_state(vix: float) -> str:
    return "calm" if vix < 15 else "normal" if vix < 20 else "elevated" if vix < 30 else "stressed"


def _fg_trend(fg: FearGreed) -> str:
    ref = fg.month_ago if fg.month_ago is not None else fg.week_ago
    if ref is None:
        return ""
    when = "a month" if fg.month_ago is not None else "a week"
    delta = fg.score - ref
    if abs(delta) < 3:
        return f", flat vs {when} ago"
    return f", {'up' if delta > 0 else 'down'} from {ref:.0f} {when} ago"


def market_regime(fg: FearGreed | None, indices: list[IndexQuote]) -> tuple[str, str]:
    """(label, number-backed detail) of the current risk regime."""
    by_symbol = {i.symbol.upper(): i for i in indices}
    spy, vix = by_symbol.get("SPY"), by_symbol.get("^VIX")
    parts: list[tuple[float, float]] = []
    bits: list[str] = []
    s_fg = s_trend = None
    if fg is not None:
        s_fg = clamp((fg.score - 50.0) / 50.0, -1, 1)
        parts.append((s_fg, 0.45))
        bits.append(f"CNN Fear & Greed {fg.score:.0f} ({fg.rating.title()}{_fg_trend(fg)})")
    if vix is not None and vix.price:
        parts.append((clamp((20.0 - vix.price) / 10.0, -1, 1), 0.25))
        bits.append(f"VIX {vix.price:.1f} ({_vix_state(vix.price)})")
    if spy is not None and len(spy.spark) >= 5 and spy.spark[0] > 0:
        ret = (spy.spark[-1] / spy.spark[0] - 1.0) * 100.0
        off_high = (spy.spark[-1] / max(spy.spark) - 1.0) * 100.0
        s_trend = math.tanh(ret / 3.0)
        parts.append((s_trend, 0.30))
        where = "at its 1M high" if off_high > -0.25 else f"{pct(abs(off_high), sign=False)} below its 1M high"
        bits.append(f"S&P 500 {pct(ret)} over 1M, {where}")
    if not parts:
        return "Unknown", "Market gauges are unavailable right now."
    risk = sum(s * w for s, w in parts) / sum(w for _, w in parts)
    rating = fg.rating.title() if fg is not None else None
    if s_fg is not None and s_trend is not None and s_fg <= -0.1 and s_trend >= 0.2:
        label = f"Wall of worry: {rating}, prices firm"
    elif s_fg is not None and s_trend is not None and s_fg >= 0.1 and s_trend <= -0.2:
        label = f"Complacent: {rating}, prices slipping"
    elif abs(risk) >= REGIME_THRESHOLD:
        base = "Risk-on" if risk > 0 else "Risk-off"
        label = f"{base}: {rating}" if rating else base
    else:  # neither side: say which gauges point where ("Neutral: Fear" read as a contradiction)
        gauges = ([f"Fear & Greed {fg.score:.0f} ({rating})"] if fg is not None else []) + (
            [f"VIX {_vix_state(vix.price)}"] if vix is not None and vix.price else [])
        label = "Mixed: " + ", ".join(gauges) if gauges else "Mixed"
    return label, "; ".join(bits) + "."
