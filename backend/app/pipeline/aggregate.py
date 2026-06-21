"""Aggregation: where product credibility lives.

We turn scored signals into the headline number and breakdowns. The overall
score is a *weighted* mean so that trusted news outweighs anonymous social
volume, and recent / higher-engagement items count for more. We also build a
per-source breakdown and an hourly sentiment timeline.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

from app.schemas import (
    AnalyzeResponse,
    Signal,
    SourceBreakdown,
    TimelinePoint,
)
from app.sentiment.base import label_for
from app.sources.registry import SourceResult
from app.pipeline.trending import compute_trending

# Half-life (hours) for recency weighting of the overall score.
RECENCY_HALF_LIFE_H = 48.0


def _recency_weight(ts: datetime | None, now: datetime) -> float:
    if ts is None:
        return 0.6  # unknown age -> mild discount
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
    return 0.3 + 0.7 * math.pow(0.5, age_h / RECENCY_HALF_LIFE_H)


def _engagement_weight(engagement: int) -> float:
    # Diminishing returns; a viral post counts more but never dominates.
    return 1.0 + min(1.0, math.log1p(max(0, engagement)) / 8.0)


def _pcts(signals: list[Signal]) -> tuple[float, float, float]:
    if not signals:
        return 0.0, 0.0, 0.0
    n = len(signals)
    bull = sum(1 for s in signals if s.label == "bullish")
    bear = sum(1 for s in signals if s.label == "bearish")
    neu = n - bull - bear
    return (
        round(100 * bull / n, 1),
        round(100 * bear / n, 1),
        round(100 * neu / n, 1),
    )


def build_response(
    ticker: str,
    company: str | None,
    results: list[SourceResult],
    signals: list[Signal],
) -> AnalyzeResponse:
    now = datetime.now(timezone.utc)
    weights = {r.source.name: r.source.weight for r in results}

    # ---- Weighted overall score ----
    num = den = 0.0
    for s in signals:
        w = (
            weights.get(s.source, 0.5)
            * _recency_weight(s.timestamp, now)
            * _engagement_weight(s.engagement)
        )
        num += w * s.score
        den += w
    overall = round(num / den, 4) if den else 0.0

    bull_pct, bear_pct, neu_pct = _pcts(signals)

    # ---- Per-source breakdown ----
    by_source: dict[str, list[Signal]] = defaultdict(list)
    for s in signals:
        by_source[s.source].append(s)

    breakdowns: list[SourceBreakdown] = []
    active = 0
    disabled = _disabled_names()
    for result in results:
        name = result.source.name
        group = by_source.get(name, [])
        if name in disabled:
            status = "disabled"
        else:
            status = result.status
        if status == "ok" and group:
            active += 1
        b_pct, be_pct, n_pct = _pcts(group)
        avg = round(sum(s.score for s in group) / len(group), 4) if group else 0.0
        breakdowns.append(
            SourceBreakdown(
                source=name,
                status=status,
                count=len(group),
                avg_score=avg,
                bullish_pct=b_pct,
                bearish_pct=be_pct,
                neutral_pct=n_pct,
                weight=result.source.weight,
                error=result.error,
            )
        )

    # ---- Hourly timeline ----
    timeline = _build_timeline(signals)

    # ---- Trending keywords ----
    trending = compute_trending(signals, ticker)

    # ---- News feed: most recent / influential first ----
    feed = sorted(
        signals,
        key=lambda s: (
            s.timestamp or datetime.min.replace(tzinfo=timezone.utc),
            s.engagement,
        ),
        reverse=True,
    )[:40]

    return AnalyzeResponse(
        ticker=ticker.upper(),
        company=company,
        generated_at=now,
        cached=False,
        overall_score=overall,
        overall_label=label_for(overall),
        total_signals=len(signals),
        active_sources=active,
        bullish_pct=bull_pct,
        bearish_pct=bear_pct,
        neutral_pct=neu_pct,
        sources=breakdowns,
        timeline=timeline,
        trending=trending,
        signals=feed,
    )


def _build_timeline(signals: list[Signal]) -> list[TimelinePoint]:
    buckets: dict[datetime, list[float]] = defaultdict(list)
    for s in signals:
        if s.timestamp is None:
            continue
        ts = s.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        bucket = ts.replace(minute=0, second=0, microsecond=0)
        buckets[bucket].append(s.score)
    points = [
        TimelinePoint(
            bucket=b,
            avg_score=round(sum(v) / len(v), 4),
            count=len(v),
        )
        for b, v in sorted(buckets.items())
    ]
    return points


def _disabled_names() -> set[str]:
    from app.config import settings

    return settings.disabled_source_set
