"""Aggregates over prepared items: tone summaries, themes, keywords, timeline, source health."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.analytics import textkit
from app.analytics.inputs import SourceRun
from app.analytics.prepare import Item, Prepared
from app.analytics.util import clamp, effective_n, tone_label, weighted_mean
from app.schemas import Keyword, SentimentStat, SourceReport, ThemeStat, TimelineBucket
from app.sources.base import CompanyRef

STAT_PRIOR = 2.0  # neutral pseudo-items in SentimentStat.score (light Bayesian shrinkage)
TIMELINE_WINDOW = timedelta(days=7)
MAX_THEMES = 12
MAX_KEYWORDS = 15


@dataclass
class Summary:
    """Weighted tone of an item set (scored items only)."""

    mean: float | None  # raw weighted mean, -1..1
    n: int
    n_eff: float
    weight: float
    bullish: int
    bearish: int
    neutral: int
    spread: float  # weighted standard deviation of item scores
    mean_confidence: float
    outlets: int

    @property
    def shrunk(self) -> float:
        """Mean pulled toward 0 by `STAT_PRIOR` pseudo-items (what users see)."""
        if self.mean is None:
            return 0.0
        return self.mean * self.n / (self.n + STAT_PRIOR)

    @property
    def confidence(self) -> float:
        """Volume (effective n) × consistency (low spread) × engine confidence, 0..1."""
        if not self.n:
            return 0.0
        volume = self.n_eff / (self.n_eff + 6.0)
        consistency = 1.0 - min(1.0, self.spread / 0.6)
        return clamp(volume * (0.6 + 0.25 * consistency + 0.15 * self.mean_confidence))

    def stat(self) -> SentimentStat:
        score = round(self.shrunk, 3)
        return SentimentStat(
            score=score, label=tone_label(score), n=self.n, bullish=self.bullish, bearish=self.bearish,
            neutral=self.neutral, confidence=round(self.confidence, 3),
        )


def summarize(items: Iterable[Item]) -> Summary:
    scored = [it for it in items if it.scored]
    mean, wsum = weighted_mean((it.score, it.weight) for it in scored)
    spread = 0.0
    if mean is not None and wsum > 0:
        spread = math.sqrt(sum(it.weight * (it.score - mean) ** 2 for it in scored) / wsum)
    outlets = {o for it in scored for o in it.outlets()}
    return Summary(
        mean=mean, n=len(scored), n_eff=effective_n(it.weight for it in scored), weight=wsum,
        bullish=sum(it.label == "bullish" for it in scored),
        bearish=sum(it.label == "bearish" for it in scored),
        neutral=sum(it.label == "neutral" for it in scored),
        spread=spread,
        mean_confidence=(sum(it.confidence for it in scored) / len(scored)) if scored else 0.0,
        outlets=len(outlets),
    )


# --------------------------------------------------------------------------- #
# Themes & keywords
# --------------------------------------------------------------------------- #
def theme_stats(items: list[Item]) -> list[ThemeStat]:
    """Per-theme coverage and weighted tone, most covered first."""
    scored = [it for it in items if it.scored]
    if not scored:
        return []
    tagged: dict[str, list[Item]] = defaultdict(list)
    for it in scored:
        for key in it.themes:
            tagged[key].append(it)
    min_count = 2 if len(scored) >= 20 else 1
    out: list[ThemeStat] = []
    for key, members in tagged.items():
        if len(members) < min_count:
            continue
        mean, wsum = weighted_mean((it.score, it.weight) for it in members)
        out.append(ThemeStat(
            theme=key, label=textkit.theme_label(key), count=len(members),
            score=round(mean or 0.0, 3), share=round(len(members) / len(scored), 3),
        ))
    weights = {k: sum(it.weight for it in v) for k, v in tagged.items()}
    out.sort(key=lambda t: (-t.count, -weights[t.theme], t.theme))
    return out[:MAX_THEMES]


def keyword_list(items: list[Item], company: CompanyRef | None) -> list[Keyword]:
    scored = [it for it in items if it.scored]
    if not scored:
        return []
    found = textkit.keywords([it.title for it in scored], [it.score for it in scored], company, MAX_KEYWORDS)
    return [Keyword(term=t, count=int(c), score=round(float(s), 3)) for t, c, s in found if t]


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #
def timeline(items: list[Item], now: datetime) -> list[TimelineBucket]:
    """Weighted tone per time bucket over the last 7 days (non-empty buckets only).

    Hourly buckets when the dated items span <= 48 h, 6-hour buckets up to
    7 days. Buckets are aligned to UTC (00/06/12/18 h)."""
    dated = [it for it in items if it.scored and it.timestamp and now - it.timestamp <= TIMELINE_WINDOW]
    if not dated:
        return []
    span = now - min(it.timestamp for it in dated if it.timestamp)
    step = 3600 if span <= timedelta(hours=48) else 6 * 3600
    buckets: dict[int, list[Item]] = defaultdict(list)
    for it in dated:
        assert it.timestamp is not None
        ts = int(it.timestamp.timestamp())
        buckets[ts - ts % step].append(it)
    out: list[TimelineBucket] = []
    for start in sorted(buckets):
        members = buckets[start]
        mean, _ = weighted_mean((it.score, it.weight) for it in members)
        out.append(TimelineBucket(
            t=datetime.fromtimestamp(start, tz=now.tzinfo),
            score=round(mean or 0.0, 3), count=len(members),
            news=sum(it.group == "news" for it in members), social=sum(it.group == "social" for it in members),
            bullish=sum(it.label == "bullish" for it in members),
            bearish=sum(it.label == "bearish" for it in members),
        ))
    return out


# --------------------------------------------------------------------------- #
# Source health
# --------------------------------------------------------------------------- #
def source_reports(runs: list[SourceRun], prepared: Prepared) -> list[SourceReport]:
    """One row per source run: status, latency, fetched -> kept, tone of kept items."""
    by_source: dict[str, list[Item]] = defaultdict(list)
    for it in prepared.items:
        by_source[it.source].append(it)
    out: list[SourceReport] = []
    for run in runs:
        src = run.source
        kept = by_source.get(src.key, [])
        summary = summarize(kept)
        n = summary.n
        out.append(SourceReport(
            key=src.key, label=src.label, kind=src.kind, status=run.status,
            fetched=prepared.fetched.get(src.key, 0), kept=len(kept),
            score=round(summary.mean, 3) if summary.mean is not None else None,
            bullish_pct=round(summary.bullish / n, 3) if n else 0.0,
            bearish_pct=round(summary.bearish / n, 3) if n else 0.0,
            weight=float(src.weight), latency_ms=run.latency_ms, error=run.error,
            requires_key=bool(getattr(src, "requires_key", False)),
            has_metrics=bool(run.batch is not None and run.batch.metrics),
        ))
    return out
