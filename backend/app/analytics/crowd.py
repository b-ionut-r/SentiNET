"""Retail positioning (`CrowdView`) and attention (`AttentionView`) from structured metrics.

Attention heat (0..100, 50 = normal) combines "vs. its own baseline" gauges —
never absolute volume, which depends on company size:

    GDELT article volume (w 0.40)  50 + 15·z   z of the last 2 days vs the prior 28 (log scale)
    Wikipedia pageviews  (w 0.25)  50 + 15·z   same method
    Reddit mentions      (w 0.35)  50 + 20·log2((now+1)/(24h ago+1))   needs >= 5 mentions;
                                   reliability min(1, mentions/50) — except a rank
                                   breakout (into the top 25 from outside the top 100
                                   or unranked, on >= 10 mentions): at least 85 at full
                                   reliability, since the rank is relative to every
                                   tracked ticker (META 675 -> 7 on 14 mentions vs 2)

    heat = 50 + Σ w·r·(gauge − 50) / max(Σ w of available gauges, 0.6)

so a lone, thin gauge (e.g. 21 → 7 Reddit mentions) cannot by itself declare
a ticker "Quiet" or "Spiking".

`signals_24h` is reported but not scored: source sample caps make raw counts
incomparable across tickers.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal

from app.analytics.inputs import SourceRun
from app.analytics.prepare import Item
from app.analytics.util import clamp, mean, stdev
from app.schemas import AttentionView, CrowdView, ToneTrend

SPIKE_RECENT_DAYS = 2
SPIKE_BASELINE_DAYS = 28
SPIKE_MIN_BASELINE = 14
MIN_LOG_SD = 0.1  # floor for a flat baseline (~10% day-to-day variation)
REDDIT_MIN_MENTIONS = 5
REDDIT_MIN_PREV_FOR_PCT = 3
HEAT_WEIGHTS = {"news": 0.40, "wiki": 0.25, "reddit": 0.35}
HEAT_MIN_DENOMINATOR = 0.6
REDDIT_FULL_RELIABILITY = 50
BREAKOUT_RANK = 25  # a rank breakout lands inside this rank …
BREAKOUT_FROM = 100  # … from outside this one (or unranked) …
BREAKOUT_MENTIONS = 10  # … on at least this many mentions
BREAKOUT_HEAT = 85.0


def merged_metrics(runs: list[SourceRun]) -> dict[str, Any]:
    """All sources' metrics in one dict (first non-None value per key wins)."""
    out: dict[str, Any] = {}
    for run in runs:
        if run.batch is None:
            continue
        for key, value in (run.batch.metrics or {}).items():
            if value is not None and key not in out:
                out[key] = value
    return out


def as_int(value: Any) -> int | None:
    try:
        return None if value is None or isinstance(value, bool) else int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def as_float(value: Any) -> float | None:
    try:
        f = None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None
    return f if f is not None and math.isfinite(f) else None


@dataclass(frozen=True)
class Tally:
    """StockTwits bull/bear tags: one vote per account when the source reports it
    (a single prolific account cannot swing the ratio), else one per message."""

    bullish: int
    bearish: int
    per_author: bool

    @property
    def n(self) -> int:
        return self.bullish + self.bearish

    @property
    def ratio(self) -> float | None:
        return self.bullish / self.n if self.n else None

    @property
    def sample(self) -> str:
        """'35 accounts' / '64 tagged posts' — the base of the ratio."""
        return f"{self.n} {'account' if self.n == 1 else 'accounts'}" if self.per_author else f"{self.n} tagged"

    @property
    def described(self) -> str:
        """'35 StockTwits accounts tagging a stance' / '64 tagged StockTwits posts'."""
        if self.per_author:
            return f"{self.n} StockTwits {'account' if self.n == 1 else 'accounts'} tagging a stance"
        return f"{self.n} tagged StockTwits {'post' if self.n == 1 else 'posts'}"


def stocktwits_tally(metrics: dict[str, Any]) -> Tally | None:
    """Author-deduplicated tallies when present, else per-message tallies; None without tags data."""
    bull_a, bear_a = as_int(metrics.get("stocktwits_bull_authors")), as_int(metrics.get("stocktwits_bear_authors"))
    if bull_a is not None and bear_a is not None and bull_a + bear_a > 0:
        return Tally(max(bull_a, 0), max(bear_a, 0), per_author=True)
    bull, bear = as_int(metrics.get("stocktwits_bullish")), as_int(metrics.get("stocktwits_bearish"))
    if bull is None and bear is None:
        return None
    return Tally(max(bull or 0, 0), max(bear or 0, 0), per_author=False)


def crowd_view(metrics: dict[str, Any]) -> CrowdView | None:
    """Structured retail metrics; None when no crowd source reported anything.

    StockTwits counts and ratio are per tagged message, as `CrowdView` documents
    (and the UI labels them); the scoring uses the per-account `stocktwits_tally`
    and says "accounts" wherever it quotes it."""
    bull, bear = as_int(metrics.get("stocktwits_bullish")), as_int(metrics.get("stocktwits_bearish"))
    tagged = max(bull or 0, 0) + max(bear or 0, 0)
    wsb_label = metrics.get("wsb_label")
    view = CrowdView(
        stocktwits_bullish=bull, stocktwits_bearish=bear,
        stocktwits_bull_ratio=round(max(bull or 0, 0) / tagged, 3) if tagged else None,
        stocktwits_bull_authors=as_int(metrics.get("stocktwits_bull_authors")),
        stocktwits_bear_authors=as_int(metrics.get("stocktwits_bear_authors")),
        stocktwits_messages=as_int(metrics.get("stocktwits_messages")),
        stocktwits_watchers=as_int(metrics.get("stocktwits_watchers")),
        reddit_mentions=as_int(metrics.get("reddit_mentions")),
        reddit_mentions_prev=as_int(metrics.get("reddit_mentions_prev")),
        reddit_rank=as_int(metrics.get("reddit_rank")),
        reddit_rank_prev=as_int(metrics.get("reddit_rank_prev")),
        reddit_upvotes=as_int(metrics.get("reddit_upvotes")),
        wsb_sentiment=as_float(metrics.get("wsb_sentiment")),
        wsb_label=wsb_label if wsb_label in ("bullish", "bearish", "neutral") else None,
        wsb_comments=as_int(metrics.get("wsb_comments")),
        bluesky_posts=as_int(metrics.get("bluesky_posts")),
    )
    return view if any(v is not None for v in view.model_dump().values()) else None


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def spike_z(values: Sequence[float]) -> float | None:
    """Robust "how unusual is now": mean log value of the last 2 days vs the prior 28.

    Values are chronological daily counts; needs >= 14 baseline days."""
    vals = [math.log1p(max(float(v), 0.0)) for v in values]
    if len(vals) < SPIKE_RECENT_DAYS + SPIKE_MIN_BASELINE:
        return None
    recent = vals[-SPIKE_RECENT_DAYS:]
    base = vals[-(SPIKE_RECENT_DAYS + SPIKE_BASELINE_DAYS):-SPIKE_RECENT_DAYS]
    mu, sd = mean(base), stdev(base)
    if mu is None or sd is None:
        return None
    return round((sum(recent) / len(recent) - mu) / max(sd, MIN_LOG_SD), 2)


def news_volume_z(tone: ToneTrend | None) -> float | None:
    if tone is None:
        return None
    series = sorted((p for p in tone.series if p.volume is not None), key=lambda p: p.date)
    return spike_z([p.volume for p in series if p.volume is not None])


def wiki_stats(wiki: list[tuple[date, float]] | None) -> tuple[float | None, float | None]:
    """(average daily views over the last 7 days, spike z)."""
    if not wiki:
        return None, None
    series = [float(v) for _, v in sorted(wiki, key=lambda x: x[0]) if v is not None]
    if not series:
        return None, None
    last7 = series[-7:]
    return round(sum(last7) / len(last7), 1), spike_z(series)


def reddit_change_pct(crowd: CrowdView | None) -> float | None:
    """Mention growth vs 24 h earlier (needs a meaningful base)."""
    if crowd is None or crowd.reddit_mentions is None or crowd.reddit_mentions_prev is None:
        return None
    prev = crowd.reddit_mentions_prev
    if prev < REDDIT_MIN_PREV_FOR_PCT:
        return None
    return round((crowd.reddit_mentions - prev) / prev * 100.0, 1)


def reddit_breakout(crowd: CrowdView | None) -> bool:
    """Jumped into Reddit's top 25 from outside the top 100 (or unranked) on >= 10 mentions.

    The mention change cannot show it when the prior count is tiny (14 vs 2), but
    the rank is relative to every tracked ticker, so the jump itself is the signal."""
    if crowd is None or crowd.reddit_rank is None or crowd.reddit_mentions is None:
        return False
    prev = crowd.reddit_rank_prev
    return (crowd.reddit_rank <= BREAKOUT_RANK and (prev is None or prev >= BREAKOUT_FROM)
            and crowd.reddit_mentions >= BREAKOUT_MENTIONS)


def reddit_move(crowd: CrowdView | None) -> str | None:
    """'fell 67% in 24h (21 → 7)' / 'rose 240% in 24h (10 → 34)'."""
    change = reddit_change_pct(crowd)
    if change is None or crowd is None:
        return None
    verb = "rose" if change > 0 else "fell" if change < 0 else "were flat"
    size = f" {abs(change):.0f}%" if change else ""
    return f"{verb}{size} in 24h ({crowd.reddit_mentions_prev} → {crowd.reddit_mentions})"


def attention_view(tone: ToneTrend | None, wiki: list[tuple[date, float]] | None, crowd: CrowdView | None,
                   items: list[Item], now: datetime) -> AttentionView | None:
    """Composite attention vs. the ticker's own baseline; None when no gauge is available."""
    news_z = news_volume_z(tone)
    wiki_7d, wiki_z = wiki_stats(wiki)
    reddit_pct = reddit_change_pct(crowd)

    parts: dict[str, tuple[float, float]] = {}  # gauge -> (heat, reliability)
    if news_z is not None:
        parts["news"] = (clamp(50 + 15 * news_z, 0, 100), 1.0)
    if wiki_z is not None:
        parts["wiki"] = (clamp(50 + 15 * wiki_z, 0, 100), 1.0)
    if crowd is not None and crowd.reddit_mentions is not None and crowd.reddit_mentions_prev is not None:
        m, p = crowd.reddit_mentions, crowd.reddit_mentions_prev
        if max(m, p) >= REDDIT_MIN_MENTIONS:
            parts["reddit"] = (clamp(50 + 20 * math.log2((m + 1) / (p + 1)), 0, 100),
                               min(1.0, max(m, p) / REDDIT_FULL_RELIABILITY))
    if reddit_breakout(crowd):
        h, _ = parts.get("reddit", (50.0, 0.0))
        parts["reddit"] = (max(h, BREAKOUT_HEAT), 1.0)
    if not parts:
        return None
    total = max(sum(HEAT_WEIGHTS[k] for k in parts), HEAT_MIN_DENOMINATOR)
    heat = round(clamp(50 + sum(HEAT_WEIGHTS[k] * r * (h - 50) for k, (h, r) in parts.items()) / total, 0, 100))
    day_ago = now - timedelta(hours=24)
    recent = sum(1 for it in items for t in it.times() if t >= day_ago)
    return AttentionView(
        heat=heat, label=heat_label(heat), news_volume_z=news_z, wiki_views_7d=wiki_7d, wiki_views_z=wiki_z,
        reddit_change_pct=reddit_pct, signals_24h=recent,
    )


def heat_label(heat: float) -> Literal["Quiet", "Normal", "Elevated", "Spiking"]:
    return "Quiet" if heat < 35 else "Normal" if heat < 62 else "Elevated" if heat < 78 else "Spiking"
