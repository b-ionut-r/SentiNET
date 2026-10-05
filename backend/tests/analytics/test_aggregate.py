"""Tone summaries (shrinkage, confidence), themes, timeline buckets and source reports."""
from __future__ import annotations

from datetime import timedelta

import pytest

from app.analytics.aggregate import source_reports, summarize, theme_stats, timeline
from app.analytics.prepare import Item, prepare
from tests.analytics.factories import GOOGLE, NOW, STOCKTWITS, company, raw, run


def item(score: float, weight: float = 1.0, hours: float = 1.0, themes: list[str] | None = None,
         group: str = "news", label: str | None = None) -> Item:
    it = Item(id=f"{score}-{weight}-{hours}", source="google_news", source_label="Google News", source_weight=1.0,
              kind=group, title="t", timestamp=NOW - timedelta(hours=hours))  # type: ignore[arg-type]
    it.scored, it.score, it.weight, it.confidence = True, score, weight, 0.7
    it.label = label or ("bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral")
    it.themes = themes or []
    return it


def test_small_samples_shrink_toward_neutral() -> None:
    two = summarize([item(0.6), item(0.6)])
    many = summarize([item(0.6) for _ in range(40)])
    assert two.mean == pytest.approx(0.6) and many.mean == pytest.approx(0.6)
    assert two.stat().score == pytest.approx(0.3)  # 0.6 · 2 / (2 + 2)
    assert many.stat().score == pytest.approx(0.6 * 40 / 42, abs=1e-3)
    assert two.stat().confidence < many.stat().confidence


def test_disagreement_lowers_confidence() -> None:
    agree = summarize([item(0.5) for _ in range(20)])
    split = summarize([item(0.5) for _ in range(10)] + [item(-0.5) for _ in range(10)])
    assert split.stat().score == pytest.approx(0.0) and split.stat().label == "neutral"
    assert split.confidence < agree.confidence


def test_weights_drive_the_mean_and_effective_n() -> None:
    s = summarize([item(1.0, weight=9.0), item(-1.0, weight=1.0)])
    assert s.mean == pytest.approx(0.8) and s.n == 2 and s.n_eff == pytest.approx(100 / 82)


def test_unscored_items_are_ignored() -> None:
    it = item(0.9)
    it.scored = False
    assert summarize([it]).n == 0 and summarize([it]).stat().score == 0


def test_theme_stats_share_and_tone() -> None:
    items = [item(-0.6, themes=["legal"]) for _ in range(3)] + [item(0.4, themes=["earnings", "ai"])
                                                                for _ in range(5)]
    stats = {t.theme: t for t in theme_stats(items)}
    assert stats["earnings"].count == 5 and stats["earnings"].share == pytest.approx(5 / 8)
    assert stats["legal"].score == pytest.approx(-0.6)
    assert [t.theme for t in theme_stats(items)][0] in ("earnings", "ai")


def test_timeline_hourly_when_recent_six_hourly_otherwise() -> None:
    recent = [item(0.2, hours=h) for h in (0.5, 1.5, 1.7, 30)]
    buckets = timeline(recent, NOW)
    assert all(b.t.minute == 0 and b.t.second == 0 for b in buckets)
    assert sum(b.count for b in buckets) == 4 and len(buckets) == 3
    spread = [item(0.2, hours=h) for h in (1, 50, 100, 150)] + [item(0.3, hours=24 * 9)]
    buckets = timeline(spread, NOW)
    assert all(b.t.hour % 6 == 0 for b in buckets)
    assert sum(b.count for b in buckets) == 4  # the 9-day-old item is outside the 7-day window
    assert timeline([], NOW) == []


def test_source_reports_count_fetched_and_kept() -> None:
    acme = company()
    runs = [run(GOOGLE, [raw("Acme beats estimates"), raw("Oil climbs"), raw("Acme beats estimates")]),
            run(STOCKTWITS, status="error", error="timeout"), run(GOOGLE.__class__("finnhub", "Finnhub"),
                                                                  status="unconfigured")]
    prepared = prepare(acme, runs, NOW)
    reports = {r.key: r for r in source_reports(runs, prepared)}
    g = reports["google_news"]
    assert (g.fetched, g.kept, g.status) == (3, 1, "ok") and g.bullish_pct == 1.0
    assert reports["stocktwits"].status == "error" and reports["stocktwits"].error == "timeout"
    assert reports["finnhub"].status == "unconfigured" and reports["finnhub"].score is None
