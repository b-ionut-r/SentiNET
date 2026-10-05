"""Crowd metrics and the attention heat gauge."""
from __future__ import annotations

import pytest

from app.analytics.crowd import (
    attention_view,
    crowd_view,
    merged_metrics,
    reddit_change_pct,
    reddit_move,
    spike_z,
    wiki_stats,
)
from app.schemas import CrowdView
from tests.analytics.factories import APEWISDOM, NOW, STOCKTWITS, daily, run, tone_trend


def test_merged_metrics_and_crowd_view() -> None:
    runs = [run(STOCKTWITS, [], {"stocktwits_bullish": 18, "stocktwits_bearish": 6, "stocktwits_watchers": 5000}),
            run(APEWISDOM, [], {"reddit_mentions": 30, "reddit_mentions_prev": 10, "reddit_rank": 9}),
            run(STOCKTWITS, status="error")]
    view = crowd_view(merged_metrics(runs))
    assert view.stocktwits_bull_ratio == pytest.approx(0.75) and view.reddit_rank == 9
    assert view.wsb_sentiment is None and view.bluesky_posts is None
    assert crowd_view({}) is None
    assert crowd_view({"wsb_label": "euphoric", "reddit_mentions": "n/a"}) is None  # junk is not data


def test_reddit_change_needs_a_meaningful_base() -> None:
    assert reddit_change_pct(CrowdView(reddit_mentions=34, reddit_mentions_prev=10)) == pytest.approx(240.0)
    assert reddit_change_pct(CrowdView(reddit_mentions=5, reddit_mentions_prev=1)) is None  # 1 -> 5 is noise
    assert reddit_move(CrowdView(reddit_mentions=7, reddit_mentions_prev=21)) == "fell 67% in 24h (21 → 7)"


def test_spike_z_is_robust_and_needs_history() -> None:
    flat = [100.0] * 30
    assert spike_z(flat) == pytest.approx(0.0)
    assert spike_z(flat[:-2] + [400.0, 400.0]) > 10  # floored baseline spread
    noisy = [100.0, 140.0, 80.0, 120.0, 90.0] * 6
    assert 1 < spike_z(noisy[:-2] + [300.0, 320.0]) < 10
    assert spike_z([100.0] * 10) is None


def test_wiki_stats() -> None:
    avg, z = wiki_stats(daily([1000.0] * 40 + [5000.0, 6000.0]))
    assert avg == pytest.approx((1000 * 5 + 11000) / 7, abs=0.1) and z > 2
    assert wiki_stats(None) == (None, None)


def test_heat_combines_gauges_against_their_own_baselines() -> None:
    spiking = attention_view(tone_trend(volume=200, spike=900), daily([1000.0] * 40 + [5000.0, 6000.0]),
                             CrowdView(reddit_mentions=200, reddit_mentions_prev=50), [], NOW)
    assert spiking.label == "Spiking" and spiking.heat >= 78
    assert spiking.news_volume_z > 2 and spiking.wiki_views_z > 2 and spiking.reddit_change_pct == 300.0
    normal = attention_view(tone_trend(volume=200), None, None, [], NOW)
    assert normal.label == "Normal" and 40 <= normal.heat <= 60


def test_a_thin_reddit_blip_alone_cannot_declare_quiet() -> None:
    view = attention_view(None, None, CrowdView(reddit_mentions=6, reddit_mentions_prev=21), [], NOW)
    assert view is not None and view.label == "Normal" and view.reddit_change_pct == pytest.approx(-71.4)


def test_no_gauges_no_attention() -> None:
    assert attention_view(None, None, CrowdView(reddit_mentions=2, reddit_mentions_prev=1), [], NOW) is None
