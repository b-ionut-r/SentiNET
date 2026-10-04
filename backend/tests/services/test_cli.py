"""CLI: rich rendering of real-shaped analyses, JSON mode and exit codes (fake providers)."""
from __future__ import annotations

import json

import pytest
from rich.console import Console

from app import cli
from app.schemas import (
    AnalystView,
    AttentionView,
    Brief,
    Catalyst,
    CrowdView,
    InsiderView,
    Insight,
    MarketOverview,
    Narrative,
)
from tests.services.fakes import NOW, FakeWorld


@pytest.fixture
def consoles(monkeypatch) -> tuple[Console, Console]:
    out, err = Console(record=True, width=140), Console(record=True, width=140, stderr=True)
    monkeypatch.setattr(cli, "out", out)
    monkeypatch.setattr(cli, "err", err)
    return out, err


def test_analyze_renders_intel_at_a_glance(world: FakeWorld, consoles):
    world.narratives = [Narrative(id="n1", headline="Nvidia faces $1.05B lawsuit", count=14, score=-0.42,
                                  velocity_24h=6, publishers=["Reuters", "Bloomberg"], is_new=True)]
    world.attention = AttentionView(heat=81, label="Spiking", signals_24h=44)
    assert cli.main(["analyze", "nvda", "--quiet"]) == 0
    text = consoles[0].export_text()
    for needle in ("NVDA", "NVIDIA Corporation", "SentiNET  64", "Bullish: analysts and news align.",
                   "3 price-target raises in 30 days", "Components", "+0.21 across 12 articles", "n/a",
                   "What's moving it", "Nvidia faces $1.05B lawsuit", "NEW", "Spiking (81/100)",
                   "Sources", "Google News", "unconfigured", "add a free API key"):
        assert needle in text, needle


def test_analyze_json_mode(world: FakeWorld, consoles, capsys):
    assert cli.main(["analyze", "NVDA", "--json", "--quiet"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ticker"] == "NVDA" and data["verdict"]["score"] == world.score


def test_analyze_errors_have_exit_codes(world: FakeWorld, consoles):
    assert cli.main(["analyze", "!!", "--quiet"]) == 2
    assert "not a valid ticker" in consoles[1].export_text()
    world.build_error = RuntimeError("boom")
    assert cli.main(["analyze", "NVDA", "--quiet"]) == 1
    assert "Synthesis failed" in consoles[1].export_text()


def test_render_smart_money_crowd_and_catalysts(consoles):
    from app.analytics.inputs import AnalysisInputs
    from app.sources.base import CompanyRef

    a = FakeWorld().build(AnalysisInputs(company=CompanyRef(ticker="SOFI", name="SoFi", short_name="SoFi"),
                                         now=NOW, engine_name="sentinel"))
    a = a.model_copy(update={
        "analysts": AnalystView(consensus="hold", total=25, target_mean=20.3, upside_pct=29.0, upgrades_90d=2,
                                pt_raises_30d=3),
        "insiders": InsiderView(buys=3, sells=9, buy_value=1.2e6, sell_value=4.5e6, net_value=-3.3e6),
        "crowd": CrowdView(stocktwits_bull_ratio=0.71, stocktwits_bullish=42, stocktwits_bearish=17,
                           reddit_mentions=310, reddit_mentions_prev=120, reddit_rank=7),
        "catalysts": [Catalyst(date=NOW, kind="earnings", title="Q3 earnings", upcoming=True, detail="beat 7/8")],
        "insights": [Insight(kind="crowding", severity="watch", polarity="bear", title="Crowded long",
                             detail="71% bulls")],
        "brief": Brief(summary="Leaning bullish on member growth.", bull_points=["Members +35% y/y"],
                       bear_points=["Dilution from converts", "Credit losses rising"], watch=["Q3 on Oct 27"]),
    })
    cli.render_analysis(a, consoles[0])
    text = consoles[0].export_text()
    for needle in ("hold · 25 analysts", "$20.30", "+29%", "3 buys $1.20M", "net -$3.30M",
                   "71% bullish of 59 tagged", "#7 · 310 mentions/24h", "+158%", "Q3 earnings", "WATCH",
                   "Crowded long", "▲ Bull case", "▼ Bear case", "Members +35% y/y", "Credit losses rising",
                   "Watch: Q3 on Oct 27"):
        assert needle in text, needle


def test_render_market(consoles):
    from app.schemas import FearGreed, IndexQuote, TrendingTicker

    m = MarketOverview(generated_at=NOW, regime="Risk-off: Fear", regime_detail="F&G 38, VIX 21",
                       fear_greed=FearGreed(score=38, rating="fear", week_ago=45),
                       indices=[IndexQuote(symbol="SPY", name="S&P 500", price=600.1, change_pct=-0.4)],
                       trending=[TrendingTicker(symbol="NVDA", source="reddit", rank=1, mentions=900,
                                                change_pct=35.0)])
    cli.render_market(m, consoles[0])
    text = consoles[0].export_text()
    for needle in ("Risk-off: Fear", "CNN Fear & Greed: 38 fear", "1w ago 45", "SPY", "600.10", "NVDA", "+35%"):
        assert needle in text, needle


def test_bar_and_money_helpers():
    assert cli.bar(None).plain.strip() == "n/a"
    assert cli.bar(100, width=10).plain == "     │█████"
    assert cli.bar(0, width=10).plain == "█████│     "
    assert cli.bar(50, width=10).plain == "     │     "
    assert cli.money(4.43e12) == "$4.43T" and cli.money(-3.3e6) == "-$3.30M" and cli.money(None) == "n/a"


def test_env_example_has_no_inline_comments_and_parses():
    """`docker run --env-file` passes values literally: `KEY=30  # note` would crash Settings."""
    from pathlib import Path

    from app.config import Settings

    path = Path(__file__).resolve().parents[2] / ".env.example"
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            assert "#" not in value, f"inline comment in {line!r}"
            values[key.lower()] = value
    assert "monitor_interval_minutes" in values
    Settings(_env_file=None, **{k: v for k, v in values.items() if v})  # every literal value validates
