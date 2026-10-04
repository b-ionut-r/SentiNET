"""Theme classification on real headlines (Google News, 2026-10-04).

Each case lists themes that must be present and themes that must not be:
themes drive the "what's moving it" breakdown, so a wrong theme is as visible
as a missing one.
"""
from __future__ import annotations

import time

import pytest

from app.nlp.themes import THEMES, classify_themes, theme_scores
from tests.conftest import load_json_fixture

CASES: list[tuple[str, set[str], set[str]]] = [
    # (headline, must include, must not include)
    ("Morgan Stanley lowers Apple stock price target on limited upside", {"analyst"}, {"macro"}),
    ("Goldman Sachs adds Amazon stock to monthly Director's Cut list", {"analyst"}, set()),
    ("Apple's Fall Product Cycle Could Support Revenue Growth, Says Morgan Stanley", {"analyst", "product"}, set()),
    ("Tesla (TSLA) Q3 2026 deliveries slip 2% to 486,532, but beat estimates", {"earnings"}, {"macro"}),
    ("Tesla sold a lot more EVs than Wall Street expected, and the stock is surging", {"earnings"}, {"macro"}),
    ("Tesla September registrations rise across Europe, extending recovery", {"earnings"}, set()),
    ("Nike plans more job cuts to boost sputtering turnaround, forecasts steep revenue drop",
     {"labor", "guidance"}, set()),
    ("Walmart raises annual forecast after strong quarter", {"guidance"}, set()),
    ("US jury says Apple owes record $5.7 billion in haptic technology patent case", {"legal"}, set()),
    ("Meta Faces Up To $40 Billion Penalty In New Mexico Privacy Case After Jury Finds Company Misled Users",
     {"legal"}, set()),
    ("Amazon.com (AMZN) Draws Senate Questions Over AI Data Center Tax Deductions", {"regulatory", "ai"}, set()),
    ("SEC clears Tesla-crafted auto-vote plan for wide use, worrying activists", {"regulatory"}, set()),
    ("AMD to acquire World Labs in $8.2B all-stock deal (AMD:NASDAQ)", {"deals"}, set()),
    ("Constellation, Amazon sign 20-yr nuclear power agreement (CEG:NASDAQ)", {"deals"}, set()),
    ("Meta hires MongoDB CEO to lead company's enterprise ambitions", {"management"}, set()),
    ("New Apple CEO John Ternus is reportedly planning layoffs as he looks to reshape the iPhone maker",
     {"management", "labor"}, set()),
    ("NVIDIA Announces a $150 Billion Share Repurchase Authorization Increase", {"capital_return"}, set()),
    ("Target's Dividend Has Survived 8 Recessions. Here's What $10,000 Earns in Dividend Income Yearly.",
     {"capital_return"}, set()),
    ("Stocks slide on Wall Street as Treasury yields jump", {"macro"}, set()),
    ("Dow Jones Futures Rise, Oil Prices Tumble With Tesla, Jobs Due", {"macro"}, set()),
    ("Meta's Muse app surpasses 5M downloads and 3M weekly users", {"product"}, set()),
    ("Why the Tesla Roadster Event Was Delayed—and What It Means for the Stock", {"product"}, set()),
    ("Apple plans smart-home hub launch in October: report (AAPL:NASDAQ)", {"product"}, set()),
    ("Meta Faces Challenge in Enterprise Market As OpenAI Launches Autonomous Dots Agents",
     {"competition", "ai"}, set()),
    ("Target Is Cutting Prices on Thousands of Products Amid a 'Value War.' What This Means for TGT Stock.",
     {"competition"}, set()),
    ("Tesla Locks In $30 Billion Credit Line To Fund Soaring AI Spending", {"ai"}, set()),
    ("Apple (NASDAQ: AAPL) director Timothy D. Cook proposes selling 218,078 shares.", {"insider"}, set()),
    ("Block Insider Sold Shares Worth $1,327,740, According to a Recent SEC Filing", {"insider"}, set()),
    ("84,750 Shares in Amazon.com, Inc. $AMZN Bought by EMC Capital Management", {"trading"}, {"management"}),
    ("Michael Burry Buys Put Options on Nvidia, Micron, and Palantir", {"trading"}, set()),
    ("Tesla Trades at 350 Times Earnings. Here's What Has to Be True for That to Make Sense.", {"valuation"}, set()),
    ("Advanced Micro Devices (AMD) Could Be 33% Undervalued Following Its World Labs Deal", {"valuation"}, set()),
    ("Cybertruck sales in free fall as Tesla reports mediocre sales for Q3", {"earnings"}, set()),
    ("Memory chip shortage squeezes PC makers as DRAM prices soar", {"supply_chain"}, set()),
    # false friends
    ("Amazon seeks to offload $8 bln of Nvidia chips to investors- FT", set(), {"product"}),
    ("Meta taps Australia's Firmus for AI computing capacity in Southeast Asia", set(), {"management"}),
    ("Apple Is the Best-Performing Stock This Year. But It Has Still Lagged This Unassuming Dividend Stock",
     set(), {"capital_return"}),
    ("Stock Market Today, Oct. 2: Tesla Rises on Q3 Delivery Beat", {"earnings"}, {"macro"}),
    ("Phase 3 trial data lifts biotech shares", set(), {"legal"}),
    ("Netflix raises prices in the US; subscriber growth beats", set(), {"capital_return"}),
]


@pytest.mark.parametrize(("text", "must", "must_not"), CASES, ids=[c[0][:40] for c in CASES])
def test_real_headline_themes(text, must, must_not):
    themes = set(classify_themes(text))
    assert must <= themes, themes
    assert not must_not & themes, themes


def test_case_set_precision_and_recall():
    """Aggregate guard over the labeled cases (one bad pattern shows up here)."""
    expected = sum(len(m) for _t, m, _n in CASES)
    found = sum(len(m & set(classify_themes(t))) for t, m, _n in CASES)
    assert found / expected >= 0.95


def test_ranked_by_strength_and_capped():
    text = "Apple beats Q3 revenue estimates, raises guidance and boosts buyback; analysts lift price targets"
    themes = classify_themes(text)
    assert len(themes) <= 4
    scores = theme_scores(text)
    assert [scores[t] for t in themes] == sorted((scores[t] for t in themes), reverse=True)
    assert themes[0] == "earnings"
    assert classify_themes(text, max_themes=2) == themes[:2]


def test_vocabulary_and_edges():
    assert len(THEMES) >= 14
    assert all(key.islower() and label for key, label in THEMES.items())
    assert classify_themes("") == [] and classify_themes("$NVDA 🚀🚀") == []
    assert set(classify_themes("Nvidia lawsuit verdict appeal")) <= set(THEMES)


def test_coverage_on_real_feed():
    """Most company headlines carry at least one theme; generic 'why is X up
    today' stubs legitimately carry none."""
    items = load_json_fixture("nlp/headlines_amzn.json")["items"]
    tagged = sum(1 for it in items if classify_themes(it["title"]))
    assert tagged / len(items) >= 0.6


def test_fast():
    items = [it["title"] for name in ("nvda", "aapl", "meta", "tsla", "amzn") for it in
             load_json_fixture(f"nlp/headlines_{name}.json")["items"]]
    start = time.perf_counter()
    for title in items:
        classify_themes(title)
    assert time.perf_counter() - start < 0.5  # ~500 headlines
