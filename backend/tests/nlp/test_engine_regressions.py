"""Regression cases from the engine review: phrasings that used to flip the sign.

Each group pins one failure mode the benchmarks do not catch (they are template-heavy):
what a forecast is about, negators inside rule spans, clause-crossing rule gaps, size-class
compounds, the retailer Target vs. a price target, bare "record", signed percents on bad
metrics, "stock up/down", why-questions, and trader words in forum prose. All texts are
hand-written; none come from a held-out benchmark.
"""
from __future__ import annotations

import pytest

from app.nlp import lexicon as lx
from app.nlp.engine import SentinelEngine
from app.nlp.types import TextAnalysis

CASES: list[tuple[str, str, str]] = [
    # guidance: the sign is the verb's times the forecast metric's polarity
    ("Company raises loss forecast", "bearish", "news"),
    ("Airline raises fuel cost forecast", "bearish", "news"),
    ("Biotech raises cash burn guidance", "bearish", "news"),
    ("Fed raises inflation forecast", "bearish", "news"),
    ("Company lowers net loss outlook", "bullish", "news"),
    ("Fed lowers unemployment forecast", "bullish", "news"),
    ("Central bank surprises by cutting its inflation forecast", "bullish", "news"),
    ("IMF cuts its forecast for global growth", "bearish", "news"),
    ("Acme raises full-year sales forecast", "bullish", "news"),
    ("Acme cuts profit, sales forecasts", "bearish", "news"),
    ("Acme's loss forecast raised after weak quarter", "bearish", "news"),
    ("$XYZ (+5.8% pre) Acme Oil cuts full-year capex forecast by 30%", "bullish", "news"),
    # size classes are not the verb "cap"
    ("Small-cap stocks rally as yields fall", "bullish", "news"),
    ("Large-cap tech stocks lead gains", "bullish", "news"),
    ("Mega-cap stocks slide", "bearish", "news"),
    ("Mid-cap stocks outperform", "bullish", "news"),
    ("1 Large-Cap Stock to Target This Week and 2 We Find Risky", "neutral", "news"),
    # negators inside a rule's span, and regulators saying no
    ("FDA did not approve Acme's drug", "bearish", "news"),
    ("FDA refuses to approve Acme's drug", "bearish", "news"),
    ("FDA declines to approve Acme drug", "bearish", "news"),
    ("Regulators decline to clear Acme merger", "bearish", "news"),
    ("FDA rejects Acme's cancer drug", "bearish", "news"),
    ("FTC sues to block Acme's takeover of Widget Co", "bearish", "news"),
    ("FDA approves Acme's cancer drug", "bullish", "news"),
    # rule gaps never cross clauses
    ("Acme Valuation Looks Lower Despite Strong Growth Outlook", "neutral", "news"),
    ("Supplier issue cuts shipments, rival holds 2026 guidance", "bearish", "news"),
    ("Acme slashes prices, keeps its outlook", "bearish", "news"),
    ("Acme case closed but second lawsuit looms", "bearish", "news"),
    ("Acme cuts prices and boosts outlook", "bullish", "news"),
    # Target the retailer is not a price target
    ("Walmart raises wages to compete with Target", "neutral", "news"),
    ("Walmart cuts prices to take on Target", "neutral", "news"),
    ("Amazon lowers prices ahead of Target's sale", "neutral", "news"),
    ("Geode Capital Management LLC Increases Position in Target Co. (NYSE:TGT)", "neutral", "news"),
    ("Target to cut 1,800 jobs", "bearish", "news"),
    ("Target price raised to $150 from $130 at Barclays", "bullish", "news"),
    ("Barclays raises its target on Target to $150", "bullish", "news"),
    ("Analyst lifts Target price target", "bullish", "news"),
    ("SunTrust trims Nielsen target", "bearish", "news"),
    ("Acme Lasers target raised on defense deal", "bullish", "news"),
    ("Acme gets Street-low target on demand worries", "bearish", "news"),
    # a bare "record" is not good news
    ("A US Jury Orders Apple (AAPL) to Pay a Record $5.7 Billion in a Patent Case", "bearish", "news"),
    ("Meta hit with record $1.3 billion EU fine", "bearish", "news"),
    ("Apple declares quarterly dividend of $0.26 per share payable to shareholders of record as of November 10",
     "neutral", "news"),
    ("AT&T declares dividend, payable to stockholders of record on January 10", "neutral", "news"),
    ("US households now owe more than $15 trillion, a new record", "neutral", "news"),
    ("Acme Sets Another Box Office Record For Its Film Studio", "neutral", "news"),
    ("Acme posts record revenue", "bullish", "news"),
    ("Nvidia stock hits record", "bullish", "news"),
    ("Brazilian coffee exports set to hit record 50 million bags", "bullish", "news"),
    ("Acme posts record loss", "bearish", "news"),
    # signed percents take their sign from what moved
    ("Costs +12% y/y", "bearish", "news"),
    ("Inflation +3.2% y/y vs +3.0% expected", "bearish", "news"),
    ("US CPI +0.4% m/m, core +0.3%", "bearish", "news"),
    ("VIX +15%", "bearish", "news"),
    ("Short interest +20%", "bearish", "news"),
    ("Opex -8% y/y", "bullish", "news"),
    ("$ABCD (+13.1% pre) gets FDA nod", "bullish", "news"),
    ("Acme -6% after guidance", "bearish", "news"),
    # "stock up/down" is a move unless "stock up on" means hoarding
    ("Nvidia stock down after earnings", "bearish", "news"),
    ("Why Is Nvidia Stock Down Today?", "bearish", "news"),
    ("Tesla stock up on strong deliveries", "bullish", "news"),
    ("Bank stocks up on rate hopes", "bullish", "news"),
    ("Investors stock up on canned goods", "neutral", "news"),
    ("Is Nvidia stock a buy?", "neutral", "news"),
    # a company moving pay is a cost and a perk, not a market sign
    ("Walmart raises wages for store workers", "neutral", "news"),
    ("US wages rise faster than expected", "bullish", "news"),
    # common constructions the first review pass surfaced
    ("Acme beats on earnings but guidance disappoints", "bearish", "news"),
    ("Ford shakes up management days after weak profit outlook", "bearish", "news"),
    ("Acme reports third-quarter loss", "bearish", "news"),
    ("Acme reports narrower third-quarter loss", "bullish", "news"),
    ("Acme wins $10 billion Pentagon contract", "bullish", "news"),
    ("Acme wins IT services contract", "neutral", "news"),
    ("Acme cleared of wrongdoing", "bullish", "news"),
    ("Acme added to S&P 500", "bullish", "news"),
    ("Acme removed from S&P 500", "bearish", "news"),
    ("S&P 500 joins Dow in negative territory", "bearish", "news"),
    ("Fed signals fewer rate cuts", "bearish", "news"),
    ("Fed signals more rate cuts", "bullish", "news"),
    ("New CEO takes charge at Acme", "neutral", "news"),
    ("Acme takes a $2 billion charge", "bearish", "news"),
    ("Acme's CFO departs abruptly", "bearish", "news"),
    # trader words only count as positions in trading talk
    ("I've been using this for a long time and it calls the API twice", "neutral", "social"),
    ("Long story short, the board calls for a vote", "neutral", "social"),
    ("As long as rates stay where they are it's fine", "neutral", "social"),
    ("Long $TSLA here", "bullish", "social"),
    ("bought more calls today $NVDA", "bullish", "social"),
    ("loading puts on $SPY", "bearish", "social"),
    ("$CCL doubling down short", "bearish", "social"),
]


@pytest.fixture(scope="module")
def engine() -> SentinelEngine:
    return SentinelEngine()


@pytest.mark.parametrize(("text", "expected", "kind"), CASES, ids=[c[0][:60] for c in CASES])
def test_labels(engine: SentinelEngine, text: str, expected: str, kind: str) -> None:
    a = engine.analyze(text, kind)
    assert a.label == expected, (a.score, a.drivers)


def _drivers(a: TextAnalysis) -> str:
    return " | ".join(term.lower() for term, _ in a.drivers)


def test_guidance_driver_names_the_metric(engine: SentinelEngine) -> None:
    a = engine.analyze("Company raises loss forecast")
    assert a.drivers[0][0] == "raises loss forecast" and a.drivers[0][1] < 0
    canonical = engine.evidence("Biotech raises its full-year net loss outlook again").hits[0].term
    assert canonical == "net loss outlook raised"


def test_capex_forecast_is_neither_good_nor_bad(engine: SentinelEngine) -> None:
    a = engine.analyze("Acme Oil cuts full-year capex forecast by 30%")
    assert a.label == "neutral" and not a.drivers


def test_negated_rule_driver_keeps_the_negator(engine: SentinelEngine) -> None:
    ev = engine.evidence("Analysts say Acme did not beat estimates this time")
    assert any(h.value < 0 for h in ev.hits)


def test_target_the_retailer_never_becomes_a_price_target(engine: SentinelEngine) -> None:
    for text in ("Walmart raises wages to compete with Target", "Target cuts prices on thousands of products",
                 "Geode Capital Management LLC Increases Position in Target Co. (NYSE:TGT)"):
        assert "target" not in _drivers(engine.analyze(text)), text
    jobs = engine.analyze("Target to cut 1,800 jobs")
    assert "jobs" in _drivers(jobs)


def test_record_fines_and_dividend_dates_carry_no_record_driver(engine: SentinelEngine) -> None:
    for text in ("Meta hit with record $1.3 billion EU fine",
                 "AT&T declares dividend, payable to stockholders of record on January 10"):
        assert all(not term.lower() == "record" for term, _ in engine.analyze(text).drivers), text


def test_signed_percent_driver_names_the_metric(engine: SentinelEngine) -> None:
    a = engine.analyze("VIX +15%")
    assert a.drivers[0][0] == "VIX +15%" and a.drivers[0][1] < 0
    ref = engine.analyze("Inflation +3.2% y/y vs +3.0% expected")
    assert [t for t, _ in ref.drivers] == ["Inflation +3.2%"]  # the expectation is not a move


def test_why_questions_keep_their_premise_but_yes_no_questions_do_not(engine: SentinelEngine) -> None:
    why = engine.analyze("Why Is Nvidia Stock Down Today?")
    stated = engine.analyze("Nvidia stock down today")
    asked = engine.analyze("Is Nvidia stock down today?")
    assert stated.score < why.score < asked.score <= 0


def test_lexicon_counts_are_honest() -> None:
    stats = lx.lexicon_stats()
    assert lx.lexicon_size() == stats["entries"]
    assert stats["lemma_groups"] <= stats["entries"] < stats["with_inflections"]
    assert stats["valence_entries"] >= 600
    # verbs are listed once and inflected by helpers: the generated forms are not entries
    assert "plunged" in lx.DIRECTIONS and "plunged" in lx._GENERATED
