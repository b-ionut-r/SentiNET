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
    # capped upside is a bearish argument; capped downside a bullish one
    ("Morgan Stanley lowers Acme stock price target on limited upside", "bearish", "news"),
    ("Acme Stock's Upside May Be Limited as AI Agents Could Disrupt Business", "bearish", "news"),
    ("Analyst sees limited upside for Acme", "bearish", "news"),
    ("Acme shares rose sharply after the update", "bullish", "news"),
    # Form-4 pay and plan transactions are not trading decisions
    ("Acme (ACM) COO acquires 512 stock units at $24.40 each through payroll deductions.", "neutral", "news"),
    ("Acme (ACM) executive acquires 145 benefit-plan shares and has 9,248 shares withheld for taxes.", "neutral",
     "news"),
    ("Acme CFO exercises options and sells 20,000 shares", "neutral", "news"),
    ("Acme CEO buys 100,000 shares in open market purchase", "bullish", "news"),
    ("Acme director bought $2 million of stock", "bullish", "news"),
    ("Acme CEO sells 50,000 shares", "bearish", "news"),
    # regulators and enforcement
    ("SEC approves spot bitcoin ETFs", "bullish", "news"),
    ("SEC charges Acme executives with fraud", "bearish", "news"),
    ("DOJ sues Acme over pricing", "bearish", "news"),
    ("SEC drops charges against Acme", "bullish", "news"),
    ("CFTC approves one final, two proposed rules at open meeting", "neutral", "news"),
    ("FAA weighs plane approval revamp", "neutral", "news"),
    ("Acme 737 jets grounded after incident", "bearish", "news"),
    # destruction takes its sign from what is destroyed
    ("Acme's 45% Rally Wiped Out as Hopes for an AI Windfall Fizzle", "bearish", "news"),
    ("Acme's 45% rally wiped out as hopes fade", "bearish", "news"),
    ("Acme wipes out last year's losses", "bullish", "news"),
    ("Selloff wipes out $1 trillion in market value", "bearish", "news"),
    ("Acme cuts $2 billion in costs", "bullish", "news"),
    ("Fed adds $90 billion in cash to money markets", "neutral", "news"),
    # "to <rating word>" is a rating change only when a rating phrase ends there
    ("Company takes steps to reduce debt", "bullish", "news"),
    ("Chipmaker cuts prices to add market share", "neutral", "news"),
    ("Acme cuts jobs to reduce costs", "bearish", "news"),
    ("Founder offers to buy prepaid brand", "neutral", "news"),
    ("Acme upgraded to Buy at Goldman", "bullish", "news"),
    ("Acme cut to Neutral; price target lowered", "bearish", "news"),
    ("Analyst moves Acme to Sell", "bearish", "news"),
    # the other camp, idioms and emoji in posts
    ("$ACME can't stop won't stop 🚀", "bullish", "social"),
    ("$ACME going to zero", "bearish", "social"),
    ("$ACME dead money", "bearish", "social"),
    ("$ACME is cooked", "bearish", "social"),
    ("$ACME bears getting destroyed", "bullish", "social"),
    ("$ACME bulls getting destroyed", "bearish", "social"),
    ("$ACME $SPY $QQQ Black Monday coming boys and girls", "bearish", "social"),
    ("Fed cuts rates to zero", "bullish", "news"),
    # containment, pauses, 13F bots and market background
    ("Acme Q3 sales fall 6.6%, says disruption contained within guidance", "bearish", "news"),
    ("Acme says costs contained", "bullish", "news"),
    ("Stocks and bonds pause ahead of data", "neutral", "news"),
    ("Rally pauses as investors take profits", "bearish", "news"),
    ("Vanguard Group Inc. Raises Stake in Acme", "neutral", "news"),
    ("State Street Corp Cuts Stake in Acme", "neutral", "news"),
    ("Geode Capital Management LLC Sells 1,234 Shares of Acme Inc. (NYSE:ACM)", "neutral", "news"),
    ("Holding company trims its stake in Acme", "bearish", "news"),
    ("Acme (ACM) Stock Dips While Market Gains", "bearish", "news"),
    ("Oil prices, which surged last week on supply fears, slide to 3-month low", "bearish", "news"),
    # trader words only count as positions in trading talk
    ("I've been using this for a long time and it calls the API twice", "neutral", "social"),
    ("Long story short, the board calls for a vote", "neutral", "social"),
    ("As long as rates stay where they are it's fine", "neutral", "social"),
    ("Long $TSLA here", "bullish", "social"),
    ("bought more calls today $NVDA", "bullish", "social"),
    ("loading puts on $SPY", "bearish", "social"),
    ("$CCL doubling down short", "bearish", "social"),
    # final review: a distance below a high is a drawdown; the "high" is only the reference point
    ("Acme stock trades 58% below its 52-week high", "bearish", "news"),
    ("Acme Stock Is 80% Below Its All-Time High: 1 Metric That Shows How Bearish Wall Street Has Become.",
     "bearish", "news"),
    ("FTSE drops Acme: Acme Athletica stock sits 58.37 percent below its high", "bearish", "news"),
    ("Acme is now 45% off its highs", "bearish", "news"),
    ("Acme trades 20% below its IPO price", "bearish", "news"),
    ("Acme Stock Is Just 3% Below Its All-Time High", "bullish", "news"),  # near-high framing
    ("Acme shares have fallen 40% from their record high", "bearish", "news"),
    # valuation calls
    ("Acme (ACME) Could Be 30% Above Fair Value On Product Buzz", "bearish", "news"),
    ("Acme stock 15% below fair value, analyst says", "bullish", "news"),
    ("Broker says Acme's AI rally prices in far more than the numbers support", "bearish", "news"),
    ("Acme rally has gone too far, warns strategist", "bearish", "news"),
    ("The selloff in Acme has gone too far, says strategist", "bullish", "news"),
    # a move takes its sign from what moved, also for quantities that are bad news when they grow
    ("Acme recalls surge 50% after software glitch", "bearish", "news"),
    ("Token Withdrawal Queue Surges 392%: Wallet Incident Could Trigger 523,000 Token Withdrawals", "bearish",
     "news"),
    ("Fund redemptions jump 30%", "bearish", "news"),
    ("Crypto liquidations surge past $1 billion", "bearish", "news"),
    ("Validator entry queue surges as staking demand returns", "bullish", "news"),
    ("Acme expands recall to 500,000 trucks", "bearish", "news"),
    # rating headlines without a change (MarketBeat templates)
    ('Acme Motor Corporation (NYSE:ACM) Stock Now Rated "Buy" by Wall Street Analysts', "bullish", "news"),
    ('Acme Platforms Earns "Buy" Rating from Needham', "bullish", "news"),
    ('Acme Platforms Given Consensus Rating of "Moderate Buy" by Brokerages', "bullish", "news"),
    ('Acme Given Average Rating of "Reduce" by Analysts', "bearish", "news"),
    ("Acme Has a Consensus Rating of Hold", "neutral", "news"),
    ("Top-rated buy-and-hold stocks for 2026", "neutral", "news"),
    # "top"/"best" after a possessive or determiner are adjectives, not a beat
    ("Here are Wall Street's top analyst calls", "neutral", "news"),
    ("Acme tops analysts' estimates", "bullish", "news"),
    # chart levels: a support level is a price, not "support"; holding it is a non-move
    ("Acme Coin Holds Key Support Levels", "neutral", "news"),
    ("Acme coin falls below key support at $60,000", "bearish", "news"),
    ("Acme coin breaks key support", "bearish", "news"),
    # a rally that cracks has ended
    ("Acme's AI rally cracked", "bearish", "news"),
    ("Chip rally crumbles", "bearish", "news"),
    ("Selloff falters as buyers return", "bullish", "news"),
    # "Bros." is an abbreviation, and "Warner Bros. Discovery" a name, not a find
    ("Warner Bros. Discovery to be acquired; terms not disclosed", "neutral", "news"),
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


def test_limited_upside_is_a_negative_driver_and_does_not_shrink_the_cut(engine: SentinelEngine) -> None:
    a = engine.analyze("Morgan Stanley lowers Acme stock price target on limited upside")
    drivers = dict(a.drivers)
    assert drivers["limited upside"] < 0
    plain = engine.analyze("Morgan Stanley lowers Acme stock price target")
    assert a.score < plain.score  # the trailing adjective no longer diminishes the cut


def test_insider_plan_transactions_carry_no_driver(engine: SentinelEngine) -> None:
    a = engine.analyze("Acme (ACM) COO acquires 512 stock units at $24.40 each through payroll deductions.")
    assert a.drivers == [] and a.score == 0.0
    planned = engine.analyze("Acme CEO sells 50,000 shares under 10b5-1 plan")
    unplanned = engine.analyze("Acme CEO sells 50,000 shares")
    assert unplanned.score < planned.score <= 0


def test_wiped_out_driver_names_what_was_wiped(engine: SentinelEngine) -> None:
    a = engine.analyze("Acme's 45% rally wiped out as hopes fade")
    assert a.drivers[0][0].lower().endswith("rally wiped out") and a.drivers[0][1] < 0


def test_level_drivers_keep_their_qualifier_and_drop_generic_nouns(engine: SentinelEngine) -> None:
    a = engine.analyze("$ACME Stocks that break through 52 week highs", "social")
    assert a.drivers[0][0] == "52 week highs"
    assert all(term.lower() not in ("stocks", "stocks highs") for term, _ in a.drivers)


def test_relative_clause_is_background_not_the_subject(engine: SentinelEngine) -> None:
    a = engine.analyze("Oil prices, which surged last week on supply fears, slide to 3-month low")
    assert a.label == "bearish"
    assert not any("fears, slide" in term for term, _ in a.drivers)


def test_neutral_confidence_separates_found_nothing_from_said_neutral(engine: SentinelEngine) -> None:
    nothing = engine.analyze("Acme to present at investor conference on Tuesday")
    said = engine.analyze("Acme quarterly results in line with estimates")
    assert nothing.label == said.label == "neutral"
    assert nothing.confidence == pytest.approx(0.5, abs=0.05)  # a default, not a finding
    assert said.confidence > nothing.confidence + 0.15


def test_news_polar_confidence_map_is_monotone_and_bounded() -> None:
    from app.nlp.engine import NEWS_POLAR_CALIBRATION, _interpolate

    xs = [i / 100 for i in range(5, 99)]
    ys = [_interpolate(x, NEWS_POLAR_CALIBRATION) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:], strict=False))
    assert 0.0 < ys[0] and ys[-1] <= 0.9


def test_lexicon_counts_are_honest() -> None:
    stats = lx.lexicon_stats()
    assert lx.lexicon_size() == stats["entries"]
    assert stats["lemma_groups"] <= stats["entries"] < stats["with_inflections"]
    assert stats["valence_entries"] >= 600
    # verbs are listed once and inflected by helpers: the generated forms are not entries
    assert "plunged" in lx.DIRECTIONS and "plunged" in lx._GENERATED


def test_distance_from_a_high_has_one_negative_driver(engine: SentinelEngine) -> None:
    for text in ("Acme stock trades 58% below its 52-week high", "Acme Stock Is 80% Below Its All-Time High"):
        a = engine.analyze(text)
        assert a.drivers and all(v < 0 for _, v in a.drivers), (text, a.drivers)  # no "52-week high" +
    deeper = engine.analyze("Acme stock trades 80% below its record high").score
    assert deeper < engine.analyze("Acme stock trades 12% below its record high").score < 0


def test_rating_templates_name_the_rating_and_moderate_is_not_a_move(engine: SentinelEngine) -> None:
    a = engine.analyze('Acme Platforms Given Consensus Rating of "Moderate Buy" by Brokerages')
    assert [t for t, _ in a.drivers] == ['Given Consensus Rating of "Moderate Buy"']
    direct = engine.analyze('Acme Platforms Earns "Buy" Rating from Needham')
    assert direct.drivers[0] == ('Earns "Buy" Rating', direct.drivers[0][1]) and direct.drivers[0][1] > 0
    assert direct.score > a.score > 0  # one broker's stance outweighs a bot's consensus summary
    assert engine.analyze("Acme Has a Moderate Buy rating").score >= 0


def test_negative_quantity_subjects_flip_the_move(engine: SentinelEngine) -> None:
    a = engine.analyze("Acme recalls surge 50% after software glitch")
    assert a.drivers[0][0] == "recalls surge 50%" and a.drivers[0][1] < 0
    assert not any(v > 0 for _, v in a.drivers)
    assert engine.analyze("Acme recalls 1 million vehicles").label == "bearish"  # the verb keeps its sense


def test_sitting_out_a_move_is_mild_and_signed_by_what_was_missed(engine: SentinelEngine) -> None:
    missed = engine.evidence("Acme and Widget Sit Out the Fintech Rally").hits
    avoided = engine.evidence("Acme sat out the selloff").hits
    assert [h.source for h in missed] == ["rule:sit_out"] and missed[0].value < 0
    assert [h.source for h in avoided] == ["rule:sit_out"] and avoided[0].value > 0
    assert abs(missed[0].value) < 1.0  # weaker than any actual move
    # an estimate miss is not "missing the rally"
    assert engine.evidence("Acme misses estimates as gains in services slow").hits[0].source == "rule:miss"


def test_holding_steady_is_a_stated_non_move(engine: SentinelEngine) -> None:
    a = engine.analyze("Acme shares hold steady")
    assert a.label == "neutral" and not a.drivers and a.confidence >= 0.7


def test_chart_support_is_a_level_not_praise(engine: SentinelEngine) -> None:
    held = engine.analyze("Acme Coin Holds Key Support")
    assert held.label == "neutral" and not held.drivers and held.confidence >= 0.7  # said, not defaulted
    broke = engine.analyze("Acme coin falls below key support at $60,000")
    assert broke.drivers and all(v < 0 for _, v in broke.drivers), broke.drivers


def test_bros_abbreviation_does_not_end_the_sentence() -> None:
    from app.nlp.rules import normalize

    assert normalize("Replacing Warner Bros. Discovery") == "Replacing Warner Bros Discovery"
    assert not SentinelEngine().analyze("Moderna to Join Nasdaq-100, Replacing Warner Bros. Discovery").drivers[1:]

