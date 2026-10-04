"""Sentinel engine: hand-written finance cases + engine contract.

The labelled cases cover the constructions the engine must get right in market
text: analyst actions (direction from ratings/numbers), earnings vs. estimates,
guidance, price moves with magnitude, compositional metrics ("loss narrowed"),
negation, contrast/concession, hedges/questions/listicles, legal and corporate
events, macro, the social register, and neutral "false friends".
"""
from __future__ import annotations

import time

import pytest

from app.nlp.engine import NEUTRAL_BAND, SentinelEngine, VaderEngine, label_for
from app.nlp.types import TextAnalysis

CASES: list[tuple[str, str, str]] = [
    # analyst actions
    ("Morgan Stanley upgrades Nvidia to Overweight", "bullish", "news"),
    ("Goldman downgrades Apple to Sell", "bearish", "news"),
    ("Apple cut to Neutral from Buy at Goldman", "bearish", "news"),
    ("Tesla raised to Buy from Hold at Jefferies", "bullish", "news"),
    ("Intelsat downgraded to underweight from neutral at J.P. Morgan", "bearish", "news"),
    ("Audentes Therapeutics upgraded to outperform from in line at Evercore ISI", "bullish", "news"),
    ("$ESS: BTIG Research cuts to Neutral", "bearish", "news"),
    ("Citi initiates coverage of Snowflake with a Buy rating", "bullish", "news"),
    ("Patterson-UTI Energy started at sell at Deutsche Bank", "bearish", "news"),
    ("Barclays initiates Ford at Equal Weight", "neutral", "news"),
    ("UBS reiterates Buy on Microsoft", "bullish", "news"),
    ("Jefferies maintains Hold rating on Intel", "neutral", "news"),
    ("Morgan Stanley raises Nvidia price target to $250 from $220", "bullish", "news"),
    ("Citi cuts Nike price target to $90 from $110", "bearish", "news"),
    ("Wedbush lowers its price target on Rivian", "bearish", "news"),
    ("Altria stock price target raised to $54 from $50 at Stifel Nicolaus", "bullish", "news"),
    ("Analysts slash Boeing price target after 737 MAX grounding", "bearish", "news"),
    ("KeyBanc price target on Snap to $14 from $18", "bearish", "news"),
    ("Piper Sandler bumps Amazon target to $210", "bullish", "news"),
    ("Hedge Funds Are Dumping Lumber Liquidators Holdings", "bearish", "news"),
    ("Hedge Funds Have Never Been This Bullish On Apple", "bullish", "news"),
    ("Hedge Funds Have Never Been Less Bullish On Westlake Chemical", "bearish", "news"),
    ("Insiders are buying shares of Pfizer", "bullish", "news"),
    ("Spotify double downgrade reflects slower revenue growth", "bearish", "news"),
    # earnings
    ("Nvidia beats estimates on record data-center revenue", "bullish", "news"),
    ("CIGNA EPS beats by $0.11, beats on revenue", "bullish", "news"),
    ("Bellerophon Therapeutics EPS misses by $0.33", "bearish", "news"),
    ("Sanofi EPS beats by €0.06, misses on revenue", "neutral", "news"),
    ("Macy's quarterly sales fall short of estimates", "bearish", "news"),
    ("AbbVie Reports Estimate-Beating Fourth-Quarter Earnings Results", "bullish", "news"),
    ("Twitter tops expectations with first $1 billion quarterly revenue", "bullish", "news"),
    ("Results came in line with expectations", "neutral", "news"),
    ("Netflix subscriber growth disappoints; shares slide 8%", "bearish", "news"),
    ("Tyson Foods Q1 adj. EPS $1.66 vs. $1.58 a year ago; FactSet consensus $1.63", "bullish", "news"),
    ("Macy's Q3 sales $5.173 bln vs. $5.404 bln; FactSet consensus $5.321 bln", "bearish", "news"),
    ("Kellogg Q4 EPS 42 cents vs. per-share loss 24 cents a year ago", "bullish", "news"),
    ("Better-than-expected iPhone sales lift Apple shares", "bullish", "news"),
    ("Worse-than-expected margins hit Target", "bearish", "news"),
    ("Hero MotoCorp profit beats estimates despite lower sales", "bullish", "news"),
    ("The company beat estimates but guided below consensus", "bearish", "news"),
    ("Company X reports net loss of $5 million", "bearish", "news"),
    ("GameStop net sales decrease of 25.7% in the holiday quarter", "bearish", "news"),
    ("$PHAT - Phathom Pharmaceuticals EPS of -$9.30", "bearish", "news"),
    ("Smaller-than-expected loss sends shares higher", "bullish", "news"),
    # guidance
    ("Apple raises full-year guidance", "bullish", "news"),
    ("Intel cuts 2024 revenue forecast", "bearish", "news"),
    ("Matson withdraws its financial outlook for the full year 2020", "bearish", "news"),
    ("Greenbrier beats by $0.20, misses on revs; suspending FY20 guidance", "bearish", "news"),
    ("Despite strong sales, the company cut its full-year guidance", "bearish", "news"),
    ("$BYND - JPMorgan reels in expectations on Beyond Meat", "bearish", "news"),
    ("Oppenheimer cuts estimates on Yum China", "bearish", "news"),
    ("Retailer guides fourth-quarter sales below consensus", "bearish", "news"),
    ("Company boosts annual profit outlook after strong quarter", "bullish", "news"),
    # price moves and levels
    ("Uber Shares Tumble 6% In Premarket As London Again Revokes Operating License", "bearish", "news"),
    ("Tesla stock soars 20% after record deliveries", "bullish", "news"),
    ("Nvidia shares fall 12% after earnings miss", "bearish", "news"),
    ("Aurora Cannabis stock slides 16% premarket", "bearish", "news"),
    ("Why Diabetes Stock Insulet Jumped 13.3% in January", "bullish", "news"),
    ("$EBAY - EBay +6% after selling StubHub", "bullish", "news"),
    ("$VRNS - Varonis -6% on wider-than-expected loss forecast", "bearish", "news"),
    ("Dow up 300 points as tech stocks rally", "bullish", "news"),
    ("S&P 500 sinks 3% in worst day since March", "bearish", "news"),
    ("Stocks give up gains", "bearish", "news"),
    ("Wall Street slips from record highs at open", "bearish", "news"),
    ("AMD stock hits highest level in 13 years", "bullish", "news"),
    ("Peloton shares hit a 52-week low", "bearish", "news"),
    ("Futures dip from all-time highs", "bearish", "news"),
    ("Stocks close at record highs", "bullish", "news"),
    ("Investors Who Bought Lendlease Shares A Year Ago Are Now Up 53%", "bullish", "news"),
    ("Fossil Has Lost Half of Its Value This Year", "bearish", "news"),
    ("Dow futures pare gains as Home Depot's stock sinks 5%", "bearish", "news"),
    ("Wall Street bounces back from a steep sell-off", "bullish", "news"),
    # compositional metrics (release language)
    ("Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007", "bullish", "news"),
    ("Net sales decreased to EUR 3.2 mn from EUR 4.5 mn", "bearish", "news"),
    ("Operating loss narrowed to EUR 1.2 mn from EUR 3.4 mn", "bullish", "news"),
    ("The company's net loss widened to EUR 5 mn", "bearish", "news"),
    ("Costs rose sharply in the quarter", "bearish", "news"),
    ("The company cut costs by 15%", "bullish", "news"),
    ("Operating expenses decreased 8% year-on-year", "bullish", "news"),
    ("Gross margin improved to 42% from 39%", "bullish", "news"),
    ("Net debt fell to its lowest level in a decade", "bullish", "news"),
    ("Unemployment rate falls to record low", "bullish", "news"),
    ("Jobless claims rise more than expected", "bearish", "news"),
    ("Inflation eases for a third straight month", "bullish", "news"),
    ("Pre-tax loss totalled EUR 0.3 mn compared to a loss of EUR 2.2 mn", "bullish", "news"),
    ("It moved to an operating profit of EUR 2 mn from a loss of EUR 1 mn", "bullish", "news"),
    ("Componenta's net sales doubled to EUR131m from EUR76m", "bullish", "news"),
    ("Order intake grew by 23% to EUR 450 mn", "bullish", "news"),
    ("Higher raw material costs weighed on margins", "bearish", "news"),
    ("Lower interest rates boosted housing demand", "bullish", "news"),
    ("Operating profit, excluding non-recurring items, rose to EUR 5.1 mn", "bullish", "news"),
    ("Sales growth slowed to 2% in the third quarter", "bearish", "news"),
    ("U.S. Claims for Jobless Benefits Fall to Lowest Since April", "bullish", "news"),
    # negation
    ("Exxon Mobil found not guilty of fraud in climate case", "bullish", "news"),
    ("Coronavirus won't hinder PayPal's growth", "bullish", "news"),
    ("Fed's Rosengren says U.S. unlikely to have economic downturn in 2020", "bullish", "news"),
    ("Why NIO Hasn't Fallen Totally Off the Cliff", "bullish", "news"),
    ("Kraft Heinz: Why Dividend Investors Should Not Buy", "bearish", "news"),
    ("Airline avoids bankruptcy with new financing deal", "bullish", "news"),
    ("No signs of a slowdown in demand, CEO says", "bullish", "news"),
    ("Revenue did not meet expectations", "bearish", "news"),
    ("Margins are no longer improving", "bearish", "news"),
    ("Investors shrug off weak data as stocks climb", "bullish", "news"),
    # contrast / concession / background
    ("Revenue grew, but margins collapsed", "bearish", "news"),
    ("Despite the lawsuit, Apple shares rallied", "bullish", "news"),
    ("Markets trending lower despite positive jobs report", "bearish", "news"),
    ("Although sales rose, the company swung to a loss", "bearish", "news"),
    ("Stocks rally even as recession fears linger", "bullish", "news"),
    # hedges, questions, listicles
    ("Is Chimerix Inc (CMRX) A Good Stock To Buy?", "neutral", "news"),
    ("3 Top Dividend Stocks to Buy Now", "neutral", "news"),
    ("Should You Buy Tesla Before Earnings?", "neutral", "news"),
    ("Apple reportedly plans to cut 1,000 jobs", "bearish", "news"),
    # legal, regulatory, corporate events
    ("SEC opens probe into Vale over dam disaster", "bearish", "news"),
    ("Germany Fines Steelmakers 646 Million Euros for Price Fixing", "bearish", "news"),
    ("Judge dismisses shareholder lawsuit against Meta", "bullish", "news"),
    ("China suspends planned tariffs on some U.S. goods", "bullish", "news"),
    ("U.S. lifts sanctions on aluminum giant Rusal", "bullish", "news"),
    ("Retailer files for Chapter 11 bankruptcy protection", "bearish", "news"),
    ("Auditor raises substantial doubt about its ability to continue as a going concern", "bearish", "news"),
    ("Nasdaq sends delisting notice to Luckin Coffee", "bearish", "news"),
    ("Short seller Hindenburg Research targets Adani Group", "bearish", "news"),
    ("Danaos Corporation announces $55M equity offering", "bearish", "news"),
    ("Ardagh prices $500 million senior secured notes offering", "neutral", "news"),
    ("Data breach exposes records of 100 million customers", "bearish", "news"),
    ("FDA approves Vertex cystic fibrosis drug", "bullish", "news"),
    ("FDA issues complete response letter for Spectrum's lung cancer drug", "bearish", "news"),
    ("Toyota recalls 1 million vehicles over airbag defect", "bearish", "news"),
    ("Amazon to create 500 new jobs in Florida", "bullish", "news"),
    ("Lowe's says it will shut 34 stores in Canada", "bearish", "news"),
    ("Walmart plans to open 500 new stores in China", "bullish", "news"),
    ("Intel to lay off 15% of its workforce", "bearish", "news"),
    ("Board approves $10 billion share buyback", "bullish", "news"),
    ("Company raises quarterly dividend by 10%", "bullish", "news"),
    ("Carnival suspends its dividend", "bearish", "news"),
    ("$KRC - Kilroy Realty declares $0.485 dividend", "neutral", "news"),
    ("KBR wins $23.7M Army contract", "bullish", "news"),
    # macro
    ("Fed cuts interest rates by 50 basis points", "bullish", "news"),
    ("Fed signals more rate hikes ahead as inflation persists", "bearish", "news"),
    ("Recession fears ease as GDP growth beats forecasts", "bullish", "news"),
    ("Trade war escalates as U.S. slaps tariffs on Chinese goods", "bearish", "news"),
    ("Consumer confidence falls to 3-year low", "bearish", "news"),
    ("Oil prices plunge on supply glut", "bearish", "news"),
    ("Economic volatility is at a record low", "bullish", "news"),
    ("Probability of a recession pretty low: strategist", "bullish", "news"),
    # social register
    ("$TSLA to the moon 🚀🚀🚀", "bullish", "social"),
    ("bought more calls, diamond hands 💎🙌", "bullish", "social"),
    ("loaded up on puts, this is going to dump hard", "bearish", "social"),
    ("$GME rug pull incoming, bagholders getting rekt", "bearish", "social"),
    ("short squeeze is on, shorts are trapped", "bullish", "social"),
    ("Bought the dip on $AMD", "bullish", "social"),
    ("$SPY bull trap, be careful", "bearish", "social"),
    ("this stock is garbage, selling everything", "bearish", "social"),
    ("Long $WMT", "bullish", "social"),
    ("lol what a dumpster fire 🤡", "bearish", "social"),
    ("$NVDA ripping today, new ATH", "bullish", "social"),
    ("Gap down Monday, been bleeding all week", "bearish", "social"),
    ("$AAPL earnings tomorrow, what are your plays?", "neutral", "social"),
    ("$RIOT We goin way way up", "bullish", "social"),
    ("Bears you know what to do 😂", "neutral", "social"),
    ("$HOOD red to green move incoming", "bullish", "social"),
    # neutral news (false friends)
    ("Apple to hold annual shareholder meeting on March 1", "neutral", "news"),
    ("Fed Chair Powell speaks at Jackson Hole", "neutral", "news"),
    ("Microsoft names new head of cloud division", "neutral", "news"),
    ("Netflix is spending $420 million to produce more local content in India", "neutral", "news"),
    ("Best Buy to report earnings next week", "neutral", "news"),
    ("Shares outstanding totaled 1.2 billion at quarter end", "neutral", "news"),
    ("The dividend record date is May 5", "neutral", "news"),
    ("Rocket Lab schedules its next Electron launch", "neutral", "news"),
    ("The value of the order is EUR 2.5 mn", "neutral", "news"),
    ("According to Gran, the company has no plans to move all production to Russia", "neutral", "news"),
    ("Advanced Micro Devices to present at investor conference", "neutral", "news"),
    ("The company was established in 1995 and is headquartered in Helsinki", "neutral", "news"),
    ("Shares were little changed in early trading", "neutral", "news"),
    ("Prices for the new models range from $799 to $1,099", "neutral", "news"),
    ("Merck's vericiguat studied in heart failure patients", "neutral", "news"),
]


@pytest.fixture(scope="module")
def engine() -> SentinelEngine:
    return SentinelEngine()


@pytest.mark.parametrize(("text", "expected", "kind"), CASES, ids=[c[0][:60] for c in CASES])
def test_labels(engine: SentinelEngine, text: str, expected: str, kind: str) -> None:
    result = engine.analyze(text, kind)
    assert result.label == expected, (result.score, result.drivers)


def test_case_count_and_balance() -> None:
    assert len(CASES) >= 120
    labels = {label for _, label, _ in CASES}
    assert labels == {"bullish", "bearish", "neutral"}


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_neutral_band_and_label_for() -> None:
    assert NEUTRAL_BAND == 0.05
    assert label_for(0.0) == "neutral"
    assert label_for(NEUTRAL_BAND) == "neutral"
    assert label_for(-NEUTRAL_BAND) == "neutral"
    assert label_for(0.0501) == "bullish"
    assert label_for(-0.0501) == "bearish"
    assert label_for(1.0) == "bullish" and label_for(-1.0) == "bearish"


def test_score_returns_one_result_per_text(engine: SentinelEngine) -> None:
    texts = ["Apple beats estimates", "", "   ", "Tesla shares plunge 12%"]
    out = engine.score(texts)
    assert len(out) == len(texts)
    assert all(isinstance(a, TextAnalysis) for a in out)
    assert out[1].label == "neutral" and out[1].score == 0.0 and out[1].confidence == 0.0
    assert engine.score([]) == []


def test_kinds_may_be_short_or_none(engine: SentinelEngine) -> None:
    texts = ["calls printing 🚀", "calls printing 🚀"]
    a, b = engine.score(texts, ["social"])  # second falls back to the news register
    assert a.label == "bullish"
    assert b.score <= a.score
    assert len(engine.score(texts, None)) == 2


def test_outputs_are_bounded_and_label_consistent(engine: SentinelEngine) -> None:
    for text, _, kind in CASES:
        a = engine.analyze(text, kind)
        assert -1.0 <= a.score <= 1.0
        assert 0.0 <= a.confidence <= 1.0
        assert a.label == label_for(a.score)
        assert a.themes == [] and a.events == []  # textintel fills these


def test_deterministic(engine: SentinelEngine) -> None:
    texts = [t for t, _, _ in CASES]
    first = [(a.score, a.label, a.confidence, a.drivers) for a in engine.score(texts)]
    again = [(a.score, a.label, a.confidence, a.drivers) for a in SentinelEngine().score(texts)]
    assert first == again


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #
def test_drivers_are_meaningful_terms(engine: SentinelEngine) -> None:
    a = engine.analyze("Morgan Stanley raises Nvidia price target to $250 from $220")
    assert a.drivers and a.drivers[0][0] == "price target raised to $250 from $220"
    assert a.drivers[0][1] > 0
    b = engine.analyze("China's Pinduoduo posts bigger loss as costs surge; shares tumble")
    terms = [t.lower() for t, _ in b.drivers]
    assert "costs surge" in terms and "shares tumble" in terms
    assert all(v < 0 for _, v in b.drivers)


def test_drivers_capped_signed_and_highlightable(engine: SentinelEngine) -> None:
    for text, label, kind in CASES:
        a = engine.analyze(text, kind)
        assert len(a.drivers) <= 5
        for term, impact in a.drivers:
            assert term.strip() and -1.0 <= impact <= 1.0
        if label != "neutral" and a.drivers:
            top_sign = 1 if a.drivers[0][1] > 0 else -1
            assert top_sign == (1 if label == "bullish" else -1) or len(a.drivers) > 1, (text, a.drivers)
        short = [t for t, _ in a.drivers if len(t.split()) <= 3 and "$" not in t]
        normalized = " ".join(text.replace("-", " ").lower().split())
        for term in short:  # short drivers are verbatim spans the UI can highlight
            assert " ".join(term.replace("-", " ").lower().split()) in normalized or term in (
                "beats", "miss", "beat", "misses", "job cuts", "legal relief", "fine"), (term, text)


def test_negated_driver_includes_negator(engine: SentinelEngine) -> None:
    a = engine.analyze("Exxon Mobil found not guilty of fraud")
    assert any(t.lower().startswith("not guilty") for t, _ in a.drivers)
    assert a.label == "bullish"


def test_mixed_signals_are_neutral_but_explained(engine: SentinelEngine) -> None:
    a = engine.analyze("Sanofi EPS beats by €0.06, misses on revenue")
    assert a.label == "neutral"
    signs = {v > 0 for _, v in a.drivers}
    assert signs == {True, False}


# --------------------------------------------------------------------------- #
# Magnitude, confidence, modifiers
# --------------------------------------------------------------------------- #
def test_move_magnitude_scales_with_percent(engine: SentinelEngine) -> None:
    big = engine.analyze("Nvidia shares plunge 25% after guidance cut").score
    mid = engine.analyze("Nvidia shares fall 8%").score
    small = engine.analyze("Nvidia shares fall 0.5%").score
    assert big < mid < small <= 0


def test_numbers_decide_price_target_direction(engine: SentinelEngine) -> None:
    up = engine.analyze("Stifel price target on Altria to $54 from $50")
    down = engine.analyze("Stifel price target on Altria to $46 from $50")
    assert up.label == "bullish" and down.label == "bearish"


def test_confidence_reflects_evidence_strength(engine: SentinelEngine) -> None:
    strong = engine.analyze("Apple beats estimates and raises full-year guidance; shares jump 8%")
    weak = engine.analyze("Apple shares edge higher")
    assert strong.confidence > weak.confidence
    assert strong.score > weak.score


def test_hedges_and_questions_dampen(engine: SentinelEngine) -> None:
    plain = engine.analyze("Apple will cut 1,000 jobs")
    hedged = engine.analyze("Apple may cut 1,000 jobs, sources say")
    assert plain.score < hedged.score <= 0
    assert engine.analyze("Is Tesla a buy?").label == "neutral"
    statement = engine.analyze("Tesla stock is a strong buy, says Wedbush")
    assert statement.label == "bullish"


def test_contrast_weights_later_clause(engine: SentinelEngine) -> None:
    a = engine.analyze("Revenue grew, but margins collapsed")
    b = engine.analyze("Margins collapsed, but revenue grew")
    assert a.score < b.score


def test_concessive_clause_is_background(engine: SentinelEngine) -> None:
    a = engine.analyze("Despite weak guidance, shares rallied 6%")
    assert a.label == "bullish"
    b = engine.analyze("Despite strong sales, shares plunged 9%")
    assert b.label == "bearish"


def test_negation_flips_and_shrinks(engine: SentinelEngine) -> None:
    pos = engine.analyze("Demand is strong and growing")
    neg = engine.analyze("Demand is not strong")
    assert pos.score > 0 > neg.score
    assert abs(neg.score) < pos.score


def test_not_only_is_not_negation(engine: SentinelEngine) -> None:
    assert engine.analyze("Not only did revenue jump 20%, margins improved too").label == "bullish"


def test_composition_sign_follows_metric(engine: SentinelEngine) -> None:
    assert engine.analyze("Profit rose 12%").label == "bullish"
    assert engine.analyze("Costs rose 12%").label == "bearish"
    assert engine.analyze("Losses narrowed sharply").label == "bullish"
    assert engine.analyze("Losses widened sharply").label == "bearish"
    assert engine.analyze("The company eased investor concerns").score >= 0


def test_level_words_need_context(engine: SentinelEngine) -> None:
    assert engine.analyze("The high court will hear the case").label == "neutral"
    assert engine.analyze("Shares hit an all-time high").label == "bullish"
    assert engine.analyze("Shares retreat from all-time highs").label == "bearish"


def test_repeated_emoji_has_diminishing_returns(engine: SentinelEngine) -> None:
    one = engine.analyze("$TSLA 🚀", "social").score
    many = engine.analyze("$TSLA " + "🚀" * 12, "social").score
    assert many >= one
    assert many - one < 0.35


def test_cashtag_words_do_not_leak_into_sentiment(engine: SentinelEngine) -> None:
    # $RIOT / $FUN / $HOPE are tickers, not words
    assert engine.analyze("$RIOT earnings call at 4pm", "social").label == "neutral"
    assert engine.analyze("$FUN reports Q3 on Thursday", "news").label == "neutral"


# --------------------------------------------------------------------------- #
# Register
# --------------------------------------------------------------------------- #
def test_trader_slang_only_counts_in_social(engine: SentinelEngine) -> None:
    social = engine.analyze("buying calls on $AMD", "social")
    news = engine.analyze("Senator calls on AMD to explain chip exports", "news")
    assert social.label == "bullish"
    assert news.label == "neutral"


def test_other_camp_nouns_are_not_stance_in_social(engine: SentinelEngine) -> None:
    assert engine.analyze("bulls are going to be so mad today", "social").score <= 0.1
    assert engine.analyze("bears in shambles 🚀", "social").label == "bullish"


def test_news_kinds_share_the_news_register(engine: SentinelEngine) -> None:
    text = "Company beats estimates and raises guidance"
    scores = {k: engine.analyze(text, k).score for k in ("news", "analysis", "filing", None)}
    assert len(set(scores.values())) == 1


# --------------------------------------------------------------------------- #
# Baseline & speed
# --------------------------------------------------------------------------- #
def test_vader_baseline_engine() -> None:
    v = VaderEngine()
    a, b = v.score(["This is great!", ""])
    assert v.name == "vader" and a.label == "bullish" and b.label == "neutral"


def test_beats_vader_on_the_hand_written_cases(engine: SentinelEngine) -> None:
    vader = VaderEngine()
    texts = [t for t, _, _ in CASES]
    kinds = [k for _, _, k in CASES]
    gold = [g for _, g, _ in CASES]
    s_acc = sum(a.label == g for a, g in zip(engine.score(texts, kinds), gold)) / len(gold)
    v_acc = sum(a.label == g for a, g in zip(vader.score(texts, kinds), gold)) / len(gold)
    assert s_acc > v_acc + 0.3


def test_throughput(engine: SentinelEngine) -> None:
    texts = [t for t, _, _ in CASES] * 12
    engine.score(texts[:50])  # warm-up
    start = time.perf_counter()
    engine.score(texts)
    rate = len(texts) / (time.perf_counter() - start)
    # ~4-5k/s headlines single-threaded on a dev box; generous floor for shared CI runners
    assert rate > 1000, f"{rate:.0f} texts/s"
