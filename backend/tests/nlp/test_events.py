"""Event detection on real headline phrasing (Google News / StockTwits, 2026-10-04).

The analyst cases assert the exact firm and price target, because the
analytics layer turns them into "Morgan Stanley raised PT to $245" evidence.
"""
from __future__ import annotations

import time

import pytest

from app.nlp.events import (
    EVENT_LABELS,
    EVENT_POLARITY,
    EVENT_THEMES,
    canonical_firm,
    detect_events,
    is_known_firm,
)
from app.nlp.text import strip_publisher_suffix
from app.nlp.themes import THEMES
from tests.conftest import load_json_fixture


def _keys(text: str) -> list[str]:
    return [e.key for e in detect_events(text)]


def _one(text: str, key: str):
    matches = [e for e in detect_events(text) if e.key == key]
    assert matches, f"{key} not found in {text!r}: {detect_events(text)}"
    return matches[0]


# --------------------------------------------------------------------------- #
# Analyst actions: firm + target, direction from the numbers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("text", "key", "firm", "value"), [
    ("Okta Stock Rises After Morgan Stanley Raises Price Target to $245", "pt_raise", "Morgan Stanley", 245.0),
    ("RBC Raises Price Target on DexCom to $105 From $90, Keeps Outperform Rating", "pt_raise",
     "RBC Capital Markets", 105.0),
    ("Strategy Shares Rise 2.7% After Citi Raises Price Target to $240", "pt_raise", "Citi", 240.0),
    ("B. Riley Raises Price Target on OptimizeRx to $11 From $10, Keeps Buy Rating", "pt_raise", "B. Riley", 11.0),
    ("Update: BofA Securities Raises Price Target on Johnson & Johnson to $278 From $263, Keeps Neutral Rating",
     "pt_raise", "Bank of America", 278.0),
    ("BMO Capital Raises Price Target on Marathon Petroleum to $455 From $365, Keeps Outperform Rating", "pt_raise",
     "BMO Capital Markets", 455.0),
    ("Metlen shares rise after Wood lifts price target by 45%", "pt_raise", "Wood & Company", None),  # % is the change
    ("TD Synnex: Morgan Stanley lifts price target after strong quarter", "pt_raise", "Morgan Stanley", None),
    ("Redwire (RDW) Nudges Higher As BofA Lifts Price Target", "pt_raise", "Bank of America", None),
    # Nordic / MarketScreener formats: "to €67 (64)", "to SEK 315 (310)"
    ("SEB raises price target for Vaisala to €67 (64), reiterates buy", "pt_raise", "SEB", 67.0),
    ("SB1 Markets raises price target for Lime to SEK 315 (310), reiterates buy", "pt_raise", "SB1 Markets", 315.0),
    ("Jefferies raises price target for Huhtamäki to €32.25 (31), reiterates hold", "pt_raise", "Jefferies", 32.25),
    ("Goldman Sachs lowers price target for Trelleborg to 549 kronor (551), reiterates buy - BN", "pt_cut",
     "Goldman Sachs", 549.0),
    ("DNB Carnegie cuts price target for Green Landscaping to SEK24 (36), reiterates buy", "pt_cut", "DNB Carnegie",
     24.0),
    ("Mizuho Cuts Price Target on Westlake to $72 From $88, Keeps Neutral Rating", "pt_cut", "Mizuho", 72.0),
    # Moomoo/Futu template: a ticker tag glued to the name split the phrase (live AAPL)
    ("Morgan Stanley lowers Apple (AAPL.US) price target to $355: Roadmap is exciting, but valuation is full",
     "pt_cut", "Morgan Stanley", 355.0),
    ("Nike Stock In Focus As Barclays Cuts Price Target To $48 Ahead Of Q1 Earnings", "pt_cut", "Barclays", 48.0),
    ("Baird Cuts Price Target on Fair Isaac to $1,070 From $1,549, Keeps Outperform Rating", "pt_cut", "Baird", 1070.0),
    ("B. Riley Cuts Price Target on Canaan to $1.50 From $2, Keeps Buy Rating", "pt_cut", "B. Riley", 1.5),
    ("Akamai in focus as DA Davidson cuts price target (AKAM:NASDAQ)", "pt_cut", "D.A. Davidson", None),
    ("Oklo Price Target Slashed By UBS – But This Air Force Nuclear Project Offers A Key Catalyst", "pt_cut", "UBS",
     None),
    ("Northrop Grumman Sinks to 52-Week Low as RBC Cuts Rating, Slashes Price Target by $115", "pt_cut",
     "RBC Capital Markets", None),  # "$115" is the size of the cut, not the target
    # the verb says "raises" but the numbers say cut
    ("BofA Securities Raises Price Target on Virtu Financial to $69 From $72, Keeps Buy Rating", "pt_cut",
     "Bank of America", 69.0),
    ("HSBC Upgrades Target to Buy and Raises Price Target to $190", "analyst_upgrade", "HSBC", 190.0),
    ("BofA Securities Upgrades Grocery Outlet to Buy From Neutral, Raises Price Target to $15 From $12.50",
     "pt_raise", "Bank of America", 15.0),
    ("Wedbush Downgrades Foghorn Therapeutics to Neutral From Outperform, Cuts Price Target to $2 From $10",
     "analyst_downgrade", "Wedbush", 2.0),
    ("Freedom Capital downgrades AtriCure stock rating to hold, raises price target to $57", "analyst_downgrade",
     "Freedom Broker", 57.0),
    ("Goldman Sachs added Amazon to its conviction list with a $375 price target", "analyst_top_pick",
     "Goldman Sachs", 375.0),
    ("Nvidia Stock Gains After Morgan Stanley Names It a 'Top Semiconductor Pick'", "analyst_top_pick",
     "Morgan Stanley", None),
    ("$TGT HSBC upgraded Target to Buy from Hold today and raised its target to 190 from 125.", "pt_raise", "HSBC",
     190.0),
])
def test_analyst_actions_extract_firm_and_target(text, key, firm, value):
    event = _one(text, key)
    assert event.firm == firm
    assert event.value == value
    assert event.polarity == EVENT_POLARITY[key]


def test_initiations_take_polarity_from_the_rating():
    assert _one("Jefferies initiates coverage of Rivian with a Buy rating", "analyst_initiate").polarity == "bull"
    assert _one("Morgan Stanley initiates Snowflake at Underweight", "analyst_initiate").polarity == "bear"
    assert _one("Bernstein initiates coverage on Arm with Market Perform", "analyst_initiate").polarity == "neutral"


@pytest.mark.parametrize(("text", "firm", "value"), [
    # Moomoo/Futu template (live AAPL, Oct 3): the ticker tag glued to the name
    # ("Apple(AAPL.US)") split the phrase, so Citi's Buy/$365 initiation was lost.
    ("Citi Initiates Apple(AAPL.US) With Buy Rating, Announces Target Price $365", "Citi", 365.0),
    ("Citi Initiates Apple (AAPL.US) With Buy Rating, Announces Target Price $365", "Citi", 365.0),
    ("HSBC Initiates Target (NYSE: TGT) With Buy Rating", "HSBC", None),
    ("Jefferies Initiates Tencent (00700.HK) at Buy, Target Price HK$700", "Jefferies", 700.0),
])
def test_initiations_behind_ticker_tags(text, firm, value):
    event = _one(text, "analyst_initiate")
    assert (event.firm, event.value, event.polarity) == (firm, value, "bull")


@pytest.mark.parametrize("text", [
    "Target Cuts Prices On Nearly 2,000 Products Ahead Of Holidays: A Look At The Key Highlights $TGT $WMT",
    "Target Slashes Prices On Nearly 2K Items Amid Inflation Pressure",
    "Target hits sales target for the holiday quarter",
    "Apple's upgrade cycle could boost iPhone sales",
    "Verizon completes network upgrade in Chicago",
    "Moody's affirms AAR Corp.'s Ba2 rating, lowers outlook to negative on MRO deal",
])
def test_not_analyst_actions(text):
    assert not {"pt_raise", "pt_cut", "analyst_upgrade", "analyst_downgrade", "guidance_cut", "guidance_raise"} & set(
        _keys(text))


def test_firm_canonicalization():
    assert canonical_firm("BofA Securities") == "Bank of America"
    assert canonical_firm("B of A") == "Bank of America"
    assert canonical_firm("J.P. Morgan") == "JPMorgan"
    assert canonical_firm("Morgan Stanley's") == "Morgan Stanley"
    assert canonical_firm("Unknown Capital") == "Unknown Capital"
    assert is_known_firm("keybanc") and not is_known_firm("Acme")
    # common-word firm aliases match only with their capitalization
    assert _one("Benchmark raises Nvidia price target to $250", "pt_raise").firm == "Benchmark"
    assert not any(e.firm for e in detect_events("The benchmark index lifted its target date fund"))


# --------------------------------------------------------------------------- #
# Hypothetical vs. reported
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "Will Chevron (CVX) Beat Estimates Again in Its Next Earnings Report?",
    "Why Coca-Cola (KO) is Poised to Beat Earnings Estimates Again",
    "Constellation Brands' Pre-Q2 Earnings: Is It Poised to Beat Estimates?",
    "Can Nvidia beat estimates and lift the stock?",
    "Is a Microsoft Stock Split Coming After 23 Years?",
    "Stock-Split Watch: Is Meta Platforms Next?",
    "By 2001, His Amazon Shares Had Fallen 93% And Wall Street Was Predicting The Company Would Go Bankrupt.",
])
def test_hypotheticals_are_not_events(text):
    assert not {"earnings_beat", "stock_split", "bankruptcy"} & set(_keys(text))


@pytest.mark.parametrize(("text", "key"), [
    ("Should You Buy, Hold or Sell Costco Stock After Its Q4 Earnings Beat?", "earnings_beat"),
    ("Buy, Hold, or Fade Tesla Stock After Its Q3 Delivery Beat?", "earnings_beat"),
    ("Should You Bet on TMUS After Dividend Hike Amid Macro Uncertainty?", "dividend_raise"),
    ("Will Dividend Hike Change City Holding's (CHCO) Narrative", "dividend_raise"),
    ("Why Did AMD, HPE, MRNA Stocks Surge To 52-Week Highs Last Week?", "high_52w"),
])
def test_reported_events_inside_questions_count(text, key):
    assert key in _keys(text)


# --------------------------------------------------------------------------- #
# Other families on real phrasing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("text", "key", "polarity"), [
    ("Tesla (TSLA) Q3 2026 deliveries slip 2% to 486,532, but beat estimates", "earnings_beat", "bull"),
    ("Tesla's car business back on growth path as deliveries beat forecasts", "earnings_beat", "bull"),
    ("Nike plans more job cuts to boost sputtering turnaround, forecasts steep revenue drop", "guidance_cut", "bear"),
    ("Nike cuts full-year outlook as tariffs bite", "guidance_cut", "bear"),
    ("Walmart raises annual forecast after strong quarter", "guidance_raise", "bull"),
    ("NVIDIA Announces a $150 Billion Share Repurchase Authorization Increase", "buyback", "bull"),
    ("New Apple CEO John Ternus is reportedly planning layoffs as he looks to reshape the iPhone maker", "layoffs",
     "bear"),
    ("Apple hit with $5.7 billion jury verdict over haptic patents", "lawsuit", "bear"),
    ("Meta Faces Up To $40 Billion Penalty In New Mexico Privacy Case After Jury Finds Company Misled Users",
     "lawsuit", "bear"),
    ("AMD to acquire World Labs in $8.2B all-stock deal (AMD:NASDAQ)", "m_and_a", "neutral"),
    ("Paramount's $110 Billion Warner Bros. Deal Just Cleared Its Final Strategic Hurdle", "m_and_a", "neutral"),
    ("Meta taps MongoDB CEO Desai to drive enterprise AI push", "exec_hire", "neutral"),
    ("MongoDB CEO Chirantan Desai Steps Down to Join Meta as Chief Enterprise Platform Officer", "exec_departure",
     "bear"),
    ("ThetaRay Appoints Former TMX and Instinet Executive Laure Richmond as Chief Financial Officer", "exec_hire",
     "neutral"),
    ("Apple SVP Jennifer Newstead sells $806,495 in AAPL stock", "insider_sell", "bear"),
    ("After an October 1 stock grant, Apple (AAPL) officer John Ternus proposes a share sale.", "insider_sell", "bear"),
    ("GameStop CEO Cohen Continues Buying Spree With $10.6 Million Stock Purchase", "insider_buy", "bull"),
    ("Independent Director of Presurance Holdings Joseph Sarafa Buys 22% More Shares", "insider_buy", "bull"),
    ("SpaceX's Debut Quarter Fails To Lift Off: Short Seller Chanos Flags Starship Forecast Cut", "short_report",
     "bear"),
    ("Nvidia stock hits new all-time high, market cap at $5.7 trillion", "all_time_high", "bull"),
    ("Teekay Tankers Ltd stock hits 52-week high at 103.23 USD", "high_52w", "bull"),
    ("Li Auto Stock Hits 52-Week Low: What's Going On?", "low_52w", "bear"),
    ("Every 25 shares are expected to become one as Scienture aims to meet Nasdaq's minimum share-price rule.",
     "stock_split", "bear"),
    ("Nvidia announces 10-for-1 stock split", "stock_split", "bull"),
    ("Getty Images in talks with lenders on rescue financing - report", "bankruptcy", "bear"),
    ("Spirit Airlines files for bankruptcy again", "bankruptcy", "bear"),
    ("Ford recalls 1.9 million vehicles over rearview camera issue", "recall", "bear"),
    ("Tesla stock jumps 5% on better-than-expected vehicle deliveries report", "price_up", "bull"),
    ("Meta Shares Slide 4%, Shaving Billions from Zuckerberg's Paper Wealth", "price_down", "bear"),
    ("Acme prices $300 million convertible notes offering", "offering", "bear"),
])
def test_event_families(text, key, polarity):
    assert _one(text, key).polarity == polarity


def test_price_moves_carry_signed_percent():
    assert _one("Apple Stock Drops 1.5% as Local AI Moves Onto Desktops", "price_down").value == -1.5
    assert _one("Tesla Surges 5% as 486,532 Deliveries Top Company Consensus", "price_up").value == 5.0


@pytest.mark.parametrize(("text", "key"), [
    ("Tesla Bears Retreat as Sell Ratings Hit 3-Year Low: 'Don't Bet Against Musk'", "low_52w"),
    ("SLS Short Interest Hits All-Time High: Sellas Bulls Smell A Short Squeeze", "all_time_high"),
    ("Alphabet Stock Is Down 15% From Its All-Time High. Now Is the Perfect Time to Buy", "all_time_high"),
    ("Block director Anthony Eisen sells $1.33 million in stock", "offering"),
    ("Oura puts off initial public offering due to market uncertainty", "offering"),
    ("CLSA raises Micron stock price target on strong memory pricing outlook", "guidance_raise"),
    ("Tesla Faces Q3 Delivery Test — JPMorgan Cuts Forecast, Flags Weakness In Two Key Markets", "guidance_cut"),
    ("Here's Why Investors Should Give Delta Air Stock a Miss Currently", "earnings_miss"),
    ("Hindenburg Disaster 1937: The 32 Seconds That Ended the Airship Age", "short_report"),
    ("Apple picking 2026: Your guide to 9 orchards in Illinois and beyond", "recall"),
    ("$QBTS and $NVDA And when I first bought Nividia. QBTS going to be in that number.", "m_and_a"),
    ("Tesla wins deal for Model Y fleet in Europe", "m_and_a"),
    ("Fed officials recall the 2008 crisis as they weigh rate cuts", "recall"),
    ("Block Insider Sold Shares Worth $1,327,740, According to a Recent SEC Filing", "offering"),
])
def test_false_friends(text, key):
    assert key not in _keys(text)


def test_long_posts_are_judged_on_their_lead():
    post = ("$NVDA strong close today. " + "Lots of people talking about everything under the sun here. " * 6
            + "Also they announced a partnership with someone and a merger and a product launch.")
    assert _keys(post) == []
    assert "buyback" in _keys("$NVDA hit an all-time high Friday after a $150B buyback boost, " + "x " * 200)


def test_events_are_ordered_and_deduplicated():
    events = detect_events("Mizuho cuts Acme price target to $40; Mizuho cuts Acme price target again to $35 "
                           "as Acme stock falls 5%")
    keys = [e.key for e in events]
    assert keys.count("pt_cut") == 1  # one per (key, firm)
    assert keys.index("pt_cut") < keys.index("price_down")
    assert detect_events("") == [] and detect_events(None) == []  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Families on the real keyword-search fixture
# --------------------------------------------------------------------------- #
# Each family's 40 headlines come from a keyword search, so not all contain
# the event (e.g. "Will X Beat Estimates Again?" previews, "Hindenburg" the
# airship, Moody's outlook changes). Floors sit under measured rates.
@pytest.mark.parametrize(("family", "keys", "floor"), [
    ("pt_raise", {"pt_raise", "pt_cut"}, 0.95),      # 39/40 (+1 with no PT at all)
    ("pt_cut", {"pt_cut"}, 0.97),                    # 40/40
    ("upgrade", {"analyst_upgrade"}, 0.9),           # 37/40
    ("downgrade", {"analyst_downgrade"}, 0.9),       # 37/40
    ("initiate", {"analyst_initiate"}, 0.97),        # 40/40
    ("miss", {"earnings_miss"}, 0.75),               # 32/40
    ("guidance", {"guidance_raise", "guidance_cut"}, 0.8),  # 34/40 (rest: Moody's/Fitch outlooks)
    ("layoffs", {"layoffs"}, 0.97),                  # 40/40
    ("legal", {"lawsuit", "settlement", "investigation"}, 0.8),  # 34/40 (rest: police probes)
    ("exec", {"exec_departure", "exec_hire"}, 0.9),  # 39/40
    ("dividend", {"dividend_raise", "dividend_cut"}, 0.95),  # 40/40
    ("insider", {"insider_buy", "insider_sell"}, 0.9),  # 39/40
    ("recall", {"recall"}, 0.97),                    # 40/40
    ("bankrupt", {"bankruptcy"}, 0.78),              # 33/40
    ("mna", {"m_and_a"}, 0.75),                      # 30/39 (rest: licensing deals, "Big Billion Days Sale")
])
def test_family_recall_on_real_headlines(family, keys, floor):
    titles = [strip_publisher_suffix(t) for t in load_json_fixture("nlp/event_headlines.json")["families"][family]]
    hits = sum(1 for t in titles if keys & set(_keys(t)))
    assert hits / len(titles) >= floor, f"{family}: {hits}/{len(titles)}"


# Families added from 2026-10-05 searches. Positive floors sit under measured
# recall; the neg_* families are traps that must not yield the trap event.
@pytest.mark.parametrize(("family", "keys", "floor"), [
    ("reuters_guidance_above", {"guidance_raise"}, 0.85),           # 89/98 (was 4)
    ("reuters_guidance_below", {"guidance_cut"}, 0.85),             # 92/100 (was 2)
    ("guidance_strong_weak", {"guidance_raise", "guidance_cut"}, 0.85),  # 92/100
    ("exec_step_down", {"exec_departure"}, 0.82),                   # 88/100 (was 13)
    ("exec_ousted", {"exec_departure"}, 0.9),                       # 90/94 (was 3)
    ("exec_named", {"exec_hire"}, 0.9),                             # 94/100
])
def test_family_recall_on_new_real_headlines(family, keys, floor):
    test_family_recall_on_real_headlines(family, keys, floor)


@pytest.mark.parametrize(("family", "key", "ceiling"), [
    ("neg_top_wall_street", "earnings_beat", 0.03),       # "Top Wall Street Analyst Research Calls" (was 85/99)
    ("neg_payment_settlement", "settlement", 0.0),         # stablecoin/card settlement is plumbing, not legal
    ("neg_insider_plan_units", "insider_buy", 0.05),       # RSU/deferred-unit accruals are not open-market buys
    ("neg_preview_after_beat", "earnings_beat", 0.03),     # "heads into October 20 earnings after a Q2 beat"
])
def test_trap_families_stay_quiet(family, key, ceiling):
    titles = [strip_publisher_suffix(t) for t in load_json_fixture("nlp/event_headlines.json")["families"][family]]
    hits = [t for t in titles if key in _keys(t)]
    assert len(hits) / len(titles) <= ceiling, hits


@pytest.mark.parametrize(("text", "want"), [
    # adjective "top" / Street views are not results
    ("Here Are Wednesday's Top Wall Street Analyst Research Calls: Ally, Ciena", []),
    ("Top Wall Street Forecasters Revamp Nvidia Price Expectations Ahead Of Q3 Earnings", []),
    ("Apple To $355? Here Are 10 Top Analyst Forecasts For Thursday", []),
    ("Tesla Deliveries Top Wall Street's Forecast. Its Stock is Jumping.", ["earnings_beat", "price_up"]),
    ("Nvidia's quarterly revenue forecast beats estimates", ["guidance_raise"]),
    ("Microsoft Stock Delivers Strongest Quarter in 26 Years as Analysts Boost Outlook", []),
    # the canonical Reuters guidance phrasing
    ("Nvidia forecasts third-quarter revenue above estimates on strong AI demand", ["guidance_raise"]),
    ("Target forecasts annual profit below Wall Street estimates", ["guidance_cut"]),
    ("Intel forecasts weak fourth-quarter revenue", ["guidance_cut"]),
    ("Nvidia sees fourth-quarter revenue of $65 billion, above estimates", ["guidance_raise"]),
    ("Salesforce Gives Lukewarm Outlook That Fuels Disruption Fears", ["guidance_cut"]),
    # expectations and previews are not results
    ("SoFi Technologies stock heads toward October 27 results after Q2 beat", []),
    ("Nvidia Stock Heads Into Earnings With Biggest Short Position In S&P 500 — HSBC Sees 'Beat And Raise' Quarter",
     []),
    ("CELH Stock Heads For 16-Month Lows After Q2 Earnings Miss Sparks Massive Selloff", ["low_52w", "earnings_miss"]),
    # payments "settlement" vs legal settlements
    ("Should Stablecoin Card Settlement Require Action From SoFi Stock Investors?", []),
    ("SoFi, Mastercard launch stablecoin settlement", ["product_launch"]),
    ("Meta agrees to $725 million settlement of privacy class action", ["settlement"]),
    ("Purchasers of CAE common shares may claim compensation from a $38.25 million class action settlement",
     ["settlement"]),
    # "$N deal for X": goods are supply deals, companies are M&A
    ("AT&T, Corning Enter $3 Billion Deal for Fiber", ["partnership"]),
    ("Amazon strikes $38 billion deal for OpenAI compute", ["partnership"]),
    ("Raytheon nets $24.4 billion deal for SM-6 interceptors", ["contract_win"]),
    ("Chevron clinches $53 billion deal for Hess", ["m_and_a"]),
    ("Google's $32 billion deal for Wiz clears DOJ antitrust review", ["m_and_a"]),
    ("Curium strikes up to $8 billion deal for radiopharma peer Lantheus", ["m_and_a"]),
    ("Fox Shakes Up Streaming With $22 Billion Deal for Roku", ["m_and_a"]),
    # priced bids and pursuits (live GME feed): the bid is the deal
    ("GME CEO Ryan Cohen May Reportedly Withdraw GameStop's $56B eBay Bid", ["m_and_a"]),
    ("Unilever rejects $50 billion takeover approach from Kraft Heinz", ["m_and_a"]),
    ("Paramount's Warner Bid Faces Shareholder Revolt", ["m_and_a"]),
    ("GME's Ryan Cohen Isn't Done Chasing eBay, Remains Committed To Cracking A Deal", ["m_and_a"]),
    ("GME Reportedly Wants To Buy eBay But Retail Wonders How", ["m_and_a"]),
    # ...but not a priced buyback that "offers" a lesson, a share offer, a stock bid price, or buying more
    ("Nvidia's $235 Billion Buyback Offers a Powerful Lesson for Founders", ["buyback"]),
    ("Company Receives Nasdaq Notice on Minimum Bid Price", ["delisting"]),
    ("$GME GME - Cohen Buys AGAIN, but no Secret October Deal Coming", []),
    ("Everyone wants to buy the dip in Nvidia", []),
    # authorizing more shares for a deal is dilution (live GME)
    ("GME Stock Rises After Hours — GameStop Shareholders Back Bigger Share Count To Support Proposed eBay "
     "Acquisition", ["price_up", "offering", "m_and_a"]),
    # launches that are not products
    ("AT&T, T-Mobile, and Verizon Launch Joint Venture to expand satellite coverage", ["partnership"]),
    ("Target Rolls Out Fresh Price Cuts on Apparel", []),
    # plan accruals are not open-market buying
    ("AT&T (T) COO acquires 512 stock units at $24.40 each through payroll deductions.", []),
    ("AT&T Executive Lori M. Lee Acquires 368.852 Deferred Stock Units Under Company Benefit Plan", []),
    ("Meta director acquires 1,000 RSUs under equity plan", []),
    ("Ford Executive Doug Field Buys $2 Million In Company Stock", ["insider_buy"]),
    # the common CEO-change phrasings
    ("Apple's Tim Cook to step down as CEO; John Ternus named successor", ["exec_departure", "exec_hire"]),
    ("Tim Cook steps down as Apple CEO", ["exec_departure"]),
    ("Starbucks ousts CEO Laxman Narasimhan, names Brian Niccol", ["exec_departure"]),
    ("Pat Gelsinger out as Intel CEO", ["exec_departure"]),
    ("Kohl's fires CEO Ashley Buchanan", ["exec_departure"]),
    ("Target CEO Brian Cornell to hand reins to Michael Fiddelke", ["exec_departure"]),
    ("Meta's chief AI scientist Yann LeCun leaves", ["exec_departure"]),
    ("Apple Stock Approaches Buy Point Following Launch Of New iPhone, New CEO", ["product_launch", "exec_hire"]),
    ("Microsoft Stock Has Grown 14-Fold Since Satya Nadella Became CEO in 2014", []),
    ("Disney cuts hundreds more jobs for third round of layoffs under new CEO", ["layoffs"]),
    ("Founder and CEO J.W. Roth Named CEO of the Year at Business Awards", []),
])
def test_review_cases(text, want):
    assert sorted(_keys(text)) == sorted(want)


@pytest.mark.parametrize("text", [
    # live TWLO/GME (2026-10-05): sell-to-cover / plan-mandated insider sales are no
    # offering (no dilution) and no insider's call on the stock
    "The 13,898-share sale by Twilio (TWLO)'s CEO was required by company plans, not discretionary.",
    "Mandated tax sales total 9,084 shares for Twilio (TWLO) CFO Aidan Viggiano.",
    "Twilio CEO Khozema Shipchandler Executes RSU Tax Withholding Share Sales",
    "To cover stock-award taxes, Twilio (TWLO)'s Shipchandler proposes selling shares.",
    "To cover taxes on vested stock grants, an officer proposes selling Twilio (NYSE: TWLO) shares.",
    "To cover vesting taxes, GameStop (GME) officer Daniel Moore proposes selling 7,297 shares.",
    "Acme CFO sells 5,000 shares in sell-to-cover transaction",
])
def test_non_discretionary_insider_sales_carry_no_event(text):
    assert not {"offering", "insider_sell"} & set(_keys(text)), _keys(text)


@pytest.mark.parametrize(("text", "key"), [
    ("Twilio (NYSE:TWLO) Stock: Insider Khozema Shipchandler Sells 13,898 Shares", "insider_sell"),  # live
    ("The $5 million share sale by Nvidia's CEO raises eyebrows", "insider_sell"),  # discretionary, named after
    ("Acme CFO sells shares under 10b5-1 plan", "insider_sell"),  # a plan the insider chose
    ("Acme announces $200 million share sale to fund expansion", "offering"),  # the company's own sale
])
def test_discretionary_and_company_sales_keep_their_event(text, key):
    keys = _keys(text)
    assert key in keys and not ({"offering", "insider_sell"} - {key}) & set(keys), keys


@pytest.mark.parametrize(("text", "polarity"), [
    ("Apple wins appeal in Masimo patent case", "bull"),
    ("Judge dismisses lawsuit against Nvidia", "bull"),
    ("Judge refuses to dismiss lawsuit against Nvidia", "bear"),
    ("Apple hit with $5.7 billion jury verdict over haptic patents", "bear"),
])
def test_lawsuit_outcome_sets_polarity(text, polarity):
    assert _one(text, "lawsuit").polarity == polarity


def test_vocabularies_are_consistent():
    assert set(EVENT_POLARITY) == set(EVENT_LABELS)
    assert set(EVENT_THEMES) <= set(EVENT_LABELS)
    assert set(EVENT_THEMES.values()) <= set(THEMES)
    assert set(EVENT_POLARITY.values()) <= {"bull", "bear", "neutral"}
    spec_keys = ("analyst_upgrade analyst_downgrade analyst_initiate pt_raise pt_cut earnings_beat earnings_miss "
                 "guidance_raise guidance_cut record_results buyback dividend_raise dividend_cut layoffs lawsuit "
                 "investigation settlement m_and_a partnership contract_win product_launch recall exec_departure "
                 "exec_hire offering bankruptcy delisting short_report insider_buy insider_sell all_time_high low_52w "
                 "stock_split price_up price_down").split()
    assert set(spec_keys) <= set(EVENT_LABELS)


def test_fast_for_500_headlines():
    families = load_json_fixture("nlp/event_headlines.json")["families"]
    titles = [t for ts in families.values() for t in ts][:500]
    detect_events(titles[0])
    start = time.perf_counter()
    for t in titles:
        detect_events(t)
    assert time.perf_counter() - start < 1.0  # ~0.1 s measured
