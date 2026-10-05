"""Relevance: is this text about the company, as opposed to the word?

Real headlines are the test data. Labeled sets: TGT/XYZ (tuned against),
Ford/Snap/Oracle/Visa (held out — rules were not edited for them), and
generic-sense headlines for apple/meta/block/target that must score low.
"""
from __future__ import annotations

import pytest

from app.nlp.relevance import company_terms, explain_relevance, relevance
from app.nlp.text import is_meaningful, strip_publisher_suffix
from app.sources.base import CompanyRef
from tests.conftest import load_json_fixture

THRESHOLD = 0.35  # what the pipeline uses to keep a headline


def _company(**kw) -> CompanyRef:
    return CompanyRef(**kw)


NVDA = _company(ticker="NVDA", name="NVIDIA Corporation", short_name="Nvidia", aliases=[],
                industry="Semiconductors", sector="Technology")
AAPL = _company(ticker="AAPL", name="Apple Inc.", short_name="Apple", aliases=[], industry="Consumer Electronics",
                sector="Technology")
META = _company(ticker="META", name="Meta Platforms, Inc.", short_name="Meta", aliases=["Facebook"],
                industry="Internet Content & Information", sector="Communication Services")
TGT = _company(ticker="TGT", name="Target Corporation", short_name="Target", aliases=[], industry="Discount Stores",
               sector="Consumer Defensive")
XYZ = _company(ticker="XYZ", name="Block, Inc.", short_name="Block", aliases=["Square"],
               industry="Software - Infrastructure", sector="Technology")
HRB = _company(ticker="HRB", name="H&R Block, Inc.", short_name="H&R Block", aliases=[], industry="Personal Services",
               sector="Consumer Cyclical")
F = _company(ticker="F", name="Ford Motor Company", short_name="Ford", aliases=[], industry="Auto Manufacturers",
             sector="Consumer Cyclical")
T = _company(ticker="T", name="AT&T Inc.", short_name="AT&T", aliases=[], industry="Telecom Services",
             sector="Communication Services")
A = _company(ticker="A", name="Agilent Technologies, Inc.", short_name="Agilent", aliases=[],
             industry="Diagnostics & Research", sector="Healthcare")
SNAP = _company(ticker="SNAP", name="Snap Inc.", short_name="Snap", aliases=["Snapchat"],
                industry="Internet Content & Information", sector="Communication Services")
ON = _company(ticker="ON", name="ON Semiconductor Corporation", short_name="onsemi", aliases=[],
              industry="Semiconductors", sector="Technology")
BTC = _company(ticker="BTC-USD", name="Bitcoin USD", short_name="Bitcoin", aliases=[], quote_type="CRYPTOCURRENCY")
SPY = _company(ticker="SPY", name="SPDR S&P 500 ETF Trust", short_name="SPDR S&P 500", aliases=[], quote_type="ETF")


def _precision_recall(items: list[dict], company: CompanyRef) -> tuple[float, float, list[str], list[str]]:
    tp = fp = fn = 0
    false_pos: list[str] = []
    false_neg: list[str] = []
    for item in items:
        title = strip_publisher_suffix(item["title"], item.get("publisher"))
        predicted = is_meaningful(title) and relevance(title, company) >= THRESHOLD
        if predicted and item["relevant"]:
            tp += 1
        elif predicted:
            fp += 1
            false_pos.append(title)
        elif item["relevant"]:
            fn += 1
            false_neg.append(title)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return precision, recall, false_pos, false_neg


# --------------------------------------------------------------------------- #
# Labeled real-headline sets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ticker", ["tgt", "xyz"])
def test_tuned_sets_are_classified_correctly(ticker):
    """Google News for "Target"/"Block" is ~half retail-target / H&R Block / city-block noise."""
    data = load_json_fixture(f"nlp/headlines_{ticker}.json")
    company = CompanyRef(**data["company"])
    precision, recall, fps, fns = _precision_recall(data["items"], company)
    assert precision >= 0.97, fps
    assert recall >= 0.97, fns


@pytest.mark.parametrize(("key", "min_precision", "min_recall"), [
    # Measured P/R. Ford's misses are F-150 consumer listicles; Oracle/Visa
    # "misses" are quote-page boilerplate and automated price ticks ("Visa
    # stock pre-market at EUR 318.35: plus 0.27 percent") that is_meaningful
    # drops on purpose — they carry no information beyond the quote.
    ("F", 0.93, 0.97),     # 0.954 / 1.000
    ("SNAP", 0.97, 0.97),  # 1.000 / 1.000
    ("ORCL", 0.97, 0.97),  # 1.000 / 0.989
    ("V", 0.97, 0.86),     # 1.000 / 0.885
])
def test_held_out_sets(key, min_precision, min_recall):
    """Sets captured after the rules were written and never tuned against."""
    data = load_json_fixture("nlp/relevance_heldout.json")["sets"][key]
    company = CompanyRef(**data["company"])
    precision, recall, fps, fns = _precision_recall(data["items"], company)
    assert precision >= min_precision, fps
    assert recall >= min_recall, fns


@pytest.mark.parametrize(("sense", "company"), [("apple", AAPL), ("meta", META), ("block", XYZ), ("target", TGT)])
def test_generic_word_headlines_score_low(sense, company):
    """Real headlines where the brand word is used in its everyday sense."""
    data = load_json_fixture("nlp/ambiguous_names.json")
    genuine = set(data["company_sense"].get(sense, []))
    for title in data["headlines"][sense]:
        score = relevance(strip_publisher_suffix(title), company)
        if title in genuine:
            assert score >= THRESHOLD, title
        else:
            assert score < THRESHOLD, title


@pytest.mark.parametrize("ticker", ["nvda", "aapl", "meta"])
def test_unambiguous_search_results_mostly_relevant(ticker):
    """Google News for a ticker+name query: most results mention the company
    and those that do are kept."""
    data = load_json_fixture(f"nlp/headlines_{ticker}.json")
    company = CompanyRef(**data["company"])
    titles = [strip_publisher_suffix(i["title"], i["publisher"]) for i in data["items"]]
    kept = [t for t in titles if relevance(t, company) >= THRESHOLD]
    assert len(kept) / len(titles) >= 0.7


# --------------------------------------------------------------------------- #
# Mention classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("company", "text"), [
    (NVDA, "$NVDA puts are the only play"),
    (NVDA, "Nvidia Adds Record $150 Billion to Stock Buyback"),
    (NVDA, "Shares of NVDA rose 3%"),
    (AAPL, "Apple stock jumps after iPhone sales beat"),
    (AAPL, "Morgan Stanley lowers Apple price target"),
    (AAPL, "Apple rating cut at Jefferies"),
    (META, "Meta stock falls after Zuckerberg AI spending plan"),
    (META, "META to the moon"),
    (META, "Facebook outage hits millions"),
    (TGT, "Target cuts prices on 2,000 items"),
    (TGT, "HSBC upgraded Target to Buy"),
    (TGT, "$TGT earnings tomorrow"),
    (XYZ, "Block lays off 1,000 workers, Jack Dorsey memo says"),
    (XYZ, "Square's Cash App adds bitcoin feature"),
    (XYZ, "Block to join S&P 500"),
    (HRB, "H&R Block tax prep fees rise"),
    (F, "Ford recalls 300,000 F-150 trucks"),
    (F, "$F up 2%"),
    (F, "Ford Motor (NYSE:F) earnings beat"),
    (T, "AT&T raises dividend"),
    (T, "NYSE:T falls"),
    (A, "Agilent beats estimates"),
    (A, "NYSE:A shares drop"),
    (SNAP, "Snap shares plunge 20% after earnings"),
    (SNAP, "Snapchat adds AI lenses"),
    (SNAP, "SNAP stock soars"),
    (ON, "$ON jumps"),
    (ON, "onsemi to cut jobs"),
    (BTC, "Bitcoin hits record above $120,000"),
    (BTC, "$BTC.X breaking out"),
    (SPY, "$SPY puts printing"),
])
def test_company_mentions(company, text):
    assert relevance(text, company) >= 0.8, explain_relevance(text, company)


@pytest.mark.parametrize(("company", "text"), [
    (AAPL, "Big Apple restaurants face rent hikes"),
    (AAPL, "Apple pie recipe for fall"),
    (AAPL, "Apple cider vinegar benefits"),
    (AAPL, "apple ratings for best varieties"),
    (META, "A meta-analysis of statin trials"),
    (META, "The meta of Fortnite has shifted"),
    (TGT, "Analysts raise price target on Nvidia"),
    (TGT, "Police target drug ring in Ohio"),
    (TGT, "Fed's inflation target remains 2%"),
    (XYZ, "H&R Block tax prep fees rise"),
    (XYZ, "Police block road after crash"),
    (XYZ, "Celtics block party in Game 3"),
    (F, "Harrison Ford stars in new film"),
    (F, "Gerald Ford's legacy revisited"),
    (F, "Grade F for the city budget"),
    (T, "Mr. T spotted in Chicago"),
    (A, "A stock to watch"),
    (SNAP, "Oh snap, the weather turned"),
    (SNAP, "SNAP benefits cut for millions"),
    (ON, "Stocks on the move today"),
])
def test_word_senses_and_false_friends(company, text):
    assert relevance(text, company) < THRESHOLD, explain_relevance(text, company)


def test_evidence_tiers():
    assert relevance("$NVDA ripping", NVDA) == 1.0
    assert relevance("NASDAQ:NVDA ripping", NVDA) >= 0.95
    assert 0.85 <= relevance("NVDA ripping", NVDA) < 1.0
    # A cue alone (CEO name) is weaker evidence than the name.
    assert THRESHOLD <= relevance("Jensen Huang says demand is insane", NVDA) < 0.8


def test_roundups_are_capped():
    result = explain_relevance("7 AI stocks to buy: Nvidia, AMD, Broadcom, Micron, Intel, Qualcomm, Marvell", NVDA)
    assert result.roundup
    assert result.score <= 0.4
    assert relevance("$NVDA $AMD $AVGO $MU $INTC movers", NVDA) <= 0.4


def test_secondary_subject_is_discounted():
    result = explain_relevance("Microsoft stock rises; Nvidia is a supplier", NVDA)
    assert result.secondary
    assert THRESHOLD <= result.score < relevance("Nvidia is a supplier to Microsoft", NVDA)


def test_no_mention_scores_zero():
    assert relevance("Fed holds rates steady", NVDA) == 0.0
    assert relevance("", NVDA) == 0.0
    assert explain_relevance("Fed holds rates steady", NVDA).evidence == []


def test_word_tickers_need_context():
    # Bare single-letter / dictionary-word tickers are never enough on their own.
    assert relevance("F shares rise", F) < THRESHOLD
    assert relevance("T stock rises after earnings", T) < THRESHOLD
    # Acronym tickers count once market vocabulary confirms them.
    assert relevance("SNAP stock soars", SNAP) >= 0.8
    assert relevance("SNAP benefits cut", SNAP) < THRESHOLD


def test_company_terms():
    assert company_terms(NVDA) >= {"nvidia", "nvda", "$nvda"}
    assert company_terms(XYZ) >= {"block", "square", "xyz"}
    assert company_terms(None) == frozenset()


def test_speed_500_mentions():
    import time

    titles = [strip_publisher_suffix(i["title"], i["publisher"])
              for t in ("nvda", "aapl", "meta", "tgt", "xyz") for i in load_json_fixture(f"nlp/headlines_{t}.json")["items"]]
    start = time.perf_counter()
    for title in titles:
        relevance(title, TGT)
    assert time.perf_counter() - start < 1.0


# --------------------------------------------------------------------------- #
# Brokerages: the firm as the author of research is not news about the firm
# --------------------------------------------------------------------------- #
def test_bank_feed_excludes_its_research_on_other_stocks():
    """JPM feed, labeled before the broker rule existed: ~40% of 'JPMorgan'
    headlines are JPMorgan rating, targeting or opining on other stocks.
    Before the rule: precision 0.62; now 0.98 at full recall."""
    data = load_json_fixture("nlp/relevance_jpm.json")
    company = CompanyRef(**data["company"])
    precision, recall, fps, fns = _precision_recall(data["items"], company)
    assert precision >= 0.95, fps
    assert recall >= 0.97, fns


GS = _company(ticker="GS", name="The Goldman Sachs Group, Inc.", short_name="Goldman Sachs",
              industry="Capital Markets", sector="Financial Services")
MS = _company(ticker="MS", name="Morgan Stanley", short_name="Morgan Stanley", industry="Capital Markets",
              sector="Financial Services")


@pytest.mark.parametrize(("text", "company", "about_firm"), [
    ("Goldman Sachs raises Nvidia price target to $250", GS, False),
    ("Goldman Sachs strategists see S&P 500 at 7,500", GS, False),
    ("Goldman Sachs sees Fed cutting twice", GS, False),
    ("Goldman Sachs says Nvidia remains a top pick", GS, False),
    ("Goldman Sachs lowers its S&P 500 target", GS, False),
    ("Nvidia upgraded to Buy at Goldman Sachs", GS, False),
    ("Goldman Sachs adds Amazon to conviction list", GS, False),
    ("Morgan Stanley Has Strong Message For Nvidia Stock Investors", MS, False),
    ("Nvidia Stock Gains After Morgan Stanley Names It a 'Top Semiconductor Pick'", MS, False),
    ("Morgan Stanley warns of AI bubble", MS, False),
    ("Goldman Sachs profit jumps 40% on dealmaking boom", GS, True),
    ("Goldman Sachs expects record investment banking fees", GS, True),
    ("Goldman Sachs raises its dividend by 50%", GS, True),
    ("Goldman Sachs names new CFO", GS, True),
    ("Goldman Sachs stock hits record high", GS, True),
    ("Morgan Stanley sees record wealth inflows in third quarter", MS, True),
    ("Morgan Stanley beats estimates as trading revenue surges", MS, True),
    ("RBC Capital Maintains Morgan Stanley (MS) With Buy Rating", MS, True),
    # live Google News, 2026-10-04
    ("Amazon Just Joined Goldman Sachs' Conviction List: 5 New Top Stock Picks With Massive Upside", GS, False),
    ("Phillips 66 (NYSE:PSX) Stock Price Target Raised at The Goldman Sachs Group", GS, False),
    ("Goldman Sachs adds Amazon stock to monthly Director's Cut list", GS, False),
    ("Goldman Sachs adds three partners to its tech banking team", GS, True),
    ("Goldman Sachs' profit jumps on trading", GS, True),
])
def test_broker_as_author_vs_broker_as_subject(text, company, about_firm):
    assert (relevance(text, company) >= THRESHOLD) is about_firm, explain_relevance(text, company).evidence


def test_non_brokers_keep_their_opinion_verbs():
    assert relevance("Apple says iPhone demand is strong", AAPL) >= 0.8
    assert relevance("Nvidia sees record data center sales next quarter", NVDA) >= 0.8


def test_medical_amd_is_not_the_chipmaker():
    amd = _company(ticker="AMD", name="Advanced Micro Devices, Inc.", short_name="AMD", industry="Semiconductors",
                   sector="Technology")
    for text in ("Deep Learning May Guide Earlier Neovascular AMD Treatment",
                 "AAO 2026 Preview: Emerging Therapies for Wet AMD",
                 "Week in Review: Global Rates of Corneal Transplants, AMD Risk in Older Women"):
        assert relevance(text, amd) < THRESHOLD, text
    for text in ("AMD to acquire World Labs in $8.2B all-stock deal", "\"Zen 5\" AMD Ryzen Processors for Agentic AI"):
        assert relevance(text, amd) >= 0.9, text


def test_sponsored_venues_are_places_not_companies():
    sofi = _company(ticker="SOFI", name="SoFi Technologies, Inc.", short_name="SoFi", industry="Credit Services",
                    sector="Financial Services")
    for text, company in [("Bruno Mars' The Romantic Tour makes fans swoon at SoFi Stadium show", sofi),
                          ("Slipknot rocks SoFi Stadium", sofi),
                          ("Timberwolves beat Lakers at Target Center", TGT)]:
        assert relevance(text, company) < THRESHOLD, text
    assert relevance("SoFi stock jumps after record member growth", sofi) >= 0.9


# --------------------------------------------------------------------------- #
# Review round 2 (live 2026-10-05 feeds)
# --------------------------------------------------------------------------- #
SPY_LIVE = _company(ticker="SPY", name="State Street SPDR S&P 500 ETF Trust", short_name="S&P 500",
                    aliases=["SPDR S&P 500"], quote_type="ETF")
JPM = _company(ticker="JPM", name="JPMorgan Chase & Co.", short_name="JPMorgan Chase", aliases=["JPMorgan"],
               industry="Banks - Diversified", sector="Financial Services")
MS = _company(ticker="MS", name="Morgan Stanley", short_name="Morgan Stanley", aliases=[], industry="Capital Markets",
              sector="Financial Services")
MU = _company(ticker="MU", name="Micron Technology, Inc.", short_name="Micron", aliases=[], industry="Semiconductors")
GE = _company(ticker="GE", name="GE Aerospace", short_name="GE Aerospace", aliases=[], industry="Aerospace & Defense")


@pytest.mark.parametrize(("company", "text"), [
    # legal/regulatory actions with the company as object (were 0-0.25 live)
    (META, "New Mexico wants Meta to pay $40B in penalties after data privacy trial"),
    (META, "New Mexico reportedly seeks up to $40B in penalties from Meta over data privacy"),
    (META, "Eight Years On, Cambridge Analytica Scandal Catches Up with Meta in Santa Fe"),
    (META, "EU fines Meta €1.2 billion over data transfers"),
    (AAPL, "EU fines Apple €500 million under DMA"),
    (META, "Jury finds Meta liable in child safety case"),
    # the name after a reported-speech "that", a comma, a clause-leading colon
    (META, "Cramer Warned Viewers That Meta's AI Would Crush This Stock"),
    (AAPL, "Analysts say that Apple will raise prices"),
    (META, "Overnight, Meta deleted all her accounts"),
    (META, "Meta: Muse Is Nice, But Not Enough"),
    (META, "Texas Electric Utility Only Considered Gas to Power $10B Meta Data Center"),
    # coordinated subjects
    (F, "Ford and JPMorganChase launch Michigan LIFT manufacturing platform"),
    (F, "Ford, JPMorgan Chase and Michigan establish $3B manufacturing initiative"),
    # social tags and short tickers with a stock noun
    (BTC, "#Bitcoin to 150k by March"),
    (BTC, "#BTC dumping hard rn"),
    (NVDA, "#NVDA breaking out, RSI 70"),
    (NVDA, "$NVIDIA gets $350 target from Cantor"),
    (MU, "MU stock soars after earnings"),
    (GE, "GE stock hits record"),
    # a bank's own news, not its research
    (JPM, "JPMorgan's Dimon warns of 'cockroaches' in credit markets"),
    (JPM, "JPMorgan's Dimon says AI will cut jobs at the bank"),
    (MS, "Morgan Stanley sets new wealth management target of $10 trillion"),
    (MS, "Morgan Stanley's Pick says firm will keep hiring"),
])
def test_review_round_two_mentions(company, text):
    assert relevance(text, company) >= 0.6, explain_relevance(text, company)


@pytest.mark.parametrize(("company", "text"), [
    (F, "Sylvester Stallone Shares How Francis Ford Coppola Cried Over His Film's Failure"),
    (F, "I'm Troy Ford - a stock broker's perfect life is upended"),
    (META, "A meta data study of clinical trials"),
    (JPM, "JPMorgan's Kolanovic says stocks will fall 20%"),
    (MS, "Nvidia Stock Gains After Morgan Stanley Names It a 'Top Semiconductor Pick'"),
    (SPY_LIVE, "Why Nvidia Stock Is Down Today"),
    (SPY_LIVE, "AMD Climbs 3% as Chip Stocks Extend Their Run; Arm Jumps 8%, NVIDIA Rises 2%"),
    (SPY_LIVE, "A $1,000 Bet on Marvell in 2016 Crushed the Market With 2140% Returns"),
])
def test_review_round_two_non_mentions(company, text):
    assert relevance(text, company) < THRESHOLD, explain_relevance(text, company)


@pytest.mark.parametrize(("company", "text"), [
    (T, "Is AT&T an Undervalued Dividend Stock to Buy for Passive Income Investors?"),
    (BTC, "Bitcoin: ETF Inflows, Fed Hikes, Failed Regulation, And A Rally That Makes No Sense"),
    (BTC, "Bitcoin beats Gold, SPY, Silver, QQQ in Iran war"),
    (NVDA, "Nvidia Sets Record as Ali Tracks Two ETFs for AI Interest"),
    (NVDA, "Is Nvidia a Stock to Buy Now?"),
    (SPY_LIVE, "Dow, S&P 500, Nasdaq Futures Rise As Earnings Roll In: NKE, NFLX, TSLA In Focus"),
])
def test_single_stock_pieces_are_not_roundups(company, text):
    result = explain_relevance(text, company)
    assert not result.roundup and result.score >= 0.8, result


@pytest.mark.parametrize("text", [
    "Stock Futures Are Rising With Earnings Season, Fed Minutes Ahead",
    "Wall Street's AI Party Is on Edge as Soaring Yields Raise Risks",
    "Stocks Settle Higher as Fed Rate Hike Concerns Ease",
    "Is a stock market correction coming? Most Americans think so",
])
def test_market_wide_news_is_about_an_index_fund(text):
    assert relevance(text, SPY_LIVE) == pytest.approx(0.6)
    assert relevance(text, NVDA) == 0.0


def test_hashtag_lists_and_soups_stay_low():
    assert relevance("📢 Stocks Trending NOW: #MU #ACN #LQDA #IBM #GOOG #MSFT #CTVA #ABAT", MU) < THRESHOLD
    assert relevance("Who's hiring? #GraphicDesigner #artists #3D #NFT #Crypto #eth #Bitcoin", BTC) < THRESHOLD
    assert relevance("🤖 AI Agent Upgrade: #AAPL is now a BUY 📈 Reason: RSI <65 #stocks #AI #trading #invest",
                     AAPL) >= 0.8  # the company's tag leads: the post is about it
