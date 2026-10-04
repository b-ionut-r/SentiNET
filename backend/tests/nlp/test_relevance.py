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
    # "misses" are quote-page boilerplate that is_meaningful drops on purpose.
    ("F", 0.93, 0.97),     # 0.954 / 1.000
    ("SNAP", 0.97, 0.97),  # 1.000 / 1.000
    ("ORCL", 0.97, 0.97),  # 1.000 / 0.989
    ("V", 0.97, 0.95),     # 1.000 / 0.962
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
