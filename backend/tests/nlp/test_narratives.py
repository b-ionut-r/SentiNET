"""Duplicate detection and story clustering on real headline sets.

Fixtures (tests/fixtures/nlp/headlines_*.json) are real Google News RSS
results with hand-labeled stories. Protocol: NVDA/AAPL/META/TGT/XYZ were used
for tuning, TSLA/AMZN for validation, AMD was labeled before any tuning and
first scored untouched (see app/nlp/narratives.py docstring). The thresholds
below sit a few points under measured values: they guard against regressions
without pinning exact numbers.
"""
from __future__ import annotations

import random
import time
from datetime import UTC, datetime, timedelta
from itertools import combinations

import pytest

from app.nlp.narratives import PROGRAM_SPAN_H, _Doc, _join_programs, cluster_narratives, find_duplicates
from app.nlp.text import strip_publisher_suffix
from app.nlp.types import Cluster, ClusterItem
from app.sources.base import CompanyRef
from tests.conftest import load_json_fixture

NVDA = CompanyRef(ticker="NVDA", name="NVIDIA Corporation", short_name="Nvidia", industry="Semiconductors")
T0 = datetime(2026, 9, 30, 14, 0, tzinfo=UTC)


def _items(ticker: str) -> tuple[list[ClusterItem], list[str | None], CompanyRef]:
    data = load_json_fixture(f"nlp/headlines_{ticker}.json")
    items, labels = [], []
    for k, it in enumerate(data["items"]):
        if it.get("relevant") is False:  # the pipeline drops these before clustering
            continue
        items.append(ClusterItem(
            id=str(k), title=strip_publisher_suffix(it["title"], it.get("publisher")),
            timestamp=datetime.fromisoformat(it["timestamp"]), publisher=it.get("publisher"),
        ))
        labels.append(it.get("story"))
    return items, labels, CompanyRef(**data["company"])


def _verdict(a: str | None, b: str | None) -> bool | None:
    """True = same story, False = different, None = don't care ('?' or an
    'a|b' item whose either reading is acceptable)."""
    if a == "?" or b == "?":
        return None
    if a is None or b is None:
        return False
    oa, ob = set(a.split("|")), set(b.split("|"))
    if not oa & ob:
        return None if "?" in oa | ob else False
    return True if len(oa) == len(ob) == 1 else None


def _scores(clusters: list[Cluster], items: list[ClusterItem], labels: list[str | None]) -> tuple[float, float]:
    """(pairwise F1, B-cubed F1) against the labels."""
    where = {iid: k for k, c in enumerate(clusters) for iid in c.item_ids}
    together = [where[it.id] for it in items]
    tp = fp = fn = 0
    for a, b in combinations(range(len(items)), 2):
        v = _verdict(labels[a], labels[b])
        if v is None:
            continue
        same = together[a] == together[b]
        tp += v and same
        fp += (not v) and same
        fn += v and not same
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    bp = br = 0.0
    scored = [a for a in range(len(items)) if labels[a] != "?"]
    for a in scored:
        hits = shared = should = 0
        for b in range(len(items)):
            v = None if a == b else _verdict(labels[a], labels[b])
            if v is None:
                continue
            should += v
            if together[a] == together[b]:
                shared += 1
                hits += v
        bp += (hits + 1) / (shared + 1)
        br += (hits + 1) / (should + 1)
    bp, br = bp / len(scored), br / len(scored)
    return 2 * p * r / (p + r), 2 * bp * br / (bp + br)


def _item(k: int, title: str, hours: float = 0.0, publisher: str | None = None, weight: float = 1.0) -> ClusterItem:
    return ClusterItem(id=f"i{k}", title=title, timestamp=T0 + timedelta(hours=hours), publisher=publisher,
                       weight=weight)


def _cluster_of(clusters: list[Cluster], item_id: str) -> Cluster:
    return next(c for c in clusters if item_id in c.item_ids)


# --------------------------------------------------------------------------- #
# Story clustering on labeled real sets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("ticker", "min_f1", "min_b3"), [
    ("nvda", 0.92, 0.92),  # tuning      measured .95 / .94 (rivals' milestones kept apart)
    ("aapl", 0.83, 0.89),  # tuning      .87 / .92
    ("meta", 0.45, 0.78),  # tuning      .50 / .81 (many overlapping "Muse" angles)
    ("tgt", 0.76, 0.83),   # tuning      .82 / .87
    ("xyz", 0.93, 0.92),   # tuning      .99 / .96
    ("tsla", 0.80, 0.84),  # validation  .84 / .87
    ("amzn", 0.78, 0.86),  # validation  .82 / .89
    ("amd", 0.80, 0.84),   # test: .89 / .89 at first (untouched) scoring; .86 / .87 now
])
def test_labeled_story_sets(ticker, min_f1, min_b3):
    items, labels, company = _items(ticker)
    clusters = cluster_narratives(items, company, max_clusters=len(items))
    f1, b3 = _scores(clusters, items, labels)
    assert f1 >= min_f1, f"{ticker}: pairwise F1 {f1:.3f}"
    assert b3 >= min_b3, f"{ticker}: B-cubed F1 {b3:.3f}"


def test_every_item_lands_in_exactly_one_cluster():
    items, _labels, company = _items("nvda")
    clusters = cluster_narratives(items, company, max_clusters=len(items))
    ids = [iid for c in clusters for iid in c.item_ids]
    assert sorted(ids) == sorted(it.id for it in items)
    for c in clusters:
        assert c.item_ids[0] == c.representative_id
        assert c.title == next(it.title for it in items if it.id == c.representative_id)


def test_spec_example_lawsuit_paraphrases_group():
    items = [
        _item(0, "Nvidia hit with $1.05B lawsuit over options", 0),
        _item(1, "Nvidia faces $1.05B suit from 1993 advisor", 3),
        _item(2, "Nvidia adds $150 billion to stock buyback", 1),
        _item(3, "Nvidia boosts share repurchase authorization by $150B", 2),
        _item(4, "Morgan Stanley names Nvidia its top semiconductor pick", 4),
    ]
    clusters = cluster_narratives(items, NVDA)
    assert set(_cluster_of(clusters, "i0").item_ids) == {"i0", "i1"}
    assert set(_cluster_of(clusters, "i2").item_ids) == {"i2", "i3"}  # "share repurchase" == "buyback"
    assert _cluster_of(clusters, "i4").item_ids == ["i4"]


def test_real_buyback_story_is_one_cluster_with_readable_terms():
    items, labels, company = _items("nvda")
    clusters = cluster_narratives(items, company, max_clusters=len(items))
    buyback_ids = {it.id for it, lab in zip(items, labels, strict=True) if lab == "buyback"}
    biggest = clusters[0]
    assert buyback_ids <= set(biggest.item_ids)
    assert "$150B" in biggest.terms and "buyback" in biggest.terms
    assert not any("nvidia" in t.lower() for c in clusters for t in c.terms)  # own name is never a story term


def test_same_session_move_headlines_join_the_story_of_the_day():
    """'Why is Tesla stock surging today?' carries no topic words; the
    session's price move ties it to the delivery-beat story that day."""
    items, labels, company = _items("tsla")
    clusters = cluster_narratives(items, company, max_clusters=len(items))
    surging = next(it for it in items if it.title == "Why is Tesla stock surging today?")
    beat = next(it for it in items if it.title == "Tesla Stock Jumps After Deliveries Top Estimates")
    assert _cluster_of(clusters, surging.id) is _cluster_of(clusters, beat.id)


def test_stories_stay_apart_from_each_other():
    items = [
        _item(0, "Apple hit with $5.7 billion jury verdict over haptic patents", 0),
        _item(1, "US jury says Apple owes record $5.7 billion in haptic technology patent case", 1),
        _item(2, "Apple plans smart-home hub launch in October: report", 2),
        _item(3, "Apple to enter smart-home market with Siri AI hub", 3),
        _item(4, "Morgan Stanley lowers Apple stock price target on limited upside", 4),
        _item(5, "Morgan Stanley lowers Apple (AAPL.US) price target to $355", 5),
    ]
    company = CompanyRef(ticker="AAPL", name="Apple Inc.", short_name="Apple")
    clusters = cluster_narratives(items, company)
    assert sorted(sorted(c.item_ids) for c in clusters) == [["i0", "i1"], ["i2", "i3"], ["i4", "i5"]]


def test_representative_prefers_declarative_trusted_headline():
    items = [
        _item(0, "Why Is Nvidia Stock Rising After Its Huge $150 Billion Buyback?", 0, "Motley Fool"),
        _item(1, "Nvidia adds record $150 billion to stock buyback", 1, "Reuters"),
        _item(2, "Nvidia's $150 billion buyback: what it means", 2, "Some Blog"),
    ]
    clusters = cluster_narratives(items, NVDA)
    assert len(clusters) == 1
    assert clusters[0].representative_id == "i1"


def test_syndicated_copies_cluster_with_their_original():
    items = [
        _item(0, "Morgan Stanley lowers Apple stock price target on limited upside", 0),
        _item(1, "Morgan Stanley lowers Apple stock price target on limited upside By Investing.com", 1),
        _item(2, "Apple hit with $5.7 billion jury verdict over haptic patents", 2),
    ]
    clusters = cluster_narratives(items, CompanyRef(ticker="AAPL", name="Apple Inc.", short_name="Apple"))
    assert set(_cluster_of(clusters, "i0").item_ids) == {"i0", "i1"}


def test_ranking_max_clusters_and_singletons():
    items = [_item(k, f"Nvidia adds record $150 billion to stock buyback variant {k}", k) for k in range(3)]
    items += [_item(10, "Nvidia opens graduate fellowship applications", 1, weight=0.5),
              _item(11, "Nvidia supplier secret weapon under the radar", 2, weight=0.4)]
    clusters = cluster_narratives(items, NVDA, max_clusters=2)
    assert len(clusters) == 2
    assert len(clusters[0].item_ids) == 3  # heaviest story first
    assert clusters[1].item_ids == ["i10"]  # singletons follow, by weight


def test_edge_cases():
    assert cluster_narratives([]) == []
    one = cluster_narratives([_item(0, "Nvidia adds $150 billion to buyback")], NVDA)
    assert len(one) == 1 and one[0].item_ids == ["i0"]
    no_time = [ClusterItem(id="a", title="Nvidia adds $150 billion to buyback"),
               ClusterItem(id="b", title="Nvidia boosts buyback by $150 billion"),
               ClusterItem(id="c", title=""), ClusterItem(id="d", title="$NVDA")]
    clusters = cluster_narratives(no_time, None)  # no company, no timestamps, junk titles
    assert sorted(i for c in clusters for i in c.item_ids) == ["a", "b", "c", "d"]
    assert set(_cluster_of(clusters, "a").item_ids) == {"a", "b"}


def test_deterministic_and_input_order_independent():
    items, _labels, company = _items("aapl")
    first = cluster_narratives(items, company, max_clusters=len(items))
    again = cluster_narratives(items, company, max_clusters=len(items))
    assert [c.item_ids for c in first] == [c.item_ids for c in again]
    shuffled = items[:]
    random.Random(7).shuffle(shuffled)
    other = cluster_narratives(shuffled, company, max_clusters=len(items))
    partition = {frozenset(c.item_ids) for c in first}
    moved = sum(1 for c in other if frozenset(c.item_ids) not in partition)
    assert moved <= 3  # tie-breaks may differ; stories don't


def test_fast_for_500_headlines():
    pool: list[ClusterItem] = []
    for ticker in ("nvda", "aapl", "meta", "tsla", "amzn", "amd"):
        items, _labels, _company = _items(ticker)
        pool += [ClusterItem(id=f"{ticker}{it.id}", title=it.title, timestamp=it.timestamp) for it in items]
    pool = pool[:500]
    start = time.perf_counter()
    clusters = cluster_narratives(pool, NVDA, max_clusters=50)
    elapsed = time.perf_counter() - start
    assert clusters
    assert elapsed < 2.0, f"{elapsed:.2f}s"  # ~0.3 s on a laptop; generous for CI


# --------------------------------------------------------------------------- #
# Duplicates (syndication)
# --------------------------------------------------------------------------- #
def test_find_duplicates_is_a_partition_with_lowest_index_first():
    titles = ["A b c d e f", "x", "A b c d e f", "", "y z"]
    groups = find_duplicates(titles)
    assert sorted(i for g in groups for i in g) == list(range(len(titles)))
    assert all(g == sorted(g) for g in groups)
    assert [0, 2] in groups
    assert find_duplicates([]) == []


@pytest.mark.parametrize("pair", [
    # attribution suffix
    ("Wells Fargo reiterates Amazon stock Overweight on AWS pricing power",
     "Wells Fargo reiterates Amazon stock Overweight on AWS pricing power By Investing.com"),
    # money spelled differently
    ("Amazon seeks to offload $8 bln of Nvidia chips to investors- FT",
     "Amazon seeks to offload $8 billion of Nvidia chips to investors, FT reports"),
    # truncated copy
    ("HSBC Upgrades Target (TGT) Rating to Buy with Price Target of $1",
     "HSBC Upgrades Target (TGT) Rating to Buy with Price Target of $190"),
    # curly vs straight quotes, case
    ("Amazon Just Joined Goldman Sachs' Conviction List: 5 New Top Stock Picks With Massive Upside",
     "Amazon Just Joined Goldman Sachs’ Conviction List: 5 New Top Stock Picks With Massive Upside"),
    ("Meridian Mining applies for block admission of 1.6M shares",
     "Meridian Mining Applies For Block Admission Of 1.6 Million Ordinary Shares To LSE"),
    # the same story from Google News and Bing News (casing differs by outlet)
    ("Amazon Stock Pays $0 in Dividends. Here's Why Long-Term Investors Should Own It Anyway.",
     "Amazon stock pays $0 in dividends. Here's why long-term investors should own it anyway."),
    ("What a 20-Year Deal With Amazon Means for Constellation Energy Stock",
     "What a 20-year deal with Amazon means for Constellation Energy stock"),
])
def test_real_syndicated_copies_are_duplicates(pair):
    assert find_duplicates(list(pair)) == [[0, 1]]


@pytest.mark.parametrize("pair", [
    # same template, different filings / days
    ("Block Insider Sold Shares Worth $1,327,740, According to a Recent SEC Filing",
     "Block Insider Sold Shares Worth $1,360,080, According to a Recent SEC Filing"),
    ("H&R Block Inc. stock underperforms Monday when compared to competitors",
     "H&R Block Inc. stock underperforms Friday when compared to competitors"),
    # same story, different headlines (not copies)
    ("Nvidia Adds $150 Billion to Massive Stock Buyback, the Largest Ever",
     "Nvidia Adds Record $150 Billion to Stock Buyback"),
    ("Morgan Stanley lowers Apple stock price target on limited upside",
     "Morgan Stanley Has Strong Verdict for Apple Stock Investors"),
    ("Why Apple Stock Is Climbing Today", "Why Apple Stock Is Falling Today"),
])
def test_distinct_headlines_are_not_duplicates(pair):
    assert find_duplicates(list(pair)) == [[0], [1]]


def test_duplicates_on_real_feed():
    data = load_json_fixture("nlp/headlines_xyz.json")
    titles = [strip_publisher_suffix(i["title"], i.get("publisher")) for i in data["items"]]
    groups = [g for g in find_duplicates(titles) if len(g) > 1]
    piper = [g for g in groups if titles[g[0]].startswith("Piper Sandler reiterates Block stock rating")]
    assert len(piper) == 1 and len(piper[0]) == 3  # original + two "By Investing.com" copies
    for g in groups:  # every group is one headline, give or take attribution/truncation
        words = [set(titles[i].lower().split()) for i in g]
        assert all(len(words[0] & w) / len(words[0] | w) >= 0.5 for w in words[1:])


def test_find_duplicates_scales():
    rng = random.Random(3)
    vocab = [f"w{k}" for k in range(400)]
    titles = [" ".join(rng.sample(vocab, 9)) for _ in range(1000)]
    titles += [t + " By Investing.com" for t in titles[:100]]
    start = time.perf_counter()
    groups = find_duplicates(titles)
    assert time.perf_counter() - start < 1.0
    assert sum(len(g) > 1 for g in groups) == 100


# --------------------------------------------------------------------------- #
# Review round 2: features and display (live NVDA, 2026-10-05)
# --------------------------------------------------------------------------- #
def _cluster(titles: list[str], company: CompanyRef = NVDA) -> list[Cluster]:
    items = [ClusterItem(id=str(i), title=t, timestamp=T0 + timedelta(hours=i), publisher="X")
             for i, t in enumerate(titles)]
    return cluster_narratives(items, company)


def test_currency_codes_join_the_same_buyback_story():
    clusters = _cluster(["Nvidia authorizes USD 150 billion increase to share repurchase program",
                         "Nvidia announces $150 billion buyback",
                         "Nvidia adds $150B to buyback, lifting total to $235B"])
    assert len(clusters) == 1
    assert clusters[0].terms[0] == "$150B"  # money shown upper-case, words not ("$150B buyback")
    assert all(t == t.split()[0] or not t.split()[-1].isupper() for t in clusters[0].terms)


def test_a_rivals_market_cap_milestone_is_not_the_companys_record_story():
    clusters = _cluster(["Nvidia stock hits record high, market cap nears $6 trillion",
                         "Nvidia hits record high as AI rally extends",
                         "AMD Reaches a $1 Trillion Market Cap. Can It Finally Dethrone Nvidia?",
                         "AMD hits $1 trillion market cap for first time"])
    groups = sorted(sorted(c.item_ids) for c in clusters)
    assert ["0", "1"] in groups and not any({"0", "2"} <= set(g) for g in groups)
    assert "record high high" not in clusters[0].terms


def test_story_terms_never_span_pronouns_or_punctuation():
    clusters = _cluster(["Nvidia CEO Jensen Huang Just Reaffirmed His Jaw-Dropping Projection for 2030",
                         "Jensen Huang Reaffirmed His Jaw-Dropping Projection: Here's Why",
                         "Nvidia Settles Advisor Dispute: Early Payout Expected"])
    terms = {t.lower() for c in clusters for t in c.terms}
    assert not {"reaffirmed jaw", "dispute early"} & terms, terms


# --------------------------------------------------------------------------- #
# Final review: over-merging on one shared word, fragmentation of continuing
# developments (live SOFI / AAPL / NVDA / GME, 2026-10-04)
# --------------------------------------------------------------------------- #
SOFI = CompanyRef(ticker="SOFI", name="SoFi Technologies, Inc.", short_name="SoFi", industry="Credit Services")
AAPL = CompanyRef(ticker="AAPL", name="Apple Inc.", short_name="Apple", industry="Consumer Electronics")
GME = CompanyRef(ticker="GME", name="GameStop Corp.", short_name="GameStop", industry="Specialty Retail")


def _timed(rows: list[tuple[str, float]], company: CompanyRef) -> list[list[str]]:
    """Cluster (title, hours after T0) rows; return the groups as sorted id lists."""
    items = [ClusterItem(id=str(i), title=t, timestamp=T0 + timedelta(hours=h), publisher="X")
             for i, (t, h) in enumerate(rows)]
    return sorted(sorted(c.item_ids) for c in cluster_narratives(items, company, max_clusters=len(items)))


def test_one_shared_word_is_not_a_story():
    # one cluster with terms ['Compelling Entry', 'Robinhood Sit', 'Technologies Stock', 'Sit', 'Peak']
    clusters = _cluster(["Nu Holdings Jumps 3% After Ruling Out Monzo Deal; SoFi and Robinhood Sit Out the Rally",
                         "SoFi Technologies (NASDAQ: SOFI) Stock Sits 50% Below Peak, Presenting A Compelling Entry "
                         "Point For Investors"], SOFI)
    assert len(clusters) == 2
    assert not any("Technologies" in t for c in clusters for t in c.terms)  # part of the company's name


def test_a_word_shared_by_two_product_names_is_not_a_story():
    groups = _timed([("Apple iPhone 18 Pro Max AT&T Glitch Requires Device Replacements", 0),
                     ("Apple Says Some AT&T iPhone 18 Pro Max Users Must Replace Phones After Service-Loss Bug", 1),
                     ("Apple to replace iPhone 18 Pro Max facing AT&T glitch (AAPL)", 2),
                     ("AAPL Stock Ends Week Lower — Apple's Vision Pro And Smart Glasses Head Reportedly Defects "
                      "To OpenAI", 3)], AAPL)
    assert ["0", "1", "2"] in groups and ["3"] in groups


def test_opposite_calls_from_different_firms_are_different_stories():
    groups = _timed([("Morgan Stanley lowers Apple stock price target on limited upside", 0),
                     ("Morgan Stanley Maintains Apple(AAPL.US) With Buy Rating, Cuts Target Price to $355", 1),
                     ("Apple stock gains 1.02 percent as Morgan Stanley trims target", 30),
                     ("Citi Initiates Apple(AAPL.US) With Buy Rating, Announces Target Price $365", 50)], AAPL)
    assert ["0", "1", "2"] in groups and ["3"] in groups


def test_a_buyback_raised_again_is_one_story_whatever_the_amount():
    # live NVDA: "$150 Billion Buyback Plan" and "$235 Billion Buyback" (the same program:
    # a $150B increase taking capacity to $235B) were two narratives and two catalysts
    groups = _timed([("Nvidia Stock Pops on $150 Billion Buyback Plan", 0),
                     ("Nvidia unveils massive $150B increase to share buyback program", 1),
                     ("Nvidia Stock Climbs After $150 Billion Buyback Bombshell", 2),
                     ("Nvidia's $235 Billion Buyback Shows Who Is Really Winning the AI Boom", 140),
                     ("Nvidia Wraps Its AI Empire in Insurance, Guardrails and a $235 Billion Buyback", 141)], NVDA)
    assert groups == [["0", "1", "2", "3", "4"]]


def test_an_insider_buying_spree_is_one_story_but_not_with_insider_selling():
    # live GME: six of eight narrative slots were the CEO's buys, one per amount
    groups = _timed([("GameStop CEO Ryan Cohen buys $26.4 million of GME stock", 0),
                     ("GameStop stock jumps as CEO Ryan Cohen buys $26 million in shares", 1),
                     ("Billionaire GameStop CEO Ryan Cohen Buys 1.2 Million Shares for $26.4 Million", 2),
                     ("GameStop shares rise as CEO Ryan Cohen buys $10.6 million in stock", 170),
                     ("GameStop CEO Ryan Cohen Buys $10.6M in Shares, GME Shares Rise", 171),
                     ("GameStop(GME.US) Officer Buys US$17.08 Million in Common Stock", 230),
                     ("GameStop Officer Daniel Moore Files Form 144 to Sell 7,297 Shares for RSU Tax Withholding", 231),
                     ("To cover vesting taxes, GameStop (GME) officer Daniel Moore proposes selling 7,297 shares.",
                      232)], GME)
    assert ["0", "1", "2", "3", "4", "5"] in groups
    assert ["6", "7"] in groups


def test_program_follow_ups_join_only_within_the_window():
    docs = [_Doc(vec={}, raw={}, anchors=frozenset(), surfaces={}, program=frozenset({"insider_buy"}))] * 4
    hour = 3600.0
    times: list[float | None] = [0.0, hour, (PROGRAM_SPAN_H - 1) * hour, (2 * PROGRAM_SPAN_H + 5) * hour]
    # [2] follows [1] within the window (a spree goes on); [3] comes long after the last follow-up
    assert sorted(_join_programs([[0], [1], [2], [3]], docs, times)) == [[0, 1, 2], [3]]


def test_another_companys_program_does_not_join():
    groups = _timed([("Nvidia Stock Pops on $150 Billion Buyback Plan", 0),
                     ("Nvidia unveils massive $150B increase to share buyback program", 1),
                     ("Nvidia stock dips as AMD announces $12 billion buyback", 30),
                     ("Nvidia slips as AMD unveils $12 billion buyback", 31)], NVDA)
    assert ["0", "1"] in groups and ["2", "3"] in groups
