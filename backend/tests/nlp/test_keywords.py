"""Keyword chips on real headline sets (Google News, 2026-10-04)."""
from __future__ import annotations

import time

import pytest

from app.nlp.keywords import extract_keywords
from app.nlp.text import strip_publisher_suffix
from app.sources.base import CompanyRef
from tests.conftest import load_json_fixture

NVDA = CompanyRef(ticker="NVDA", name="NVIDIA Corporation", short_name="Nvidia")


def _feed(ticker: str) -> tuple[list[str], CompanyRef]:
    data = load_json_fixture(f"nlp/headlines_{ticker}.json")
    titles = [strip_publisher_suffix(i["title"], i.get("publisher")) for i in data["items"]
              if i.get("relevant") is not False]
    return titles, CompanyRef(**data["company"])


def _terms(keywords: list[tuple[str, int, float]]) -> list[str]:
    return [term for term, _count, _score in keywords]


@pytest.mark.parametrize(("ticker", "expected"), [
    ("nvda", {"buyback", "record high", "Morgan Stanley", "Jensen Huang", "market cap", "top pick"}),
    ("tgt", {"price cuts", "price target", "holiday", "turnaround"}),
    ("meta", {"Muse", "enterprise", "MongoDB CEO", "OpenAI", "privacy"}),
    ("tsla", {"deliveries", "estimates", "credit lines", "Roadster"}),
    ("amzn", {"data centers", "nuclear power", "Constellation Energy", "Goldman Sachs", "conviction list"}),
    ("amd", {"World Labs", "Nvidia", "HPE"}),
])
def test_real_feeds_surface_the_stories(ticker, expected):
    titles, company = _feed(ticker)
    terms = _terms(extract_keywords(titles, None, company))
    assert expected <= set(terms), terms


@pytest.mark.parametrize("ticker", ["nvda", "aapl", "meta", "tgt", "xyz", "tsla", "amzn", "amd"])
def test_chips_exclude_company_names_filler_and_publishers(ticker):
    titles, company = _feed(ticker)
    terms = _terms(extract_keywords(titles, None, company))
    words = {w.lower() for t in terms for w in t.split()}
    own = {company.short_name.lower(), company.ticker.lower()}
    concepts = {"price target", "price cuts", "record high", "market cap", "top pick", "data centers"}
    assert not {w.lower() for t in terms if t not in concepts for w in t.split()} & own, terms
    assert not words & {"stock", "stocks", "shares", "investors", "says", "today", "why", "yahoo", "zacks",
                        "benzinga", "usd", "nasdaq"}, terms
    for a in terms:  # no chip restates another ("Musk" / "Elon Musk", "target" / "price target")
        for b in terms:
            assert a == b or not set(a.lower().split()) <= set(b.lower().split()), (a, b)


def test_syndicated_copies_count_once():
    copies = ["Nvidia adds record $150 billion to stock buyback"] * 3 + [
        "Nvidia adds record $150 billion to stock buyback By Investing.com",
        "Nvidia buyback: what the $150 billion plan means",
        "Morgan Stanley names Nvidia a top pick",
        "Morgan Stanley sees more upside for Nvidia",
    ]
    keywords = dict((t, c) for t, c, _s in extract_keywords(copies, None, NVDA))
    assert keywords["buyback"] == 2  # five headlines, two distinct
    assert keywords["Morgan Stanley"] == 2


def test_counts_and_mean_scores():
    texts = ["Acme announces share buyback", "Acme layoffs hit 500 workers", "Acme buyback lifts shares",
             "Acme layoffs spread"]
    scores = [0.5, -0.6, 0.7, -0.2]
    out = {t: (c, s) for t, c, s in extract_keywords(texts, scores, CompanyRef(ticker="ACME", name="Acme Inc.",
                                                                                  short_name="Acme"))}
    assert out["buyback"] == (2, 0.6)
    assert out["layoffs"] == (2, -0.4)
    assert "Acme" not in out


def test_bigrams_and_concepts_are_preferred():
    texts = ["Morgan Stanley raises price target on Acme", "Acme price target cut by Morgan Stanley",
             "Acme expands data centers in Texas", "Acme data center capex doubles",
             "Acme buys back stock in share repurchase", "Acme announces buyback"]
    terms = _terms(extract_keywords(texts, None, CompanyRef(ticker="ACME", name="Acme Inc.", short_name="Acme")))
    assert {"Morgan Stanley", "price target", "data centers", "buyback"} <= set(terms)
    assert not {"Morgan", "Stanley", "price", "target", "data"} & set(terms)


def test_capitalization_reflects_usage():
    titles, company = _feed("nvda")
    terms = set(_terms(extract_keywords(titles, None, company, top_n=30)))
    assert "Michael Burry" in terms and "Jensen Huang" in terms  # names
    assert "chips" in terms and "Chips" not in terms  # written lower-case in sentence-case headlines
    titles, company = _feed("amzn")
    terms = set(_terms(extract_keywords(titles, None, company, top_n=30)))
    assert "Synopsys" in terms and "Anthropic" in terms  # only ever capitalized
    titles, company = _feed("aapl")
    assert "iPhone" in _terms(extract_keywords(titles, None, company))


def test_arguments_edges_and_limits():
    texts = ["Acme buyback approved", "Acme buyback expanded"]
    company = CompanyRef(ticker="ACME", name="Acme Inc.", short_name="Acme")
    assert extract_keywords([], None, company) == []
    # (texts, company, scores) order is accepted too
    assert extract_keywords(texts, company, [0.4, 0.2]) == extract_keywords(texts, [0.4, 0.2], company)
    assert extract_keywords(texts, [0.4], company)[0] == ("buyback", 2, 0.2)  # short score list -> 0.0 for the rest
    titles, nv = _feed("nvda")
    assert len(extract_keywords(titles, None, nv, top_n=5)) == 5
    counts = [c for _t, c, _s in extract_keywords(titles, None, nv, top_n=40)]
    assert min(counts) >= 2  # with 8+ texts, one-off terms are dropped


def test_fast_for_500_texts():
    texts: list[str] = []
    for ticker in ("nvda", "aapl", "meta", "tsla", "amzn", "amd"):
        texts += _feed(ticker)[0]
    texts = texts[:500]
    start = time.perf_counter()
    assert extract_keywords(texts, [0.1] * len(texts), NVDA)
    assert time.perf_counter() - start < 1.0  # ~0.1 s measured


def test_two_letter_acronyms_make_chips():
    texts = ["Nvidia AI chips demand", "AI chips boom lifts Nvidia", "Nvidia Q3 revenue beats",
             "Nvidia third-quarter revenue tops estimates", "AI boom: Nvidia AI chips sold out", "EU fines Nvidia",
             "EU probes Nvidia deal"]
    terms = set(_terms(extract_keywords(texts, None, NVDA)))
    assert "AI chips" in terms and "EU" in terms and any(t.startswith("Q3") for t in terms), terms


def test_everyday_words_inside_a_long_company_name_stay():
    fslr = CompanyRef(ticker="FSLR", name="First Solar, Inc.", short_name="First Solar")
    texts = ["First Solar shares jump as solar demand surges", "Solar stocks rally; First Solar leads",
             "First Solar wins solar panel order", "Tariffs lift US solar makers", "First Solar raises outlook",
             "Solar installers see record quarter"]
    terms = set(_terms(extract_keywords(texts, None, fslr)))
    assert "solar" in {t.lower() for t in terms} and not {"First Solar", "First"} & terms, terms
    aal = CompanyRef(ticker="AAL", name="American Airlines Group Inc.", short_name="American Airlines")
    texts = ["American Airlines cuts flights as airlines face fuel costs", "Airlines rally on lower oil",
             "Delta, United, American report strong bookings", "Airlines brace for strike"]
    terms = set(_terms(extract_keywords(texts, None, aal)))
    assert "airlines" in {t.lower() for t in terms}, terms
    assert "Delta United" not in terms  # bigrams never span punctuation


def test_generic_verbs_are_not_chips():
    texts = ["Acme continues to support the rally", "Acme acquires 512 stock units", "Acme files new claim",
             "Acme support level holds", "Acme continues buyback", "Acme files patent", "Acme acquires startup",
             "Acme changed its plan", "Acme reportedly plans buyback"]
    terms = {t.lower() for t in _terms(extract_keywords(texts, None, CompanyRef(ticker="ACME", name="Acme Inc.",
                                                                                   short_name="Acme")))}
    assert not {"continues", "continue", "support", "acquires", "files", "claim", "units", "changed",
                "reportedly"} & terms, terms
    assert "buyback" in terms


def test_roles_labels_and_generic_nouns_are_no_chip_alone():
    # live (GME, AAPL, TGT, LULU): "officer", "director", "money", "hours", "CEO", "Pro",
    # "products", "items", "bullish", "bears" padded the chips
    gme = CompanyRef(ticker="GME", name="GameStop Corp.", short_name="GameStop")
    titles = ["GameStop director Nat Turner buys $254,540 in stock",
              "GameStop Director Nat Turner Acquires 10,462 Shares",
              "GameStop officer proposes selling shares to cover taxes",
              "GameStop Officer Files Form 144 to cover taxes",
              "GME Pops 2% After-Hours After CEO Buys Shares",
              "GME Rises After Hours as CEO adds to stake",
              "Bullish traders pile into GameStop as bears retreat",
              "Bulls and bears battle over GameStop products and items",
              "GameStop money: CEO has $1B in cash",
              "GameStop is putting its cash and money to work"]
    terms = {t.lower() for t in _terms(extract_keywords(titles, None, gme))}
    assert "nat turner" in terms
    assert not terms & {"officer", "director", "ceo", "hours", "after-hours", "bullish", "bears", "bulls", "money",
                        "cash", "products", "items"}, terms


def test_weak_words_still_count_next_to_a_name():
    titles = ["MongoDB CEO Dev Ittycheria joins Meta to lead enterprise AI",
              "Meta hires MongoDB CEO for enterprise push",
              "Meta taps MongoDB CEO as enterprise chief"]
    terms = _terms(extract_keywords(titles, None, CompanyRef(ticker="META", name="Meta Platforms, Inc.",
                                                             short_name="Meta")))
    assert "MongoDB CEO" in terms, terms
