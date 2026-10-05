"""Publisher identity, trust tiers and press-release detection."""
from __future__ import annotations

from collections import Counter

import pytest

from app.nlp.publishers import (
    PRESS_RELEASE_TRUST,
    UNKNOWN_TRUST,
    canonical_publisher,
    is_known_publisher,
    is_press_release,
    publisher_trust,
)
from tests.conftest import load_json_fixture


@pytest.mark.parametrize(("raw", "canonical"), [
    ("Barron's on MSN", "Barron's"),
    ("Yahoo! Finance Canada", "Yahoo Finance"),
    ("finance.yahoo.com", "Yahoo Finance"),
    ("Investing.com Nigeria", "Investing.com"),
    ("https://www.reuters.com/markets/us/nvidia-buyback", "Reuters"),
    ("https://www.cnbc.com/2026/a.html", "CNBC"),
    ("Bloomberg.com", "Bloomberg"),
    ("bloomberg.co.jp", "Bloomberg"),
    ("The Wall Street Journal", "WSJ"),
    ("Motley Fool", "The Motley Fool"),
    ("Zacks Investment Research", "Zacks"),
])
def test_canonical_names(raw, canonical):
    assert canonical_publisher(raw) == canonical
    assert is_known_publisher(raw)


def test_unknown_outlet_keeps_cleaned_name():
    assert canonical_publisher("Some&nbsp;Local Paper") == "Some Local Paper"
    assert not is_known_publisher("Some Local Paper")
    assert publisher_trust("Some Local Paper") == UNKNOWN_TRUST
    assert canonical_publisher(None) is None
    assert canonical_publisher("") is None
    assert publisher_trust(None) == UNKNOWN_TRUST


def test_trust_tiers_are_ordered():
    wire = publisher_trust("Reuters")
    mainstream = publisher_trust("Yahoo Finance")
    aggregator = publisher_trust("MarketBeat")
    auto_content = publisher_trust("ETF Daily News")
    assert wire > mainstream > aggregator > auto_content
    assert wire == publisher_trust("Bloomberg") == publisher_trust("WSJ")
    assert publisher_trust("GlobeNewswire") == PRESS_RELEASE_TRUST


@pytest.mark.parametrize("publisher", ["GlobeNewswire", "PR Newswire", "Business Wire", "NVIDIA Newsroom",
                                       "Apple Investor Relations"])
def test_press_release_channels(publisher):
    assert is_press_release(publisher)
    assert publisher_trust(publisher) == PRESS_RELEASE_TRUST


@pytest.mark.parametrize("publisher", ["Reuters", "techi.com", "Some Random Blog", "MarketBeat", None])
def test_not_press_release_channels(publisher):
    # Low-quality content sites share the low weight but are not wires.
    assert not is_press_release(publisher)


@pytest.mark.parametrize("title", [
    "SHAREHOLDER ALERT: Pomerantz Law Firm Investigates Claims On Behalf of Investors of Target Corporation - TGT",
    "Block, Inc. Announces Pricing of $1.5 Billion Senior Notes Offering",
    "ROSEN, A LEADING LAW FIRM, Encourages Investors with Losses to Secure Counsel",
])
def test_press_release_by_wording(title):
    assert is_press_release(title=title)


def test_news_wording_is_not_press_release():
    assert not is_press_release("WSJ", "Nvidia Adds Record $150 Billion to Stock Buyback")


def test_captured_outlets_are_mostly_known():
    """The table covers the outlets that actually show up in Google News
    results for large caps (500 captured headlines, 5 tickers)."""
    counts: Counter[str] = Counter()
    for ticker in ("nvda", "aapl", "meta", "tgt", "xyz"):
        for item in load_json_fixture(f"nlp/headlines_{ticker}.json")["items"]:
            counts[item["publisher"]] += 1
    known = sum(n for name, n in counts.items() if is_known_publisher(name))
    assert known / sum(counts.values()) >= 0.9


@pytest.mark.parametrize(("variants", "canonical"), [
    (("CCN.com", "CCN", "ccn.com"), "CCN"),                        # live: one outlet, not two
    (("Cryptonews.net", "Cryptonews"), "Cryptonews"),
    (("Decrypt News", "decrypt", "decrypt.co"), "Decrypt"),
    (("tradersunion", "Traders Union"), "Traders Union"),
    (("MT Newswires", "mtnewswires.com"), "MT Newswires"),
])
def test_name_and_domain_variants_are_one_outlet(variants, canonical):
    assert {canonical_publisher(v) for v in variants} == {canonical}


@pytest.mark.parametrize(("publisher", "title", "expected"), [
    ("CNBC", "Notice how Nvidia stock behaves", False),
    ("Reuters", "Nvidia today announced record revenue", False),  # wording never demotes a major newsroom
    ("PR Newswire", "Nvidia stock rises", True),
    ("Unknown Blog", "SHAREHOLDER ALERT: class action filed against Acme", True),
    ("Unknown Blog", "Notice how Nvidia stock behaves", False),
    (None, "Notice of Annual Meeting of Shareholders", True),
])
def test_press_release_wording_only_for_unknown_outlets(publisher, title, expected):
    assert is_press_release(publisher, title) is expected
