"""Whose event is it? `detect_events(text, company)` keeps only the events
that happened to `company`, so a bystander never inherits another firm's
bankruptcy, short report or price move.

Real live headlines (2026-10-04/05 feeds) are marked "live"; the rest are the
headline shapes that produced false red-flag alerts in review.
"""
from __future__ import annotations

import pytest

from app.nlp.events import detect_events
from app.nlp.relevance import explain_relevance
from app.sources.base import CompanyRef

C = {
    "T": CompanyRef(ticker="T", name="AT&T Inc.", short_name="AT&T", industry="Telecom Services"),
    "NVDA": CompanyRef(ticker="NVDA", name="NVIDIA Corporation", short_name="Nvidia", industry="Semiconductors"),
    "TSLA": CompanyRef(ticker="TSLA", name="Tesla, Inc.", short_name="Tesla", industry="Auto Manufacturers"),
    "AAPL": CompanyRef(ticker="AAPL", name="Apple Inc.", short_name="Apple", industry="Consumer Electronics"),
    "SMCI": CompanyRef(ticker="SMCI", name="Super Micro Computer, Inc.", short_name="Super Micro",
                       aliases=["Supermicro"], industry="Computer Hardware"),
    "AMD": CompanyRef(ticker="AMD", name="Advanced Micro Devices, Inc.", short_name="AMD", industry="Semiconductors"),
    "SATS": CompanyRef(ticker="SATS", name="EchoStar Corporation", short_name="EchoStar", industry="Telecom Services"),
    "META": CompanyRef(ticker="META", name="Meta Platforms, Inc.", short_name="Meta", aliases=["Facebook"],
                       industry="Internet Content & Information"),
    "NKLA": CompanyRef(ticker="NKLA", name="Nikola Corporation", short_name="Nikola", industry="Auto Manufacturers"),
    "F": CompanyRef(ticker="F", name="Ford Motor Company", short_name="Ford", industry="Auto Manufacturers"),
    "SOFI": CompanyRef(ticker="SOFI", name="SoFi Technologies, Inc.", short_name="SoFi", industry="Credit Services"),
}


@pytest.mark.parametrize(("ticker", "text", "want"), [
    # live: the headline that produced a fabricated "Bankruptcy" alert on T
    ("T", "EchoStar Unit Dish DBS Files for Bankruptcy After Delays In Crucial AT&T Transaction", []),
    ("SATS", "EchoStar Unit Dish DBS Files for Bankruptcy After Delays In Crucial AT&T Transaction", ["bankruptcy"]),
    ("T", "Dish DBS files for Chapter 11 bankruptcy as AT&T deal stalls", []),
    ("T", "AT&T says Dish bankruptcy won't hurt its deal", []),
    # appositives: the company only qualifies another entity
    ("NVDA", "Nvidia supplier Super Micro hit by Hindenburg short report", []),
    ("SMCI", "Nvidia supplier Super Micro hit by Hindenburg short report", ["short_report"]),
    ("NVDA", "Hindenburg short seller report targets Super Micro, a key Nvidia partner", []),
    ("TSLA", "Tesla rival Nikola files for bankruptcy", []),
    ("NKLA", "Tesla rival Nikola files for bankruptcy", ["bankruptcy"]),
    ("NVDA", "Nvidia partner Wistron faces delisting notice", []),
    ("NVDA", "Nvidia-backed CoreWeave prices $2B stock offering", []),
    ("NVDA", "Super Micro shares plunge as Hindenburg short report raises questions about Nvidia ties", []),
    ("TSLA", "Tesla stock rises. Nikola, which makes electric trucks, files for bankruptcy.", ["price_up"]),
    # whose price moved
    ("NVDA", "Nvidia stock falls 3% while AMD jumps 5%", ["price_down"]),
    ("AMD", "Nvidia stock falls 3% while AMD jumps 5%", ["price_up"]),
    ("AMD", "AMD stock jumps 5% as Nvidia falls 3%", ["price_up"]),
    ("NVDA", "AMD stock jumps 5% as Nvidia falls 3%", ["price_down"]),
    ("F", "Tesla stock rises 1.1% in September while Ford and GM fall over 10%.", ["price_down"]),  # live
    ("SOFI", "Nu Holdings Jumps 3% After Ruling Out Monzo Deal; SoFi and Robinhood Sit Out the Rally", []),  # live
    ("NVDA", "Intel, AMD Slide 4.2% as Nvidia Unveils N1X PC Chip", ["product_launch"]),  # live
    ("AAPL", "Apple Stock Falls After Samsung Unveils New Phone", ["price_down"]),
    # the company's own events survive, as subject or as object
    ("NVDA", "Nvidia shares fall after Hindenburg short report", ["price_down", "short_report"]),
    ("NVDA", "Nvidia beats estimates as AMD misses estimates", ["earnings_beat"]),
    ("AMD", "Nvidia beats estimates as AMD misses estimates", ["earnings_miss"]),
    ("AAPL", "Masimo sues Apple over patents", ["lawsuit"]),
    ("AAPL", "Burford stock jumps after jury orders Apple to pay", ["lawsuit"]),
    ("META", "Lawsuit filed against Meta over teen safety", ["lawsuit"]),
    ("NKLA", "Hindenburg discloses short position in Nikola", ["short_report"]),
    ("TSLA", "Layoffs hit Tesla as demand slows", ["layoffs"]),
    ("TSLA", "Tesla recalls 2 million vehicles over Autopilot", ["recall"]),
    ("AAPL", "Stock jumps 5% after Apple beats estimates", ["price_up", "earnings_beat"]),
    ("NVDA", "Nvidia Stock Rises, Intel Files For Bankruptcy", ["price_up"]),
    ("NVDA", "Nvidia and AMD unveil new chips", ["product_launch"]),
    ("NVDA", "Nvidia rises as rival AMD unveils new chip", []),
    # multi-sentence posts: subject-less statements inherit the company (or its products/executives)
    ("NVDA", "$NVDA YE target $325 • Q2 revenue +106% YoY • Data Center revenue +117% • Rubin ramp underway "
             "• ~$235B buyback authorization", ["buyback"]),
    ("AAPL", "Apple skips AI summit. Neither Tim Cook nor CEO John Ternus attended. Shares trade at 333.02, "
             "up 1.10% today", ["price_up"]),
])
def test_events_belong_to_their_subject(ticker, text, want):
    assert sorted(e.key for e in detect_events(text, C[ticker])) == sorted(want)


def test_without_company_every_event_is_kept():
    text = "Tesla rival Nikola files for bankruptcy"
    assert [e.key for e in detect_events(text)] == ["bankruptcy"]


def test_texts_not_naming_the_company_keep_their_events():
    """The owner is unknown; relevance decides whether the text counts."""
    assert [e.key for e in detect_events("Shares jump 8% after earnings beat", C["NVDA"])] == \
        ["price_up", "earnings_beat"]


@pytest.mark.parametrize(("ticker", "text"), [
    ("T", "EchoStar Unit Dish DBS Files for Bankruptcy After Delays In Crucial AT&T Transaction"),
    ("T", "EchoStar's Dish DBS files for bankruptcy protection; AT&T transaction delayed"),
    ("NVDA", "Nvidia supplier Super Micro hit by Hindenburg short report"),
    ("TSLA", "Tesla rival Nikola files for bankruptcy"),
    ("NVDA", "Nvidia-backed CoreWeave prices $2B stock offering"),
])
def test_bystander_mentions_are_secondary(ticker, text):
    """Kept as background (>= 0.35) but below the 0.5 that event insights need,
    so a bystander never gets a red-flag alert even when events are text-level."""
    result = explain_relevance(text, C[ticker])
    assert result.secondary and 0.35 <= result.score <= 0.45, result


@pytest.mark.parametrize(("ticker", "text"), [
    ("NVDA", "Nvidia and its suppliers rally"),
    ("NVDA", "Nvidia partners with OpenAI on data centers"),
    ("AAPL", "Apple's suppliers face tariff pressure"),
])
def test_own_partners_and_suppliers_stay_primary(ticker, text):
    result = explain_relevance(text, C[ticker])
    assert not result.secondary and result.score >= 0.8, result
