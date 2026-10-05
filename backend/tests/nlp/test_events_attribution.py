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
    "GME": CompanyRef(ticker="GME", name="GameStop Corp.", short_name="GameStop", industry="Specialty Retail"),
    "EBAY": CompanyRef(ticker="EBAY", name="eBay Inc.", short_name="eBay", industry="Internet Retail"),
    "TWLO": CompanyRef(ticker="TWLO", name="Twilio Inc.", short_name="Twilio", industry="Software - Infrastructure"),
    "SNPS": CompanyRef(ticker="SNPS", name="Synopsys, Inc.", short_name="Synopsys", industry="Software - Application"),
    "TGT": CompanyRef(ticker="TGT", name="Target Corporation", short_name="Target", industry="Discount Stores"),
    "MRNA": CompanyRef(ticker="MRNA", name="Moderna, Inc.", short_name="Moderna", industry="Biotechnology"),
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
    # a market or sector clause owns its own move (live AAPL headline)
    ("AAPL", "Apple's iPhone And Google Pixel Q2 Sales Shine Even As Global Smartphone Market Drops To 13-Year Low",
     []),
    ("AAPL", "Apple stock drops 3% as market falls", ["price_down"]),
    ("T", "Corning stock rises on $3B AT&T fiber deal", []),  # live: Corning's move, AT&T's deal
    # multi-sentence posts: subject-less statements inherit the company (or its products/executives)
    ("NVDA", "$NVDA YE target $325 • Q2 revenue +106% YoY • Data Center revenue +117% • Rubin ramp underway "
             "• ~$235B buyback authorization", ["buyback"]),
    ("AAPL", "Apple skips AI summit. Neither Tim Cook nor CEO John Ternus attended. Shares trade at 333.02, "
             "up 1.10% today", ["price_up"]),
])
def test_events_belong_to_their_subject(ticker, text, want):
    assert sorted(e.key for e in detect_events(text, C[ticker])) == sorted(want)


@pytest.mark.parametrize(("ticker", "text", "want"), [
    # live (GME feed, 2026-10-04): a deal belongs to both parties — the bidder
    # (as subject or possessor) as much as the target. GameStop's $56B eBay bid
    # was missed entirely: no pattern for a priced bid, and the person-led clause
    # ("As Ryan Cohen Pushes ...") was read as another subject.
    ("GME", "GME CEO Ryan Cohen May Reportedly Withdraw GameStop's $56B eBay Bid", ["m_and_a"]),
    ("GME", "GameStop Steps Up EBAY Exposure To 6.5% As Ryan Cohen Pushes $56B Takeover Vision", ["m_and_a"]),
    ("EBAY", "GameStop Steps Up EBAY Exposure To 6.5% As Ryan Cohen Pushes $56B Takeover Vision", ["m_and_a"]),
    ("GME", "GME's Ryan Cohen Isn't Done Chasing eBay, Remains Committed To Cracking A Deal", ["m_and_a"]),
    ("GME", "GME Reportedly Wants To Buy eBay But Retail Wonders How; eBay Stock Soars", ["m_and_a"]),
    ("EBAY", "GME Reportedly Wants To Buy eBay But Retail Wonders How; eBay Stock Soars", ["m_and_a", "price_up"]),
    ("GME", "The Clock Is Ticking on GameStop's eBay Acquisition Play as Warrants Near Expiration", ["m_and_a"]),
    # a person's clause still never makes another company's deal, or a price move, the company's
    ("AAPL", "Apple stock rises as Microsoft agrees to buy Activision", ["price_up"]),
    ("AAPL", "Apple stock rises as Warren Buffett's Berkshire buys Occidental stake", ["price_up"]),
    ("NVDA", "Nvidia stock falls as Elon Musk unveils new Tesla chip", ["price_down"]),
])
def test_deals_belong_to_every_party(ticker, text, want):
    assert sorted(e.key for e in detect_events(text, C[ticker])) == sorted(want)


@pytest.mark.parametrize(("ticker", "text", "want"), [
    # live TWLO/TGT (2026-10-05): CNBC's daily column title. A headline passive belongs
    # to the name right before it in its comma clause.
    ("TWLO", "Twilio downgraded, Synopsys upgraded: Wall Street's top analyst calls", ["analyst_downgrade"]),
    ("SNPS", "Twilio downgraded, Synopsys upgraded: Wall Street's top analyst calls", ["analyst_upgrade"]),
    ("TGT", "Target upgraded, Moderna downgraded: Wall Street's top analyst calls", ["analyst_upgrade"]),
    ("MRNA", "Target upgraded, Moderna downgraded: Wall Street's top analyst calls", ["analyst_downgrade"]),
    ("AAPL", "Apple slips as AMD stock downgraded to Sell", []),
    # coordinated names share the call; the company's own passive and active objects stay
    ("NVDA", "Nvidia, AMD downgraded at Citi on AI capex worries", ["analyst_downgrade"]),
    ("AMD", "Nvidia, AMD downgraded at Citi on AI capex worries", ["analyst_downgrade"]),
    ("TWLO", "Twilio downgraded at HSBC despite Muse-induced hype", ["analyst_downgrade"]),  # live
    ("TWLO", "HSBC Just Downgraded Twilio Stock. Here's Why.", ["analyst_downgrade"]),  # live
    ("TGT", "Target stock upgraded to Buy at HSBC as analyst sees traffic-driven recovery gaining momentum",
     ["analyst_upgrade"]),  # live
    ("TWLO", "Twilio gets downgraded to Reduce at HSBC", ["analyst_downgrade"]),
])
def test_passive_rating_changes_belong_to_the_name_before_them(ticker, text, want):
    assert sorted(e.key for e in detect_events(text, C[ticker])) == sorted(want)


def test_ticker_tags_do_not_hide_mentions():
    """Ticker tags are blanked for the patterns, but the company named only by
    its tag is still found as the subject."""
    events = detect_events("Citi Initiates Apple(AAPL.US) With Buy Rating, Announces Target Price $365", C["AAPL"])
    assert [(e.key, e.firm, e.value) for e in events] == [("analyst_initiate", "Citi", 365.0)]
    assert [e.key for e in detect_events("SoFi Technologies (NASDAQ: SOFI) Stock Craters 43% In 2026",
                                         C["SOFI"])] == ["price_down"]


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
