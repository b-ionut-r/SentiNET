"""A pending acquisition of the company leads the read; stale listing flags do not."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from app.analytics.build import build_analysis
from app.analytics.deals import deadlines, pending_deal, quoted_amount
from app.analytics.prepare import Item
from tests.analytics.factories import (
    GOOGLE,
    NOW,
    STOCKTWITS,
    analysts,
    company,
    filing,
    inputs,
    news_flow,
    post,
    quote,
    raw,
    run,
    technicals,
)

GOPRO = company("GPRO", "GoPro, Inc.", "GoPro")
NEWS = news_flow([f"GoPro schedules investor day number {i}" for i in range(10)])

# The live GPRO 8-Ks (Sep 2026), as decoded by the SEC transform.
AGREEMENT = filing(33, "8-K", "Entered a material agreement: GoPro, Inc., a Delaware corporation, entered into an "
                   "Agreement and Plan of Merger with Action Acquisitions LLC, a Delaware limited liability company, "
                   "and Starman Optical, Inc., a Delaware corporation and a wholly owned subsidiary of Parent.",
                   ["1.01", "9.01"], "high", "neutral")
TERMS = filing(34, "8-K", "Other material event: The Merger Agreement provides that, upon the terms and subject to "
               "the satisfaction or waiver of the conditions set forth therein, Merger Sub will merge with and into "
               "GoPro, with GoPro continuing as the surviving corporation and a subsidiary of Parent.",
               ["8.01", "9.01"], "high", "neutral")
DELISTING = filing(72, "8-K", "Delisting notice / listing-rule failure", ["3.01"], "high", "bear")
COMPLIANCE = filing(18, "8-K", "Other material event: The Company received notice from Nasdaq that it has regained "
                    "compliance with the minimum bid price requirement under the Nasdaq Listing Rules.",
                    ["8.01", "9.01"], "high", "bear")


def test_target_side_merger_agreement_is_a_pending_deal() -> None:
    deal = pending_deal([TERMS, AGREEMENT, DELISTING], GOPRO, NOW.date())
    assert deal is not None and deal.buyer == "Action Acquisitions LLC"
    assert deal.filed == AGREEMENT.date and deal.items == ("1.01", "9.01")  # the agreement entry is quoted


def test_acquirer_side_and_closed_or_broken_deals_are_not_pending() -> None:
    acme = company()
    buyer = filing(10, "8-K", "Entered a material agreement: Acme Corporation entered into an Agreement and Plan of "
                   "Merger with Widget Inc. and Merger Sub, a wholly owned subsidiary of the Company. Merger Sub will "
                   "merge with and into Widget, with Widget surviving as a wholly owned subsidiary of the Company.",
                   ["1.01"], "high", "neutral")
    assert pending_deal([buyer], acme, NOW.date()) is None
    closed = filing(3, "8-K", "Completed an acquisition or disposition: the Company consummated the merger "
                    "contemplated by the Merger Agreement.", ["2.01", "3.01", "5.01"], "high", "neutral")
    assert pending_deal([AGREEMENT, TERMS, closed], GOPRO, NOW.date()) is None
    broken = filing(5, "8-K", "Terminated a material agreement: the Company terminated the Merger Agreement.",
                    ["1.02"], "high", "bear")
    assert pending_deal([AGREEMENT, TERMS, broken], GOPRO, NOW.date()) is None
    old = filing(400, "8-K", AGREEMENT.title, ["1.01"], "high", "neutral")
    assert pending_deal([old], GOPRO, NOW.date()) is None


def test_pending_deal_leads_the_read_and_discounts_price_anchored_components() -> None:
    # Live GPRO: the verdict leaned on one analyst's stale $0.50 Sell target while the merger
    # agreement rendered only as grey dots on the chart; delisting/compliance 8-Ks were alerts.
    base = dict(quote=quote(price=1.35, market_cap=2.8e8), technicals=technicals(r1m=-20, r3m=88, vs200=32),
                analysts=analysts(mean=4.0, total=1, upside=-63.0, price=1.35))
    a = build_analysis(inputs(GOPRO, [run(GOOGLE, NEWS)], filings=[COMPLIANCE, AGREEMENT, TERMS, DELISTING], **base))
    plain = build_analysis(inputs(GOPRO, [run(GOOGLE, NEWS)], filings=[DELISTING], **base))
    v = a.verdict
    assert v.headline.startswith(f"{v.label}, but a pending acquisition dominates: GoPro agreed to be acquired by "
                                 f"Action Acquisitions LLC")
    assert v.reasons[0].ref == "deal" and v.reasons[0].text.startswith(
        "Agreed to be acquired by Action Acquisitions LLC (merger agreement, 8-K ")
    first = a.insights[0]
    assert first.severity == "alert" and first.title.startswith("Pending acquisition: merger agreement")
    assert first.kind == "deal"  # a deal, not a risk (the schema's own insight kind)
    assert first.detail.startswith("GoPro agreed to be acquired by Action Acquisitions LLC. Form 8-K")
    assert "(the “" not in first.detail and '("' not in first.detail  # no defined-term legalese
    titles = [i.title for i in a.insights]
    assert not [t for t in titles if t.startswith("Red-flag filing")]  # superseded / resolved listing notices
    comps = {c.key: c for c in v.components}
    plain_comps = {c.key: c for c in plain.verdict.components}
    assert abs(comps["analysts"].score - 50) < abs(plain_comps["analysts"].score - 50)
    assert "deal pending" in comps["analysts"].detail and "deal pending" in comps["technicals"].detail
    assert "pending acquisition dominates" in a.brief.summary
    assert a.brief.watch[0].startswith("Deal outcome: pending acquisition by Action Acquisitions LLC")
    assert not any(b.startswith(("Extended", "Overbought", "Oversold")) for b in a.brief.bear_points + a.brief.bull_points)
    # Without the deal, the unresolved delisting notice is still an alert.
    assert any(i.title.startswith("Red-flag filing: Delisting notice") for i in plain.insights)


def test_regained_compliance_clears_the_earlier_delisting_notice() -> None:
    found = build_analysis(inputs(GOPRO, [run(GOOGLE, NEWS)], filings=[COMPLIANCE, DELISTING])).insights
    assert not [i for i in found if i.title.startswith("Red-flag filing")]
    # A compliance notice dated *before* a new delisting notice does not clear it.
    later = filing(5, "8-K", "Delisting notice / listing-rule failure", ["3.01"], "high", "bear")
    again = build_analysis(inputs(GOPRO, [run(GOOGLE, NEWS)], filings=[later, COMPLIANCE])).insights
    assert [i for i in again if i.title.startswith("Red-flag filing: Delisting")]


# --------------------------------------------------------------------------- #
# A deal in play (news): the company as bidder, target or partner
# --------------------------------------------------------------------------- #
GAMESTOP = company("GME", "GameStop Corp.", "GameStop")
# Live GME (2026-10-04): the $56B eBay bid (~4.5x GameStop's market cap) ranked as a minor story.
EBAY_BID = [
    raw("GameStop's $56B eBay takeover offer draws skepticism", 30, "Reuters"),
    raw("Cohen doubles down on GameStop's $56B takeover offer for eBay", 20, "Bloomberg"),
    raw("GameStop holders back offering of new shares to fund takeover offer for eBay", 40, "CNBC"),
    raw("GameStop takeover offer: eBay warrants expire on Oct 30", 50, "MarketWatch"),
]
QUIET = news_flow([f"GameStop schedules store event number {i}" for i in range(8)], start=60)


def gme(market_cap: float = 12.5e9, news=None, **kw):
    return build_analysis(inputs(GAMESTOP, [run(GOOGLE, (news if news is not None else EBAY_BID) + QUIET)],
                                 quote=quote(price=24.7, market_cap=market_cap), **kw))


def test_company_deal_coverage_is_a_deal_in_play() -> None:
    a = gme()
    ins = a.insights[0]
    assert (ins.kind, ins.severity, ins.polarity) == ("deal", "alert", "neutral")
    assert ins.title == "Deal in play: $56B, 4.5× its market cap"
    assert "4 articles from 4 outlets on M&A involving GameStop" in ins.detail
    assert "a deal this size would transform the company" in ins.detail
    assert a.verdict.headline.endswith("; a $56B deal (4.5× its market cap) is in play.")
    assert "A deal is in play:" in a.brief.summary and "quoted at $56B, 4.5× its market cap" in a.brief.summary
    assert a.brief.watch[0].startswith("Deal in play ($56B, 4.5× its market cap): ")
    # The story itself gets full intensity and leads; its catalyst carries the quoted value.
    deal_story = next(n for n in a.narratives if "m_and_a" in n.events)
    assert a.narratives[0].id == deal_story.id
    story_catalyst = next(c for c in a.catalysts if c.kind == "news" and not c.upcoming)
    assert "quoted $56B, 4.5× its market cap" in (story_catalyst.detail or "")
    # New shares to fund it are a dilution risk, and the explicit deadline an upcoming catalyst.
    dilution = next(i for i in a.insights if i.title == "Dilution risk: new shares to fund the deal")
    assert dilution.kind == "risk" and dilution.polarity == "bear" and "offering of new shares" in dilution.detail
    deadline = next(c for c in a.catalysts if c.title.startswith("Deal deadline"))
    assert deadline.upcoming and deadline.date.date() == date(2026, 10, 30)
    assert deadline.title == "Deal deadline: warrants expire"


def test_deal_in_play_severity_follows_its_size_vs_the_company() -> None:
    assert gme(market_cap=300e9).insights[0].title == "Deal in play: $56B, 19% of its market cap"  # material: alert
    mid = next(i for i in gme(market_cap=1e12).insights if i.kind == "deal")
    assert mid.severity == "watch" and mid.title == "Deal in play: $56B, 5.6% of its market cap"
    assert not [i for i in gme(market_cap=10e12).insights if i.kind == "deal"]  # a bolt-on for this company
    # Without a quoted value it is still in play, but never an alert.
    unpriced = [replace(r, title=r.title.replace("$56B ", "")) for r in EBAY_BID]
    plain = next(i for i in gme(news=unpriced).insights if i.kind == "deal")
    assert plain.severity == "watch" and plain.title == "Deal in play" and "No deal value is quoted yet" in plain.detail


def test_stale_or_uncorroborated_deal_talk_is_not_in_play() -> None:
    stale = [replace(r, timestamp=r.timestamp - timedelta(days=9)) for r in EBAY_BID]
    assert not [i for i in gme(news=stale).insights if i.kind == "deal"]
    single = [replace(r, publisher="The Motley Fool") for r in EBAY_BID]
    assert not [i for i in gme(news=single).insights if i.kind == "deal"]
    # A signed agreement to be acquired (8-K) leads instead; the news view is not repeated.
    agreement = filing(10, "8-K", "Entered a material agreement: GameStop Corp. entered into an Agreement and Plan "
                       "of Merger with Parent Holdings Inc. and Merger Sub, a wholly owned subsidiary of Parent.",
                       ["1.01"], "high", "neutral")
    kinds = [i.title for i in gme(filings=[agreement]).insights if i.kind == "deal"]
    assert len(kinds) == 1 and kinds[0].startswith("Pending acquisition")


def test_quoted_deal_value_and_explicit_deadlines() -> None:
    assert quoted_amount(["Withdraw GameStop’s $56B eBay Bid", "Ryan Cohen Pushes $56B Takeover Vision",
                          "Cohen's $10.6 Million Stock Purchase"]) == (56e9, "USD")
    assert quoted_amount(["Rival agrees to acquire Foo for $1.2 billion"]) == (1.2e9, "USD")
    assert quoted_amount(["takeover offer valued at £9 billion"]) == (9e9, "GBP")
    assert quoted_amount(["CEO buys $26.4 million of stock", "$2B convertible notes offer"]) is None

    def item(title: str, hours: float, social: bool = False) -> Item:
        return Item(id=title, source="x", source_label="x", source_weight=1.0, kind="social" if social else "news",
                    title=title, timestamp=NOW - timedelta(hours=hours), publisher="Reuters")

    found = deadlines([item("The tender offer expires in 26 days", 48), item("Warrants expire on Nov 2", 5),
                       item("The warrants expire in 3 days", 2, social=True),  # a post never dates a deadline
                       item("Shareholder vote is due on Sep 1", 5),  # already past
                       item("As warrants near expiration", 5)], NOW.date())
    assert [(d.when, d.what) for d in found] == [(date(2026, 10, 26), "Tender offer expires"),
                                                (date(2026, 11, 2), "Warrants expire")]
    assert deadlines([item("Vote ends on Jan 15", 5)], NOW.date())[0].when == date(2027, 1, 15)
    # End to end, a StockTwits post's countdown never becomes a catalyst.
    social = run(STOCKTWITS, [post("$GME the warrants expire in 26 days", 1, "ape")])
    a = build_analysis(inputs(GAMESTOP, [run(GOOGLE, EBAY_BID + QUIET), social], quote=quote(market_cap=12.5e9)))
    assert [c.date.date() for c in a.catalysts if c.title.startswith("Deal deadline")] == [date(2026, 10, 30)]


def test_only_deal_coverage_dominates_while_the_company_is_being_acquired() -> None:
    # Live GPRO: 'A pending acquisition dominates: …' was followed by 'The dominant story is ‘GoPro Inc. stock
    # outperforms competitors on strong trading day’: 2 articles from 1 outlet' (an auto-written recap).
    recap = [raw("GoPro Inc. stock outperforms competitors on strong trading day", 2 + i, "MarketWatch")
             for i in range(2)]
    other = news_flow([f"GoPro beats estimates as strong camera demand surges {i}" for i in range(6)])
    a = build_analysis(inputs(GOPRO, [run(GOOGLE, NEWS + recap + other)], filings=[AGREEMENT, TERMS]))
    assert "pending acquisition dominates" in a.brief.summary
    assert "dominant story" not in a.brief.summary and "outperforms competitors" not in a.brief.summary
    deal_news = news_flow([f"GoPro merger vote set as Action Acquisitions deal nears {i}" for i in range(6)])
    b = build_analysis(inputs(GOPRO, [run(GOOGLE, NEWS + deal_news)], filings=[AGREEMENT, TERMS]))
    assert "The dominant story is ‘GoPro merger vote set" in b.brief.summary
