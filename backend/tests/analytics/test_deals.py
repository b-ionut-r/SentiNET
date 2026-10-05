"""A pending acquisition of the company leads the read; stale listing flags do not."""
from __future__ import annotations

from app.analytics.build import build_analysis
from app.analytics.deals import pending_deal
from tests.analytics.factories import (
    GOOGLE,
    NOW,
    analysts,
    company,
    filing,
    inputs,
    news_flow,
    quote,
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
    assert v.reasons[0].ref == "deal" and v.reasons[0].text.startswith("Pending acquisition by Action Acquisitions LLC")
    first = a.insights[0]
    assert first.severity == "alert" and first.title.startswith("Pending acquisition: merger agreement")
    titles = [i.title for i in a.insights]
    assert not [t for t in titles if t.startswith("Red-flag filing")]  # superseded / resolved listing notices
    comps = {c.key: c for c in v.components}
    plain_comps = {c.key: c for c in plain.verdict.components}
    assert abs(comps["analysts"].score - 50) < abs(plain_comps["analysts"].score - 50)
    assert "deal pending" in comps["analysts"].detail and "deal pending" in comps["technicals"].detail
    assert "pending acquisition dominates" in a.brief.summary
    assert a.brief.watch[0].startswith("Deal outcome: pending acquisition by Action Acquisitions LLC")
    # Without the deal, the unresolved delisting notice is still an alert.
    assert any(i.title.startswith("Red-flag filing: Delisting notice") for i in plain.insights)


def test_regained_compliance_clears_the_earlier_delisting_notice() -> None:
    found = build_analysis(inputs(GOPRO, [run(GOOGLE, NEWS)], filings=[COMPLIANCE, DELISTING])).insights
    assert not [i for i in found if i.title.startswith("Red-flag filing")]
    # A compliance notice dated *before* a new delisting notice does not clear it.
    later = filing(5, "8-K", "Delisting notice / listing-rule failure", ["3.01"], "high", "bear")
    again = build_analysis(inputs(GOPRO, [run(GOOGLE, NEWS)], filings=[later, COMPLIANCE])).insights
    assert [i for i in again if i.title.startswith("Red-flag filing: Delisting")]
