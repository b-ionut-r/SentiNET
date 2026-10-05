"""Dated catalyst list and the delta vs. the previous snapshot."""
from __future__ import annotations

import pytest

from app.analytics.build import build_analysis
from app.analytics.catalysts import describe_action, grade_polarity
from app.analytics.delta import build_delta
from app.schemas import SentimentStat, Verdict
from tests.analytics.factories import (
    GOOGLE,
    action,
    analysts,
    company,
    earnings,
    ex_dividend,
    filing,
    inputs,
    insider,
    insiders,
    news_flow,
    quote,
    run,
    snapshot,
)

ACME = company()
NEWS = news_flow([f"Acme schedules investor day number {i}" for i in range(6)])


def test_describe_action() -> None:
    assert describe_action(action(1, "UBS", "up", "Buy", 120, 100, "Neutral")) == (
        "UBS upgrades to Buy", "from Neutral · PT $100.00 → $120.00 (+20%)", "bull")
    assert describe_action(action(1, "Barclays", "main", "Equal-Weight", 90, 100))[0] == \
        "Barclays cuts target to $90.00 · Equal-Weight"
    assert describe_action(action(1, "Loop Capital", "init", "Hold", 22))[2] == "neutral"
    assert describe_action(action(1, "Loop Capital", "init", "Outperform", 22))[2] == "bull"
    assert grade_polarity("Market Perform") == "neutral" and grade_polarity("Underperform") == "bear"
    assert grade_polarity("Sector Outperform") == "bull"


def test_catalyst_ordering_and_filters() -> None:
    a = build_analysis(inputs(
        ACME, [run(GOOGLE, NEWS)], quote=quote(),
        earnings=earnings(days_until=12), calendar_catalysts=[ex_dividend(3)],
        analysts=analysts(actions=[action(2, "UBS", "up", "Buy", 120, 100, "Neutral"),
                                   action(4, "Bernstein", "reit", "Outperform", 130, 130),  # flat: skipped
                                   action(45, "Jefferies", "down", "Hold", 90, 110, "Buy")]),  # > 30 days
        insiders=insiders([insider(5, "Ann Lee", "buy", 250_000, "Director"),
                           insider(8, "Bo Chan", "sell", 4e6, "Chief Executive Officer"),
                           insider(9, "Cy Dee", "sell", 200_000)]),  # small sale: not a catalyst
        filings=[filing(6, "8-K", "Entered a material agreement", ["1.01"], "medium"),
                 filing(3, "8-K", "Shareholder vote results", ["5.07"], "low"),
                 filing(4, "10-Q", "Quarterly report", [], "medium")],
    ))
    kinds = [(c.kind, c.upcoming) for c in a.catalysts]
    assert kinds[:2] == [("dividend", True), ("earnings", True)]  # soonest upcoming first
    recent = [c for c in a.catalysts if not c.upcoming]
    assert [c.date for c in recent] == sorted((c.date for c in recent), reverse=True)
    titles = [c.title for c in a.catalysts]
    assert "UBS upgrades to Buy" in titles
    assert not any("Bernstein" in t or "Jefferies" in t for t in titles)
    assert "Ann Lee (Director) bought $250K" in titles
    assert "Bo Chan (Chief Executive Officer) sold $4M" in titles and not any("Cy Dee" in t for t in titles)
    assert "Entered a material agreement" in titles
    assert not any("vote" in t or "Quarterly" in t for t in titles)


def test_money_follows_what_each_figure_is_denominated_in() -> None:
    # Live: SHOP.TO (reports in USD) and VOD.L (in EUR) printed EPS estimates without a symbol, and every
    # non-USD listing printed insider values bare ("6 buys (240K)") although they are USD from every source.
    from app.schemas import Profile, Quote

    def run_for(currency: str, reporting: str | None):
        profile = Profile(symbol="X", name="X Corp", financial_currency=reporting)
        return build_analysis(inputs(ACME, [run(GOOGLE, NEWS)], profile=profile, earnings=earnings(days_until=12),
                                     quote=Quote(price=215.0, market_cap=280e9, currency=currency),
                                     analysts=analysts(actions=[action(2, "Wedbush", "main", "Outperform", 176, 160)]),
                                     insiders=insiders([insider(5, "Ann Lee", "buy", 250_000, "Director")])))

    shop = run_for("CAD", "USD")
    cat = {c.kind: c for c in shop.catalysts}
    assert cat["earnings"].detail.startswith("EPS est. $1.25 (range $1.10–$1.40) · revenue est. $5.2B")
    assert cat["insider"].title == "Ann Lee (Director) bought $250K"  # USD whatever the listing
    # Broker *action* targets on a cross-listed company quote its US line ($176 vs a C$215 price): % only.
    assert cat["analyst"].title == "Wedbush raises target (+10%) · Outperform"
    vod = run_for("GBp", "eur")
    assert {c.kind: c for c in vod.catalysts}["earnings"].detail.startswith("EPS est. €1.25")
    unknown = run_for("GBp", None)
    assert {c.kind: c for c in unknown.catalysts}["earnings"].detail.startswith("EPS est. 1.25 (range")  # never guessed
    assert "1 buy ($250K) / 0 sells ($0)" in next(c.detail for c in unknown.verdict.components if c.key == "insiders")
    adr = run_for("USD", "TWD")  # an ADR (TSM): per-ADR estimates are USD although the company reports in TWD
    assert {c.kind: c for c in adr.catalysts}["earnings"].detail.startswith("EPS est. $1.25")


def verdict(score: int, stance: str) -> Verdict:
    return Verdict(score=score, label="x", stance=stance, confidence="medium", confidence_value=0.5,  # type: ignore[arg-type]
                   headline="h")


def test_delta_notes() -> None:
    prev = snapshot(20, 50, 0.0, "neutral", price=100.0)
    sharp = build_delta(prev, verdict(64, "bullish"), SentimentStat(score=0.2), quote(price=104.0), [])
    assert sharp.sentinel_change == 14 and sharp.score_change == pytest.approx(0.2)
    assert sharp.note.startswith("Sentiment improved sharply (+14: 50 → 64), now bullish (was neutral) since")
    assert "price +4.0%" in sharp.note
    small = build_delta(prev, verdict(48, "neutral"), SentimentStat(score=-0.01), None, [])
    assert small.note.startswith("Little changed (−2)") and small.price_change_pct is None
    soft = build_delta(prev, verdict(43, "bearish"), SentimentStat(), quote(price=100.0), [])
    assert soft.note.startswith("Sentiment softened (−7: 50 → 43), now bearish")
    assert build_delta(None, verdict(60, "bullish"), SentimentStat(), None, []).note is None


# --------------------------------------------------------------------------- #
# News catalysts: only material developments, never noise (live findings)
# --------------------------------------------------------------------------- #
def _story(headline: str, lead_events: list[str], events: list[str] | None = None, impact: float = 0.6,
           score: float = 0.4, days_ago: float = 1.0):
    from datetime import timedelta

    from app.analytics.narratives import Story
    from app.analytics.prepare import Item
    from app.nlp.types import DetectedEvent
    from app.schemas import Narrative
    from tests.analytics.factories import NOW

    lead = Item(id="i-" + headline[:8], source="google_news", source_label="Google News", source_weight=1.0,
                kind="news", title=headline, scored=True, score=score,
                events=[DetectedEvent(key=k, polarity="bull") for k in lead_events])
    other = Item(id="j-" + headline[:8], source="bing_news", source_label="Bing News", source_weight=1.0,
                 kind="news", title=headline + " (copy)", scored=True, score=score)
    when = NOW - timedelta(days=days_ago)
    narrative = Narrative(id="n-" + headline[:8], headline=headline, count=3, publishers=["Reuters", "CNBC"],
                          score=score, first_seen=when, last_seen=when, impact=impact)
    return Story(narrative=narrative, members=[lead, other], material_events=events or lead_events)


def _news_catalysts(stories, analyst_view=None, earnings_view=None):
    from types import SimpleNamespace

    from app.analytics.catalysts import _news
    from tests.analytics.factories import NOW

    facts = SimpleNamespace(inputs=SimpleNamespace(analysts=analyst_view, earnings=earnings_view), now=NOW,
                            stories=stories, deal_in_play=None)
    return _news(facts)  # type: ignore[arg-type]


def test_news_catalysts_skip_noise() -> None:
    out = _news_catalysts([
        _story("Acme opens second factory in Texas", ["expansion"]),                          # kept
        _story("Acme director sells shares", ["insider_sell"]),                                 # Form 4 covers it
        _story("Dow, S&P 500, Nasdaq open higher; Acme at 52-week high", ["high_52w"]),          # tape, not news
        _story("Acme partners with Globex on chips", ["partnership"], impact=0.40),              # below the bar
        _story("Morgan Stanley lowers Acme price target", ["pt_cut", "product_launch"],         # analyst story
               events=["pt_cut", "product_launch"]),
    ], analyst_view=analysts(actions=[action(1, "Morgan Stanley", "main", "Overweight", 90, 100)]))
    assert [c.title for c in out] == ["Acme opens second factory in Texas"]


def test_retrospective_results_stories_are_not_fresh_catalysts() -> None:
    # Live: 'Target Q2 2025 Earnings: Results, Market Reaction & History' was dated as a fresh beat.
    e = earnings(days_until=40)  # last report ~91 days ago
    out = _news_catalysts([_story("Acme Q2 earnings: results, market reaction and history",
                                  ["earnings_beat", "guidance_raise"]),
                           _story("Acme beats estimates and raises guidance", ["earnings_beat", "buyback"])],
                          earnings_view=e)
    assert [c.title for c in out] == ["Acme beats estimates and raises guidance"]
    assert out[0].detail.startswith("Buyback")  # the stale results event is not claimed as fresh


def test_mixed_story_catalyst_is_neutral() -> None:
    mixed = _story("Acme faces sell signals as reversal looms", ["product_launch"], score=0.3)
    mixed.members[0].score = 0.0  # its headline carries no tone: the cluster mean is not its polarity
    mixed.core_share = 0.4
    (c,) = _news_catalysts([mixed])
    assert c.polarity == "neutral"


def test_long_8k_titles_are_split_into_label_and_trimmed_detail() -> None:
    desc = ("NVIDIA Corporation entered into a definitive agreement to acquire Hugging Face, Inc. The transaction "
            "includes an approximately $11.9 billion purchase price payable to Hugging Face shareholders, subject to "
            "certain adjustments and customary closing conditions.")
    a = build_analysis(inputs(ACME, [run(GOOGLE, NEWS)], quote=quote(), filings=[
        filing(3, "8-K", f"Other material event: {desc}", ["8.01"], "high"),
        filing(5, "8-K/A", "Amended: Director/officer departure or appointment: The Board appointed a new CFO.",
               ["5.02"], "high")]))
    by_title = {c.title: c for c in a.catalysts if c.kind == "filing"}
    assert set(by_title) == {"Other material event", "Director/officer departure or appointment (amended)"}
    detail = by_title["Other material event"].detail
    assert detail.startswith("Form 8-K · items 8.01 · NVIDIA Corporation entered") and detail.endswith("…")
    assert len(detail) < 200
