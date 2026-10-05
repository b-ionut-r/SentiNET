"""End-to-end analytics on six realistic situations, judged like an analyst would."""
from __future__ import annotations

import re

import pytest

from app.analytics.build import build_analysis
from app.schemas import Analysis
from tests.analytics import scenarios
from tests.analytics.factories import NOW, snapshot

DIGIT = re.compile(r"\d")


def build(name: str) -> Analysis:
    return build_analysis(scenarios.ALL[name]())


def comp(a: Analysis, key: str):
    return next(c for c in a.verdict.components if c.key == key)


def insight(a: Analysis, title_part: str):
    return next((i for i in a.insights if title_part.lower() in i.title.lower()), None)


# --------------------------------------------------------------------------- #
# Invariants that must hold for every situation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(scenarios.ALL))
def test_invariants(name: str) -> None:
    inputs = scenarios.ALL[name]()
    a = build_analysis(inputs)
    v = a.verdict
    assert a.generated_at == inputs.now == NOW
    assert 0 <= v.score <= 100 and 0.0 <= v.confidence_value <= 1.0
    assert [c.key for c in v.components] == ["news", "social", "analysts", "insiders", "momentum", "technicals"]
    for c in v.components:
        assert c.available == (c.score is not None)
        assert c.score is None or 0 <= c.score <= 100
    # Every reason, insight and brief point is backed by at least one number (data-gap notes excepted).
    for text in [r.text for r in v.reasons] + [i.detail for i in a.insights if i.kind != "quality"] \
            + a.brief.bull_points + a.brief.bear_points:
        assert DIGIT.search(text), text
    if any(c.available for c in v.components):
        assert DIGIT.search(v.headline), v.headline
    assert len(a.insights) <= 8 and len(v.reasons) <= 4 and len(a.signals) <= 200
    assert all(0 < r.weight <= 1 for r in v.reasons)
    # Narratives are ranked by impact and their members are present among the signals.
    impacts = [n.impact for n in a.narratives]
    assert impacts == sorted(impacts, reverse=True)
    ids = {s.id for s in a.signals}
    for n in a.narratives:
        assert set(n.signal_ids) <= ids
        assert all(s.narrative_id == n.id for s in a.signals if s.id in n.signal_ids)
    # The JSON contract round-trips.
    assert Analysis.model_validate_json(a.model_dump_json()) == a


@pytest.mark.parametrize("name", list(scenarios.ALL))
def test_deterministic(name: str) -> None:
    assert build(name).model_dump_json() == build(name).model_dump_json()


# --------------------------------------------------------------------------- #
# 1. Bullish large cap: upgrades, beat, improving global tone
# --------------------------------------------------------------------------- #
def test_bullish_large_cap_reads_bullish_with_evidence() -> None:
    a = build("bullish_large_cap")
    v = a.verdict
    assert v.score >= 70 and v.stance == "bullish" and v.label in ("Bullish", "Strongly Bullish")
    assert v.confidence in ("medium", "high")
    assert "Strong Buy" in v.headline and "+25%" in v.headline
    assert all(r.polarity == "bull" for r in v.reasons[:3])
    analyst_reason = next(r for r in v.reasons if r.ref == "analysts")
    assert "40 analysts" in analyst_reason.text and "25%" in analyst_reason.text
    assert any(r.text.startswith("Top story") and "Morgan Stanley upgrades" in r.text for r in v.reasons)

    for key in ("news", "analysts", "momentum", "technicals"):
        assert comp(a, key).score > 60, key
    assert 40 <= comp(a, "insiders").score <= 50  # routine selling at a mega-cap is mild

    smart = insight(a, "Analysts turning more bullish")
    assert smart is not None and smart.polarity == "bull"
    assert "Morgan Stanley" in smart.detail and "upgrade" in smart.detail
    assert insight(a, "90-day high") is not None
    assert not [i for i in a.insights if i.kind == "risk"]

    # Catalysts: earnings first (upcoming), then dated analyst actions with target moves.
    assert a.catalysts[0].kind == "earnings" and a.catalysts[0].upcoming
    ms = next(c for c in a.catalysts if c.kind == "analyst" and c.title.startswith("Morgan Stanley"))
    assert ms.polarity == "bull" and "$220.00 → $260.00" in (ms.detail or "")
    assert not any(c.title.startswith("Bernstein") for c in a.catalysts)  # a flat reiteration is noise
    assert len(a.brief.bull_points) >= 3 and "Strong Buy" in a.brief.summary


def test_bullish_case_still_states_the_material_negatives() -> None:
    # Live NVDA: 5 bull points vs 1 bear point although insiders sold $1.37B with no buys.
    a = build("bullish_large_cap")
    bear = a.brief.bear_points
    selling = next((b for b in bear if b.startswith("Insider selling")), None)
    assert selling is not None
    assert "2 open-market sales worth $40M" in selling and "bought nothing in 180 days" in selling
    assert "largest: John Doe (Director) $25M" in selling
    assert "(0.01% of market cap)" in selling and "routine-sized for the company" in selling
    # Pre-scheduled (10b5-1) selling is claimed only when the Form 4 filings say so.
    assert "10b5-1" not in selling
    inputs = scenarios.bullish_large_cap()
    assert inputs.insiders is not None
    for t in inputs.insiders.transactions:
        if t.kind == "sell" and t.insider == "John Doe":
            t.text = "Sale at price 120.00 per share. (10b5-1 plan)"
    planned = next(b for b in build_analysis(inputs).brief.bear_points if b.startswith("Insider selling"))
    assert "1 of 2 sales under a pre-arranged 10b5-1 trading plan" in planned
    # Counterpoints rank below everything that moves the score; the bull case is unchanged.
    assert a.brief.bull_points[0].startswith(("Analysts", "News", "Story", "Price", "Sentiment"))
    assert all(any(ch.isdigit() for ch in b) for b in bear)


@pytest.mark.parametrize("name", ["bullish_large_cap", "meme_stock", "crypto", "lawsuit"])
def test_price_confirms_sentiment_but_does_not_outshout_it(name: str) -> None:
    # Live finding: technicals were the largest contributor in 5 of 9 tickers (a normal uptrend read 75-85).
    a = build(name)
    c = next(x for x in a.verdict.components if x.key == "technicals")
    if not c.available:
        return
    contributions = {x.key: abs((x.score or 50) - 50) * x.weight for x in a.verdict.components if x.available}
    share = contributions["technicals"] / (sum(contributions.values()) or 1)
    nominal = c.weight / sum(x.weight for x in a.verdict.components if x.available)
    assert share <= nominal * 1.6, (share, nominal)


def test_news_counts_share_one_base() -> None:
    # Live NVDA: '197 articles' counted syndicated copies while '66 bullish vs 14 bearish' counted 166 uniques.
    a = build("bullish_large_cap")
    news = comp(a, "news")
    reason = next(r.text for r in a.verdict.reasons if r.ref == "news")
    assert f"across {a.news.n} articles" in news.detail
    assert f"{a.news.bullish} bullish / {a.news.bearish} bearish" in news.detail
    assert f"across {a.news.n} articles (+" in reason and "syndicated cop" in reason


def test_bullish_large_cap_syndication_and_sources() -> None:
    a = build("bullish_large_cap")
    top = a.narratives[0]
    assert top.count >= 3 and len(top.publishers) >= 3 and "Reuters" in top.publishers
    rep = next(s for s in a.signals if s.id == top.signal_ids[0])
    assert rep.duplicates >= 1
    yahoo = next(s for s in a.sources if s.key == "yahoo_news")
    assert yahoo.fetched == 3 and yahoo.kept == 0  # all three were syndicated copies of Google items
    assert next(s for s in a.sources if s.key == "tradestie").status == "empty"


# --------------------------------------------------------------------------- #
# 2. Meme stock: euphoric crowd vs bearish news, short report, dilution
# --------------------------------------------------------------------------- #
def test_meme_stock_flags_crowding_divergence_and_risks() -> None:
    a = build("meme_stock")
    v = a.verdict
    # The crowd panel reports tagged messages (57 of 60, as CrowdView documents); the scoring
    # uses one vote per account (34 of 37) and says so. A crowded long reads as a contrarian
    # caution, so the social component stays well short of "strongly bullish".
    assert a.crowd.stocktwits_bullish == 57 and a.crowd.stocktwits_bull_ratio == pytest.approx(0.95)
    assert 55 <= comp(a, "social").score < 75  # euphoric posts (+0.88) are capped at the crowding level
    assert comp(a, "news").score <= 35 and comp(a, "analysts").score <= 25
    # The crowd cannot carry the verdict on its own.
    assert v.score < 55
    assert "crowded-long retail (92% bullish of 37 accounts)" in v.headline

    crowded = insight(a, "Crowded long")
    assert crowded is not None and crowded.polarity == "bear"
    assert "92% of 37 StockTwits accounts tagging a stance" in crowded.detail
    divergence = insight(a, "Crowd bullish, news bearish")
    assert divergence is not None and divergence.kind == "divergence"
    assert insight(a, "Short-seller report").severity == "alert"
    assert insight(a, "Dilution risk") is not None
    reddit = insight(a, "Reddit mentions +300%")
    assert reddit is not None and "160 mentions vs 40" in reddit.detail
    spike = insight(a, "Global news volume spiking")
    assert spike is not None and "10.7×" in spike.detail
    assert a.attention is not None and a.attention.label == "Spiking"


def test_meme_stock_overbought_dampens_technicals() -> None:
    a = build("meme_stock")
    tech = comp(a, "technicals")
    assert "RSI 86" in tech.detail
    assert 60 < tech.score < 80  # +140% in 3M but RSI 86: strong, dampened, not maxed out


# --------------------------------------------------------------------------- #
# 3. Crypto: no analysts / insiders / earnings
# --------------------------------------------------------------------------- #
def test_crypto_renormalizes_without_analysts_and_insiders() -> None:
    a = build("crypto")
    for key in ("analysts", "insiders"):
        c = comp(a, key)
        assert not c.available and c.score is None and c.confidence == 0
    assert a.verdict.score >= 65 and a.verdict.stance == "bullish"  # not dragged to 50 by missing pieces
    assert not any("analyst" in i.detail.lower() for i in a.insights)
    assert not any(c.kind in ("earnings", "insider", "analyst") for c in a.catalysts)
    assert "analyst" not in a.brief.summary.lower()
    assert a.analysts is None and a.insiders is None and a.earnings is None


# --------------------------------------------------------------------------- #
# 4. Lawsuit narrative
# --------------------------------------------------------------------------- #
def test_lawsuit_story_dominates_and_is_flagged() -> None:
    a = build("lawsuit")
    v = a.verdict
    top = a.narratives[0]
    assert "lawsuit" in top.headline.lower() and top.label == "bearish"
    assert top.count >= 7 and len(top.publishers) >= 5 and "lawsuit" in top.events
    assert v.stance == "bearish"
    assert "lawsuit" in v.headline.lower()  # the story behind the read, at a glance
    assert v.reasons[1].text.startswith("Top story") and v.reasons[1].ref == top.id
    legal = insight(a, "Legal/regulatory overhang")
    assert legal is not None and legal.severity == "alert"
    assert "law-firm solicitation" in legal.detail
    # Law-firm press releases are weighted far below wire coverage.
    pr = [s for s in a.signals if s.publisher == "PR Newswire"]
    wire = [s for s in a.signals if s.publisher == "Reuters"]
    assert pr and wire and max(s.weight for s in pr) < min(s.weight for s in wire)


# --------------------------------------------------------------------------- #
# 5. Nearly no data
# --------------------------------------------------------------------------- #
def test_thin_coverage_is_shrunk_and_honest() -> None:
    a = build("thin_coverage")
    v = a.verdict
    assert 40 <= v.score <= 60 and v.stance == "neutral"
    assert v.confidence == "low"
    assert "thin evidence" in v.headline and "2 relevant items" in v.headline
    assert a.sentiment.n == 2  # the two generic market headlines were filtered as irrelevant
    thin = insight(a, "Thin coverage")
    assert thin is not None and "2 relevant items" in thin.detail
    assert comp(a, "news").confidence < 0.3
    assert a.narratives == []


# --------------------------------------------------------------------------- #
# 6. Everything down
# --------------------------------------------------------------------------- #
def test_all_sources_down_degrades_gracefully() -> None:
    a = build("all_sources_down")
    v = a.verdict
    assert v.score == 50 and v.label == "Neutral" and v.confidence == "low"
    assert v.headline.startswith("No read") and v.reasons == []
    assert not any(c.available for c in v.components)
    assert a.signals == [] and a.narratives == [] and a.timeline == []
    assert {s.status for s in a.sources} == {"error"}
    assert all(s.error for s in a.sources)
    outage = insight(a, "No news or social data")
    assert outage is not None and outage.severity == "alert" and "All 5 text sources failed" in outage.detail
    pending = insight(a, "not loaded this run")
    assert pending is not None and pending.severity == "info"  # GDELT is slow, not down
    assert not insight(a, "Thin coverage")
    assert a.brief.summary.startswith("No read on Acme")


# --------------------------------------------------------------------------- #
# What changed since the last look
# --------------------------------------------------------------------------- #
def test_delta_against_previous_snapshot() -> None:
    inputs = scenarios.bullish_large_cap()
    inputs.previous = snapshot(30, sentinel=55, score=0.05, label="neutral", price=190.0,
                               narratives=["Acme stock jumps 6% after record data-center sales"])
    a = build_analysis(inputs)
    d = a.delta
    assert d.previous_at == inputs.previous.at
    assert d.sentinel_change == a.verdict.score - 55 and d.sentinel_change >= 12
    assert d.price_change_pct == pytest.approx(5.26, abs=0.01)
    assert "improved sharply" in d.note and "now bullish (was neutral)" in d.note
    # The record-sales story (headlined by its non-recap member) was in the previous snapshot.
    same = next(n for n in a.narratives if "record data-center sales" in n.headline)
    assert not same.is_new
    assert any(n.is_new for n in a.narratives)
    assert set(d.new_narratives) == {n.headline for n in a.narratives if n.is_new}
    jump = insight(a, "SentiNET jumped")
    assert jump is not None and f"55 → {a.verdict.score}" in jump.detail


def test_no_previous_snapshot_marks_nothing_new() -> None:
    a = build("bullish_large_cap")
    assert a.delta.previous_at is None and a.delta.note is None
    assert not any(n.is_new for n in a.narratives)


def test_bear_case_says_each_insider_fact_once_and_never_invents_a_trend() -> None:
    # Live TWLO: both 'Heavy insider selling: …' and 'Insider selling only: …' were bear points, and
    # 'Analysts turning cautious' was asserted while revisions were 2 raises / 0 cuts (the stock had
    # simply run +30% in a month past the mean target).
    from tests.analytics.factories import analysts, company, inputs, insider, insiders, news_flow, quote, run
    from tests.analytics.factories import GOOGLE, technicals

    twlo = company("TWLO", "Twilio Inc.", "Twilio")
    heavy = insiders([insider(20 + i, f"Seller {i}", "sell", 30e6) for i in range(16)])  # $480M of a $45B cap
    a = build_analysis(inputs(twlo, [run(GOOGLE, news_flow([f"Twilio expands growth plan {i}" for i in range(8)]))],
                              quote=quote(price=294.58, market_cap=45e9), insiders=heavy,
                              technicals=technicals(r1m=30.0, r3m=41.0, vs200=67.0, rsi=66.0),
                              analysts=analysts(mean=1.9, total=30, upside=-10.7, price=294.58)))
    bear = a.brief.bear_points
    assert len([b for b in bear if "insider" in b.lower()]) == 1
    assert not any(b.startswith("Analysts turning cautious") for b in bear)
    assert any(b.startswith("Price has run past the mean analyst target ($263.06, −11%) after +30% in 1M") for b in bear)
    assert "Analysts rate it Buy, but the stock already trades above the mean target ($263.06, −11%)" in a.brief.summary


def test_next_catalyst_is_the_most_material_not_the_soonest() -> None:
    # Live TGT: 'Next catalyst: ex-dividend date in 36 days (Nov 10)' while earnings were Nov 18;
    # the watch list also carried the dividend *payment* date.
    from datetime import timedelta

    from app.schemas import Catalyst
    from tests.analytics.factories import GOOGLE, company, earnings, ex_dividend, inputs, news_flow, run

    pay = Catalyst(date=NOW + timedelta(days=57), kind="dividend", title="Dividend payment",
                   detail="$1.16/share", upcoming=True)
    built = inputs(company(), [run(GOOGLE, news_flow([f"Acme expands growth plan {i}" for i in range(8)]))],
                   earnings=earnings(days_until=44), calendar_catalysts=[ex_dividend(36), pay])
    brief = build_analysis(built).brief
    assert "Next catalyst: earnings in 44 days" in brief.summary
    assert not any(w.startswith("Dividend payment") for w in brief.watch)
    assert any(w.startswith("Ex-dividend date in 36 days") for w in brief.watch)
    far = inputs(company(), [run(GOOGLE, news_flow([f"Acme expands growth plan {i}" for i in range(8)]))],
                 earnings=earnings(days_until=80), calendar_catalysts=[ex_dividend(36)])
    assert "Next catalyst: ex-dividend date in 36 days" in build_analysis(far).brief.summary


def test_delta_tells_a_coverage_change_from_a_sentiment_change() -> None:
    # Live GME: momentum went 34.7 -> 61.4 between two runs minutes apart only because GDELT loaded;
    # the note also embedded '… since Oct 5 03:06 UTC' next to the UI's own local time.
    first = scenarios.bullish_large_cap()
    first.tone = None
    before = build_analysis(first)
    second = scenarios.bullish_large_cap()
    from tests.analytics.factories import tone_trend
    second.tone = tone_trend(base=0.9, recent=-0.2, percentile=0.02)  # GDELT arrives, and it is souring
    second.previous = snapshot(0.2, sentinel=before.verdict.score, score=before.sentiment.score,
                               label=before.verdict.stance, price=200.0)
    second.previous_components = before.verdict.components
    after = build_analysis(second)
    assert abs(after.verdict.score - before.verdict.score) >= 5
    note = after.delta.note or ""
    assert "UTC" not in note and "since the last look" in note
    assert note.startswith("Data coverage changed") and "momentum −" in note and "(now available)" in note
    # Without the previous components the note falls back to the score move.
    second.previous_components = None
    assert build_analysis(second).delta.note.startswith("Sentiment")
