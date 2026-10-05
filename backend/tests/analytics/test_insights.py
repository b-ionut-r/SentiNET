"""Each insight fires only when its evidence clears the threshold — and says so with numbers."""
from __future__ import annotations

from datetime import timedelta

import pytest

from app.analytics.build import build_analysis
from app.schemas import Insight, Quote
from tests.analytics.factories import (
    APEWISDOM,
    GOOGLE,
    NOW,
    STOCKTWITS,
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
    post,
    quote,
    raw,
    run,
    technicals,
    tone_trend,
)

ACME = company()
NEUTRAL_NEWS = news_flow([f"Acme schedules investor day number {i}" for i in range(12)])
NEGATIVE_NEWS = news_flow([f"Acme shares fall after weak guidance warning {i}" for i in range(10)])
UPBEAT_NEWS = news_flow([f"Acme beats estimates as strong demand surges {i}" for i in range(10)])


def insights(*runs, **intel) -> list[Insight]:
    return build_analysis(inputs(ACME, list(runs), **intel)).insights


def titled(found: list[Insight], part: str) -> Insight | None:
    return next((i for i in found if part.lower() in i.title.lower()), None)


def stocktwits(bull: int, bear: int):
    return run(STOCKTWITS, [post(f"$ACME position update {i}", 1, f"u{i}") for i in range(3)],
               {"stocktwits_bullish": bull, "stocktwits_bearish": bear, "stocktwits_messages": bull + bear})


# --------------------------------------------------------------------------- #
# Divergences
# --------------------------------------------------------------------------- #
def test_crowd_vs_news_divergence_needs_a_statistically_clear_crowd() -> None:
    weak = insights(run(GOOGLE, NEGATIVE_NEWS), stocktwits(7, 3))  # 70% of 10: noise vs the 62% norm
    assert titled(weak, "Crowd bullish") is None
    clear = insights(run(GOOGLE, NEGATIVE_NEWS), stocktwits(14, 1))  # 93% of 15: z ≈ 2.5
    d = titled(clear, "Crowd bullish, news bearish")
    assert d is not None and d.kind == "divergence" and "93% of 15 tagged" in d.detail
    assert "across 10 articles" in d.detail


def test_price_vs_news_divergence_is_volatility_aware() -> None:
    falling = technicals(r1m=-12.0, r3m=-5.0, vol=30.0, r5d=-2.0)  # −12% vs a ~8.7% monthly σ
    found = insights(run(GOOGLE, UPBEAT_NEWS), technicals=falling)
    i = titled(found, "Price falling despite upbeat news")
    assert i is not None and i.polarity == "bear" and "−12% over 1 month (1.4σ)" in i.detail
    wild = technicals(r1m=-12.0, r3m=-5.0, vol=120.0, r5d=-2.0)  # same move, but routine for a 120%-vol name
    assert titled(insights(run(GOOGLE, UPBEAT_NEWS), technicals=wild), "Price falling") is None


# --------------------------------------------------------------------------- #
# Crowding & attention
# --------------------------------------------------------------------------- #
def test_crowding_extremes() -> None:
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), stocktwits(26, 4)), "Crowded long").polarity == "bear"
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), stocktwits(13, 1)), "Crowded long") is None  # 14 tagged
    cap = titled(insights(run(GOOGLE, NEUTRAL_NEWS), stocktwits(5, 15)), "capitulation")
    assert cap is not None and cap.polarity == "bull" and "Only 25% of 20 tagged" in cap.detail


def test_reddit_attention_spike_needs_volume() -> None:
    def reddit(now: int, prev: int):
        return run(APEWISDOM, [], {"reddit_mentions": now, "reddit_mentions_prev": prev, "reddit_rank": 12,
                                   "reddit_rank_prev": 60, "reddit_tracked": 700})

    hot = titled(insights(run(GOOGLE, NEUTRAL_NEWS), reddit(48, 12)), "Reddit mentions")
    assert hot is not None and hot.title == "Reddit mentions +300% in 24h" and "#12 (from #60) of 700" in hot.detail
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), reddit(9, 4)), "Reddit mentions") is None
    fading = titled(insights(run(GOOGLE, NEUTRAL_NEWS), reddit(5, 40)), "fading")
    assert fading is not None and "fell 88%" in fading.detail


def test_gdelt_volume_spike() -> None:
    found = insights(run(GOOGLE, NEGATIVE_NEWS), tone=tone_trend(volume=300, spike=1500))
    spike = titled(found, "Global news volume spiking")
    assert spike is not None and spike.severity == "alert" and "5.0× its 4-week norm" in spike.detail


# --------------------------------------------------------------------------- #
# Reversals & momentum
# --------------------------------------------------------------------------- #
def test_tone_flip_and_extremes() -> None:
    flipped = insights(run(GOOGLE, NEUTRAL_NEWS), tone=tone_trend(base=0.6, recent=-0.4))
    flip = titled(flipped, "flipped negative")
    assert flip is not None and "7-day tone is −0.40 vs +0.60" in flip.detail
    low = titled(flipped, "90-day low")
    assert low is not None and low.severity == "watch"
    calm = insights(run(GOOGLE, NEUTRAL_NEWS), tone=tone_trend(base=0.5, recent=0.6, percentile=0.6))
    assert titled(calm, "flipped") is None and titled(calm, "90-day") is None


# --------------------------------------------------------------------------- #
# Smart money
# --------------------------------------------------------------------------- #
def test_gdelt_extremes_are_judged_as_shown_and_strong_momentum_is_never_silent() -> None:
    # Live NVDA: GDELT 7d tone at the 10.1th percentile ("10th pct" in the component) missed the
    # "90-day low" bar, so the rail said "Nothing unusual" while the headline named the drag.
    from app.schemas import ToneTrend

    low = tone_trend(base=0.54, recent=0.30, percentile=0.101)
    found = insights(run(GOOGLE, NEUTRAL_NEWS), tone=low)
    low_ins = titled(found, "90-day low")
    assert low_ins is not None and "10th percentile" in low_ins.detail
    # A sharply cooling GDELT tone that is not at an extreme still surfaces through the
    # momentum component (it moves the verdict, so the rail must say why).
    cooling = ToneTrend.model_validate({**tone_trend(base=0.9, recent=0.2, percentile=0.2).model_dump()})
    found = insights(run(GOOGLE, NEUTRAL_NEWS), tone=cooling)
    m = next((i for i in found if i.kind == "momentum"), None)
    assert m is not None and m.polarity == "bear" and "GDELT" in m.detail and "/100" in m.detail
    assert m.title == "Sentiment cooling"
    # A sign flip is told once, by the reversal insight (not again as momentum).
    flip = insights(run(GOOGLE, NEUTRAL_NEWS), tone=tone_trend(base=0.6, recent=-0.1, percentile=0.2))
    assert titled(flip, "flipped negative") is not None and not [i for i in flip if i.kind == "momentum"]
    # Calm tone: no momentum insight at all.
    calm = insights(run(GOOGLE, NEUTRAL_NEWS), tone=tone_trend(base=0.3, recent=0.32, percentile=0.55))
    assert not [i for i in calm if i.kind == "momentum"]


def test_headline_turn_insight_discounts_news_cycle_decay() -> None:
    hot = news_flow([f"Acme beats estimates as strong demand surges record {i}" for i in range(10)], start=60)
    calmer = news_flow([f"Acme expands growth plan {i}" for i in range(10)], start=1)
    found = insights(run(GOOGLE, hot + calmer))
    assert titled(found, "Headline tone turned") is None  # +big → still-positive is decay, not a turn
    souring = news_flow([f"Acme shares fall after weak guidance warning {i}" for i in range(10)], start=1)
    neutral_before = news_flow([f"Acme schedules investor day number {i}" for i in range(10)], start=60)
    turned = titled(insights(run(GOOGLE, neutral_before + souring)), "Headline tone turned down")
    assert turned is not None and turned.polarity == "bear"


def test_analyst_revision_waves() -> None:
    ups = analysts(mean=2.0, total=25, upside=15.0, up90=2, actions=[
        action(3, "Goldman Sachs", "up", "Buy", 120, 100, "Neutral"), action(10, "UBS", "up", "Buy", 118, 104, "Neutral")])
    i = titled(insights(run(GOOGLE, NEUTRAL_NEWS), analysts=ups), "Analysts turning more bullish")
    assert i is not None and "2 upgrades (Goldman Sachs and UBS)" in i.detail
    cuts = analysts(mean=2.5, total=25, upside=5.0, actions=[
        action(2, "Barclays", "main", "Equal-Weight", 90, 100), action(6, "Citigroup", "main", "Neutral", 88, 95),
        action(9, "BofA", "main", "Neutral", 85, 99)])
    c = titled(insights(run(GOOGLE, NEUTRAL_NEWS), analysts=cuts), "Analysts turning cautious")
    assert c is not None and "3 price-target cuts (Barclays, Citigroup and BofA)" in c.detail
    one = analysts(mean=2.0, total=25, upside=15.0, actions=[action(3, "UBS", "up", "Buy", 120, 100)])
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), analysts=one), "Analysts turning") is None


def test_insider_buying_signals() -> None:
    two = insiders([insider(10, "Ann Lee", "buy", 300_000), insider(25, "Bo Chan", "buy", 150_000)])
    i = titled(insights(run(GOOGLE, NEUTRAL_NEWS), insiders=two, quote=quote()), "Insider cluster buying")
    assert i is not None and i.severity == "watch" and "2 insiders bought $450K" in i.detail
    three = insiders([insider(10, "A", "buy", 1e5), insider(12, "B", "buy", 1e5), insider(14, "C", "buy", 1e5)])
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), insiders=three), "cluster").severity == "alert"
    ceo = insiders([insider(5, "Dana Smith", "buy", 750_000, "Chief Executive Officer")])
    c = titled(insights(run(GOOGLE, NEUTRAL_NEWS), insiders=ceo), "bought")
    assert c is not None and c.title == "Chief Executive Officer bought $750K"
    small = insiders([insider(5, "Dana Smith", "buy", 50_000, "Director")])
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), insiders=small), "bought") is None


def vodafone_insiders():
    """Live VOD.L (2026-10-05): five directors bought $1.4K–$84K while colleagues sold $20.2M.
    Values are USD (Yahoo converts the London filings), ~$1.6 a share."""
    from app.schemas import InsiderTxn

    def txn(days: float, who: str, kind: str, value: float) -> InsiderTxn:
        return InsiderTxn(date=(NOW - timedelta(days=days)).date(), insider=who, kind=kind,  # type: ignore[arg-type]
                          shares=round(value / 1.62), value=value)

    buys = [txn(59, "Scott Petty", "buy", 36_969), txn(59, "Joakim Reiter", "buy", 83_727),
            txn(59, "Jean-Francois van Boxmeer", "buy", 36_005), txn(63, "Stephen Carter", "buy", 4_672),
            txn(64, "Simon Dingemans", "buy", 1_402)]
    sells = [txn(14, "Joakim Reiter", "sell", 850_500), txn(56, "Shameel Joosub", "sell", 1_195_725),
             txn(60, "Marika Auramo", "sell", 1_223_601)] + [
        txn(66, f"Executive {i}", "sell", 16_925_444 / 8) for i in range(8)]
    return insiders(buys + sells)


def test_token_insider_buying_next_to_heavy_selling_is_not_a_cluster() -> None:
    # Live VOD.L: ALERT 'Insider cluster buying: 5 insiders bought 163K in 90 days' while insiders
    # sold $20.2M — and the insiders component read 67 ("Insider buying").
    pence = Quote(price=126.8, market_cap=29.4e9, currency="GBp")
    a = build_analysis(inputs(ACME, [run(GOOGLE, NEUTRAL_NEWS)], insiders=vodafone_insiders(), quote=pence))
    assert titled(a.insights, "cluster") is None
    part = next(c for c in a.verdict.components if c.key == "insiders")
    assert part.score is not None and part.score < 50
    assert part.detail.startswith("5 buys ($163K) / 11 sells ($20.2M)")  # insider values are USD everywhere
    selling = next(b for b in a.brief.bear_points if "insider" in b.lower())
    assert "$20.2M" in selling and "token-sized" in selling
    assert not any("cluster" in b.lower() for b in a.brief.bull_points)


def test_cluster_buying_needs_material_purchases() -> None:
    # Dividend-reinvestment-sized buys by several insiders are routine, not a cluster.
    dribs = insiders([insider(10 + i, f"Director {i}", "buy", 3_000) for i in range(5)])
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), insiders=dribs, quote=quote()), "cluster") is None
    # Two material buyers who together put $100K+ in still are — with the selling stated next to them.
    real = insiders([insider(10, "Ann Lee", "buy", 80_000), insider(20, "Bo Chan", "buy", 60_000),
                     insider(30, "Cy Diaz", "buy", 2_000), insider(40, "Di Eng", "sell", 900_000)])
    c = titled(insights(run(GOOGLE, NEUTRAL_NEWS), insiders=real, quote=quote()), "cluster")
    assert c is not None and c.severity == "watch"
    assert c.detail.startswith("2 insiders bought $140K on the open market in the last 90 days "
                               "(+1 smaller buyer under $25K)")
    assert "$900K sold over the same period" in c.detail
    # The same buyers next to > 10x their purchases in discretionary sales: token.
    dwarfed = insiders([insider(10, "Ann Lee", "buy", 80_000), insider(20, "Bo Chan", "buy", 60_000),
                        insider(40, "Di Eng", "sell", 1.6e6)])
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), insiders=dwarfed, quote=quote()), "cluster") is None


def test_insider_share_of_market_cap_never_mixes_currencies() -> None:
    # Live VOD.L: '0.07% of market cap' divided USD insider sales by a GBP market cap.
    from app.analytics.build import market_cap_usd

    pence = Quote(price=126.8, market_cap=29.4e9, currency="GBp")
    view = vodafone_insiders()
    cap = market_cap_usd(inputs(ACME, [], insiders=view, quote=pence))
    assert cap == pytest.approx(29.4e9 / 1.268 * 1.62, rel=0.01)  # shares outstanding × the USD price insiders paid
    a = build_analysis(inputs(ACME, [run(GOOGLE, NEUTRAL_NEWS)], insiders=view, quote=pence))
    detail = next(c for c in a.verdict.components if c.key == "insiders")
    assert "share of market cap" not in detail.detail
    assert "0.05% of market cap" in next(b for b in a.brief.bear_points if "insider" in b.lower())
    # Without USD trade prices there is no honest conversion: the share is skipped and the detail says why.
    unpriced = view.model_copy(update={"transactions": [t.model_copy(update={"shares": None})
                                                        for t in view.transactions]})
    b = build_analysis(inputs(ACME, [run(GOOGLE, NEUTRAL_NEWS)], insiders=unpriced, quote=pence))
    part = next(c for c in b.verdict.components if c.key == "insiders")
    assert part.detail.endswith("share of market cap n/a (GBP market cap, USD trades)")
    assert not any("of market cap" in x for x in b.brief.bear_points + [r.text for r in b.verdict.reasons])
    # A USD listing compares directly.
    assert market_cap_usd(inputs(ACME, [], insiders=view, quote=quote(market_cap=5e9))) == 5e9


def test_heavy_insider_selling_is_relative_to_market_cap() -> None:
    sells = insiders([insider(20, "A", "sell", 60e6), insider(40, "B", "sell", 60e6)])
    heavy = insights(run(GOOGLE, NEUTRAL_NEWS), insiders=sells, quote=quote(market_cap=2e9))  # 6% of cap
    assert titled(heavy, "Heavy insider selling") is not None
    routine = insights(run(GOOGLE, NEUTRAL_NEWS), insiders=sells, quote=quote(market_cap=2e12))
    assert titled(routine, "Heavy insider selling") is None


# --------------------------------------------------------------------------- #
# Catalysts
# --------------------------------------------------------------------------- #
def test_catalyst_windows() -> None:
    soon = titled(insights(run(GOOGLE, NEUTRAL_NEWS), earnings=earnings(days_until=5)), "Earnings in 5 days")
    assert soon is not None and soon.severity == "watch" and "beat EPS estimates in 6 of the last 8" in soon.detail
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), earnings=earnings(days_until=12)), "Earnings").severity == "info"
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), earnings=earnings(days_until=30)), "Earnings") is None
    exdiv = titled(insights(run(GOOGLE, NEUTRAL_NEWS), calendar_catalysts=[ex_dividend(4)]), "Ex-dividend")
    assert exdiv is not None and exdiv.title == "Ex-dividend in 4 days" and "$0.25/share" in exdiv.detail
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), calendar_catalysts=[ex_dividend(11)]), "Ex-dividend") is None


# --------------------------------------------------------------------------- #
# Risks
# --------------------------------------------------------------------------- #
def test_red_flag_filings() -> None:
    restatement = filing(20, "8-K", "Non-reliance on prior financials (restatement): audit committee review",
                         ["4.02"], "high", "bear")
    i = titled(insights(run(GOOGLE, NEUTRAL_NEWS), filings=[restatement]), "Red-flag filing")
    assert i is not None and i.severity == "alert" and "item 4.02" in i.detail
    old = filing(200, "8-K", "Non-reliance on prior financials (restatement)", ["4.02"], "high", "bear")
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), filings=[old]), "Red-flag") is None
    dilution = filing(10, "8-K", "Unregistered sale of equity (dilution)", ["3.02"], "medium", "bear")
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), filings=[dilution]), "Dilution") is not None


def test_red_flag_news_needs_corroboration() -> None:
    lone = [raw("Acme warns of possible bankruptcy filing", 3, "SmallBlog")]
    assert titled(insights(run(GOOGLE, NEUTRAL_NEWS + lone)), "Bankruptcy") is None
    major = [raw("Acme warns of possible bankruptcy filing", 3, "Reuters")]
    b = titled(insights(run(GOOGLE, NEUTRAL_NEWS + major)), "Bankruptcy")
    assert b is not None and b.severity == "alert" and "(Reuters)" in b.detail


def test_regulatory_theme_alone_is_not_a_legal_overhang() -> None:
    sec_story = news_flow(["SEC custody rule warning hits Acme and other custodians",
                           "Acme shares fall on SEC custody rule warning", "SEC rule is weak news for Acme, warning"])
    a = build_analysis(inputs(ACME, [run(GOOGLE, sec_story + NEUTRAL_NEWS)]))
    regulatory = next(t for t in a.themes if t.theme == "regulatory")
    assert regulatory.count == 3 and regulatory.score < -0.1  # negative regulatory coverage…
    assert titled(a.insights, "Legal") is None  # …but no lawsuit/probe event: not a legal overhang
    probe = news_flow(["Regulators open probe into Acme billing practices", "Acme probe widens to sales unit",
                       "Acme shares fall as probe of billing practices widens"])
    legal = titled(insights(run(GOOGLE, probe + NEUTRAL_NEWS)), "Legal/regulatory overhang")
    assert legal is not None and "on investigation" in legal.detail


# --------------------------------------------------------------------------- #
# Data quality & ranking
# --------------------------------------------------------------------------- #
def test_quality_notes_distinguish_slow_from_failed() -> None:
    found = insights(run(GOOGLE, NEUTRAL_NEWS), run(STOCKTWITS, status="error", error="timeout"),
                     intel_status={"tone": "error: still loading after 12s; ready on next refresh",
                                   "analysts": "error: UpstreamError: HTTP 500", "quote": "ok"})
    pending = titled(found, "not loaded this run")
    assert pending is not None and pending.severity == "info"
    failed = titled(found, "Some market data unavailable")
    assert failed is not None and "Analyst ratings could not be loaded" in failed.detail
    assert "tone" not in failed.detail.lower()  # slow GDELT is not reported as a failure


def test_ranking_and_cap() -> None:
    found = build_analysis(__import__("tests.analytics.scenarios", fromlist=["meme_stock"]).meme_stock()).insights
    ranks = [{"alert": 0, "watch": 1, "info": 2}[i.severity] for i in found]
    assert ranks == sorted(ranks) and len(found) <= 8
    assert len({i.title for i in found}) == len(found)


def test_reddit_rank_breakout_from_a_tiny_base_is_not_invisible() -> None:
    # Live META: rank 675 -> 7 on 14 mentions vs 2 — the % change needs >= 3 prior mentions, so the
    # breakout produced no insight, a 'Normal' heat and no mention in the summary.
    def reddit(now: int, prev: int, rank: int, rank_prev: int | None):
        return run(APEWISDOM, [], {"reddit_mentions": now, "reddit_mentions_prev": prev, "reddit_rank": rank,
                                   "reddit_rank_prev": rank_prev, "reddit_tracked": 670})

    a = build_analysis(inputs(ACME, [run(GOOGLE, NEUTRAL_NEWS), reddit(14, 2, 7, 675)]))
    hit = titled(a.insights, "Reddit breakout")
    assert hit is not None and hit.title == "Reddit breakout: #7 from #675"
    assert hit.detail.startswith("14 mentions in 24h vs 2 a day earlier") and "of 670 tracked" in hit.detail
    assert a.attention is not None and a.attention.label in ("Elevated", "Spiking")
    assert "Reddit rank jumped to #7 from #675 (14 mentions)" in a.brief.summary
    # A move inside the top ranks, or a thin one, is not a breakout.
    for args in ((14, 10, 7, 40), (6, 2, 7, 675), (30, 2, 60, 675)):
        assert titled(insights(run(GOOGLE, NEUTRAL_NEWS), reddit(*args)), "Reddit breakout") is None


def test_small_reddit_moves_stay_out_of_the_summary() -> None:
    # Live GME: '8 -> 6 (fell 25%)' was reported in the summary.
    def reddit(now: int, prev: int):
        return run(APEWISDOM, [], {"reddit_mentions": now, "reddit_mentions_prev": prev, "reddit_rank": 24,
                                   "reddit_rank_prev": 16})

    assert "Reddit" not in build_analysis(inputs(ACME, [run(GOOGLE, NEUTRAL_NEWS), reddit(6, 8)])).brief.summary
    big = build_analysis(inputs(ACME, [run(GOOGLE, NEUTRAL_NEWS), reddit(48, 12)])).brief.summary
    assert "Reddit mentions rose 300% in 24h (12 → 48)" in big


def test_the_rail_explains_the_headlines_main_drag_when_nothing_else_fired() -> None:
    # Live NVDA: 'Nothing unusual' on the rail while the headline named the main drag.
    a = build_analysis(inputs(ACME, [run(GOOGLE, UPBEAT_NEWS)], analysts=analysts(mean=4.0, total=12, upside=-30.0)))
    assert a.verdict.stance == "bullish" and "the main drag is" in a.verdict.headline
    drag = next((i for i in a.insights if i.kind != "quality"), None)
    assert drag is not None and drag.polarity == "bear" and drag.title == "Analysts cautious"
    assert "the main drag on the" in drag.detail and "points)" in drag.detail
