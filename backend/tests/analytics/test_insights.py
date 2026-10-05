"""Each insight fires only when its evidence clears the threshold — and says so with numbers."""
from __future__ import annotations

from app.analytics.build import build_analysis
from app.schemas import Insight
from tests.analytics.factories import (
    APEWISDOM,
    GOOGLE,
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
    pending = titled(found, "still loading")
    assert pending is not None and pending.severity == "info"
    failed = titled(found, "Some market data unavailable")
    assert failed is not None and "Analyst ratings could not be loaded" in failed.detail
    assert "tone" not in failed.detail.lower()  # slow GDELT is not reported as a failure


def test_ranking_and_cap() -> None:
    found = build_analysis(__import__("tests.analytics.scenarios", fromlist=["meme_stock"]).meme_stock()).insights
    ranks = [{"alert": 0, "watch": 1, "info": 2}[i.severity] for i in found]
    assert ranks == sorted(ranks) and len(found) <= 8
    assert len({i.title for i in found}) == len(found)
