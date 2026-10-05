"""Component calibration and the composite: monotonic, shrunk, baseline-aware, renormalized."""
from __future__ import annotations

import pytest

from app.analytics.aggregate import Summary
from app.analytics.composite import (
    DEGRADED_MAX_DISTANCE,
    NEWS_BASELINE,
    WEIGHTS,
    Part,
    analysts_part,
    compose,
    insiders_part,
    momentum_part,
    news_part,
    social_part,
    stocktwits_strength,
    technicals_part,
    text_strength,
)
from app.analytics.crowd import Tally
from app.schemas import CrowdView
from tests.analytics.factories import NOW, action, analysts, insider, insiders, technicals, tone_trend


def summary(mean: float | None, n: int, spread: float = 0.3, conf: float = 0.7) -> Summary:
    return Summary(mean=mean, n=n, n_eff=n * 0.7, weight=float(n), bullish=n // 2, bearish=n // 5,
                   neutral=n - n // 2 - n // 5, spread=spread, mean_confidence=conf, outlets=max(1, n // 2),
                   coverage=n)


# --------------------------------------------------------------------------- #
# Text components
# --------------------------------------------------------------------------- #
def test_text_strength_shrinks_small_samples_and_respects_baseline() -> None:
    assert text_strength(0.4, 2) < text_strength(0.4, 50)
    assert text_strength(NEWS_BASELINE, 100, NEWS_BASELINE) == 0
    assert text_strength(0.0, 100, NEWS_BASELINE) < 0  # flat headlines are below the typical +0.04
    assert -1 < text_strength(-5.0, 1000) < 0


def test_news_part_scores_and_explains() -> None:
    upbeat = news_part(summary(0.3, 60))
    flat = news_part(summary(0.04, 60))
    bad = news_part(summary(-0.3, 60))
    assert upbeat.score > 70 and bad.score < 30 and 48 <= flat.score <= 52
    assert "+0.29 across 60 articles" in upbeat.detail  # shown tone is lightly shrunk: 0.3 · 60 / 62 and upbeat.reason.startswith("News flow positive")
    assert upbeat.phrase.startswith("upbeat news") and upbeat.strong
    assert not news_part(summary(None, 0)).available
    with_av = news_part(summary(0.3, 60), av_sentiment=-0.4, av_articles=30)
    assert with_av.score < upbeat.score and "Alpha Vantage" in with_av.detail


def test_news_shrinks_by_effective_sample_size() -> None:
    # 12 articles where a few heavy ones dominate the weight are worth less than 12 even ones.
    even = summary(0.3, 12)
    concentrated = summary(0.3, 12)
    concentrated.n_eff = 2.5
    assert news_part(concentrated).score < news_part(even).score - 3


def test_soft_news_is_called_soft_not_negative() -> None:
    soft = news_part(summary(-0.04, 80))
    assert soft.score < 45 and soft.reason.startswith("News flow softer than usual")
    assert soft.phrase.startswith("soft news") and "typical is +0.04" in soft.reason


def tags(bull: int, bear: int, per_author: bool = False) -> tuple[CrowdView, Tally]:
    return (CrowdView(stocktwits_bullish=bull, stocktwits_bearish=bear, stocktwits_bull_ratio=bull / (bull + bear)),
            Tally(bull, bear, per_author))


def test_stocktwits_judged_against_its_bullish_baseline() -> None:
    def social(bull: int, bear: int, per_author: bool = False) -> Part:
        return social_part(summary(None, 0), *tags(bull, bear, per_author))

    typical, bullish, euphoric = social(62, 38), social(80, 20), social(95, 5)
    assert euphoric.score < bullish.score  # crowded beyond 85%: folds back
    bearish, tiny = social(30, 70), social(3, 0)  # < 5 tagged: ignored
    assert 47 <= typical.score <= 53
    assert 62 <= bullish.score <= 75 and bearish.score < 30
    assert not tiny.available
    assert "80% of 100 tagged StockTwits posts are bullish (62% is typical)" in bullish.reason


def test_crowded_stocktwits_never_adds_bullish_points() -> None:
    """Live finding: 98% of 64 tags read 95.9 on its own, contradicting the crowding insight."""
    ratios = [social_part(summary(None, 0), *tags(b, 100 - b)).score for b in (85, 90, 95, 100)]
    assert ratios == sorted(ratios, reverse=True)  # beyond 85% the signal tapers back
    euphoric = social_part(summary(None, 0), *tags(98, 2))
    assert 50 < euphoric.score < 60  # reads like a ~65% bullish crowd: near the norm
    assert "crowding, which adds no further conviction" in euphoric.reason
    assert euphoric.phrase.startswith(("crowded-long retail", "mildly bullish retail")) and not euphoric.strong
    unanimous = social_part(summary(None, 0), *tags(100, 0))
    assert abs(unanimous.score - 50) < 2  # 100% bullish is a positioning risk, not more conviction
    # One vote per account: 34 bullish accounts out of 35 (not 63 messages from a handful of accounts).
    by_author = social_part(summary(None, 0), *tags(34, 1, per_author=True))
    assert "35 StockTwits accounts tagging a stance" in by_author.reason and "(35 accounts)" in by_author.detail
    assert stocktwits_strength(0.62, 50) == 0


def test_wsb_sentiment_counts_with_volume() -> None:
    quiet = social_part(summary(None, 0), CrowdView(wsb_sentiment=0.6, wsb_comments=2, wsb_label="bullish"))
    busy = social_part(summary(None, 0), CrowdView(wsb_sentiment=0.6, wsb_comments=900, wsb_label="bullish"))
    assert busy.score > quiet.score > 50


# --------------------------------------------------------------------------- #
# Analysts
# --------------------------------------------------------------------------- #
def test_analysts_rating_and_upside_are_monotonic() -> None:
    scores = [analysts_part(analysts(mean=m, total=30, upside=15.0), NOW).score for m in (1.2, 2.0, 2.6, 3.4, 4.2)]
    assert scores == sorted(scores, reverse=True)
    ups = [analysts_part(analysts(mean=2.2, total=30, upside=u), NOW).score for u in (-20, 0, 10, 30, 60)]
    assert ups == sorted(ups)


def test_analysts_revisions_and_mixed_reads() -> None:
    raising = analysts(mean=2.2, total=20, upside=12.0, up90=3, actions=[
        action(3, "Goldman Sachs", "up", "Buy", 120, 100), action(8, "UBS", "main", "Buy", 125, 110),
        action(12, "Citigroup", "main", "Buy", 118, 105)])
    flat = analysts(mean=2.2, total=20, upside=12.0, actions=[action(200, "UBS", "reit", "Buy", 100, 100)])
    assert analysts_part(raising, NOW).score > analysts_part(flat, NOW).score + 5
    assert "30d PT: 3 up / 0 down" in analysts_part(raising, NOW).detail
    hold_upside = analysts_part(analysts(mean=2.9, total=25, upside=35.0), NOW)
    assert hold_upside.reason.startswith("Analysts mixed:") and ", but mean target" in hold_upside.reason


def test_analysts_without_ratings_or_recent_actions_are_unavailable() -> None:
    stale = analysts(mean=2.0, total=0, upside=None, actions=[action(400, "Wedbush", "reit", "Underperform", 10, 10)])
    stale.mean_rating = None
    stale.consensus = None
    assert not analysts_part(stale, NOW).available
    assert not analysts_part(None, NOW).available


def test_analyst_downside_phrase() -> None:
    p = analysts_part(analysts(mean=4.0, total=6, upside=-45.0), NOW)
    assert p.phrase == "a Sell consensus with targets 45% below the price"


def test_mean_and_median_targets_that_disagree_mean_targets_at_the_price() -> None:
    # Live AAPL: mean $328.09 (−1.7%, dragged by a $215 low) but median $340 (+1.9%) was written
    # "Analysts cautious: Buy consensus …; mean target is 1.7% below the price".
    split = analysts(mean=2.5, total=44, upside=-1.7, price=333.69)
    split.target_median, split.consensus = 340.0, "buy"  # the provider's consensus for AAPL
    p = analysts_part(split, NOW)
    assert p.facts["upside"].split and p.facts["upside"].value == 0.0
    assert p.reason.startswith("Analysts mixed: Buy consensus (mean 2.50 from 44 analysts), but targets sit at about "
                               "the price (mean $328.02, −1.7%; median $340.00, +1.9%)")
    assert "target ≈ price" in p.detail
    # One outlier target skews the mean: the median carries the read.
    skew = analysts(mean=2.0, total=30, upside=29.0, price=15.77)
    skew.target_median = 19.0
    q = analysts_part(skew, NOW)
    assert q.facts["upside"].skewed and q.facts["upside"].value == pytest.approx(20.5, abs=0.1)
    assert "median target $19.00 is 21% above the price (mean $20.34, +29%)" in q.reason
    # Agreeing mean and median: unchanged wording.
    agree = analysts(mean=2.0, total=30, upside=40.0, price=234.0)
    agree.target_median = 325.0
    assert "mean target $327.60 is 40% above the price" in analysts_part(agree, NOW).reason


def test_buy_consensus_with_no_upside_left_is_mixed_not_cautious() -> None:
    # Live TWLO: 'Analysts cautious: Buy consensus …; mean target $263.04 is 11% below the price'.
    p = analysts_part(analysts(mean=1.9, total=30, upside=-10.7, price=294.58), NOW)
    assert p.reason.startswith("Analysts mixed: Buy consensus (mean 1.90 from 30 analysts), but mean target")


def test_revision_counts_are_pluralized() -> None:
    # Live TGT: '90d: 1 upgrades / 0 downgrades'.
    one = analysts(mean=2.2, total=20, upside=12.0, up90=1, actions=[action(10, "UBS", "up", "Buy", 120, 100)])
    assert "90d: 1 upgrade / 0 downgrades" in analysts_part(one, NOW).detail


def test_targets_are_written_in_the_quote_currency() -> None:
    # Live VOD.L (GBp) read 'mean target $121.82'; 7203.T (JPY) 'mean target $3,698.63'.
    from app.analytics.util import money

    pence = analysts_part(analysts(mean=2.9, total=16, upside=-3.9, price=126.8), NOW, currency="GBp")
    assert "mean target 121.85p is 3.9% below the price" in pence.reason
    yen = analysts_part(analysts(mean=2.2, total=20, upside=28.0, price=2889.55), NOW, currency="JPY")
    assert "mean target ¥3,699 is 28% above the price" in yen.reason
    assert money(13.3e12, currency="JPY") == "¥13.3T" and money(5e8, currency="GBp") == "£5M"
    assert money(237.91, price=True, currency="CAD") == "C$237.91" and money(12.4, True, "NOK") == "12.40 NOK"
    assert money(3.74e9, currency=None) == "3.74B"  # unknown reporting currency: no symbol is invented


# --------------------------------------------------------------------------- #
# Insiders
# --------------------------------------------------------------------------- #
def test_insider_cluster_buying_beats_single_buy_beats_nothing() -> None:
    cluster = insiders_part(insiders([insider(10, "A", "buy", 400_000, "Chief Executive Officer"),
                                      insider(15, "B", "buy", 250_000), insider(20, "C", "buy", 150_000)]),
                            5e9, NOW)
    single = insiders_part(insiders([insider(10, "A", "buy", 100_000)]), 5e9, NOW)
    assert cluster.score > single.score > 50
    assert "by 3 insiders" in cluster.reason
    assert not insiders_part(insiders([]), 5e9, NOW).available


def test_insider_selling_is_mild_and_scaled_by_market_cap() -> None:
    sells = insiders([insider(10, "A", "sell", 50e6), insider(30, "B", "sell", 50e6)])
    mega = insiders_part(sells, 3e12, NOW)
    small = insiders_part(sells, 2e9, NOW)
    assert 44 <= mega.score < 50 and "routine-sized" in mega.reason
    assert small.score < mega.score and small.score >= 30  # 5% of market cap sold: bearish, not catastrophic
    assert "5.00% of market cap" in small.reason


def test_insider_buys_count_by_size_and_against_the_selling() -> None:
    # Live VOD.L read 67 ("Insider buying") on $240K bought vs $20.2M sold: every buyer counted in full.
    material = insiders_part(insiders([insider(10, "A", "buy", 60_000), insider(12, "B", "buy", 60_000)]), 5e9, NOW)
    dribs = insiders_part(insiders([insider(10, "A", "buy", 1_500), insider(12, "B", "buy", 4_500)]), 5e9, NOW)
    assert material.score > dribs.score > 50
    assert material.facts["material_buyers"] == 2 and dribs.facts["material_buyers"] == 0
    dwarfed = insiders_part(insiders([insider(10, "A", "buy", 60_000), insider(12, "B", "buy", 60_000),
                                      insider(20, "C", "sell", 12e6)]), 5e9, NOW)
    assert dwarfed.score < 50 and dwarfed.facts["token"] == pytest.approx(0.1)
    assert dwarfed.reason.startswith("Net insider selling: 2 open-market purchases ($120K) by 2 insiders")
    assert dwarfed.reason.endswith("token-sized next to $12M of sales")  # never 'discretionary' (see below)
    # Pre-arranged (10b5-1) sales do not make discretionary buying token.
    plan = insiders([insider(10, "A", "buy", 60_000), insider(12, "B", "buy", 60_000),
                     insider(20, "C", "sell", 12e6).model_copy(update={"text": "Sale under a 10b5-1 trading plan"})])
    assert insiders_part(plan, 5e9, NOW).facts["token"] == 1.0
    assert "1 buy ($60K) / 0 sells" in insiders_part(insiders([insider(10, "A", "buy", 60_000)]), 5e9, NOW).detail


def test_old_buys_decay() -> None:
    fresh = insiders_part(insiders([insider(5, "A", "buy", 300_000)]), 5e9, NOW)
    old = insiders_part(insiders([insider(170, "A", "buy", 300_000)]), 5e9, NOW)
    assert fresh.score > old.score > 50


# --------------------------------------------------------------------------- #
# Momentum & technicals
# --------------------------------------------------------------------------- #
def test_momentum_from_gdelt_and_headline_shift() -> None:
    improving = momentum_part(tone_trend(base=0.0, recent=1.0), summary(None, 0), summary(None, 0))
    worsening = momentum_part(tone_trend(base=0.5, recent=-0.5), summary(None, 0), summary(None, 0))
    assert improving.score > 75 and worsening.score < 25
    assert worsening.reason.startswith("Sentiment deteriorating")
    cooling = momentum_part(None, summary(0.10, 40, spread=0.2), summary(0.40, 40, spread=0.2))
    # Still above the typical tone, just less so: a decisive drop, but worded as what it is.
    assert cooling.score < 45 and cooling.reason.startswith("Headline tone normalizing")
    assert cooling.strong and cooling.phrase.startswith("fading headline optimism")
    noisy = momentum_part(None, summary(0.10, 6, spread=0.6), summary(0.30, 6, spread=0.6))
    assert noisy.score > cooling.score  # a small, noisy shift is damped
    assert not momentum_part(None, summary(0.1, 3), summary(0.2, 3)).available


def test_headline_decay_toward_typical_tone_is_not_bearish_momentum() -> None:
    # Live NVDA/GME cold runs (no GDELT yet): headlines +0.13 in 48h vs +0.27 before read as
    # momentum 34–36 and became the headline's "main drag", though both windows sat above the
    # typical +0.04 — the news cycle decaying after an event day, not sentiment deteriorating.
    decay = momentum_part(None, summary(0.13, 82, spread=0.3), summary(0.27, 64, spread=0.3))
    assert 42 <= decay.score < 50 and not decay.strong
    assert decay.reason.startswith("Headline tone normalizing") and "typical is +0.04" in decay.reason
    assert decay.phrase is None  # a mild shift-only lean is never named in the headline
    assert decay.facts["normalizing"]
    # The same size of move *away* from typical is a real turn and counts in full.
    souring = momentum_part(None, summary(-0.10, 82, spread=0.3), summary(0.04, 64, spread=0.3))
    assert souring.score < decay.score - 3
    assert souring.reason.startswith("Sentiment deteriorating") and not souring.facts["normalizing"]
    # With GDELT present the shift is one input among others: the GDELT wording leads.
    with_gdelt = momentum_part(tone_trend(base=0.5, recent=0.5), summary(0.13, 82, spread=0.3),
                               summary(0.27, 64, spread=0.3))
    assert not with_gdelt.facts["normalizing"]
    # And a lone 48h shift is shrunk by its sample: a decisive turn reads strongly only on volume.
    thin = momentum_part(None, summary(-0.25, 6, spread=0.2), summary(0.15, 6, spread=0.2))
    thick = momentum_part(None, summary(-0.25, 60, spread=0.2), summary(0.15, 60, spread=0.2))
    assert thick.score < 40 < thin.score and thin.score > thick.score + 12 and thick.strong


def test_reverting_change_discounts_only_the_return_toward_typical() -> None:
    from app.analytics.composite import REVERSION_CREDIT, reverting_change

    assert reverting_change(0.30, 0.14, 0.04) == pytest.approx(REVERSION_CREDIT * -0.16)
    assert reverting_change(-0.27, -0.14, 0.04) == pytest.approx(REVERSION_CREDIT * 0.13)  # LULU-style recovery
    assert reverting_change(0.04, -0.10, 0.04) == pytest.approx(-0.14)  # away from typical: full
    # Crossing typical: the part up to it is decay, the rest counts in full.
    assert reverting_change(0.24, -0.06, 0.04) == pytest.approx(-(REVERSION_CREDIT * 0.20 + 0.10))


def test_technicals_are_volatility_scaled_and_rsi_dampened() -> None:
    calm = technicals_part(technicals(r1m=5, r3m=10, vs50=4, vs200=8, rsi=60, vol=12))
    wild = technicals_part(technicals(r1m=5, r3m=10, vs50=4, vs200=8, rsi=60, vol=90))
    assert calm.score > wild.score > 50  # +10% in 3M means more for a 12%-vol stock
    hot = technicals_part(technicals(r1m=40, r3m=80, vs50=30, vs200=60, rsi=88, vol=60))
    cool = technicals_part(technicals(r1m=40, r3m=80, vs50=30, vs200=60, rsi=65, vol=60))
    assert hot.score < cool.score and "overbought" in hot.reason
    down = technicals_part(technicals(r1m=-15, r3m=-25, vs50=-12, vs200=-20, rsi=30, vol=40))
    assert down.score < 35 and down.phrase.startswith("weak price action")
    assert not technicals_part(None).available


def test_technicals_calibration_keeps_price_from_outshouting_sentiment() -> None:
    """Live finding: SPY at +3.3% in 3M (~0.55σ) read 75 and 'a strong price trend'."""
    spy = technicals_part(technicals(r1m=0.6, r3m=3.3, vs50=2.0, vs200=6.8, rsi=54, vol=12))
    assert 51 <= spy.score <= 58 and spy.phrase.startswith("a firm tape") and not spy.strong
    # An ordinary uptrend (the market's usual drift) is the baseline, not a bullish signal.
    drift = technicals_part(technicals(r1m=0.8, r3m=2.4, vs50=0.9, vs200=3.8, rsi=55, vol=18))
    assert drift.score == pytest.approx(50, abs=1) and drift.phrase is None
    one_sigma = technicals_part(technicals(r1m=8.7, r3m=15.0, vs50=8.7, vs200=15.0, rsi=60, vol=30))
    assert 60 <= one_sigma.score <= 68
    slide = technicals_part(technicals(r1m=-8, r3m=-15, vs50=-7, vs200=-12, rsi=40, vol=30))
    assert 28 <= slide.score <= 38 and slide.phrase.startswith("weak price action")
    surge = technicals_part(technicals(r1m=30, r3m=60, vs50=25, vs200=55, rsi=70, vol=45))
    assert 78 <= surge.score <= 90 and surge.phrase.startswith("a strong price trend")


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def part(key: str, score: float | None, conf: float = 0.8) -> Part:
    return Part(key, score=score, confidence=conf)  # type: ignore[arg-type]


def test_compose_renormalizes_and_contributions_add_up() -> None:
    parts = [part("news", 80), part("social", 60), part("analysts", 70), part("insiders", None),
             part("momentum", 55), part("technicals", 75)]
    c = compose(parts)
    assert 60 < c.score < 80
    assert sum(c.contributions.values()) == pytest.approx(c.score - 50, abs=0.5)
    assert set(c.contributions) == {"news", "social", "analysts", "momentum", "technicals"}
    assert c.coverage == pytest.approx(1 - WEIGHTS["insiders"])


def test_compose_pulls_toward_neutral_when_coverage_is_low() -> None:
    lone = compose([part("technicals", 90), part("news", None)])
    full = compose([part(k, 90) for k in WEIGHTS])
    assert full.score == 90
    assert 60 < lone.score < 72  # 15% of the evidence: price alone cannot read "Strongly Bullish"


def test_low_confidence_components_weigh_less() -> None:
    sure = compose([part("news", 80, conf=0.9), part("analysts", 30, conf=0.9)])
    unsure_news = compose([part("news", 80, conf=0.1), part("analysts", 30, conf=0.9)])
    assert unsure_news.score < sure.score


def test_no_components_is_exactly_neutral() -> None:
    c = compose([part(k, None) for k in WEIGHTS])
    assert c.score == 50 and c.contributions == {} and c.coverage == 0


def test_degraded_cap_limits_the_read_to_leaning() -> None:
    parts = [part("social", 95), part("technicals", 90), part("momentum", 80), part("news", None)]
    free, capped = compose(parts), compose(parts, max_distance=DEGRADED_MAX_DISTANCE)
    assert free.score >= 70 and capped.score == 61
    assert sum(capped.contributions.values()) == pytest.approx(capped.score - 50, abs=0.5)


def test_net_negative_revisions_are_counted_as_they_are() -> None:
    # Live LULU: 'Bearish: negative news (…) and analyst downgrades (1 in 90d, 1 PT cuts in 30d) align.'
    from app.analytics.composite import cautious_revisions

    assert cautious_revisions(1, 1) == "cautious analyst revisions (1 downgrade in 90d, 1 PT cut in 30d)"
    assert cautious_revisions(1, 0) == "an analyst downgrade in the last 90 days"
    assert cautious_revisions(3, 0) == "analyst downgrades (3 in 90d)"
    assert cautious_revisions(0, 1) == "an analyst target cut in the last 30 days"
    assert cautious_revisions(0, 4) == "analyst target cuts (4 in 30d)"
    view = analysts(mean=3.4, total=20, upside=25.0, actions=[action(5, "UBS", "main", "Neutral", 90, 100)], down90=1)
    part = analysts_part(view, NOW)
    assert part.phrase == "cautious analyst revisions (1 downgrade in 90d, 1 PT cut in 30d)"


def test_insider_sales_are_never_called_discretionary_and_sell_to_cover_is_recognised() -> None:
    # Live VOD.L: '… token-sized next to $20.2M of discretionary sales' while the rows read only 'Sold at price
    # 1.63 per share.' (no plan status), and three executives each sold 850,831 shares the day after
    # 1,805,752-share award rows (47%: tax withholding).
    from datetime import timedelta

    from app.analytics.composite import insider_sales, sell_to_cover
    from app.schemas import InsiderTxn

    def row(days: int, who: str, kind: str, shares: float, value: float | None = None) -> InsiderTxn:
        return InsiderTxn(date=(NOW - timedelta(days=days)).date(), insider=who, kind=kind, shares=shares,  # type: ignore[arg-type]
                          value=value, text="Sold at price 1.63 per share." if kind == "sell" else None)

    txns = [row(69, "Scott Petty", "other", 1_805_752), row(68, "Scott Petty", "sell", 850_831, 1_387_705),
            row(69, "Joakim Reiter", "other", 1_805_752), row(68, "Joakim Reiter", "sell", 850_831, 1_387_705),
            row(68, "Ahmed Essam", "sell", 2_500_000, 4_012_500),  # no award row: a plain sale
            row(40, "Ann Lee", "exercise", 10_000), row(40, "Ann Lee", "sell", 10_000, 50_000),  # exercise sold in full
            row(60, "Joakim Reiter", "buy", 53_059, 83_727), row(60, "Scott Petty", "buy", 23_428, 36_969)]
    view = insiders(txns)
    assert {(t.insider, t.shares) for t in sell_to_cover(view)} == {("Scott Petty", 850_831), ("Joakim Reiter", 850_831)}
    sales = insider_sales(view)
    assert sales.cover == pytest.approx(2 * 1_387_705) and sales.free == pytest.approx(4_012_500 + 50_000)
    part = insiders_part(view, 37e9, NOW)
    assert "discretionary" not in (part.reason or "")
    assert part.reason.endswith("token-sized next to $4.06M of sales (not counting $2.78M of likely tax "
                                "sell-to-cover)")
