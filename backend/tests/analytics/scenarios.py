"""Six realistic, fully synthetic market situations used by the scenario tests and the demo."""
from __future__ import annotations

from app.analytics.inputs import AnalysisInputs
from tests.analytics.factories import (
    APEWISDOM,
    BING,
    BLUESKY,
    GOOGLE,
    STOCKTWITS,
    TRADESTIE,
    YAHOO,
    action,
    analysts,
    company,
    earnings,
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


def bullish_large_cap() -> AnalysisInputs:
    """Mega-cap with an analyst upgrade wave, record results and an improving global tone."""
    acme = company("ACME", "Acme Corporation", "Acme")
    google = news_flow([
        "Morgan Stanley upgrades Acme to Overweight, raises price target to $260",
        "Acme beats quarterly revenue estimates as AI chip demand surges",
        "Acme stock jumps 6% after record data-center sales",
        "Goldman raises Acme price target to $255 on strong AI demand",
        "Acme expands buyback by $20 billion after record quarter",
        "Acme wins $3 billion cloud contract from federal agency",
        "Acme AI chip launch draws strong early orders from hyperscalers",
        "Jefferies upgrades Acme to Buy on data-center momentum",
        "Acme to present at Citi technology conference next week",
        "What to watch on Acme's earnings call: data-center margins",
        "Acme shares hit record as AI rally broadens",
        "Analyst warning: Acme valuation looks stretched after rally",
        "Citi raises Acme price target to $240, keeps Buy rating",
        "Acme record data-center sales lift supplier stocks",
    ], hours_step=5.0)
    yahoo = [raw("Morgan Stanley upgrades Acme to Overweight, raises price target to $260", 2.0, "Yahoo Finance"),
             raw("Acme beats quarterly revenue estimates as AI chip demand surges", 7.0, "Barron's"),
             raw("Acme beats quarterly revenue estimates as AI chip demand surges", 7.5, "MarketWatch")]
    bing = news_flow(["Acme stock jumps 6% after record data-center sales",
                      "Acme record data-center sales: three takeaways for investors"], start=12.0)
    tagged = ([post(f"$ACME breakout looks strong, adding calls #{i}", 1 + i * 0.4, f"bull{i}", "bullish", 3)
               for i in range(21)]
              + [post(f"$ACME overvalued here, taking profits #{i}", 2 + i * 0.5, f"bear{i}", "bearish")
                 for i in range(9)])
    return inputs(
        acme,
        [run(GOOGLE, google), run(YAHOO, yahoo), run(BING, bing),
         run(STOCKTWITS, tagged, {"stocktwits_bullish": 21, "stocktwits_bearish": 9, "stocktwits_messages": 30,
                                  "stocktwits_watchers": 412_000}),
         run(APEWISDOM, [], {"reddit_mentions": 30, "reddit_mentions_prev": 25, "reddit_rank": 14,
                             "reddit_rank_prev": 17, "reddit_tracked": 690}),
         run(TRADESTIE, status="empty")],
        quote=quote(200.0, 400e9),
        technicals=technicals(r1m=6.0, r3m=18.0, vs50=7.0, vs200=20.0, rsi=64.0, vol=32.0, r5d=2.5),
        analysts=analysts(mean=1.4, total=40, upside=25.0, price=200.0, up90=2, actions=[
            action(2, "Morgan Stanley", "up", "Overweight", 260, 220, frm="Equal-Weight"),
            action(5, "Goldman Sachs", "main", "Buy", 255, 230),
            action(9, "Jefferies", "up", "Buy", 250, 210, frm="Hold"),
            action(14, "Citigroup", "main", "Buy", 240, 210),
            action(20, "Bernstein", "reit", "Outperform", 245, 245),
        ]),
        insiders=insiders([insider(12, "Jane Roe", "sell", 15e6, "Chief Financial Officer"),
                           insider(40, "John Doe", "sell", 25e6, "Director")]),
        earnings=earnings(days_until=25, beat_rate=0.875),
        tone=tone_trend(base=0.4, recent=1.1, volume=900.0),
    )


def meme_stock() -> AnalysisInputs:
    """Retail mania (StockTwits 95% bullish, WSB #2, Reddit +300%) against bearish news and a Sell consensus."""
    blast = company("BLST", "Blast Holdings Inc.", "Blast")
    news = news_flow([
        "Short-seller report accuses Blast of inflating revenue",
        "Blast plunges as company announces $500 million share offering",
        "Blast share offering raises dilution worries for retail holders",
        "Blast misses quarterly revenue estimates, shares fall",
        "Analysts warn Blast is overvalued after 300% run",
        "Blast short-seller report sparks volatile trading session",
        "Blast plunges as company announces $500 million share offering",
        "Why Blast stock is so volatile this week",
        "Blast stock soars again as retail traders pile in",
    ], hours_step=4.0)
    crowd = ([post(f"$BLST to the moon 🚀🚀 squeeze incoming #{i}", 0.2 + i * 0.1, f"ape{i}", "bullish", 12)
              for i in range(57)]
             + [post(f"$BLST bagholders everywhere, dumping #{i}", 1 + i, f"bear{i}", "bearish") for i in range(3)])
    return inputs(
        blast,
        [run(GOOGLE, news), run(STOCKTWITS, crowd, {"stocktwits_bullish": 57, "stocktwits_bearish": 3,
                                                    "stocktwits_bull_authors": 34, "stocktwits_bear_authors": 3,
                                                    "stocktwits_messages": 60, "stocktwits_watchers": 98_000}),
         run(APEWISDOM, [], {"reddit_mentions": 160, "reddit_mentions_prev": 40, "reddit_rank": 3,
                             "reddit_rank_prev": 25, "reddit_upvotes": 2400, "reddit_tracked": 690}),
         run(TRADESTIE, [], {"wsb_rank": 2, "wsb_comments": 1200, "wsb_sentiment": 0.45, "wsb_label": "bullish"})],
        quote=quote(42.0, 3.1e9),
        technicals=technicals(r1m=85.0, r3m=140.0, vs50=60.0, vs200=120.0, rsi=86.0, vol=180.0, r5d=30.0),
        analysts=analysts(mean=4.0, total=6, upside=-45.0, price=42.0, actions=[
            action(6, "Wedbush", "down", "Underperform", 18, 25, frm="Neutral")], down90=1),
        tone=tone_trend(base=-0.2, recent=-0.9, volume=150.0, spike=1600.0),
    )


def crypto() -> AnalysisInputs:
    """Bitcoin: no analysts, insiders, earnings or filings — those components must be n/a, not neutral."""
    btc = company("BTC-USD", "Bitcoin", "Bitcoin", quote_type="CRYPTOCURRENCY")
    news = news_flow([
        "Bitcoin ETF inflows hit record as institutions buy the dip",
        "Bitcoin rally extends as ETF inflows hit record",
        "Bitcoin jumps above $90,000 on strong ETF demand",
        "Bitcoin miners expand capacity as hashrate hits record",
        "Bitcoin volatility falls to multi-year low",
        "Bitcoin traders eye Fed decision next week",
        "Regulators open probe into crypto exchange; Bitcoin steady",
        "Bitcoin dominance rises as altcoins slump",
    ], hours_step=6.0)
    social = [post(f"$BTC.X strong bid, buying every dip #{i}", 0.5 + i * 0.3, f"hodl{i}", "bullish", 5)
              for i in range(40)] + [post(f"$BTC.X looks weak, puts #{i}", 1 + i, f"bear{i}", "bearish")
                                     for i in range(6)]
    sky = [raw("Bitcoin breaking out, strong ETF inflows again", 3.0, "Bluesky", author="satoshi.bsky.social")]
    return inputs(
        btc,
        [run(GOOGLE, news), run(STOCKTWITS, social, {"stocktwits_bullish": 40, "stocktwits_bearish": 6,
                                                     "stocktwits_messages": 46, "stocktwits_watchers": 680_000}),
         run(BLUESKY, sky, {"bluesky_posts": 1}),
         run(APEWISDOM, [], {"reddit_mentions": 120, "reddit_mentions_prev": 110, "reddit_rank": 1,
                             "reddit_rank_prev": 1, "reddit_tracked": 140})],
        quote=quote(91_000.0, 1.8e12),
        technicals=technicals(r1m=9.0, r3m=35.0, vs50=9.0, vs200=21.0, rsi=68.0, vol=38.0),
        tone=tone_trend(base=0.2, recent=0.6, volume=3000.0),
        intel_status={"profile": "ok", "quote": "ok", "technicals": "ok", "tone": "ok", "wiki": "empty"},
    )


def lawsuit() -> AnalysisInputs:
    """A $1.05B lawsuit dominates coverage; law firms pile in with solicitation releases."""
    nimbus = company("NMBS", "Nimbus Systems Inc.", "Nimbus")
    suit = [
        raw("Nimbus hit with $1.05 billion lawsuit over founder options grant", 3, "Reuters"),
        raw("Nimbus hit with $1.05 billion lawsuit over founder options grant", 4, "Yahoo Finance"),
        raw("Nimbus hit with $1.05 billion lawsuit over founder options grant", 5, "MarketWatch"),
        raw("Nimbus faces $1.05 billion lawsuit from early advisor over options", 6, "Bloomberg"),
        raw("Early advisor sues Nimbus seeking $1.05 billion in options damages", 8, "CNBC"),
        raw("Nimbus shares fall as $1.05 billion options lawsuit weighs", 9, "Barron's"),
        raw("Nimbus lawsuit: what the $1.05 billion options claim means for investors", 20, "The Motley Fool"),
        raw("Nimbus options lawsuit draws regulator probe of grant practices", 26, "Reuters"),
    ]
    solicitations = [
        raw(f"SHAREHOLDER ALERT: {firm} investigates claims on behalf of Nimbus investors", 10 + i, "PR Newswire")
        for i, firm in enumerate(["Pomerantz Law Firm", "Rosen Law Firm", "Levi & Korsinsky"])
    ]
    other = news_flow([
        "Nimbus launches new storage product line for enterprises",
        "Nimbus to report third-quarter results on October 28",
        "Nimbus product launch event draws enterprise customers",
    ], start=30.0)
    chatter = [post(f"$NMBS lawsuit looks like a nothingburger, holding #{i}", 2 + i, f"user{i}", None)
               for i in range(8)]
    return inputs(
        nimbus,
        [run(GOOGLE, suit + other), run(BING, solicitations), run(STOCKTWITS, chatter, {
            "stocktwits_bullish": 4, "stocktwits_bearish": 5, "stocktwits_messages": 8})],
        quote=quote(58.0, 12e9),
        technicals=technicals(r1m=-6.0, r3m=-2.0, vs50=-4.0, vs200=3.0, rsi=41.0, vol=35.0, r5d=-5.0,
                              trend="sideways"),
        analysts=analysts(mean=2.4, total=14, upside=11.0, price=58.0),
        earnings=earnings(days_until=26, beat_rate=0.5),
        filings=[filing(2, "8-K", "Other material event: litigation update", ["8.01"], "low")],
    )


def thin_coverage() -> AnalysisInputs:
    """A micro-cap nobody writes about: two relevant headlines and a price history."""
    tiny = company("TNYB", "Tiny Biologics Inc.", "Tiny Biologics")
    news = [raw("Tiny Biologics doses first patient in phase 1 trial", 30, "GlobeNewswire"),
            raw("Tiny Biologics to present at biotech investor forum", 50, "Benzinga"),
            raw("Biotech stocks rally on rate-cut hopes", 5, "Reuters"),
            raw("Small caps outperform as yields fall", 8, "CNBC")]
    return inputs(
        tiny,
        [run(GOOGLE, news), run(STOCKTWITS, status="empty")],
        quote=quote(2.1, 45e6),
        technicals=technicals(r1m=3.0, r3m=-8.0, vs50=1.0, vs200=-12.0, rsi=52.0, vol=95.0, trend="sideways"),
        intel_status={"quote": "ok", "technicals": "ok", "analysts": "empty", "insiders": "empty",
                      "earnings": "empty", "tone": "empty", "wiki": "empty"},
    )


def all_sources_down() -> AnalysisInputs:
    """Every text source and intel feed failed (provider outage / network loss)."""
    acme = company("ACME", "Acme Corporation", "Acme")
    runs = [run(GOOGLE, status="error", error="TimeoutError: timed out after 10s"),
            run(BING, status="error", error="UpstreamError: HTTP 503"),
            run(YAHOO, status="error", error="UpstreamError: HTTP 429 (rate limited)"),
            run(STOCKTWITS, status="error", error="ConnectError: connection reset"),
            run(APEWISDOM, status="error", error="TimeoutError: timed out after 10s")]
    status = {k: "error: TimeoutError: timed out after 15s" for k in
              ("profile", "quote", "technicals", "analysts", "earnings", "insiders", "filings", "wiki")}
    status["tone"] = "error: still loading after 12s; ready on next refresh"
    return inputs(acme, runs, intel_status=status)


ALL = {
    "bullish_large_cap": bullish_large_cap,
    "meme_stock": meme_stock,
    "crypto": crypto,
    "lawsuit": lawsuit,
    "thin_coverage": thin_coverage,
    "all_sources_down": all_sources_down,
}
