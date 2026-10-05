"""Raw items -> kept, de-duplicated, scored and weighted items."""
from __future__ import annotations

from datetime import timedelta

import pytest

from app.analytics import textkit
from app.analytics.build import build_analysis
from app.analytics.prepare import item_weight, prepare
from tests.analytics.factories import (
    FINNHUB,
    GOOGLE,
    NOW,
    STOCKTWITS,
    company,
    inputs,
    post,
    raw,
    run,
)

ACME = company()


def kept(runs, comp=ACME):
    return prepare(comp, runs, NOW)


def test_relevance_filter_drops_unrelated_items() -> None:
    p = kept([run(GOOGLE, [raw("Acme beats estimates"), raw("Oil prices climb on supply worries"),
                           raw("$ACME breakout on volume")])])
    assert [it.title for it in sorted(p.items, key=lambda i: i.title)] == ["$ACME breakout on volume",
                                                                            "Acme beats estimates"]
    assert p.dropped["irrelevant"] == 1 and p.fetched["google_news"] == 3 and p.kept["google_news"] == 2


def test_ticker_specific_items_survive_with_a_relevance_floor() -> None:
    p = kept([run(FINNHUB, [raw("Company reports steady quarter", ticker_specific=True)])])
    assert len(p.items) == 1 and p.items[0].relevance == pytest.approx(0.7)


def test_ticker_keyed_feeds_cannot_lift_a_sister_company_back_in() -> None:
    # Live VOD.L: a ticker-keyed feed's 'Vodafone Idea …' items were floored at 0.7 although the title
    # names only the separately listed Indian affiliate.
    sister = "Acme Industries shares slump on forklift recall"
    p = kept([run(FINNHUB, [raw(sister, ticker_specific=True),
                            raw(sister + " again", extra={"provider_relevance": 0.9}),
                            raw("Acme Industries and Acme both rally", ticker_specific=True),
                            raw("Company reports steady quarter", ticker_specific=True)])])
    assert sorted(it.title for it in p.items) == ["Acme Industries and Acme both rally",
                                                  "Company reports steady quarter"]
    assert p.dropped["irrelevant"] == 2


@pytest.mark.real_nlp
def test_real_sister_company_titles_get_no_feed_floor() -> None:
    pytest.importorskip("app.nlp.relevance")
    vod = company("VOD.L", "Vodafone Group Plc", "Vodafone")
    titles = ["Vodafone Idea shares slump as funding talks drag on", "Vodafone shares rise after Q1 trading update"]
    p = kept([run(FINNHUB, [raw(t, ticker_specific=True) for t in titles])], vod)
    assert [it.title for it in p.items] == [titles[1]]


def test_snippet_mention_counts_but_less_than_title_mention() -> None:
    p = kept([run(GOOGLE, [raw("Chip stocks rally on AI demand", body="Acme shares rose 4% after the report")])])
    assert len(p.items) == 1 and p.items[0].relevance == pytest.approx(0.8 * 0.85, abs=1e-3)


def test_roundups_are_capped() -> None:
    p = kept([run(GOOGLE, [raw("Five chip stocks to watch including Acme", extra={"symbols": 5})])])
    assert p.items[0].relevance == pytest.approx(0.85 * 0.85, abs=1e-3)  # title names Acme: mild cut
    p = kept([run(GOOGLE, [raw("Chip stocks to watch this week", body="Acme, Bolt, Core, Dyna", extra={"symbols": 5})])])
    assert p.items[0].relevance <= 0.4


def test_stale_empty_and_future_items() -> None:
    old = raw("Acme announces dividend", hours=24 * 30)
    future = raw("Acme to report earnings soon", hours=-30)
    p = kept([run(GOOGLE, [old, future, raw("$ACME", hours=1)])])
    assert p.dropped["stale"] == 1 and p.dropped["empty"] == 1
    assert len(p.items) == 1 and p.items[0].timestamp is None  # future-dated -> undated, never trusted


def test_syndicated_copies_collapse_into_the_most_trusted_outlet() -> None:
    title = "Acme beats quarterly estimates as demand surges"
    p = kept([run(GOOGLE, [raw(title, 5, "Benzinga"), raw(title, 3, "Reuters"), raw(title, 4, "Zacks")])])
    assert len(p.items) == 1
    rep = p.items[0]
    assert rep.publisher == "Reuters" and rep.duplicates == 2 and rep.coverage == 3
    assert set(rep.outlets()) == {"Reuters", "Benzinga", "Zacks"}
    assert p.dropped["duplicate"] == 2


def test_weight_components() -> None:
    p = kept([run(GOOGLE, [raw("Acme beats estimates", 1, "Reuters"), raw("Acme beats estimates again", 1, "Zacks"),
                           raw("Acme wins contract", 96, "Reuters")])])
    by_title = {it.title: it for it in p.items}
    fresh, older = by_title["Acme beats estimates"], by_title["Acme wins contract"]
    zacks = by_title["Acme beats estimates again"]
    assert fresh.weight > zacks.weight  # same age, more trusted outlet
    assert fresh.weight > older.weight  # 96 h old: decayed by the 72 h half-life
    older.timestamp = NOW - timedelta(days=20)
    assert item_weight(older, NOW) > 0  # floored, never zero


def test_engagement_and_social_half_life() -> None:
    quiet = post("$ACME looking strong today", 1, "a", likes=0)
    loud = post("$ACME breaking out on heavy volume", 1, "b", likes=400)
    stale = post("$ACME looking strong this week", 72, "c", likes=0)
    p = kept([run(STOCKTWITS, [quiet, loud, stale])])
    w = {it.author: it.weight for it in p.items}
    assert w["b"] > w["a"] > w["c"]
    assert w["c"] / w["a"] < 0.3  # 72 h is two social half-lives


def test_prolific_voices_are_down_weighted() -> None:
    spam = [raw(f"Acme stock position increased by Fund {i} LLC", 2, "MarketBeat") for i in range(16)]
    p = kept([run(GOOGLE, spam + [raw("Acme beats estimates", 2, "Reuters")])])
    mb = [it for it in p.items if it.publisher == "MarketBeat"]
    assert len(mb) == 16
    single = item_weight(mb[0], NOW)
    assert mb[0].weight == pytest.approx(single * 0.5, rel=1e-3)  # sqrt(4 / 16)


def test_engine_failure_is_reported_never_faked(monkeypatch) -> None:
    def broken(texts, kinds=None, company=None):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(textkit, "analyze", broken)
    a = build_analysis(inputs(ACME, [run(GOOGLE, [raw("Acme beats estimates"), raw("Acme wins contract")])]))
    assert all(s.confidence == 0 and s.score == 0 for s in a.signals) and len(a.signals) == 2
    assert a.sentiment.n == 0
    news = next(c for c in a.verdict.components if c.key == "news")
    assert not news.available
    alert = next(i for i in a.insights if i.title == "Sentiment engine failed")
    assert alert.severity == "alert" and "model file missing" in alert.detail
    assert a.verdict.confidence == "low"


def test_signal_ids_are_stable_and_unique() -> None:
    items = [raw("Acme beats estimates", url="https://x.com/a"), raw("Acme beats estimates (updated)",
                                                                     url="https://x.com/a")]
    a1 = prepare(ACME, [run(GOOGLE, items)], NOW)
    a2 = prepare(ACME, [run(GOOGLE, items)], NOW)
    assert [i.id for i in a1.items] == [i.id for i in a2.items]
    assert len({i.id for i in a1.items}) == 2


@pytest.mark.parametrize(("title", "noise"), [
    ("Acme Corporation $ACME Stock Sold by Addison Advisors LLC", True),
    ("Natural Investments LLC Buys 1,234 Shares of Acme Corporation $ACME", True),
    ("Position in Acme Corp. $ACME Raised by Smith Capital Management", True),
    ("Berkshire Hathaway buys stake in Acme", False),
    ("Acme CEO buys 450K shares of Acme stock", False),
    ("SoftBank sells entire Acme stake", False),
])
def test_auto_generated_holdings_stories_are_dropped(title: str, noise: bool) -> None:
    p = kept([run(GOOGLE, [raw(title, 2, "MarketBeat")])])
    assert (p.dropped["boilerplate"] == 1) is noise and (len(p.items) == 0) is noise


@pytest.mark.parametrize(("title", "publisher", "noise"), [
    # Real MarketBeat templates (live captures): a holdings verb plus the template's ticker tag.
    ("Trivest Advisors Ltd Purchases 84,160 Shares of Acme Corporation $ACME", "marketbeat.com", True),
    ("Acme Corporation $ACME Shares Sold by Denver PWM LLC", "Yahoo Finance", True),
    ("Evoke Wealth LLC Sells 229,221 Shares of Acme Corporation (NASDAQ:ACME)", "Zacks", True),
    # Activist and strategic stakes are material ownership news (live false drops before the fix).
    ("SoftBank Group Sells Stake in Acme", "Reuters", False),
    ("Acme Stake Raised by SoftBank Group", "Bloomberg", False),
    ("Elliott Management Takes Stake in Acme", "CNBC", False),
    ("Acme Stake Cut by Trian Partners", "Reuters", False),
    ("Toyota Group Acquires Stake in Acme", "Reuters", False),
    ("Saudi PIF Investments Boosts Stake in Acme", "Bloomberg", False),
    ("Berkshire Hathaway Cuts Stake in Acme", "Reuters", False),
])
def test_holdings_filter_keeps_activist_and_strategic_stakes(title: str, publisher: str, noise: bool) -> None:
    p = kept([run(GOOGLE, [raw(title, 2, publisher)])])
    assert (p.dropped["boilerplate"] == 1) is noise


def test_posts_sharing_a_headline_stay_social() -> None:
    # Live META case: 4 of a story's '6 articles' were Bluesky reposts of the headline.
    from tests.analytics.factories import BLUESKY

    title = "Acme stock enjoys best month since 2022 on AI momentum"
    p = kept([run(GOOGLE, [raw(title, 3, "Reuters"), raw(title, 4, "CNBC")]),
              run(BLUESKY, [raw(title, 2, "Bluesky", author=f"user{i}.bsky.social") for i in range(4)])])
    news = [it for it in p.items if it.group == "news"]
    social = [it for it in p.items if it.group == "social"]
    assert len(news) == 1 and news[0].coverage == 2 and set(news[0].outlets()) == {"Reuters", "CNBC"}
    assert len(social) == 1 and social[0].coverage == 4  # the reposts collapse among themselves


def test_ticker_keyed_roundups_stay_capped() -> None:
    # The ticker-specific floor must not lift a multi-ticker roundup that never names the company.
    p = kept([run(FINNHUB, [raw("Chip stocks to watch this week", ticker_specific=True, extra={"symbols": 6})])])
    assert p.items[0].relevance <= 0.4
    p = kept([run(FINNHUB, [raw("Acme and four peers to watch", ticker_specific=True, extra={"symbols": 5})])])
    assert p.items[0].relevance >= 0.7  # names the company: only a mild roundup cut


def test_price_recaps_count_less_than_developments() -> None:
    # Live SOFI: 'Stock Craters 43% In 2026' recaps held 20% of the news weight and turned the
    # news tone negative, restating the drawdown the technicals component already measures.
    from app.analytics.prepare import PRICE_RECAP_WEIGHT, price_recap

    import dataclasses

    items = kept([run(GOOGLE, [raw("Acme stock plunges", 3, "Reuters"),
                               raw("Acme plunges after fraud probe announced", 3, "Bloomberg"),
                               raw("Acme drops as fraud allegations mount", 3, "CNBC")])]).items
    by_title = {it.title: it for it in items}
    recap = by_title["Acme stock plunges"]
    probe = by_title["Acme plunges after fraud probe announced"]  # a development that also moved the price
    driven = by_title["Acme drops as fraud allegations mount"]  # only a price event, but 'fraud' drives the tone
    assert price_recap(recap) and not price_recap(probe) and not price_recap(driven)
    full = item_weight(dataclasses.replace(recap, events=[]), NOW)
    assert recap.weight == pytest.approx(full * PRICE_RECAP_WEIGHT, rel=1e-3)
    assert driven.weight == pytest.approx(item_weight(driven, NOW), rel=1e-3) and driven.weight > recap.weight


def test_machine_written_daily_recaps_count_as_price_recaps() -> None:
    # Live SHOP.TO: five 'Shopify Inc. Cl A stock rises <weekday>, outperforms market' items (+0.63 each, no
    # events) kept full weight — news tone +0.18 with them, +0.09 without; GPRO's 'stock outperforms
    # competitors on strong trading day' became the summary's dominant story.
    from app.analytics.prepare import PRICE_RECAP_WEIGHT, price_recap

    import dataclasses

    titles = ["Acme Corp. stock rises Monday, outperforms market",
              "Acme Corp. stock outperforms competitors on strong trading day",
              "Acme Corp. stock underperforms Tuesday when compared to competitors despite daily gains",
              "Acme Corp. slips Friday, underperforms market"]
    items = kept([run(GOOGLE, [raw(t, 3, o) for t, o in zip(titles, ["MarketWatch", "Reuters", "CNBC", "Zacks"],
                                                                 strict=True)])]).items
    assert len(items) == 4 and all(price_recap(it) for it in items)
    for it in items:
        assert it.weight == pytest.approx(item_weight(dataclasses.replace(it, title="x"), NOW) * PRICE_RECAP_WEIGHT,
                                          rel=1e-3)
    # A development wearing the same words is not a recap; nor is an earnings beat 'outperforming expectations'.
    probe = kept([run(GOOGLE, [raw("Acme Corp. stock underperforms Tuesday when compared to competitors; "
                                   "SEC opens probe", 3)])]).items[0]
    beat = kept([run(GOOGLE, [raw("Acme outperforms market expectations as revenue beats", 3)])]).items[0]
    assert not price_recap(probe) and not price_recap(beat)


def test_user_posts_start_below_published_reporting() -> None:
    # Live NVDA: '$NVDA Looking for a huge day Monday. New ATHs all week. 🚀🚀🚀🚀🚀' (0.89) outweighed
    # Barron's (0.85): social trust was fixed at 1.0 while outlets carry their own trust <= 1.
    from app.analytics.prepare import SOCIAL_TRUST

    items = kept([run(GOOGLE, [raw("Acme expands growth plan", 3, "Benzinga")]),
                  run(STOCKTWITS, [post("$ACME new highs all week 🚀🚀🚀", 3, "u1", likes=3)])]).items
    news = next(it for it in items if it.group == "news")
    social = next(it for it in items if it.group == "social")
    assert social.trust == SOCIAL_TRUST and social.weight < news.weight


def test_signal_list_keeps_half_for_reporting_when_there_is_enough() -> None:
    from app.analytics.build import select_signals
    from app.analytics.prepare import Item

    def item(i: int, group: str, weight: float) -> Item:
        return Item(id=f"{group}{i}", source="s", source_label="S", source_weight=1.0,
                    kind="social" if group == "social" else "news", title=f"{group} {i}", weight=weight)

    def picked(n_news: int, n_social: int) -> tuple[int, int]:
        items = sorted([item(i, "news", 0.1) for i in range(n_news)] + [item(i, "social", 0.9) for i in range(n_social)],
                       key=lambda it: -it.weight)
        out = select_signals(items, [], limit=200)
        return sum(s.kind == "news" for s in out), sum(s.kind == "social" for s in out)

    assert picked(150, 300) == (100, 100)  # heavier chatter cannot take more than half of the list
    assert picked(40, 300) == (40, 160)  # little reporting: all of it, chatter fills the rest
    assert picked(300, 10) == (190, 10)  # no chatter to speak of: reporting fills the list
