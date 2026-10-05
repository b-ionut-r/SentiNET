"""Story clusters -> ranked narratives (impact, velocity, NEW flag, quality bars)."""
from __future__ import annotations

from app.analytics.narratives import build_narratives, headline_tokens, same_headline
from app.analytics.prepare import prepare
from tests.analytics.factories import GOOGLE, NOW, STOCKTWITS, company, post, raw, run, snapshot

ACME = company()


def stories(signals, previous=None, social=None):
    runs = [run(GOOGLE, signals)] + ([run(STOCKTWITS, social)] if social else [])
    items = prepare(ACME, runs, NOW).items
    return build_narratives(items, ACME, NOW, previous)


def coverage(title: str, outlets: list[str], hours: float = 3.0):
    return [raw(title + ("" if i == 0 else f" — {o} report"), hours + i, o) for i, o in enumerate(outlets)]


def test_bigger_fresher_sharper_story_ranks_first() -> None:
    lawsuit = coverage("Acme hit with $1.05 billion lawsuit over options grant", ["Reuters", "Bloomberg", "CNBC",
                                                                                  "Barron's", "MarketWatch"])
    launch = coverage("Acme launches storage product line for enterprises", ["Benzinga", "Zacks"], hours=60)
    out = stories(lawsuit + launch)
    assert out[0].narrative.headline.startswith("Acme hit with $1.05 billion lawsuit")
    top = out[0].narrative
    assert top.count == 5 and len(top.publishers) == 5 and top.label == "bearish"
    assert top.velocity_24h == 5 and top.first_seen <= top.last_seen
    assert top.impact > out[1].narrative.impact
    assert "lawsuit" in top.events and "legal" in top.themes
    assert all(0 <= s.narrative.impact <= 1 for s in out)


def test_members_are_tagged_and_syndicated_copies_count() -> None:
    title = "Acme beats estimates as cloud revenue surges"
    out = stories([raw(title, 2, "Reuters"), raw(title, 3, "Zacks"), raw(title, 4, "Benzinga"),
                   raw("Acme beats estimates as cloud revenue surges, shares jump", 5, "CNBC")])
    n = out[0].narrative
    assert n.count == 4 and set(n.publishers) == {"Reuters", "Zacks", "Benzinga", "CNBC"}
    assert all(m.narrative_id == n.id for m in out[0].members)
    assert n.url == out[0].members[0].url


def test_weak_singletons_and_single_outlet_floods_are_not_narratives() -> None:
    out = stories([
        raw("Is Acme stock a buy right now?", 2, "Reuters"),  # question
        raw("Acme to present at investor conference", 2, "Zacks"),  # neutral, mid-trust
        *[raw(f"Acme stock position raised by Fund {i} Capital Holdings LLC filing", 3, "MarketBeat")
          for i in range(6)],  # auto-generated flood from one outlet
    ])
    assert out == []


def test_strong_singleton_from_a_major_outlet_qualifies() -> None:
    out = stories([raw("Acme plunges after fraud probe announced by regulators", 2, "Reuters")])
    assert len(out) == 1 and out[0].narrative.count == 1 and out[0].narrative.label == "bearish"


def test_social_chatter_is_not_clustered_into_narratives() -> None:
    out = stories([], social=[post(f"$ACME to the moon 🚀 squeeze calls #{i}", 1, f"u{i}") for i in range(20)])
    assert out == []


def test_new_flag_compares_with_previous_snapshot_headlines() -> None:
    old = coverage("Acme beats estimates as cloud revenue surges", ["Reuters", "Zacks"])
    new = coverage("Acme names new chief financial officer from rival chipmaker", ["CNBC", "Barron's"])
    prev = snapshot(26, 60, 0.2, narratives=["Acme beats estimates as cloud revenue surges strongly"])
    out = {s.narrative.headline.split(" — ")[0]: s.narrative for s in stories(old + new, previous=prev)}
    assert not out["Acme beats estimates as cloud revenue surges"].is_new
    assert out["Acme names new chief financial officer from rival chipmaker"].is_new


def test_story_that_predates_the_last_look_is_never_new() -> None:
    # Live false positive: every article was older than the previous snapshot, yet the
    # paraphrased headline failed a strict token-Jaccard test.
    old_story = coverage("Acme names new chief financial officer from rival chipmaker", ["CNBC", "Barron's"], hours=30)
    prev = snapshot(26, 60, 0.2, narratives=["Acme beats estimates as cloud revenue surges"])
    (story,) = stories(old_story, previous=prev)
    assert not story.narrative.is_new


def test_paraphrased_headline_is_the_same_story() -> None:
    ignore = frozenset({"sofi", "technologies", "$sofi"})
    previous = headline_tokens("SoFi, Mastercard activate stablecoin settlement for SoFi Bank card programme", ignore)
    current = headline_tokens("SOFI Stock Rises After $25B Card Program Moves To Stablecoin Settlement On "
                              "Mastercard Network", ignore)
    assert same_headline(current, previous)
    # Generic market words and the company name never make two stories one.
    a = headline_tokens("Morgan Stanley cuts Apple stock price target", frozenset({"apple"}))
    b = headline_tokens("Jefferies raises Apple stock price target", frozenset({"apple"}))
    assert not same_headline(a, b)
    assert not same_headline(frozenset(), a)


def test_shared_member_articles_mean_not_new() -> None:
    from app.analytics.prepare import prepare as prep

    items = prep(ACME, [run(GOOGLE, coverage("Acme opens giant factory in Ohio to build robots", ["CNBC", "Reuters"]))],
                 NOW).items
    prev = snapshot(26, 60, 0.2, narratives=["Something else entirely happened at the company"])
    fresh = build_narratives(items, ACME, NOW, prev)
    assert fresh[0].narrative.is_new
    for it in items:
        it.narrative_id = None
    known = build_narratives(items, ACME, NOW, prev, previous_ids=[[items[0].id, "other"]])
    assert not known[0].narrative.is_new


def test_story_tone_is_anchored_on_its_headline() -> None:
    # A catch-all cluster (live BTC case): a neutral headline plus unrelated bullish members.
    # Its tone must follow the headline's core, and it must not serve as directional evidence.
    from app.analytics.narratives import _story

    items = prepare(ACME, [run(GOOGLE, [
        raw("Acme faces sequential sell signals as Monday reversal looms", 2, "Reuters"),
        raw("Acme sequential signals point to reversal, analysts note", 3, "Bloomberg"),
        raw("Acme reversal signals: record surge and strong rally for sequential buyers", 4, "CNBC"),
        raw("Acme buyers love the strong rally as signals surge", 5, "Barron's"),
    ])], NOW).items
    rep = next(it for it in items if it.title.startswith("Acme faces"))
    story = _story(rep, items, NOW)
    assert story is not None and story.lead is rep
    assert rep.score == 0.0 and abs(story.narrative.score) < 0.1  # the cluster mean is clearly bullish
    assert story.spread > 0.35 and story.core_share < 0.5
    assert not story.directional

    bullish_rep = next(it for it in items if it.title.startswith("Acme reversal signals"))
    aligned = _story(bullish_rep, [it for it in items if it.score > 0.3], NOW)
    assert aligned is not None and aligned.directional and aligned.narrative.score > 0.3


def test_events_from_one_peripheral_member_do_not_tag_the_story() -> None:
    lawsuit = [raw("Acme hit with lawsuit over options grant", 2, "Reuters"),
               raw("Acme lawsuit over options grant widens", 3, "Bloomberg"),
               raw("Acme lawsuit over options grant: what investors should know", 4, "CNBC"),
               raw("Acme options grant lawsuit as buyback announced", 30, "Zacks")]
    (story,) = stories(lawsuit)
    assert "lawsuit" in story.material_events
    assert "buyback" not in story.material_events and "buyback" not in story.narrative.events
