"""Story clusters -> ranked narratives (impact, velocity, NEW flag, quality bars)."""
from __future__ import annotations

from app.analytics.narratives import build_narratives, jaccard
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


def test_jaccard() -> None:
    assert jaccard(frozenset("ab"), frozenset("ab")) == 1.0
    assert jaccard(frozenset(), frozenset("a")) == 0.0
