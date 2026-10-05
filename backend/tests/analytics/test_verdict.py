"""Verdict reasons: story selection and labels."""
from __future__ import annotations

from types import SimpleNamespace

from app.analytics.composite import WEIGHTS, Part
from app.analytics.narratives import Story, featured
from app.analytics.prepare import Item
from app.analytics.verdict import reasons
from app.nlp.types import DetectedEvent
from app.schemas import Narrative


def story(sid: str, headline: str, score: float, impact: float, events: list[DetectedEvent] | None = None,
          drivers: list[tuple[str, float]] | None = None) -> Story:
    lead = Item(id=sid, source="google_news", source_label="Google News", source_weight=1.0, kind="news",
                title=headline, scored=True, score=score, events=events or [], drivers=drivers or [])
    material = [e.key for e in lead.events if e.key not in ("price_up", "price_down", "all_time_high")]
    n = Narrative(id=sid, headline=headline, count=6, publishers=["Reuters", "CNBC", "Barron's"], score=score,
                  impact=impact)
    return Story(narrative=n, members=[lead], material_events=material)


def facts(stories: list[Story], contributions: dict[str, float]) -> SimpleNamespace:
    parts = {k: Part(k) for k in WEIGHTS}
    for key, points in contributions.items():
        parts[key] = Part(key, score=50 + 10 * (1 if points > 0 else -1), reason=f"{key} evidence")  # type: ignore[arg-type]
    return SimpleNamespace(composite=SimpleNamespace(contributions=contributions, parts=parts), stories=stories,
                           deal=None)


def test_counter_story_is_never_shown_without_the_story_it_counters() -> None:
    # Live META / ETH-USD: the top (bearish) story was cut to fit 4 reasons while its
    # "Counter-story" (bullish, aligned with the news flow) survived, countering nothing.
    top = story("a", "Meta stock slides as Muse faces OpenAI threat", -0.55, 0.83,
                [DetectedEvent("competition", "bear")])
    other = story("b", "Wall Street keeps raising Meta price targets", 0.58, 0.6, [DetectedEvent("pt_raise", "bull")])
    out = reasons(facts([top, other], {"technicals": 6.0, "analysts": 5.0, "news": 3.0}))
    texts = [r.text for r in out]
    assert any(t.startswith("Story: ‘Wall Street keeps raising") for t in texts)
    assert not any(t.startswith(("Counter-story", "Top story")) for t in texts)
    # With room for both, the labels say how they relate.
    roomy = [r.text for r in reasons(facts([top, other], {"technicals": 6.0, "news": 3.0}))]
    assert any(t.startswith("Top story: ‘Meta stock slides") for t in roomy)
    assert any(t.startswith("Counter-story: ‘Wall Street keeps raising") for t in roomy)


def test_price_recap_stories_are_not_featured_unless_nothing_else_matters() -> None:
    # Live SOFI/LULU/TWLO: 'Stock craters 43%', 'hits 52-week low/high' were the top or bull/bear
    # "story" while the technicals component already counted the same move.
    recap = story("r", "Acme stock craters 43% in 2026", -0.8, 0.6,
                  [DetectedEvent("price_down", "bear", span="stock craters 43%")], [("Stock Craters 43%", -0.9)])
    launch = story("l", "Acme launches its new storage line", 0.3, 0.5, [DetectedEvent("product_launch", "bull")])
    assert recap.price_only and not launch.price_only
    assert featured([recap, launch]) == [launch]
    minor = story("m", "Acme launches its new storage line", 0.3, 0.2, [DetectedEvent("product_launch", "bull")])
    assert featured([recap, minor]) == [recap, minor]  # nothing else clears the bar: keep the recap
