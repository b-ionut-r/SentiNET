"""Verdict = composite score + label + honest confidence + ranked reasons + one-line headline.

Reasons are ranked by how many points they move the score: a component's
contribution (its share of the effective weight × its distance from 50), or
for a story, 10 × impact × tone strength. Every reason and headline clause is
built from the evidence numbers carried by the components and narratives.
"""
from __future__ import annotations

import math

from app.analytics.composite import LABELS, ComponentKey
from app.analytics.crowd import reddit_change_pct, reddit_move
from app.analytics.facts import Facts
from app.analytics.narratives import Story
from app.analytics.util import (
    band_label,
    clamp,
    count,
    join_and,
    polarity_of,
    quote,
    signed,
    stance_of,
    tone_polarity,
)
from app.schemas import Confidence, Reason, Verdict

MAX_REASONS = 4
MIN_REASON_POINTS = 1.0
MIN_PHRASE_POINTS = 1.0
STORY_MIN_IMPACT = 0.35
THIN_ITEMS = 8
HIGH_CONFIDENCE = 0.7
MEDIUM_CONFIDENCE = 0.4


def build_verdict(f: Facts) -> Verdict:
    comp = f.composite
    label = band_label(comp.score)
    stance = stance_of(comp.score)
    conf_label, conf_value = confidence(f)
    return Verdict(
        score=comp.score, label=label, stance=stance, confidence=conf_label,
        confidence_value=round(conf_value, 3), headline=headline(f, label, stance, conf_label),
        reasons=reasons(f), components=[p.component() for p in comp.parts.values()],
    )


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def confidence(f: Facts) -> tuple[Confidence, float]:
    """data quality (coverage, volume, freshness) × (0.5 + 0.5 · component agreement).

    Coverage is the nominal weight of available components; agreement is 1 minus
    the effective-weighted spread of component strengths (a lone component
    cannot corroborate itself: 0.5)."""
    comp = f.composite
    volume = f.overall.n_eff / (f.overall.n_eff + 12.0)
    avail = comp.available()
    if len(avail) >= 2:
        mean_x = sum(comp.effective[p.key] * p.x for p in avail)
        sd = math.sqrt(sum(comp.effective[p.key] * (p.x - mean_x) ** 2 for p in avail))
        agreement = 1.0 - min(1.0, sd / 0.6)
    else:
        agreement = 0.5
    scored = f.prepared.scored
    if scored:
        fresh = sum(1 for it in scored if (it.age_hours(f.now) or 1e9) <= 72) / len(scored)
    else:
        fresh = 0.3
    data = 0.45 * comp.coverage + 0.35 * volume + 0.20 * fresh
    value = data * (0.5 + 0.5 * agreement)
    if f.prepared.engine_error:
        value *= 0.6
    if len(f.sources_down) >= 3:
        value *= 0.85
    value = clamp(value)
    label: Confidence = "high" if value >= HIGH_CONFIDENCE else "medium" if value >= MEDIUM_CONFIDENCE else "low"
    return label, value


# --------------------------------------------------------------------------- #
# Reasons
# --------------------------------------------------------------------------- #
def reasons(f: Facts) -> list[Reason]:
    comp = f.composite
    cands: list[tuple[float, Reason]] = []
    for key, points in comp.contributions.items():
        part = comp.parts[key]
        if part.reason and abs(points) >= MIN_REASON_POINTS and abs((part.score or 50) - 50) >= 4:
            cands.append((abs(points), Reason(text=part.reason, polarity=polarity_of(part.score), weight=0, ref=key)))
    for i, story in enumerate(f.stories[:2]):
        n = story.narrative
        points = story_points(f, story)
        if points <= 0:
            continue
        lead = "Top story" if i == 0 else "Story"
        text = (f"{lead}: {quote(n.headline)} — {count(n.count, 'article')} from "
                f"{count(len(n.publishers), 'outlet')}, tone {signed(n.score)}")
        cands.append((points, Reason(text=text, polarity=tone_polarity(n.score, 0.1), weight=0, ref=n.id)))
    cands.sort(key=lambda c: -c[0])
    kept = [(p, r) for p, r in cands if p >= MIN_REASON_POINTS][:MAX_REASONS]
    if not kept:
        return []
    top = kept[0][0]
    return [r.model_copy(update={"weight": round(p / top, 3)}) for p, r in kept]


def story_points(f: Facts, story: Story) -> float:
    """How much a story explains the score: a share of the news component's contribution.

    A story aligned with the news flow gets (0.6 + 0.4·impact) of it; a
    counter-story half of that scaled by impact. Neutral-toned or minor
    stories get 0 (they inform, but do not drive the score)."""
    n = story.narrative
    if n.impact < STORY_MIN_IMPACT or abs(n.score) < 0.1:
        return 0.0
    news = f.composite.contributions.get("news", 0.0)
    if news == 0:
        return 0.0
    aligned = (n.score > 0) == (news > 0)
    return abs(news) * ((0.6 + 0.4 * n.impact) if aligned else 0.5 * n.impact)


# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
def _retail_tail(f: Facts, bullish: bool) -> str | None:
    """'retail attention is cooling (Reddit mentions fell 67% in 24h (21 → 7))' when the numbers say so."""
    change = reddit_change_pct(f.crowd)
    prev = f.crowd.reddit_mentions_prev if f.crowd else None
    move = reddit_move(f.crowd)
    if change is None or prev is None or prev < 10 or move is None:
        return None
    if change <= -50 and bullish:
        return f"retail attention is cooling (Reddit mentions {move})"
    if change >= 100:
        return f"Reddit chatter is surging (mentions {move})"
    return None


def headline(f: Facts, label: str, stance: str, conf_label: str) -> str:
    """One crisp, number-backed sentence: what drives the read and what pushes back.

    Drivers must be clear signals (strong phrases); counterweights may be mild
    leans. Components that move the score by < 1 point are never named."""
    comp = f.composite
    if not comp.available():
        return "No read: every source and data feed came back empty or failed — retry shortly."
    ranked = sorted(comp.contributions.items(), key=lambda kv: -abs(kv[1]))
    parts = comp.parts

    def side(sign: int, strong_only: bool) -> list[ComponentKey]:
        return [k for k, c in ranked if c * sign >= MIN_PHRASE_POINTS and parts[k].phrase
                and (parts[k].strong or not strong_only)]

    def text(key: ComponentKey) -> str:
        return parts[key].phrase or LABELS[key].lower()

    thin = f.overall.n < THIN_ITEMS and comp.coverage < 0.5
    if stance == "neutral":
        bulls, bears = side(1, False), side(-1, False)
        if bulls and bears:
            core = f"{join_and([text(k) for k in bulls[:2]])} offset by {join_and([text(k) for k in bears[:2]])}"
        elif bulls or bears:
            core = f"{text((bulls or bears)[0])} is the only clear signal; the rest is flat"
        else:
            core = _muted(f)
    else:
        sign = 1 if stance == "bullish" else -1
        support = side(sign, True) or side(sign, False)
        against = side(-sign, False)
        if not support:
            key = ranked[0][0]
            core = f"{LABELS[key].lower()} lead ({parts[key].detail})"
        elif len(support) >= 2:
            core = f"{text(support[0])} and {text(support[1])} align"
        else:
            core = f"driven by {text(support[0])}"
        if against:
            core += f"; the main {'drag' if sign > 0 else 'offset'} is {text(against[0])}"
        else:
            tail = _retail_tail(f, sign > 0)
            if tail:
                core += f"; {tail}"
    if thin:
        return f"{label} on thin evidence ({count(f.overall.n, 'relevant item')}): {core}."
    return f"{label}: {core}."


def _muted(f: Facts) -> str:
    if f.news.n:
        return f"signals are muted (news tone {signed(f.news.shrunk)} across {count(f.news.n, 'article')})"
    return "signals are muted"
