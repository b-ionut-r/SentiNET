"""Verdict = composite score + label + honest confidence + ranked reasons + one-line headline.

Reasons are ranked by how many points they move the score: a component's
contribution (its share of the effective weight × its distance from 50); the
top story claims a share of the news component's contribution (see
`story_points`). Every reason and headline clause is built from the evidence
numbers carried by the components and narratives.

A pending acquisition of the company (deals.py) outranks everything: it leads
the headline and the reasons, because the price now tracks the deal terms.
"""
from __future__ import annotations

import math

from app.analytics.composite import LABELS, WEIGHTS, ComponentKey
from app.analytics.crowd import reddit_change_pct, reddit_move
from app.analytics.facts import Facts
from app.analytics.narratives import STORY_MIN_IMPACT, Story, featured
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
        confidence_value=round(conf_value, 3), headline=headline(f, label, stance),
        reasons=reasons(f), components=[p.component() for p in comp.parts.values()],
    )


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def confidence(f: Facts) -> tuple[Confidence, float]:
    """data quality (coverage, volume, freshness) × (0.5 + 0.5 · component agreement).

    Coverage is the nominal weight of available components, each counted in
    proportion to its own confidence (full at >= 0.6); agreement is 1 minus the
    effective-weighted spread of component strengths (a lone component cannot
    corroborate itself: 0.5)."""
    comp = f.composite
    volume = f.overall.n_eff / (f.overall.n_eff + 12.0)
    avail = comp.available()
    if len(avail) >= 2:
        mean_x = sum(comp.effective[p.key] * p.x for p in avail)
        sd = math.sqrt(sum(comp.effective[p.key] * (p.x - mean_x) ** 2 for p in avail))
        agreement = 1.0 - min(1.0, sd / 0.6)
    else:
        agreement = 0.5
    ages = [it.age_hours(f.now) for it in f.prepared.scored]
    fresh = sum(1 for age in ages if age is not None and age <= 72) / len(ages) if ages else 0.3
    # A component counts fully once its own confidence reaches 0.6 (two headlines are not "news coverage").
    coverage = sum(WEIGHTS[p.key] * min(1.0, p.confidence / 0.6) for p in avail)
    data = 0.45 * coverage + 0.35 * volume + 0.20 * fresh
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
    ranked = featured(f.stories)
    told: list[Story] = []  # at most two: the leading story, and a second only when it runs against it
    for story in ranked[:2]:
        points = story_points(f, story)
        if points <= 0 or (told and (told[0].narrative.score > 0) == (story.narrative.score > 0)):
            continue
        told.append(story)
        cands.append((points, Reason(text="", polarity=tone_polarity(story.narrative.score, 0.1), weight=0,
                                     ref=story.narrative.id)))
    cands.sort(key=lambda c: -c[0])
    deal = _deal_reason(f)
    kept = [(p, r) for p, r in cands if p >= MIN_REASON_POINTS][:MAX_REASONS - (1 if deal else 0)]
    # Story labels are set after the cut: "Counter-story" only next to the story it counters.
    kept_ids = {r.ref for _, r in kept}
    for i, story in enumerate(told):
        n = story.narrative
        lead = ("Top story" if story is ranked[0]
                else "Counter-story" if i == 1 and told[0].narrative.id in kept_ids else "Story")
        text = (f"{lead}: {quote(n.headline)} — {count(n.count, 'article')} from "
                f"{count(len(n.publishers), 'outlet')}, tone {signed(n.score)}")
        kept = [(p, r.model_copy(update={"text": text}) if r.ref == n.id else r) for p, r in kept]
    out = []
    if kept:
        top = kept[0][0]
        out = [r.model_copy(update={"weight": round(p / top, 3)}) for p, r in kept]
    return ([deal] if deal else []) + out


def _deal_reason(f: Facts) -> Reason | None:
    d = f.deal
    if d is None:
        return None
    return Reason(text=f"Pending acquisition{d.by}: {d.excerpt} (Form {d.form}, {d.when}) — the share price now "
                       f"tracks the deal terms and the odds of closing", polarity="neutral", weight=1.0, ref="deal")


def story_points(f: Facts, story: Story) -> float:
    """How much a story explains the score: a share of the news component's contribution.

    A story aligned with the news flow gets (0.6 + 0.4·impact) of it; a
    counter-story half of that scaled by impact. Neutral-toned, mixed (its
    headline does not carry the tone) or minor stories get 0: they inform,
    but do not drive the score."""
    n = story.narrative
    if n.impact < STORY_MIN_IMPACT or not story.directional:
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


MAX_HEADLINE = 200


def headline(f: Facts, label: str, stance: str) -> str:
    """One crisp, number-backed sentence: what drives the read and what pushes back.

    Drivers must be clear signals (strong phrases); counterweights may be mild
    leans. Components that move the score by < 1 point are never named. The
    dominant story is named when it drives the news read and the sentence
    stays short enough to scan. A pending acquisition replaces all of that:
    it is the one fact that matters."""
    if f.deal is not None and f.composite.available():
        d = f.deal
        return (f"{label}, but a pending acquisition dominates: {f.name} agreed to be acquired{d.by} (merger "
                f"agreement, 8-K {d.when}) — the price tracks the deal terms, not sentiment.")
    text = _headline(f, label, stance, with_story=True)
    return text if len(text) <= MAX_HEADLINE else _headline(f, label, stance, with_story=False)


def _headline(f: Facts, label: str, stance: str, with_story: bool) -> str:
    comp = f.composite
    if not comp.available():
        return "No read: every source and data feed came back empty or failed — retry shortly."
    ranked = sorted(comp.contributions.items(), key=lambda kv: -abs(kv[1]))
    parts = comp.parts

    def side(sign: int, strong_only: bool) -> list[ComponentKey]:
        return [k for k, c in ranked if c * sign >= MIN_PHRASE_POINTS and parts[k].phrase
                and (parts[k].strong or not strong_only)]

    lead_story = _news_story_phrase(f) if with_story else None

    def text(key: ComponentKey) -> str:
        if key == "news" and lead_story:
            return lead_story
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
        if not support:  # only small or phrase-less pushes: name the largest one on the stance's side
            key = next((k for k, c in ranked if c * sign > 0), ranked[0][0])
            core = f"led by {LABELS[key].lower()} ({parts[key].detail})"
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
    if f.prepared.engine_error:
        unscored = count(len(f.prepared.items), "text")
        return f"{label} from structured data only ({unscored} could not be scored): {core}."
    if thin:
        return f"{label} on thin evidence ({count(f.overall.n, 'relevant item')}): {core}."
    return f"{label}: {core}."


def _news_story_phrase(f: Facts) -> str | None:
    """'negative news led by ‘Nimbus hit with $1.05B lawsuit…’ (8 articles, −0.58)' when one story drives the tone."""
    part = f.composite.parts["news"]
    ranked = featured(f.stories)
    if not ranked or not part.phrase or part.x == 0:
        return None
    n = ranked[0].narrative
    if n.impact < 0.5 or abs(n.score) < 0.15 or (n.score > 0) != (part.x > 0) or not ranked[0].directional:
        return None
    adj = part.phrase.split(" news", 1)[0]
    return f"{adj} news led by {quote(n.headline, 64)} ({count(n.count, 'article')}, {signed(n.score)})"


def _muted(f: Facts) -> str:
    if f.news.n:
        return f"signals are muted (news tone {signed(f.news.shrunk)} across {count(f.news.n, 'article')})"
    return "signals are muted"
