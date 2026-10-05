"""Story clusters -> ranked `Narrative`s: the few stories actually moving sentiment.

Only published media (news/analysis) that is clearly about the company
(relevance >= 0.5) is clustered; crowd chatter is summarized elsewhere.

    impact = coverage × (0.5 + 0.5·intensity) × freshness  [× 0.6 for question/listicle headlines]
    coverage  = 1 − exp(−(Σ relevance·copies + 0.5·outlets) / 5)   syndicated copies count
    intensity = max(|tone| / 0.4, 0.6 if a material event) capped at 1
    freshness = 0.35 + 0.65 · 0.5^(hours since last item / 48)

Tone is *anchored* on the representative headline: when the members whose own
tone is close to the headline's (within the cluster's spread, 0.15–0.30) hold
most of the story's weight, the story's tone is theirs — so a cluster that
mixes in unrelated items never shows its headline with a tone the headline does
not carry. Otherwise the headline speaks for a minority of the coverage and the
tone is the plain weighted mean. Either way a story is used as directional
evidence (`Story.directional`) only when the headline itself carries the tone
and the members agreeing with it hold most of the coverage.

An event is part of a story when the representative carries it or members
carrying it hold >= 35% of the story's weighted coverage (and number >= 2), so
one peripheral member cannot tag the story.

Single items, and clusters carried by a single outlet, only count when the
outlet is trusted, the item clearly about the company, the tone strong (or a
material event) and the headline a statement (not a question or listicle).

A narrative is NEW when most of its coverage is dated after the previous look
(minus 1 h of indexing lag) and none of its headlines matches a previous
narrative headline (overlap coefficient >= 0.5 with >= 3 shared content stems,
company name and generic market words ignored) — nor, when the caller knows
them, shares a member article with a previous story.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.analytics import textkit
from app.analytics.prepare import Item
from app.analytics.util import stable_id, tone_label, weighted_mean
from app.nlp.types import ClusterItem
from app.schemas import Narrative, Snapshot
from app.sources.base import CompanyRef

MIN_RELEVANCE = 0.5
MAX_NARRATIVES = 8
CLUSTER_POOL = 40
SINGLETON_MIN_TRUST = 0.9
SINGLETON_MIN_RELEVANCE = 0.8
SINGLETON_MIN_TONE = 0.3
SINGLE_OUTLET_MAX_FOCUS = 2.0  # a story only one outlet covers counts as at most 2 items
SINGLETON_FILL = 5  # single-outlet stories are listed only while fewer than this many stories are kept
WEAK_TITLE_DISCOUNT = 0.6

STORY_TONE = 0.1  # |tone| for a story (and its headline) to count as directional
ANCHOR_BAND = (0.15, 0.30)  # members within this tone distance of the headline form its core
CORE_SHARE = 0.5  # the core must hold this share of the story's weight to speak for it
MATERIAL_SHARE = 0.35  # weighted coverage share for a non-representative event to count

NEW_OVERLAP = 0.5
NEW_MIN_SHARED = 3
NEW_GRACE = timedelta(hours=1)  # indexing lag: items stamped just before the last look may be unseen
NEW_FRESH_SHARE = 0.5  # share of a story's dated coverage that must postdate the last look
NEW_SHARED_IDS = 1 / 3  # shared member articles (of the smaller story) that make two stories one

# Events that are developments in their own right (price moves merely describe the tape).
PRICE_EVENTS = frozenset({"price_up", "price_down", "all_time_high", "high_52w", "low_52w"})

_STOP = frozenset("""a an and are as at be by for from has have in into is it its of on or s says say said the
to was were will with after amid over than that this vs via new more why how what""".split())
# Words every market headline shares: they say nothing about which story it is.
_GENERIC = frozenset("""stock stocks share shares price prices investor investors market markets today week year
company inc corp co ltd update report news wall street trading trade""".split())
# Questions, listicles and "reasons to buy" pieces are opinion, not developments.
WEAK_TITLE_RE = re.compile(
    r"\?\s*$|^\s*(?:why|how|what|is|are|should|can|could|will|would|here'?s|this is)\b|"
    r"\b\d+\s+(?:reasons?|stocks?|things|ways|charts?)\b", re.IGNORECASE)


@dataclass
class Story:
    """A narrative plus the items behind it (for insights, catalysts and briefs).

    `members[0]` is the representative (its headline is the narrative's);
    `spread` is the weighted SD of all member tones and `core_share` the weight
    share of members whose tone agrees with the headline's."""

    narrative: Narrative
    members: list[Item]
    material_events: list[str]
    spread: float = 0.0
    core_share: float = 1.0

    @property
    def outlets(self) -> int:
        return len(self.narrative.publishers)

    @property
    def lead(self) -> Item:
        return self.members[0]

    @property
    def directional(self) -> bool:
        """Can this story be quoted as evidence of its tone?

        Its tone is clear, the headline itself carries that tone, and the
        members agreeing with the headline hold most of the coverage."""
        tone, rep = self.narrative.score, self.lead.score
        return (abs(tone) >= STORY_TONE and abs(rep) >= STORY_TONE and tone * rep > 0
                and self.core_share >= CORE_SHARE)


def build_narratives(items: list[Item], company: CompanyRef | None, now: datetime,
                     previous: Snapshot | None = None, limit: int = MAX_NARRATIVES,
                     previous_ids: Collection[Collection[str]] | None = None) -> list[Story]:
    """Cluster, score and rank stories; tags member items with their narrative id.

    `previous_ids` (optional): member signal ids of the previous snapshot's
    stories, the strongest evidence that a story is not new."""
    pool = [it for it in items if it.group == "news" and it.scored and it.relevance >= MIN_RELEVANCE]
    if not pool:
        return []
    by_id = {it.id: it for it in pool}
    clusters = textkit.clusters(
        [ClusterItem(id=it.id, title=it.title, timestamp=it.timestamp, score=it.score, weight=it.weight,
                     publisher=it.publisher) for it in pool],
        company, CLUSTER_POOL,
    )
    stories: list[Story] = []
    used: set[str] = set()
    for cluster in clusters:
        members = [by_id[i] for i in dict.fromkeys(cluster.item_ids) if i in by_id and i not in used]
        if not members:
            continue
        rep = by_id.get(cluster.representative_id) or members[0]
        if rep not in members:
            rep = members[0]
        story = _story(rep, members, now)
        if story is not None:
            used.update(it.id for it in members)
            stories.append(story)

    stories.sort(key=lambda s: (-s.narrative.impact, -s.narrative.count, s.narrative.id))
    # Single-outlet stories only fill out a thin list; they never crowd out corroborated ones.
    kept: list[Story] = []
    for story in stories:
        if story.outlets <= 1 and len(kept) >= SINGLETON_FILL:
            continue
        kept.append(story)
    stories = kept[:limit]
    if previous is not None and previous.n_signals > 0:
        ignore = textkit.company_terms(company)
        prev_tokens = [headline_tokens(h, ignore) for h in previous.narratives]
        prev_ids = [frozenset(ids) for ids in previous_ids or () if ids]
        for story in stories:
            story.narrative.is_new = is_new(story, previous.at, prev_tokens, prev_ids, ignore)
    for story in stories:
        for it in story.members:
            it.narrative_id = story.narrative.id
    return stories


def _story(rep: Item, members: list[Item], now: datetime) -> Story | None:
    members = [rep] + sorted((m for m in members if m is not rep), key=lambda m: (-m.weight, m.id))
    count = sum(m.coverage for m in members)
    tone, spread, core_share = anchored_tone(rep, members)
    qualified = story_events(rep, members)
    material = [k for k in qualified if k not in PRICE_EVENTS]

    outlet_counts = Counter(o for m in members for o in m.outlets())
    outlets = sorted(outlet_counts, key=lambda o: (-outlet_counts[o], -textkit.publisher_trust(o), o))
    # One item, or one outlet repeating itself, must clear the singleton bar.
    if (count < 2 or len(outlets) <= 1) and not (
        rep.trust >= SINGLETON_MIN_TRUST and rep.relevance >= SINGLETON_MIN_RELEVANCE
        and (abs(tone) >= SINGLETON_MIN_TONE or material) and not WEAK_TITLE_RE.search(rep.title)
    ):
        return None
    times = [t for m in members for t in m.times()]
    first, last = (min(times), max(times)) if times else (None, None)
    velocity = sum(1 for t in times if now - t <= timedelta(hours=24))

    intensity = max(min(1.0, abs(tone) / 0.4), 0.6 if material else 0.0)
    focus = sum(m.coverage * m.relevance for m in members)  # coverage *of this company*
    if len(outlets) <= 1:
        focus = min(focus, SINGLE_OUTLET_MAX_FOCUS)
    coverage = 1.0 - math.exp(-(focus + 0.5 * len(outlets)) / 5.0)
    age_h = (now - last).total_seconds() / 3600.0 if last else 48.0
    freshness = 0.35 + 0.65 * 0.5 ** (max(age_h, 0.0) / 48.0)
    impact = coverage * (0.5 + 0.5 * intensity) * freshness
    if WEAK_TITLE_RE.search(rep.title):
        impact *= WEAK_TITLE_DISCOUNT  # "X vs Y: which is the better buy?" is opinion, not a development

    theme_counts = Counter(t for m in members for t in m.themes)
    need = max(1, math.ceil(len(members) / 3))
    themes = [t for t, c in theme_counts.most_common(3) if c >= need]

    narrative = Narrative(
        id="n-" + stable_id(rep.id, n=10),
        headline=rep.title, count=count, publishers=outlets[:12],
        score=round(tone, 3), label=tone_label(tone),
        first_seen=first, last_seen=last, velocity_24h=velocity,
        themes=themes, events=qualified[:4],
        impact=round(min(1.0, impact), 3), url=rep.url,
        signal_ids=[m.id for m in members],
    )
    return Story(narrative=narrative, members=members, material_events=material, spread=round(spread, 3),
                 core_share=round(core_share, 3))


def anchored_tone(rep: Item, members: list[Item]) -> tuple[float, float, float]:
    """(tone, spread, core share) of a story.

    The core is every member within the cluster's weighted spread (clamped to
    0.15–0.30) of the representative's score. Tone is the core's weighted mean
    when the core holds >= half the weight, else the mean of all members (a
    mis-scored or atypical headline must not set the story's tone); `spread`
    is over all members."""
    mean, wsum = weighted_mean((m.score, m.weight) for m in members)
    if mean is None or wsum <= 0:
        return rep.score, 0.0, 1.0
    spread = math.sqrt(sum(m.weight * (m.score - mean) ** 2 for m in members if m.weight > 0) / wsum)
    band = min(max(spread, ANCHOR_BAND[0]), ANCHOR_BAND[1])
    core = [m for m in members if abs(m.score - rep.score) <= band]
    core_tone, core_w = weighted_mean((m.score, m.weight) for m in core)
    share = core_w / wsum
    tone = core_tone if core_tone is not None and share >= CORE_SHARE else mean
    return tone, spread, share


def story_events(rep: Item, members: list[Item]) -> list[str]:
    """Event keys that belong to the story (most widely carried first).

    The representative's own events always count; another event counts when
    >= 2 members carrying it hold >= 35% of the story's weighted coverage."""
    def mass(m: Item) -> float:
        return m.weight * m.coverage

    total = sum(mass(m) for m in members) or 1.0
    carriers: dict[str, list[Item]] = {}
    for m in members:
        for key in m.event_keys:
            carriers.setdefault(key, []).append(m)
    own = set(rep.event_keys)
    kept = [k for k, ms in carriers.items()
            if k in own or (len(ms) >= 2 and sum(mass(m) for m in ms) / total >= MATERIAL_SHARE)]
    return sorted(kept, key=lambda k: (-sum(mass(m) for m in carriers[k]), k))


# --------------------------------------------------------------------------- #
# NEW since the last look
# --------------------------------------------------------------------------- #
def headline_tokens(text: str, ignore: frozenset[str] = frozenset()) -> frozenset[str]:
    """Content stems that identify a story: no stopwords, company name or generic market words."""
    drop = _STOP | _GENERIC | ignore
    return frozenset(t for t in textkit.stemmed_tokens(text, drop) if len(t) > 1 and t not in drop)


def same_headline(a: frozenset[str], b: frozenset[str]) -> bool:
    """Two headlines tell the same story: overlap coefficient >= 0.5 with >= 3 shared stems
    (>= 2 when the shorter headline has only 2–3 content stems)."""
    small = min(len(a), len(b))
    shared = len(a & b)
    return small >= 2 and shared >= min(NEW_MIN_SHARED, small) and shared / small >= NEW_OVERLAP


def is_new(story: Story, since: datetime, previous: list[frozenset[str]],
           previous_ids: list[frozenset[str]], ignore: frozenset[str] = frozenset()) -> bool:
    """Most of the story's coverage postdates the last look and it matches no previous story."""
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    times = [t for m in story.members for t in m.times()]
    if not times:
        return False  # undated: cannot tell when it broke
    fresh = sum(1 for t in times if t > since - NEW_GRACE)
    if fresh / len(times) < NEW_FRESH_SHARE:
        return False
    ids = set(story.narrative.signal_ids)
    for old in previous_ids:
        shared = len(ids & old)
        if shared and shared >= NEW_SHARED_IDS * min(len(ids), len(old)):
            return False
    titles = [t for m in story.members for t in m.titles()]
    tokens = [headline_tokens(t, ignore) for t in titles]
    return not any(same_headline(a, b) for a in tokens for b in previous)
