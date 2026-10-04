"""Story clusters -> ranked `Narrative`s: the few stories actually moving sentiment.

Only published media (news/analysis) that is clearly about the company
(relevance >= 0.5) is clustered; crowd chatter is summarized elsewhere.

    impact = coverage × (0.3 + 0.7·intensity) × freshness            (0..1)
    coverage  = 1 − exp(−(items + 0.5·outlets) / 5)       syndicated copies count
    intensity = max(|tone| / 0.4, 0.6 if a material event) capped at 1
    freshness = 0.35 + 0.65 · 0.5^(hours since last item / 48)

A narrative is NEW when none of its headlines shares >= 50% of its content
tokens (Jaccard) with any narrative headline of the previous snapshot.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.analytics import textkit
from app.analytics.prepare import Item
from app.analytics.util import stable_id, tone_label, weighted_mean
from app.nlp.types import ClusterItem
from app.schemas import Narrative, Snapshot
from app.sources.base import CompanyRef

MIN_RELEVANCE = 0.5
MAX_NARRATIVES = 8
CLUSTER_POOL = 40
NEW_JACCARD = 0.5
SINGLETON_MIN_TRUST = 0.9
SINGLETON_MIN_RELEVANCE = 0.8
SINGLETON_MIN_TONE = 0.3

# Events that are developments in their own right (price moves merely describe the tape).
PRICE_EVENTS = frozenset({"price_up", "price_down", "all_time_high", "low_52w"})

_STOP = frozenset("""a an and are as at be by for from has have in into is it its of on or s says say said the
to was were will with after amid over than that this vs via new more why how what""".split())
_TOKEN_RE = re.compile(r"[a-z0-9$%][a-z0-9$%.']*")


@dataclass
class Story:
    """A narrative plus the items behind it (for insights, catalysts and briefs)."""

    narrative: Narrative
    members: list[Item]
    material_events: list[str]

    @property
    def outlets(self) -> int:
        return len(self.narrative.publishers)


def build_narratives(items: list[Item], company: CompanyRef | None, now: datetime,
                     previous: Snapshot | None = None, limit: int = MAX_NARRATIVES) -> list[Story]:
    """Cluster, score and rank stories; tags member items with their narrative id."""
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
    stories = stories[:limit]
    prev_tokens = [_tokens(h) for h in previous.narratives] if previous is not None else None
    for story in stories:
        if prev_tokens is not None:
            story.narrative.is_new = _is_new(story, prev_tokens)
        for it in story.members:
            it.narrative_id = story.narrative.id
    return stories


def _story(rep: Item, members: list[Item], now: datetime) -> Story | None:
    members = [rep] + sorted((m for m in members if m is not rep), key=lambda m: (-m.weight, m.id))
    count = sum(m.coverage for m in members)
    mean, _ = weighted_mean((m.score, m.weight) for m in members)
    tone = mean or 0.0
    events = Counter(k for m in members for k in m.event_keys)
    material = [k for k, _ in events.most_common() if k not in PRICE_EVENTS]

    if count < 2 and not (
        rep.trust >= SINGLETON_MIN_TRUST and rep.relevance >= SINGLETON_MIN_RELEVANCE
        and (abs(tone) >= SINGLETON_MIN_TONE or material)
    ):
        return None

    outlet_counts = Counter(o for m in members for o in m.outlets())
    outlets = sorted(outlet_counts, key=lambda o: (-outlet_counts[o], -textkit.publisher_trust(o), o))
    times = [t for m in members for t in m.times()]
    first, last = (min(times), max(times)) if times else (None, None)
    velocity = sum(1 for t in times if now - t <= timedelta(hours=24))

    intensity = max(min(1.0, abs(tone) / 0.4), 0.6 if material else 0.0)
    coverage = 1.0 - math.exp(-(count + 0.5 * len(outlets)) / 5.0)
    age_h = (now - last).total_seconds() / 3600.0 if last else 48.0
    freshness = 0.35 + 0.65 * 0.5 ** (max(age_h, 0.0) / 48.0)
    impact = coverage * (0.3 + 0.7 * intensity) * freshness

    theme_counts = Counter(t for m in members for t in m.themes)
    need = max(1, math.ceil(len(members) / 3))
    themes = [t for t, c in theme_counts.most_common(3) if c >= need]

    narrative = Narrative(
        id="n-" + stable_id(rep.id, n=10),
        headline=rep.title, count=count, publishers=outlets[:12],
        score=round(tone, 3), label=tone_label(tone),
        first_seen=first, last_seen=last, velocity_24h=velocity,
        themes=themes, events=[k for k, _ in events.most_common(4)],
        impact=round(min(1.0, impact), 3), url=rep.url,
        signal_ids=[m.id for m in members],
    )
    return Story(narrative=narrative, members=members, material_events=material)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t.strip(".'") for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _is_new(story: Story, previous: list[frozenset[str]]) -> bool:
    titles = [t for m in story.members for t in m.titles()]
    return not any(jaccard(_tokens(t), p) >= NEW_JACCARD for t in titles for p in previous)
