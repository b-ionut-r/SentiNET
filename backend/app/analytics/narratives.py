"""Story clusters -> ranked `Narrative`s: the few stories actually moving sentiment.

Only published media (news/analysis) that is clearly about the company
(relevance >= 0.5) is clustered; crowd chatter is summarized elsewhere.

    impact = coverage × (0.5 + 0.5·intensity) × freshness  [× 0.6 for question/listicle headlines]
    coverage  = 1 − exp(−(Σ relevance·copies + 0.5·outlets) / 5)   syndicated copies count
    intensity = max(|tone| / 0.4, 0.6 if a material event) capped at 1
    freshness = 0.35 + 0.65 · 0.5^(hours since last item / 48)

A story's tone is the weighted mean of *all* its members — the tone of the
coverage its article count describes ("12 articles, tone −0.07"), never just
the part that agrees with the headline. The headline is *anchored* on the
coverage: the members agreeing with a headline (tone within the cluster's
spread, 0.15–0.30, or on the same side of neutral) form its core, and when the
clusterer's representative speaks for only a minority (a catch-all cluster, a
mis-scored headline) the member whose core holds the largest majority fronts
the story instead; with no majority anywhere the story is mixed. A story is
used as directional evidence (`Story.directional`) only when its tone is clear,
its headline carries that tone, and the coverage agreeing with the headline
holds most of the weight.

A routine target tweak (a structured analyst action keeping the rating and
moving the target < 3%, matched by firm within 4 days) is not a development:
a story whose only material events are target revisions explained by such
actions has its intensity capped at 0.3 and is never featured as "the" story
(see `featured`), so "X maintains Buy, trims target to $355" cannot become
the top story.

The headline is a complete statement about the company: a representative
whose title is cut off ("…") or that is clearly less about the company than
another member (a peer's headline) gives way to the best member that is
neither, preferring one that carries a material event.

An event is part of a story when the representative carries it or members
carrying it hold >= 35% of the story's weighted coverage (and number >= 2), so
one peripheral member cannot tag the story.

Single items, and clusters carried by a single outlet, only count when the
outlet is trusted, the item clearly about the company, the tone strong (or a
material event) and the headline a statement (not a question or listicle).

A narrative is NEW when something in it was published since the previous look,
most of its coverage (strictly) is dated after that look (minus 1 h of indexing
lag), and none of its headlines matches a previous narrative headline (overlap
coefficient >= 0.5 with >= 3 shared content stems, company name and generic
market words ignored) — nor, when the caller knows them, shares a member
article with a previous story.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.analytics import textkit
from app.analytics.prepare import PRICE_EVENTS, Item, price_recap
from app.analytics.util import stable_id, tone_label, weighted_mean
from app.nlp.types import ClusterItem
from app.schemas import AnalystAction, Narrative, Snapshot
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
NEW_FRESH_SHARE = 0.5  # share of a story's dated coverage that must (strictly) exceed this after the last look
NEW_SHARED_IDS = 1 / 3  # shared member articles (of the smaller story) that make two stories one

# Events that are developments in their own right (price moves merely describe the tape).
STORY_MIN_IMPACT = 0.35  # a story below this impact is never quoted as "the" story
INSIDER_EVENTS = frozenset({"insider_buy", "insider_sell"})
REVISION_EVENTS = frozenset({"pt_raise", "pt_cut"})
ROUTINE_REVISION = 0.03  # |target change| below which a rating-unchanged revision is routine
ROUTINE_REVISION_INTENSITY = 0.3
REVISION_MATCH_WINDOW = timedelta(days=4)

_STOP = frozenset("""a an and are as at be by for from has have in into is it its of on or s says say said the
to was were will with after amid over than that this vs via new more why how what""".split())
# Words every market headline shares: they say nothing about which story it is.
_GENERIC = frozenset("""stock stocks share shares price prices investor investors market markets today week year
company inc corp co ltd update report news wall street trading trade""".split())
# Questions, listicles, "stocks to buy" and first-person columns are opinion, not developments.
WEAK_TITLE_RE = re.compile(
    r"\?\s*$|^\s*(?:why|how|what|is|are|should|can|could|will|would|here'?s|this is|forget|my)\b|"
    r"\bhistory\s+says\b|"
    r"(?<!S&P )(?<!Nasdaq )(?<!Russell )(?<!Dow )(?<![$\d.,])\b\d+\s+"
    r"(?:(?!million|billion|trillion|thousand)[\w&'-]+\s+){0,3}?"
    r"(?:reasons|stocks|picks|etfs|things|ways|charts|lessons|signs|mistakes)\b|"
    r"\b(?:stocks?|shares?|etfs?)\s+to\s+(?:buy|sell|watch|avoid|own|hold)\b|"
    r"\b(?:these|best|top)\s+(?:[\w&'-]+\s+){0,2}?(?:stocks|etfs|picks)\b|"
    r"(?<![$\d.,])\b1\s+(?:[\w&'-]+\s+){0,3}?(?:reason|stock|etf|pick)\b|"
    r"\bbiggest\s+(?:warning|mistake)\b|\bbetter\s+buy\b|\bon\s+my\s+radar\b", re.IGNORECASE)


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
    intensity: float = 1.0  # max(|tone| / 0.4, 0.6 if material) capped at 1 (see module docstring)
    routine: bool = False  # a rating-unchanged target tweak (see `routine_revision`)

    def cap_intensity(self, cap: float) -> None:
        """Lower the story's intensity to `cap`, rescaling its impact accordingly."""
        if self.intensity <= cap:
            return
        n = self.narrative
        n.impact = round(n.impact * (0.5 + 0.5 * cap) / (0.5 + 0.5 * self.intensity), 3)
        self.intensity = cap

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

    @property
    def price_only(self) -> bool:
        """The story is the price move itself ('stock craters 43% in 2026'): no material event,
        and its headline is a price recap. The technicals component already measures it."""
        return not self.material_events and price_recap(self.lead)

    @property
    def insider_only(self) -> bool:
        """Its only development is an insider trade, which the insiders component states from
        the Form 4 data (the story would say it twice)."""
        return bool(self.material_events) and set(self.material_events) <= INSIDER_EVENTS


def featured(stories: list[Story]) -> list[Story]:
    """Stories that may be quoted as *the* story (verdict reasons, headline, brief), ranked.

    Price recaps (they restate the tape the technicals already count) and
    routine target tweaks (not a development) are left out — unless no other
    story clears STORY_MIN_IMPACT."""
    substantive = [s for s in stories if not s.price_only and not s.routine]
    if any(s.narrative.impact >= STORY_MIN_IMPACT for s in substantive):
        return substantive
    return list(stories)


def build_narratives(items: list[Item], company: CompanyRef | None, now: datetime,
                     previous: Snapshot | None = None, limit: int = MAX_NARRATIVES,
                     previous_ids: Collection[Collection[str]] | None = None,
                     actions: Sequence[AnalystAction] = ()) -> list[Story]:
    """Cluster, score and rank stories; tags member items with their narrative id.

    `previous_ids` (optional): member signal ids of the previous snapshot's
    stories, the strongest evidence that a story is not new. `actions`
    (optional): structured analyst actions, to recognise routine target tweaks."""
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

    for story in stories:
        if routine_revision(story, actions):
            story.routine = True
            story.cap_intensity(ROUTINE_REVISION_INTENSITY)
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
    rep = headline_candidate(rep, members)
    rep, _, spread, core_share = pick_anchor(rep, members)
    members = [rep] + [m for m in members if m is not rep]
    tone = story_tone(members, rep)
    count = sum(m.coverage for m in members)
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
                 core_share=round(core_share, 3), intensity=intensity)


TRUNCATED_RE = re.compile(r"(?:\.\.\.|…)\s*$")
CLEARLY_MORE_RELEVANT = 0.1  # a member this much more about the company fronts the story instead


def _good_title(m: Item) -> bool:
    return not TRUNCATED_RE.search(m.title) and not WEAK_TITLE_RE.search(m.title)


def headline_candidate(rep: Item, members: list[Item]) -> Item:
    """The clusterer's representative, unless its title is cut off ('… eyei...') or another member
    is clearly more about the company (LULU's guidance cut fronted by 'Nike Sinks 8% …').

    The replacement has a complete statement headline, is at least as relevant, and carries one
    of the story's material events when a member does."""
    truncated = bool(TRUNCATED_RE.search(rep.title))
    floor = rep.relevance - 0.05 if truncated else rep.relevance + CLEARLY_MORE_RELEVANT
    better = [m for m in members if m is not rep and _good_title(m) and m.relevance >= floor - 1e-9]
    if not better:
        return rep

    def material(m: Item) -> bool:
        return any(k not in PRICE_EVENTS for k in m.event_keys)

    return max(better, key=lambda m: (material(m), m.relevance, m.weight, m.id))


def story_tone(members: list[Item], rep: Item) -> float:
    """Weighted mean tone of every member (the representative's own score without weight)."""
    tone, _ = weighted_mean((m.score, m.weight) for m in members)
    return rep.score if tone is None else tone


def anchored_tone(rep: Item, members: list[Item]) -> tuple[float, float, float]:
    """(tone, spread, core share) of a story told by `rep`'s headline.

    The core is the coverage agreeing with the headline: members within the
    cluster's weighted spread (clamped to 0.15–0.30) of its score, plus — for
    a directional headline — every member on the same side of neutral (a +0.8
    and a +0.4 headline tell the same bullish story). The tone is the core's
    weighted mean (used to pick the headline; the story's shown tone is the mean
    of all members, `story_tone`). `spread` is over all members, `core share`
    is the core's share of the story's weight."""
    mean, wsum = weighted_mean((m.score, m.weight) for m in members)
    if mean is None or wsum <= 0:
        return rep.score, 0.0, 1.0
    spread = math.sqrt(sum(m.weight * (m.score - mean) ** 2 for m in members if m.weight > 0) / wsum)
    band = min(max(spread, ANCHOR_BAND[0]), ANCHOR_BAND[1])
    side = 0 if abs(rep.score) < STORY_TONE else 1 if rep.score > 0 else -1
    core = [m for m in members
            if abs(m.score - rep.score) <= band or (side and m.score * side >= STORY_TONE)]
    core_tone, core_w = weighted_mean((m.score, m.weight) for m in core)
    return (core_tone if core_tone is not None else rep.score), spread, core_w / wsum


def pick_anchor(rep: Item, members: list[Item]) -> tuple[Item, float, float, float]:
    """(headline item, tone, spread, core share) for a cluster.

    The clusterer's representative tells the story when the coverage agreeing
    with it holds most of the weight. Otherwise (a catch-all cluster, or a
    mis-scored headline) the member whose agreeing coverage holds the largest
    majority takes over — so headline and tone always describe the bulk of the
    coverage. With no majority anywhere the story is mixed: it keeps the
    representative and is never used as directional evidence."""
    tone, spread, share = anchored_tone(rep, members)
    if share >= CORE_SHARE:
        return rep, tone, spread, share
    best: tuple[Item, float, float, float] | None = None
    top = max((m.relevance for m in members if _good_title(m)), default=rep.relevance)
    for m in members:
        if m is rep or not _good_title(m) or m.relevance < rep.relevance - 0.15:
            continue
        if m.relevance <= top - CLEARLY_MORE_RELEVANT + 1e-9:
            continue  # a member clearly less about the company never fronts its story
        m_tone, _, m_share = anchored_tone(m, members)
        if m_share >= CORE_SHARE and (best is None or (m_share, m.weight) > (best[3], best[0].weight)):
            best = (m, m_tone, spread, m_share)
    return best or (rep, tone, spread, share)


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
# Routine analyst target tweaks
# --------------------------------------------------------------------------- #
_FIRM_NOISE = frozenset("""group securities capital markets partners research financial company co inc llc ltd
plc lp sa ag the and of & isi""".split())
_FIRM_ALIASES = {"b of a": ("bofa", "bank of america", "b of a"), "jp morgan": ("jpmorgan", "j.p. morgan", "jp morgan"),
                 "j.p. morgan": ("jpmorgan", "j.p. morgan", "jp morgan")}


def _firm_names(firm: str) -> tuple[str, ...]:
    """Ways a headline names a research firm ('Evercore ISI Group' -> 'evercore'; never a bare
    'morgan' for Morgan Stanley, which would also match J.P. Morgan)."""
    low = " ".join(firm.lower().split())
    for key, names in _FIRM_ALIASES.items():
        if low.startswith(key):
            return names
    core = [w for w in re.findall(r"[a-z0-9.&'-]+", low) if w not in _FIRM_NOISE]
    return (" ".join(core),) if core else ()


def names_firm(title: str, firm: str) -> bool:
    text = " " + " ".join(re.findall(r"[a-z0-9.&'-]+", title.lower())) + " "
    return any(f" {name} " in text for name in _firm_names(firm))


def routine_revision(story: Story, actions: Sequence[AnalystAction]) -> bool:
    """The story's only development is a target tweak the structured data shows as routine.

    True when its material events are all target revisions and the firm its
    headline names has, within 4 days, only rating-unchanged actions moving the
    target by < 3% (an upgrade, downgrade or initiation is never routine)."""
    if not actions or not story.material_events or not set(story.material_events) <= REVISION_EVENTS:
        return False
    n = story.narrative
    when = n.last_seen or n.first_seen
    if when is None:
        return False
    matched = [a for a in actions if abs(a.date - when) <= REVISION_MATCH_WINDOW and names_firm(n.headline, a.firm)]
    if not matched:
        return False
    for a in matched:
        if a.action in ("up", "down", "init") or (a.from_grade and a.to_grade and a.from_grade != a.to_grade):
            return False
        if a.price_target and a.prior_target and abs(a.price_target / a.prior_target - 1) >= ROUTINE_REVISION:
            return False
    return any(a.price_target and a.prior_target for a in matched)


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
    """Something arrived since the last look, most of the story's coverage postdates it
    (less the indexing grace), and it matches no previous story."""
    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)
    times = [t for m in story.members for t in m.times()]
    if not times or max(times) <= since:
        return False  # undated, or nothing published since the last look (it may merely have grown)
    fresh = sum(1 for t in times if t > since - NEW_GRACE)
    if fresh / len(times) <= NEW_FRESH_SHARE:
        return False
    ids = set(story.narrative.signal_ids)
    for old in previous_ids:
        shared = len(ids & old)
        if shared and shared >= NEW_SHARED_IDS * min(len(ids), len(old)):
            return False
    titles = [t for m in story.members for t in m.titles()]
    tokens = [headline_tokens(t, ignore) for t in titles]
    return not any(same_headline(a, b) for a in tokens for b in previous)
