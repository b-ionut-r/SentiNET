"""Raw source items -> clean, relevant, de-duplicated, scored and weighted items.

Pipeline (per analysis):

1. clean title/body, drop empty/boilerplate (incl. auto-generated 13F-holdings
   stories) and stale (> 21 d) items;
2. relevance: drop < 0.35 unless the provider guarantees the ticker
   (`ticker_specific`, floored at 0.7) — never for a title naming only a sister
   company ("Vodafone Idea" on a VOD.L feed); multi-ticker roundups are capped;
3. collapse syndicated near-copies into one representative (the most trusted
   outlet), keeping the copies' outlets/times as coverage evidence;
4. score representatives with the sentiment engine (+ themes, events);
5. weight = source weight × trust (the outlet's for media, 0.6 for a user post)
   × recency × engagement × relevance
   × (0.5 + 0.5·confidence) × (1 + 0.15·ln(1 + copies)) [× 0.4 for a price
   recap], then × min(1, √(4 / items from the same outlet — or, for social
   posts, author)) so one prolific outlet or account (auto-generated 13F
   stories, spam bots) cannot dominate the aggregate.

A *price recap* ("SoFi stock craters 43% in 2026", "Nvidia hits record high")
is an item whose only events are price moves and whose strongest sentiment
driver is that price-move phrase — or a machine-written daily stock report
("Shopify Inc. Cl A stock rises Monday, outperforms market") carrying no
development: it restates the tape, which the technicals component already
measures, so it counts at 0.4× in the text aggregates instead of a second time
at full weight, and its story is never quoted as "the" story.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from app.analytics import textkit
from app.analytics.inputs import SourceRun
from app.analytics.util import clamp, finite, stable_id
from app.nlp.types import DetectedEvent
from app.schemas import Driver, SentimentLabel, Signal, SignalKind
from app.sources.base import CompanyRef, RawSignal

Group = Literal["news", "social"]

MAX_AGE = timedelta(days=21)
FUTURE_TOLERANCE = timedelta(hours=1)  # clock skew; anything later is treated as undated
MIN_RELEVANCE = 0.35
SPECIFIC_RELEVANCE = 0.7  # floor for items a provider guarantees are about the ticker
CONTEXT_DISCOUNT = 0.8  # a mention only in the snippet/parent story is weaker evidence
ROUNDUP_SYMBOLS = 4  # items tagged with this many tickers are roundups
ROUNDUP_CAP = 0.4
PROVIDER_RELEVANCE_MIN = 0.6
PRESS_RELEASE_TRUST = 0.55
# A user post is one anonymous voice: it starts below a mid-tier outlet (~0.9–1.0) and only its
# engagement (up to ×2) lifts it, so "$NVDA new ATHs all week 🚀🚀🚀" does not outrank Barron's.
SOCIAL_TRUST = 0.6
HALF_LIFE_H: dict[Group, float] = {"news": 72.0, "social": 36.0}
RECENCY_FLOOR = 0.15
UNDATED_RECENCY = 0.5
MAX_BODY = 600
MAX_DRIVERS = 5
DIVERSITY_FREE = 4  # items an outlet/author contributes before its items are down-weighted
PRICE_RECAP_WEIGHT = 0.4  # a headline that only restates the price move (see module docstring)
# Events that merely describe the tape (shared with narratives/catalysts).
PRICE_EVENTS = frozenset({"price_up", "price_down", "all_time_high", "high_52w", "low_52w"})
_TOKEN_RE = re.compile(r"[a-z0-9.%$]+")
# Machine-written daily stock reports (MarketWatch's automated recaps): the day's move against the
# market or peers and nothing else, often with no price event for the engine to see — 'Shopify Inc.
# Cl A stock rises Monday, outperforms market', 'GoPro Inc. stock outperforms competitors on strong
# trading day', 'lululemon athletica inc. stock underperforms Tuesday when compared to competitors'.
_WEEKDAY = r"(?:mon|tues|wednes|thurs|fri|satur|sun)day"
AUTO_RECAP_RE = re.compile(
    rf"\b{_WEEKDAY},\s+(?:still\s+)?(?:out|under)performs\s+(?:the\s+)?market\b|"
    rf"\bstock\s+(?:out|under)performs\s+{_WEEKDAY}\s+when\s+compared\s+to\s+competitors\b|"
    r"\bstock\s+(?:out|under)performs\s+(?:the\s+)?(?:market|competitors)\s+(?:on\s+(?:strong|weak)\s+trading\s+day|"
    r"despite\s+(?:daily\s+)?(?:losses|gains))\b", re.IGNORECASE)

# Auto-generated 13F-holdings stories ("Apple Inc. $AAPL Shares Sold by Denver PWM LLC",
# "Evoke Wealth LLC Sells 229,221 Shares of NVIDIA Corporation $NVDA"): templated filings
# noise from MarketBeat and its syndication network, with no sentiment content. Only the
# template counts — a holdings verb phrase plus the template's ticker tag (a cashtag or
# "(NASDAQ:AAPL)"), a MarketBeat publisher, or an actor ending in a legal suffix — so real
# ownership news ("Elliott Management Takes Stake in Southwest", "SoftBank Group Sells
# Stake in Nvidia") survives.
_HOLDINGS_VERB_RE = re.compile(
    r"\b(?:Stock|Shares?|Stake|Position|Holdings)\s+(?:Sold|Acquired|Bought|Purchased|Raised|Lowered|Trimmed|"
    r"Boosted|Lifted|Increased|Decreased|Reduced|Cut)\s+by\s+(?P<actor>[A-Z][^,;:]{1,80})$|"
    r"\b(?:Stake|Position|Holdings)\s+in\s+.{2,80}?\s+(?:Raised|Lowered|Trimmed|Boosted|Lifted|Increased|"
    r"Decreased|Reduced|Cut)\s+by\s+(?P<actor3>[A-Z][^,;:]{1,80})$|"
    r"^(?P<actor2>[A-Z][^,;:]{1,80}?)\s+(?:Acquires|Buys|Sells|Trims|Lowers|Raises|Boosts|Lifts|Cuts|Increases|"
    r"Decreases|Reduces|Takes|Grows|Purchases)\s+(?:(?:a\s+)?New\s+)?(?:[\d,]+\s+Shares\s+of|"
    r"(?:Stock\s+)?(?:Stake|Position|Holdings)\s+in)\b")
_TEMPLATE_TAG_RE = re.compile(r"(?<![\w$])\$[A-Z]{1,5}(?:\.[A-Z])?\b|"
                              r"\((?:NASDAQ|NYSE|NYSEARCA|NYSEAMERICAN|AMEX|OTCMKTS|OTC|BATS|CBOE)\s?:\s?[A-Z.]{1,7}\)")
_LEGAL_SUFFIX_RE = re.compile(r"\b(?:LLC|L\.L\.C\.|LP|L\.P\.|Ltd\.?|Inc\.?|Co\.?|S\.A\.|N\.A\.|PLC)\s*$")
_MARKETBEAT_RE = re.compile(r"marketbeat", re.IGNORECASE)


def is_holdings_boilerplate(title: str, publisher: str | None) -> bool:
    """True for MarketBeat-style auto-generated 13F/insider holdings headlines."""
    m = _HOLDINGS_VERB_RE.search(title)
    if m is None:
        return False
    if _TEMPLATE_TAG_RE.search(title) or (publisher and _MARKETBEAT_RE.search(publisher)):
        return True
    actor = (m.group("actor") or m.group("actor2") or m.group("actor3") or "").strip()
    return bool(_LEGAL_SUFFIX_RE.search(actor))


@dataclass
class Copy:
    """A syndicated near-copy collapsed into a representative item."""

    publisher: str | None
    timestamp: datetime | None
    title: str


@dataclass
class Item:
    """One kept piece of text with everything analytics needs about it."""

    id: str
    source: str
    source_label: str
    source_weight: float
    kind: SignalKind
    title: str
    body: str | None = None
    url: str | None = None
    author: str | None = None
    publisher: str | None = None
    timestamp: datetime | None = None
    engagement: int = 0
    user_label: SentimentLabel | None = None
    ticker_specific: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    relevance: float = 1.0
    trust: float = 1.0
    press_release: bool = False
    copies: list[Copy] = field(default_factory=list)
    scored: bool = False
    score: float = 0.0
    label: SentimentLabel = "neutral"
    confidence: float = 0.0
    drivers: list[tuple[str, float]] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    events: list[DetectedEvent] = field(default_factory=list)
    weight: float = 0.0
    narrative_id: str | None = None

    @property
    def group(self) -> Group:
        """Aggregation bucket: published media vs. crowd chatter."""
        return "social" if self.kind == "social" else "news"

    @property
    def duplicates(self) -> int:
        return len(self.copies)

    @property
    def coverage(self) -> int:
        """This item plus its syndicated copies."""
        return 1 + len(self.copies)

    @property
    def event_keys(self) -> list[str]:
        return list(dict.fromkeys(e.key for e in self.events))

    def outlets(self) -> list[str]:
        """Distinct outlets carrying this item (itself first)."""
        names = [self.publisher] + [c.publisher for c in self.copies]
        return list(dict.fromkeys(n for n in names if n))

    def times(self) -> list[datetime]:
        stamps = [self.timestamp] + [c.timestamp for c in self.copies]
        return [t for t in stamps if t is not None]

    def titles(self) -> list[str]:
        return [self.title] + [c.title for c in self.copies]

    def age_hours(self, now: datetime) -> float | None:
        if self.timestamp is None:
            return None
        return max(0.0, (now - self.timestamp).total_seconds() / 3600.0)

    def to_signal(self) -> Signal:
        return Signal(
            id=self.id, source=self.source, source_label=self.source_label, kind=self.kind,
            title=self.title, body=self.body, url=self.url, author=self.author, publisher=self.publisher,
            timestamp=self.timestamp, engagement=self.engagement,
            score=round(self.score, 3), label=self.label, confidence=round(self.confidence, 3),
            relevance=round(self.relevance, 3), weight=round(self.weight, 4),
            themes=list(self.themes), events=self.event_keys,
            drivers=[Driver(term=t, impact=round(v, 3)) for t, v in self.drivers[:MAX_DRIVERS]],
            user_label=self.user_label, narrative_id=self.narrative_id, duplicates=self.duplicates,
        )


@dataclass
class Prepared:
    items: list[Item]  # kept representatives, heaviest first
    fetched: Counter[str] = field(default_factory=Counter)  # raw items per source
    kept: Counter[str] = field(default_factory=Counter)  # representatives per source
    dropped: Counter[str] = field(default_factory=Counter)  # reason -> count
    engine_error: str | None = None  # sentiment scoring failed: items are unscored

    @property
    def scored(self) -> list[Item]:
        return [it for it in self.items if it.scored]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def prepare(company: CompanyRef | None, runs: list[SourceRun], now: datetime) -> Prepared:
    """Turn every source's raw items into kept, scored, weighted `Item`s.

    `company=None` skips the relevance filter (market-wide headlines)."""
    out = Prepared(items=[])
    candidates: list[Item] = []
    for run in runs:
        if run.batch is None:
            continue
        for raw in run.batch.signals:
            out.fetched[run.source.key] += 1
            made = _candidate(raw, run, company, now)
            if isinstance(made, str):
                out.dropped[made] += 1
            else:
                candidates.append(made)

    items = _collapse_duplicates(candidates)
    out.dropped["duplicate"] += len(candidates) - len(items)
    out.engine_error = _score(items, company)
    for it in items:
        it.weight = item_weight(it, now)
    _diversify(items)
    items.sort(key=lambda it: (-it.weight, it.id))
    _unique_ids(items)
    out.items = items
    out.kept.update(it.source for it in items)
    return out


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def _candidate(raw: RawSignal, run: SourceRun, company: CompanyRef | None, now: datetime) -> Item | str:
    """A cleaned, relevance-scored item — or the reason it was dropped."""
    src = run.source
    extra = raw.extra or {}
    kind: SignalKind = "analysis" if src.kind == "news" and extra.get("type") == "analysis" else src.kind
    publisher = textkit.publisher_name(raw.publisher or extra.get("domain"))
    title = textkit.clean(raw.title)
    if kind != "social":
        title = textkit.strip_suffix(title, raw.publisher)
    if not title or not textkit.meaningful(title):
        return "empty"
    if is_holdings_boilerplate(title, publisher or raw.publisher or extra.get("domain")):
        return "boilerplate"

    ts = raw.timestamp
    if ts is not None and ts.tzinfo is None:
        ts = None  # naive timestamps are ambiguous; never guess a timezone
    if ts is not None and ts > now + FUTURE_TOLERANCE:
        ts = None
    if ts is not None and now - ts > MAX_AGE:
        return "stale"

    body = textkit.clean(raw.body)[:MAX_BODY] or None
    rel = 1.0 if company is None else _relevance(title, body or extra.get("story"), extra, raw, company)
    if rel < MIN_RELEVANCE:
        return "irrelevant"

    group: Group = "social" if kind == "social" else "news"
    trust, is_pr = SOCIAL_TRUST, False
    if group == "news":
        trust = textkit.publisher_trust(publisher or extra.get("domain"))
        is_pr = textkit.press_release(publisher, title)
        if is_pr:
            trust = min(trust, PRESS_RELEASE_TRUST)
    return Item(
        id=stable_id(src.key, raw.url or title),
        source=src.key, source_label=src.label, source_weight=float(src.weight), kind=kind,
        title=title, body=body if body != title else None, url=raw.url, author=raw.author,
        publisher=publisher, timestamp=ts, engagement=max(0, int(raw.engagement or 0)),
        user_label=raw.user_label, ticker_specific=bool(raw.ticker_specific), extra=extra,
        relevance=round(rel, 3), trust=trust, press_release=is_pr,
    )


def _relevance(title: str, context: str | None, extra: dict[str, Any], raw: RawSignal,
               company: CompanyRef) -> float:
    """Title relevance, lifted by body context, provider relevance and ticker-keyed feeds,
    then capped for multi-ticker roundups (applied last, so a roundup from a ticker-keyed
    feed that never names the company in its title stays a roundup).

    A title that names only a separately listed sister company ("Vodafone Idea …" on a
    VOD.L feed) gets no provider/feed floor: the feed matched the brand, not the company.
    A headline naming the company only as a bystander of another company's news keeps its
    title relevance whatever lifts it — snippet, provider score or feed tag: LULU's 'Nike
    Sinks 8% …; Lululemon and On Holding Remain Flat' (0.40) came back above MIN_RELEVANCE
    (0.64) through its snippet."""
    title_rel = textkit.relevance(title, company)
    rel = _lifted(title, title_rel, context, extra, raw, company)
    if rel > title_rel > 0 and textkit.bystander(title, company):
        return clamp(title_rel)  # another company's story: neither its snippet nor a feed tag makes it this one's
    return rel


def _lifted(title: str, title_rel: float, context: str | None, extra: dict[str, Any], raw: RawSignal,
            company: CompanyRef) -> float:
    """`title_rel` lifted by body context, provider relevance and ticker-keyed feeds, then roundup-capped."""
    rel = title_rel
    if context:
        rel = max(rel, CONTEXT_DISCOUNT * textkit.relevance(f"{title}. {context[:400]}", company))
    floors = title_rel >= MIN_RELEVANCE or textkit.sister_company(title, company) is None
    provider = extra.get("provider_relevance")
    if (floors and isinstance(provider, (int, float)) and math.isfinite(provider)
            and provider >= PROVIDER_RELEVANCE_MIN):
        rel = max(rel, min(float(provider), 0.9))
    if raw.ticker_specific and floors:
        rel = max(rel, SPECIFIC_RELEVANCE)
    symbols = extra.get("symbols")
    if isinstance(symbols, int) and symbols >= ROUNDUP_SYMBOLS:
        rel = rel * 0.85 if title_rel >= 0.8 else min(rel, ROUNDUP_CAP)
    return clamp(rel)


def _collapse_duplicates(items: list[Item]) -> list[Item]:
    """Merge near-copies within each group (news with news, posts with posts).

    A social post sharing a headline is a separate (social) voice, never an extra
    "article" of the story; the most trusted (then earliest) item represents a group."""
    news = [it for it in items if it.group == "news"]
    social = [it for it in items if it.group == "social"]
    return _collapse_group(news) + _collapse_group(social)


def _collapse_group(items: list[Item]) -> list[Item]:
    if len(items) < 2:
        return list(items)

    def priority(i: int) -> tuple[Any, ...]:
        it = items[i]
        ts = it.timestamp.timestamp() if it.timestamp else float("inf")
        return (-it.trust * it.source_weight, -it.relevance, ts, i)

    order = sorted(range(len(items)), key=priority)
    groups = textkit.duplicates([items[i].title for i in order])
    reps: list[Item] = []
    seen: set[int] = set()
    for group in groups:
        members = [order[g] for g in group if 0 <= g < len(order) and order[g] not in seen]
        if not members:
            continue
        seen.update(members)
        rep = items[members[0]]
        for m in members[1:]:
            dup = items[m]
            rep.copies.append(Copy(dup.publisher, dup.timestamp, dup.title))
            rep.copies.extend(dup.copies)
            rep.ticker_specific |= dup.ticker_specific
            rep.relevance = max(rep.relevance, dup.relevance)
        reps.append(rep)
    reps.extend(items[i] for i in order if i not in seen)  # defensive: a partition missing indices
    return reps


def _score(items: list[Item], company: CompanyRef | None = None) -> str | None:
    """Run the engine over representatives; returns an error description on failure."""
    if not items:
        return None
    texts = [it.title for it in items]
    kinds = ["social" if it.group == "social" else "news" for it in items]
    try:
        results = textkit.analyze(texts, kinds, company)
        if len(results) != len(items):
            raise RuntimeError(f"engine returned {len(results)} results for {len(items)} texts")
    except Exception as exc:  # noqa: BLE001 - reported as a data-quality problem, never fabricated
        return f"{type(exc).__name__}: {exc}"[:200]
    for it, res in zip(items, results, strict=True):
        it.scored = True
        it.score = clamp(finite(res.score), -1.0, 1.0)
        it.label = res.label if res.label in ("bullish", "bearish", "neutral") else "neutral"
        it.confidence = clamp(finite(res.confidence))
        it.drivers = [(str(t), finite(v)) for t, v in (res.drivers or [])][:MAX_DRIVERS]
        it.themes = list(dict.fromkeys(res.themes or []))
        it.events = list(res.events or [])
    return None


def recency(it: Item, now: datetime) -> float:
    """Exponential decay by age (half-life 72 h news / 36 h social), floored."""
    age = it.age_hours(now)
    if age is None:
        return UNDATED_RECENCY
    return max(RECENCY_FLOOR, 0.5 ** (age / HALF_LIFE_H[it.group]))


def price_recap(it: Item) -> bool:
    """Does the item merely restate the price move (see module docstring)?

    Its events are all price events and its strongest driver is the phrase that
    triggered one of them ('Stock Craters 43%'), so 'Meta launches Muse; shares
    jump' (a product launch) or 'SoFi falls 3% as yields pressure fintech'
    (driven by the yields phrase) are not recaps. A machine-written daily recap
    (`AUTO_RECAP_RE`) is one whatever its events, unless it carries a development."""
    if AUTO_RECAP_RE.search(it.title):
        return set(it.event_keys) <= PRICE_EVENTS
    if not it.events or not it.drivers or not set(it.event_keys) <= PRICE_EVENTS:
        return False
    top = set(_TOKEN_RE.findall(max(it.drivers, key=lambda d: abs(d[1]))[0].lower()))
    if not top:
        return False
    for event in it.events:
        span = set(_TOKEN_RE.findall((event.span or "").lower()))
        if span and len(top & span) / len(top) >= 0.5:
            return True
    return False


def item_weight(it: Item, now: datetime) -> float:
    """Aggregation weight (see module docstring)."""
    engagement = 1.0 + min(1.0, math.log1p(it.engagement) / 8.0)
    confidence = 0.5 + 0.5 * (it.confidence if it.scored else 0.0)
    syndication = 1.0 + 0.15 * math.log1p(it.duplicates)
    w = it.source_weight * it.trust * recency(it, now) * engagement * it.relevance * confidence * syndication
    if it.scored and price_recap(it):
        w *= PRICE_RECAP_WEIGHT
    return round(max(w, 0.0), 6)


def voice(it: Item) -> str:
    """Who is speaking: the outlet for media, the account for social posts."""
    if it.group == "social":
        return f"author:{it.source}:{it.author}" if it.author else f"item:{it.id}"
    return f"outlet:{it.publisher}" if it.publisher else f"item:{it.id}"


def _diversify(items: list[Item]) -> None:
    counts = Counter(voice(it) for it in items)
    for it in items:
        n = counts[voice(it)]
        if n > DIVERSITY_FREE:
            it.weight = round(it.weight * math.sqrt(DIVERSITY_FREE / n), 6)


def _unique_ids(items: list[Item]) -> None:
    """Two raw items can share source+url with different titles: keep ids unique."""
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            it.id = stable_id(it.id, it.title)
        seen.add(it.id)
