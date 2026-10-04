"""Syndication detection and story-level clustering of headlines.

* `find_duplicates(titles)` groups near-identical headlines (wire copies,
  aggregator re-posts, truncations, "... By Investing.com" variants). It
  returns a partition of all indices: every index appears in exactly one
  group, the representative (lowest index — pass titles in priority order)
  first.

* `cluster_narratives(items, company)` groups headlines about the same
  *development* ("Nvidia adds $150B to buyback" / "Nvidia's record repurchase
  shows the stock is too cheap for Huang to resist"). Features are TF-IDF
  weighted content terms with the company's own name removed:
  stemmed words, concept tokens that collapse paraphrases ("share
  repurchase" == "buyback", "all-time high" == "record high"), adjacent-word
  bigrams, normalized money amounts ("$150 billion" == "$150B") and detected
  event keys / analyst firms. Clusters come from exact average-link
  agglomeration (mean pairwise cosine via cluster sum-vectors) with a tuned
  threshold. Pure Python; ~50 ms for 500 headlines.
"""
from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.nlp.events import detect_events
from app.nlp.publishers import publisher_trust
from app.nlp.relevance import company_terms
from app.nlp.text import STOPWORDS, fold, normalize_for_dedup, stem, tokenize, wordset
from app.nlp.types import Cluster, ClusterItem
from app.sources.base import CompanyRef

# Average-link threshold on mean pairwise cosine (tuned on labeled real
# headline sets in tests/fixtures/nlp, see test_narratives.py).
CLUSTER_THRESHOLD = 0.2

# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
DUP_JACCARD = 0.8
DUP_CONTAINMENT = 0.9


def _dup_tokens(title: str) -> list[str]:
    return normalize_for_dedup(title).split()


def find_duplicates(titles: list[str]) -> list[list[int]]:
    """Partition `titles` into groups of syndicated near-copies.

    Two titles are duplicates when their normalized forms are equal, their
    token Jaccard similarity is >= 0.8, or (for titles of 6+ tokens) one is
    a truncation/extension of the other (>= 90% of the shorter's tokens in the
    longer, in order). Groups are connected components; representative =
    lowest index."""
    n = len(titles)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    seqs = [_dup_tokens(t or "") for t in titles]
    sets = [frozenset(s) for s in seqs]
    by_norm: dict[str, int] = {}
    for i, seq in enumerate(seqs):
        key = " ".join(seq)
        if not key:
            continue
        if key in by_norm:
            union(by_norm[key], i)
        else:
            by_norm[key] = i

    # Candidate pairs share a token that is rare in this batch.
    df = Counter(tok for s in sets for tok in s)
    rare_cap = max(3, n // 5)
    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for tok in s:
            if df[tok] <= rare_cap:
                index[tok].append(i)
    seen: set[tuple[int, int]] = set()
    for members in index.values():
        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1:]:
                if (a, b) in seen or find(a) == find(b):
                    continue
                seen.add((a, b))
                if _near_identical(seqs[a], seqs[b], sets[a], sets[b]):
                    union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [sorted(g) for _, g in sorted(groups.items())]


def _near_identical(sa: list[str], sb: list[str], a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    inter = len(a & b)
    if inter / len(a | b) >= DUP_JACCARD:
        return True
    short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    if len(short) < 6:
        return False
    if inter / min(len(a), len(b)) < DUP_CONTAINMENT:
        return False
    # Order check: the shorter must read as a prefix/sub-run of the longer
    # (truncated "…Names It T" copies, appended attributions).
    pos = 0
    matched = 0
    for tok in short:
        try:
            pos = long_.index(tok, pos) + 1
            matched += 1
        except ValueError:
            continue
    return matched / len(short) >= DUP_CONTAINMENT


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
# Lead labels and roundup prefixes that say nothing about the story.
_LEAD_RE = re.compile(
    r"^(?:update\s*\d*|exclusive|breaking(?: news)?|watch|video|opinion|analysis|prediction|live|premarket|"
    r"market wrap|earnings call transcript|weekly recap|top (?:midday|morning) stories|stock market today[^:]*|"
    r"morning (?:brief|squawk)|[A-Z]{2,6})\s*[:|-]\s*",
    re.IGNORECASE,
)
_PAREN_TICKER_RE = re.compile(r"\((?:[A-Z]+:\s?)?[A-Z.]{1,8}(?::[A-Z]+)?\)|\$[A-Z]{1,6}\b")
_ATTRIB_RE = re.compile(r"\s+by investing\.com.*$|\s+-\s+[\w.' ]{2,30}$", re.IGNORECASE)

# Multi-word paraphrases collapsed into one concept token before stemming.
_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = tuple((re.compile(p, re.IGNORECASE), c) for p, c in (
    (r"\b(?:share|stock)[- ]repurchase(?:s| program| plan| authori[sz]ation)?\b|\brepurchas\w*\b|"
     r"\bbuy[- ]?backs?\b|\bbuy(?:ing)? back\b|\bbought back\b", " buyback "),
    (r"\b(?:all[- ]time|record)[- ]highs?\b|\bfirst record\b|\brecord territory\b", " record-high "),
    (r"\bprice[- ]targets?\b|\btarget[- ]prices?\b|\bPTs?\b", " price-target "),
    (r"\bmarket (?:cap(?:italization)?|value|valuation)\b", " market-cap "),
    (r"\btop (?:\w+ )?picks?\b", " top-pick "),
    (r"\bchief executive(?: officer)?\b", " ceo "),
    (r"\bartificial intelligence\b|\ba\.i\.", " ai "),
    (r"\b(?:law)?suits?\b|\bsue[sd]?\b|\bsuing\b|\blitigation\b|\bclass[- ]action\b", " lawsuit "),
    (r"\blay(?:s|ing)?[- ]?offs?\b|\blaid off\b|\bjob cuts?\b|\bcut(?:s|ting)? (?:\w+ )?jobs\b", " layoff "),
    (r"\bdata cent(?:er|re)s?\b", " data-center "),
    (r"\bsmart[- ]home\b", " smart-home "),
    (r"\bshort[- ]sell(?:er|ers|ing)?\b", " short-seller "),
    (r"\b52[- ]week lows?\b", " 52w-low "),
    (r"\bstock split\b", " stock-split "),
    (r"\b(?:earnings|eps|revenue|profit)s? (?:beat|tops?|topped)\b", " earnings-beat "),
    (r"\b(?:earnings|eps|revenue|profit)s? miss(?:es|ed)?\b", " earnings-miss "),
))

# Words that carry no story identity in financial headlines.
_GENERIC = wordset("""
stock stocks share shares shareholder shareholders investor investors market markets today why here heres what whats
says said say report reports reported update news analyst analysts company companies inc corp corporation co ltd plc
nasdaq nyse wall street year years week weeks month months day days time new big could would should may might will just
now still next first last best better buy buying sell selling hold amid ahead after before over know need thing things
way ways look looks looking watch watching see sees seen get gets got make makes made take takes go goes going come
comes trading trade traders price prices value worth move moves moving lot lots key keys right left long short big
huge massive major latest recent ever every much many more most less least one two three four five six seven eight
nine ten nearly almost about around above below likely set sets want wants deal deals plan plans plus via also into
amid against investing invest invested own owns owning point points case question questions answer answers
what's here's there's it's i'm don't can't won't isn't doesn't didn't let's you're they're we're
""")
# Price-move words: a move is not a story ("stock rises" links everything).
_MOVES = wordset("""
rise rises rising rose risen fall falls falling fell drop drops dropped dropping slide slides sliding slid slip slips
slipped jump jumps jumped jumping climb climbs climbed climbing gain gains gained gaining surge surges surged surging
soar soars soared soaring plunge plunges plunged plunging tumble tumbles tumbled sink sinks sank rally rallies rallied
rallying pop pops popped edge edges edged higher lower up down percent pct rebound rebounds rebounded retreat retreats
retreated sell-off selloff
""")
_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")
_PCT_RE = re.compile(r"^\d[\d.,]*%$")
_NUM_RE = re.compile(r"^\d[\d.,]*$")

_W_WORD = 1.0
_W_BIGRAM = 0.7
_W_MONEY = 1.6
_W_NUMBER = 0.6
_W_PCT = 0.35
_W_YEAR = 0.2
_W_MOVE = 0.15
_W_EVENT = 0.9
_W_FIRM = 1.2


def _focus(title: str, own: frozenset[str]) -> str:
    """Strip lead labels/tickers; for multi-story roundups keep the clauses
    that mention the company."""
    text = fold(title)
    for _ in range(2):
        text = _LEAD_RE.sub("", text, count=1)
    text = _ATTRIB_RE.sub("", text)
    text = _PAREN_TICKER_RE.sub(" ", text)
    if own and re.search(r";|\s\|\s", text):
        clauses = [c for c in re.split(r";|\s\|\s", text) if c.strip()]
        mine = [c for c in clauses if set(tokenize(c)) & own]
        if mine:
            text = " ; ".join(mine)
    return text


def _surface_case(word: str) -> str:
    return word


@dataclass
class _Doc:
    vec: dict[str, float]
    surfaces: dict[str, str]  # feature -> a surface form seen in this title


def _raw_features(title: str, own: frozenset[str]) -> tuple[dict[str, float], dict[str, str]]:
    text = _focus(title, own)
    for pattern, concept in _PHRASES:
        text = pattern.sub(concept, text)
    raw_tokens = re.findall(r"[\w$%.,&'-]+", text)
    feats: dict[str, float] = {}
    surfaces: dict[str, str] = {}
    content: list[tuple[str, str]] = []  # (feature, surface) for bigrams

    def add(feature: str, weight: float, surface: str) -> None:
        if weight > feats.get(feature, 0.0):
            feats[feature] = weight
        surfaces.setdefault(feature, surface)

    for raw in raw_tokens:
        surface = raw.strip(".,'-")
        for tok in tokenize(surface):
            if tok in own or tok.lstrip("$") in own:
                content.append(("", ""))  # break bigram adjacency at the company name
                continue
            if tok.startswith("$") and tok[1:2].isdigit():
                add(tok, _W_MONEY, surface)
                content.append((tok, surface))
            elif _PCT_RE.match(tok):
                add(tok, _W_PCT, surface)
            elif _NUM_RE.match(tok):
                if _YEAR_RE.match(tok):
                    add(tok, _W_YEAR, surface)
                elif len(tok.replace(".", "").replace(",", "")) >= 2 or "." in tok:
                    add(tok, _W_NUMBER, surface)
            elif tok in STOPWORDS or len(tok) < 2:
                continue
            elif tok in _MOVES:
                add(stem(tok), _W_MOVE, surface)
            elif tok in _GENERIC:
                content.append(("", ""))
            else:
                feature = tok if "-" in tok else stem(tok)
                add(feature, _W_WORD, surface)
                content.append((feature, surface))
    for (f1, s1), (f2, s2) in zip(content, content[1:]):
        if f1 and f2 and f1 != f2:
            add(f"{f1} {f2}", _W_BIGRAM, f"{s1} {s2}")
    for event in detect_events(title):
        if event.key in {"price_up", "price_down"}:
            continue
        add(f"ev:{event.key}", _W_EVENT, event.key)
        if event.firm:
            add(f"firm:{event.firm.lower()}", _W_FIRM, event.firm)
    return feats, surfaces


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(f, 0.0) for f, w in a.items())


def _vectorize(titles: list[str], company: CompanyRef | None) -> tuple[list[_Doc], dict[str, float]]:
    own = company_terms(company)
    raw = [_raw_features(t or "", own) for t in titles]
    n = len(titles)
    df = Counter(f for feats, _ in raw for f in feats)
    idf = {f: math.log((n + 1) / (c + 0.5)) for f, c in df.items()}
    docs: list[_Doc] = []
    for feats, surfaces in raw:
        vec = {f: w * idf[f] for f, w in feats.items() if idf[f] > 0}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        vec = {f: v / norm for f, v in vec.items()} if norm else {}
        docs.append(_Doc(vec=vec, surfaces=surfaces))
    return docs, idf


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def _average_link(docs: list[_Doc], threshold: float) -> list[list[int]]:
    """Exact average-link agglomeration: sim(A, B) = (ΣA·ΣB) / (|A||B|) for
    unit vectors, merged greedily from the most similar pair down to
    `threshold`. Only clusters that share a feature are ever compared."""
    sums: dict[int, dict[str, float]] = {i: dict(d.vec) for i, d in enumerate(docs)}
    members: dict[int, list[int]] = {i: [i] for i in range(len(docs))}
    version = dict.fromkeys(sums, 0)
    inv: dict[str, set[int]] = defaultdict(set)
    for i, vec in sums.items():
        for f in vec:
            inv[f].add(i)

    heap: list[tuple[float, int, int, int, int]] = []

    def push(a: int, b: int) -> None:
        if a == b:
            return
        a, b = min(a, b), max(a, b)
        sim = _dot(sums[a], sums[b]) / (len(members[a]) * len(members[b]))
        if sim >= threshold:
            heapq.heappush(heap, (-sim, a, b, version[a], version[b]))

    for i in sums:
        neighbours = {j for f in sums[i] for j in inv[f] if j > i}
        for j in neighbours:
            push(i, j)

    while heap:
        _neg, a, b, va, vb = heapq.heappop(heap)
        if a not in sums or b not in sums or version[a] != va or version[b] != vb:
            continue
        # merge b into a
        for f, w in sums.pop(b).items():
            sums[a][f] = sums[a].get(f, 0.0) + w
            inv[f].discard(b)
            inv[f].add(a)
        members[a].extend(members.pop(b))
        version[a] += 1
        version.pop(b)
        neighbours = {j for f in sums[a] for j in inv[f] if j != a}
        for j in neighbours:
            push(a, j)
    return [sorted(m) for m in members.values()]


def _is_weak_headline(title: str) -> bool:
    t = title.strip()
    return t.endswith("?") or bool(re.match(r"^(?:why|how|what|is|should|can|will|here's|\d+ )", t, re.IGNORECASE))


def cluster_narratives(items: list[ClusterItem], company: CompanyRef | None = None,
                       max_clusters: int = 12, threshold: float = CLUSTER_THRESHOLD) -> list[Cluster]:
    """Group `items` into stories, biggest (by item weight) first.

    Singletons are returned too (after multi-item clusters with the same
    weight), so callers can rank everything by their own impact metric;
    at most `max_clusters` clusters are returned. Each cluster's
    representative is its most central, most trusted, non-question headline;
    `terms` are its top distinguishing features in readable form."""
    if not items:
        return []
    titles = [it.title or "" for it in items]
    docs, idf = _vectorize(titles, company)
    groups = _average_link(docs, threshold)

    def group_key(g: list[int]) -> tuple[float, int, float]:
        weight = sum(max(items[i].weight, 0.0) for i in g)
        latest = max((items[i].timestamp.timestamp() for i in g if items[i].timestamp), default=0.0)
        return (-(weight + 0.25 * (len(g) - 1)), -len(g), -latest)

    groups.sort(key=lambda g: (*group_key(g), g[0]))
    out: list[Cluster] = []
    for g in groups[:max_clusters]:
        rep = _representative(g, items, docs)
        out.append(Cluster(
            item_ids=[items[rep].id] + [items[i].id for i in g if i != rep],
            representative_id=items[rep].id,
            title=items[rep].title,
            terms=_top_terms(g, docs, idf),
        ))
    return out


def _representative(group: list[int], items: list[ClusterItem], docs: list[_Doc]) -> int:
    if len(group) == 1:
        return group[0]
    centroid: dict[str, float] = defaultdict(float)
    for i in group:
        for f, w in docs[i].vec.items():
            centroid[f] += w
    max_w = max((items[i].weight for i in group), default=0.0) or 1.0

    def score(i: int) -> float:
        central = _dot(docs[i].vec, centroid) / len(group)
        trust = publisher_trust(items[i].publisher) if items[i].publisher else 0.9
        weight = max(items[i].weight, 0.0) / max_w
        words = len(items[i].title.split())
        shape = 0.8 if _is_weak_headline(items[i].title) else 1.0
        shape *= 0.85 if words < 5 or words > 22 else 1.0
        return central * (0.6 + 0.25 * trust + 0.15 * weight) * shape

    return max(group, key=lambda i: (score(i), -i))


def _top_terms(group: list[int], docs: list[_Doc], idf: dict[str, float], limit: int = 5) -> list[str]:
    counts: Counter[str] = Counter()
    surface_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for i in group:
        for f in docs[i].vec:
            counts[f] += 1
            surface = docs[i].surfaces.get(f)
            if surface:
                surface_votes[f][surface] += 1
    min_count = 2 if len(group) >= 3 else 1
    scored = []
    for f, c in counts.items():
        if c < min_count or f.startswith(("ev:", "firm:")) or _YEAR_RE.match(f) or _PCT_RE.match(f):
            continue
        weight = c / len(group) * idf.get(f, 0.0) * (1.15 if " " in f or "-" in f else 1.0)
        scored.append((weight, f))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen: list[str] = []
    covered: set[str] = set()
    for _w, f in scored:
        parts = set(f.split())
        if parts & covered and " " not in f:
            continue
        if " " in f and parts <= covered:
            continue
        chosen.append(f)
        covered |= parts
        if len(chosen) >= limit:
            break
    return [_display(f, surface_votes.get(f)) for f in chosen]


def _display(feature: str, votes: Counter[str] | None) -> str:
    if feature.startswith("$"):
        return feature[0] + feature[1:].upper() if feature[-1].isalpha() else feature
    if not votes:
        return feature.replace("-", " ")
    surface = votes.most_common(1)[0][0]
    if surface.isupper() and len(surface) <= 5:
        return surface  # AI, CEO, GPU
    lower_seen = any(s and s[0].islower() for s in votes)
    shown = surface.lower() if lower_seen or not surface[:1].isupper() else surface
    return shown.replace("-", " ") if feature in {"record-high", "price-target", "market-cap", "top-pick",
                                                   "smart-home", "data-center", "short-seller", "stock-split",
                                                   "earnings-beat", "earnings-miss"} else shown
