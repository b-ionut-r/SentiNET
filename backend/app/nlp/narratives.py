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
from itertools import pairwise

from app.nlp.events import detect_events
from app.nlp.publishers import publisher_trust
from app.nlp.relevance import company_terms
from app.nlp.text import (
    CALENDAR_WORDS,
    GENERIC_WORDS,
    HEADLINE_VERBS,
    MOVE_WORDS,
    STOPWORDS,
    fold,
    is_mostly_upper,
    is_title_case,
    normalize_for_dedup,
    stem,
    tokenize,
    wordset,
)
from app.nlp.types import Cluster, ClusterItem
from app.sources.base import CompanyRef

# Average-link threshold on mean pairwise cosine (tuned on labeled real
# headline sets in tests/fixtures/nlp, see test_narratives.py).
CLUSTER_THRESHOLD = 0.08
LINKAGE = "average"
LINKAGE_CENTROID = 0
SINGLETON_DAMPING = 0.3
MIN_EVIDENCE = 2.0
HUB_MIN_DF = 4
HUB_REFERENCE = 0.1
HUB_FLOOR = 0.2
TIME_GRACE_H = 24.0
TIME_FADE_H = 144.0
# Clusters sharing no anchor (entity, amount, bigram, concept, firm) need this
# much average-link similarity to merge: plain words link only near-paraphrases.
NONANCHOR_SIM = 0.4
CANDIDATE_MAX_DF = 0.2
# Refinement (see _refine): cores, defining features, attach/merge cut-offs.
REFINE = 1
CORE_MIN = 3
DEFINING_FRAC = 0.5
ATTACH_SIM = 0.2
CORE_MERGE_SIM = 0.35

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
    ((r"\b(?:share|stock)[- ]repurchase(?:s| program| plan| authori[sz]ation)?\b|\brepurchas\w*\b|"
      r"\bbuy[- ]?backs?\b|\bbuy(?:ing)? back\b|\bbought back\b"), " buyback "),
    ((r"\b(?:all[- ]time|record)[- ]highs?\b|\bfirst record\b|\brecord (?:territory|close|closing high)\b|"
      r"\b(?:back at|at|hits?|to|reach(?:es|ed)?|sets?|notch(?:es|ed)?|new|fresh)\s+(?:a\s+|its\s+|another\s+)?"
      r"(?:new\s+|fresh\s+|(?-i:[A-Z])\w+\s+)?record\b(?!\s+(?:revenue|sales|profit|earnings|quarter|deliveries|"
      r"buyback|repurchase|low|loss|\$|\d))"), " record-high "),
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
    ((r"\b(?:cut|cuts|cutting|slash(?:es|ed|ing)?|lower(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|trim(?:s|med|ming)?)"
      r"(?: \w+)? prices?\b|\bprice (?:cuts?|reductions?|war)\b|\bvalue war\b"), " price-cut "),
    ((r"\b(?:raise[sd]?|raising|hike[sd]?|hiking|increase[sd]?|increasing)(?: \w+)? prices?\b|"
      r"\bprice (?:hikes?|increases?)\b"), " price-hike "),
    ((r"\b(?:stock|shares|stake|position|holdings?)\s+(?:in\s+\S+\s+)?(?:acquired|bought|sold|purchased|cut|raised|"
      r"lifted|trimmed|boosted|reduced|increased|decreased|grown)\s+by\b"), " inst-holding "),
    (r"\b(?:earnings|eps|revenue|profit)s? (?:beat|tops?|topped)\b", " earnings-beat "),
    (r"\b(?:earnings|eps|revenue|profit)s? miss(?:es|ed)?\b", " earnings-miss "),
    # "third-quarter" == "Q3" == "3Q"; analysts' "expectations" == "estimates" == "consensus".
    (r"\b(?:first|1st)[- ]quarter\b|\b1q\b", " q1 "),
    (r"\b(?:second|2nd)[- ]quarter\b|\b2q\b", " q2 "),
    (r"\b(?:third|3rd)[- ]quarter\b|\b3q\b", " q3 "),
    (r"\b(?:fourth|4th)[- ]quarter\b|\b4q\b", " q4 "),
    ((r"\b(?:analysts'? |wall street(?:'s)? |street )?(?:expectations|estimates?|forecasts|consensus(?: estimates?)?|"
      r"projections)\b"), " estimates "),
))

_ACRONYMS = wordset("ai ceo cfo coo cto us usa uk eu ev evs ipo etf etfs q1 q2 q3 q4 eps gdp cpi fed sec ftc doj "
                    "fy pc gpu gpus api it hr ar vr")
# Events that recur across unrelated stories (many firms, many launches).
_GENERIC_EVENTS = frozenset({"pt_raise", "pt_cut", "analyst_upgrade", "analyst_downgrade", "analyst_initiate",
                             "product_launch", "partnership", "contract_win", "all_time_high"})
# Concept tokens that are too generic to anchor a story on their own.
_WEAK_CONCEPTS = frozenset({"price-target", "earnings-beat", "earnings-miss"})
_YEAR_RE = re.compile(r"^(?:19|20)\d\d$")
_PCT_RE = re.compile(r"^\d[\d.,]*%$")
_NUM_RE = re.compile(r"^\d[\d.,]*$")

_W_WORD = 1.0
_W_PROPER = 1.4
_W_VERB = 0.45
_W_BIGRAM = 0.5
_W_CONCEPT = 1.6
_W_MONEY = 1.6
_W_NUMBER = 0.6
_W_PCT = 0.5
_W_YEAR = 0.2
_W_MONTH = 0.3
_W_MOVE = 0.15
_W_EVENT = 0.9
_W_FIRM = 1.2
_W_TIMED = 0.8

# Events that make headlines about the same development on the same trading
# session one story ("Tesla stock jumps 5% on delivery beat" / "Why is Tesla
# stock surging today?"): event group x session day becomes a feature.
_TIMED_GROUPS = {
    "price_up": "move-up", "all_time_high": "move-up", "price_down": "move-down", "low_52w": "move-down",
    "analyst_upgrade": "analyst-bull", "pt_raise": "analyst-bull", "analyst_downgrade": "analyst-bear",
    "pt_cut": "analyst-bear", "earnings_beat": "results-beat", "record_results": "results-beat",
    "earnings_miss": "results-miss", "guidance_raise": "guide-up", "guidance_cut": "guide-down",
}
_SESSION_SHIFT_S = 6 * 3600  # US session day: ~02:00 ET cut-over
# Analyst stance in coverage that names no firm ("Scored a New 'Buy' Rating",
# "Buy Target Stock, Analyst Says"): same-session notes are one story.
_ANALYST_BULL_RE = re.compile(
    r"\b['\"]?(?:buy|outperform|overweight|strong[- ]buy)['\"]?\s+(?:rating|call|recommendation)\b|"
    r"\banalysts?\s+(?:says?|calls?|recommends?)\b[^.?!]{0,20}\bbuy\b|\bbuy\b[^.?!]{0,40}\banalysts?\s+says?\b|"
    r"\banalysts?\s+(?:prais\w+|cheer\w*|turns?\s+bullish|upbeat)\b",
    re.IGNORECASE,
)
_ANALYST_BEAR_RE = re.compile(
    r"\b['\"]?(?:sell|underperform|underweight)['\"]?\s+(?:rating|call|recommendation)\b|"
    r"\banalysts?\s+(?:says?|calls?)\b[^.?!]{0,20}\bsell\b|\bsell\b[^.?!]{0,40}\banalysts?\s+says?\b|"
    r"\banalysts?\s+(?:turns?\s+bearish|slams?|downbeat)\b",
    re.IGNORECASE,
)


def _session_day(ts: float) -> int:
    return int((ts - _SESSION_SHIFT_S) // 86400)


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


@dataclass
class _Doc:
    vec: dict[str, float]  # unit-length tf-idf vector
    raw: dict[str, float]  # un-normalized tf-idf weights (absolute evidence)
    anchors: frozenset[str]  # features specific enough to tie two headlines to one story
    surfaces: dict[str, str]  # feature -> a surface form seen in this title


_WORD_SURFACE_RE = re.compile(r"[A-Za-z][\w&'-]*")


def _learn_case(titles: list[str]) -> tuple[frozenset[str], frozenset[str]]:
    """(proper, common): words this batch writes capitalized / lower-case
    mid-sentence in sentence-case headlines. Proper nouns ("MongoDB",
    "Desai", "ByteDance", "Burry") are the strongest story identifiers;
    title-case headlines carry no signal."""
    proper: Counter[str] = Counter()
    common: Counter[str] = Counter()
    for title in titles:
        text = fold(title)
        if is_title_case(text) or is_mostly_upper(text):
            continue
        for sentence in re.split(r"[:;.!?]\s+|\s[-|]\s", text):
            for k, word in enumerate(_WORD_SURFACE_RE.findall(sentence)):
                if k == 0:
                    continue
                low = word.lower().removesuffix("'s")
                (proper if word[0].isupper() else common)[low] += 1
    names = frozenset(w for w, c in proper.items() if c > common[w] and w not in _ACRONYMS and w not in STOPWORDS)
    return names, frozenset(common)


def _is_entity_shape(surface: str) -> bool:
    """CamelCase or short all-caps brand/acronym ("OpenAI", "ByteDance", "BNP")."""
    if len(surface) < 2:
        return False
    if any(c.isupper() for c in surface[1:]) and any(c.islower() for c in surface):
        return True
    return surface.isupper() and 2 <= len(surface) <= 6 and surface.lower() not in _ACRONYMS


def _own_subject(title: str, span: str | None, own: frozenset[str]) -> bool:
    """True when the company is named right before an event span ("Tesla
    stock jumps 5%"), i.e. the move is the company's, not a rival's."""
    if not span or not own:
        return False
    text = fold(title)
    idx = text.find(span)
    if idx < 0:
        return False
    return bool(set(tokenize(text[max(0, idx - 40):idx])[-4:]) & own)


def _raw_features(title: str, own: frozenset[str], proper: frozenset[str]
                  ) -> tuple[dict[str, float], set[str], dict[str, str], set[str]]:
    text = _focus(title, own)
    for pattern, concept in _PHRASES:
        text = pattern.sub(concept, text)
    shouting = is_mostly_upper(text)
    cased = {w.lower().removesuffix("'s"): w.removesuffix("'s") for w in _WORD_SURFACE_RE.findall(text)}
    feats: dict[str, float] = {}
    anchors: set[str] = set()
    surfaces: dict[str, str] = {}
    content: list[tuple[str, str, bool]] = []  # (feature, surface, generic); "" breaks bigrams

    def add(feature: str, weight: float, surface: str, anchor: bool = False) -> None:
        if weight > feats.get(feature, 0.0):
            feats[feature] = weight
        surfaces.setdefault(feature, surface)
        if anchor:
            anchors.add(feature)

    for tok in tokenize(text):
        surface = cased.get(tok, tok)
        if tok in own or tok.lstrip("$") in own:
            content.append(("", "", False))  # the company's own name is not a story feature
        elif tok.startswith("$") and tok[1:2].isdigit():
            add(tok, _W_MONEY, tok.upper(), anchor=True)
            content.append((tok, tok.upper(), False))
        elif _PCT_RE.match(tok):
            value = float(tok[:-1].replace(",", "") or 0)
            add(tok, _W_PCT, tok, anchor=value >= 10)
        elif _NUM_RE.match(tok):
            if _YEAR_RE.match(tok):
                add(tok, _W_YEAR, tok)
            elif len(tok.replace(".", "").replace(",", "")) >= 2 or "." in tok:
                add(tok, _W_NUMBER, tok, anchor=True)
        elif tok in STOPWORDS or len(tok) < 2:
            continue
        elif tok in MOVE_WORDS:
            add(stem(tok), _W_MOVE, surface)
            content.append(("", "", False))
        elif tok in GENERIC_WORDS or tok.startswith("$"):
            content.append((tok, surface, True))
        elif tok in CALENDAR_WORDS:
            add(tok, _W_MONTH, surface)
            content.append(("", "", False))
        elif "-" in tok and tok in _CONCEPTS:
            add(tok, _W_CONCEPT, surface, anchor=tok not in _WEAK_CONCEPTS)
            content.append((tok, surface, False))
        else:
            feature = stem(tok)
            if tok in proper or (not shouting and _is_entity_shape(surface)):
                add(feature, _W_PROPER, surface, anchor=True)
            elif tok in HEADLINE_VERBS:
                add(feature, _W_VERB, surface)
            else:
                add(feature, _W_WORD, surface)
            content.append((feature, surface, False))
    for (f1, s1, g1), (f2, s2, g2) in pairwise(content):
        if f1 and f2 and f1 != f2 and not (g1 and g2):
            add(f"{f1} {f2}", _W_BIGRAM, f"{s1} {s2}", anchor=True)
    timed: set[str] = set()
    if _ANALYST_BULL_RE.search(title):
        timed.add("analyst-bull")
    if _ANALYST_BEAR_RE.search(title):
        timed.add("analyst-bear")
    for event in detect_events(title):
        group = _TIMED_GROUPS.get(event.key)
        if group and (not group.startswith("move-") or _own_subject(title, event.span, own)):
            timed.add(group)
        if event.key in {"price_up", "price_down"}:
            continue
        add(f"ev:{event.key}", _W_EVENT, event.key, anchor=event.key not in _GENERIC_EVENTS)
        if event.firm:
            add(f"firm:{event.firm.lower()}", _W_FIRM, event.firm, anchor=True)
    return feats, anchors, surfaces, timed


_CONCEPTS = frozenset(c.strip() for _p, c in _PHRASES if "-" in c)


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(f, 0.0) for f, w in a.items())


def _unit(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {f: v / norm for f, v in vec.items()} if norm else {}


def _hub_damping(vectors: list[dict[str, float]], df: Counter[str]) -> dict[str, float]:
    """Down-weight 'hub' features: frequent terms whose documents have little
    else in common (a product name like "Muse" spanning five unrelated
    stories) as opposed to story terms whose documents also share their other
    terms ("buyback" + "$150B" + "record"). Coherence of f = mean pairwise
    cosine of the documents containing f, computed without f."""
    units = [_unit(v) for v in vectors]
    by_feature: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(units):
        for f in u:
            if df[f] >= HUB_MIN_DF:
                by_feature[f].append(i)
    damping: dict[str, float] = {}
    for f, docs in by_feature.items():
        k = len(docs)
        total: dict[str, float] = defaultdict(float)
        self_sq = 0.0
        for i in docs:
            for g, w in units[i].items():
                if g != f:
                    total[g] += w
            self_sq += 1.0 - units[i][f] ** 2
        coherence = (sum(v * v for v in total.values()) - self_sq) / (k * (k - 1))
        damping[f] = min(1.0, max(HUB_FLOOR, coherence / HUB_REFERENCE))
    return damping


def _vectorize(titles: list[str], company: CompanyRef | None, times: list[float | None] | None = None
               ) -> tuple[list[_Doc], dict[str, float], frozenset[str]]:
    own = company_terms(company)
    proper, common = _learn_case(titles)
    raw = [_raw_features(t or "", own, proper) for t in titles]
    for (feats, anchors, _s, timed), ts in zip(raw, times or [None] * len(titles), strict=True):
        if ts is None:
            continue
        for group in timed:
            key = f"t:{group}@{_session_day(ts)}"
            feats[key] = _W_TIMED
            anchors.add(key)
    n = len(titles)
    df = Counter(f for feats, _a, _s, _t in raw for f in feats)
    idf = {f: math.log((n + 1) / (c + 0.5)) for f, c in df.items()}
    # Features seen once cannot link anything; keep them faint so they don't
    # swamp the shared ones in the norm.
    weighted = [{f: w * idf[f] * (SINGLETON_DAMPING if df[f] == 1 else 1.0) for f, w in feats.items() if idf[f] > 0}
                for feats, _a, _s, _t in raw]
    hubs = _hub_damping(weighted, df)
    docs: list[_Doc] = []
    for vec, (_f, anchors, surfaces, _t) in zip(weighted, raw, strict=True):
        vec = {f: w * hubs.get(f, 1.0) for f, w in vec.items()}
        docs.append(_Doc(vec=_unit(vec), raw=vec, anchors=frozenset(a for a in anchors if a in vec),
                         surfaces=surfaces))
    return docs, idf, common


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def _time_factor(a: tuple[float, int], b: tuple[float, int]) -> float:
    """Stories are time-local: full credit within a day, fading to 0.5 for
    clusters whose mean timestamps are ~4+ days apart."""
    if not a[1] or not b[1]:
        return 1.0
    gap_h = abs(a[0] / a[1] - b[0] / b[1]) / 3600.0
    return max(0.5, 1.0 - max(0.0, gap_h - TIME_GRACE_H) / TIME_FADE_H)


def _average_link(docs: list[_Doc], threshold: float, times: list[float | None] | None = None) -> list[list[int]]:
    """Exact average-link agglomeration: sim(A, B) = (ΣA·ΣB) / (|A||B|) for
    unit vectors (× a time-proximity factor), merged greedily from the most
    similar pair down to `threshold`. Two clusters are only compared when
    they share an anchor feature (entity, amount, bigram, concept, firm) and
    enough absolute evidence — common words alone never link stories."""
    clock: dict[int, tuple[float, int]] = {
        i: ((t, 1) if t is not None else (0.0, 0)) for i, t in enumerate(times or [None] * len(docs))
    }
    sums: dict[int, dict[str, float]] = {i: dict(d.vec) for i, d in enumerate(docs)}
    raws: dict[int, dict[str, float]] = {i: dict(d.raw) for i, d in enumerate(docs)}
    anchors: dict[int, set[str]] = {i: set(d.anchors) for i, d in enumerate(docs)}
    members: dict[int, list[int]] = {i: [i] for i in range(len(docs))}
    version = dict.fromkeys(sums, 0)
    # Candidate pairs share an anchor or a reasonably specific term; terms in
    # a large share of the batch only cost time (their pairs score low).
    df = Counter(f for d in docs for f in d.vec)
    cap = max(3, int(CANDIDATE_MAX_DF * len(docs)))
    keys: dict[int, set[str]] = {i: {f for f in d.vec if f in d.anchors or df[f] <= cap} for i, d in enumerate(docs)}
    by_key: dict[str, set[int]] = defaultdict(set)
    for i, feats in keys.items():
        for f in feats:
            by_key[f].add(i)

    heap: list[tuple[float, int, int, int, int]] = []

    def norm(c: int) -> float:
        return math.sqrt(sum(v * v for v in sums[c].values())) or 1.0

    norms = {c: norm(c) for c in sums}

    def evidence(a: int, b: int) -> float:
        """Shared information in absolute tf-idf units (mean per member), so a
        single ubiquitous word ("Muse" in a quarter of the batch) can't
        glue two short headlines together while a rare shared entity can."""
        ra, rb = raws[a], raws[b]
        na, nb = len(members[a]), len(members[b])
        if len(ra) > len(rb):
            ra, rb, na, nb = rb, ra, nb, na
        return sum(min(w / na, rb[f] / nb) for f, w in ra.items() if f in rb)

    def push(a: int, b: int) -> None:
        a, b = min(a, b), max(a, b)
        dot = _dot(sums[a], sums[b])
        if LINKAGE == "centroid" or LINKAGE_CENTROID:
            sim = dot / (norms[a] * norms[b])
        else:
            sim = dot / (len(members[a]) * len(members[b]))
        sim *= _time_factor(clock[a], clock[b])
        if sim < threshold or (sim < NONANCHOR_SIM and not anchors[a] & anchors[b]):
            return
        if evidence(a, b) >= MIN_EVIDENCE:
            heapq.heappush(heap, (-sim, a, b, version[a], version[b]))

    for i, feats in keys.items():
        for j in {j for f in feats for j in by_key[f] if j > i}:
            push(i, j)

    while heap:
        _neg, a, b, va, vb = heapq.heappop(heap)
        if a not in sums or b not in sums or version[a] != va or version[b] != vb:
            continue
        for f, w in sums.pop(b).items():  # merge b into a
            sums[a][f] = sums[a].get(f, 0.0) + w
        for f, w in raws.pop(b).items():
            raws[a][f] = raws[a].get(f, 0.0) + w
        anchors[a] |= anchors.pop(b)
        for f in keys.pop(b):
            by_key[f].discard(b)
            by_key[f].add(a)
            keys[a].add(f)
        members[a].extend(members.pop(b))
        ta, tb = clock[a], clock.pop(b)
        clock[a] = (ta[0] + tb[0], ta[1] + tb[1])
        version[a] += 1
        version.pop(b)
        norms[a] = norm(a)
        for j in {j for f in keys[a] for j in by_key[f] if j != a}:
            push(a, j)
    return [sorted(m) for m in members.values()]


def _span(group: list[int], times: list[float | None]) -> tuple[float, float] | None:
    stamps = [t for i in group if (t := times[i]) is not None]
    return (min(stamps), max(stamps)) if stamps else None


def _span_factor(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    """Like _time_factor, but on the gap between two clusters' time ranges:
    a follow-up inside (or next to) a week-long story's coverage window
    is not penalized for being far from the story's mean timestamp."""
    if a is None or b is None:
        return 1.0
    gap_h = max(0.0, max(a[0], b[0]) - min(a[1], b[1])) / 3600.0
    return max(0.5, 1.0 - max(0.0, gap_h - TIME_GRACE_H) / TIME_FADE_H)


def _profile(group: list[int], docs: list[_Doc]) -> tuple[dict[str, float], set[str]]:
    """(unit centroid, defining features): features carried by at least
    DEFINING_FRAC of the members — what the story *is* ("buyback", "$150B")
    as opposed to terms a few members happen to mention ("AI")."""
    total: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for i in group:
        for f, w in docs[i].vec.items():
            total[f] += w
            counts[f] += 1
    need = max(2.0, DEFINING_FRAC * len(group)) if len(group) > 1 else 1.0
    return _unit(total), {f for f, c in counts.items() if c >= need}


def _refine(groups: list[list[int]], docs: list[_Doc], times: list[float | None]) -> list[list[int]]:
    """Second pass over the average-link result. Average linkage keeps
    stories tight but leaves satellites behind: a big story's own terms are
    frequent (low idf), and a short follow-up headline is never similar to
    *every* member. Here (1) cores (>= CORE_MIN members) whose centroids are
    close and share a defining feature merge, then (2) every smaller cluster
    joins the core it shares a defining feature with and is most similar to
    (centroid cosine >= ATTACH_SIM, time-faded)."""
    cores = sorted((g for g in groups if len(g) >= CORE_MIN), key=lambda g: (-len(g), g[0]))
    small = [g for g in groups if len(g) < CORE_MIN]
    merged = True
    while merged and len(cores) > 1:
        merged = False
        profiles = [_profile(g, docs) for g in cores]
        best: tuple[float, int, int] | None = None
        for a in range(len(cores)):
            for b in range(a + 1, len(cores)):
                if not profiles[a][1] & profiles[b][1]:
                    continue
                sim = _dot(profiles[a][0], profiles[b][0]) * _span_factor(_span(cores[a], times),
                                                                          _span(cores[b], times))
                if sim >= CORE_MERGE_SIM and (best is None or sim > best[0]):
                    best = (sim, a, b)
        if best:
            _sim, a, b = best
            cores[a] = sorted(cores[a] + cores.pop(b))
            merged = True
    if not cores:
        return groups
    profiles = [_profile(g, docs) for g in cores]
    out = [list(g) for g in cores]
    for g in small:
        centroid, feats = _profile(g, docs)
        own = set(centroid)
        best_k, best_sim = -1, ATTACH_SIM
        for k, (core_centroid, defining) in enumerate(profiles):
            if not own & defining:
                continue
            sim = _dot(centroid, core_centroid) * _span_factor(_span(g, times), _span(cores[k], times))
            if sim >= best_sim:
                best_k, best_sim = k, sim
        if best_k >= 0:
            out[best_k].extend(g)
        else:
            out.append(list(g))
    return [sorted(g) for g in out]


def _is_weak_headline(title: str) -> bool:
    t = title.strip()
    return t.endswith("?") or bool(re.match(r"^(?:why|how|what|is|should|can|will|here's|\d+ )", t, re.IGNORECASE))


def cluster_narratives(items: list[ClusterItem], company: CompanyRef | None = None,
                       max_clusters: int = 12, threshold: float | None = None) -> list[Cluster]:
    """Group `items` into stories, biggest (by item weight) first.

    Singletons are returned too (after multi-item clusters with the same
    weight), so callers can rank everything by their own impact metric;
    at most `max_clusters` clusters are returned. Each cluster's
    representative is its most central, most trusted, non-question headline;
    `terms` are its top distinguishing features in readable form."""
    if not items:
        return []
    titles = [it.title or "" for it in items]
    times = [it.timestamp.timestamp() if it.timestamp else None for it in items]
    docs, idf, common = _vectorize(titles, company, times)
    groups = _average_link(docs, CLUSTER_THRESHOLD if threshold is None else threshold, times)
    if REFINE:
        groups = _refine(groups, docs, times)

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
            terms=_top_terms(g, docs, idf, common),
        ))
    return out


def _representative(group: list[int], items: list[ClusterItem], docs: list[_Doc]) -> int:
    """Most central member, preferring trusted outlets, heavier items and
    plain declarative headlines over questions, listicles and roundups."""
    if len(group) == 1:
        return group[0]
    centroid: dict[str, float] = defaultdict(float)
    for i in group:
        for f, w in docs[i].vec.items():
            centroid[f] += w
    max_w = max((items[i].weight for i in group), default=0.0) or 1.0

    def score(i: int) -> float:
        title = items[i].title or ""
        central = _dot(docs[i].vec, centroid) / len(group)
        trust = publisher_trust(items[i].publisher) if items[i].publisher else 0.9
        weight = max(items[i].weight, 0.0) / max_w
        words = len(title.split())
        shape = 0.8 if _is_weak_headline(title) else 1.0
        shape *= 0.8 if _LEAD_RE.match(fold(title)) or re.search(r";|\s\|\s", title) else 1.0
        shape *= 0.85 if words < 6 or words > 20 else 1.0
        shape *= 0.9 if title.endswith(("...", "\u2026")) or re.search(r"\s\w{1,2}$", title) else 1.0
        return central * (0.6 + 0.25 * trust + 0.15 * weight) * shape

    return max(group, key=lambda i: (score(i), -i))


_WEAK_WORDS = HEADLINE_VERBS | MOVE_WORDS | CALENDAR_WORDS
_WEAK_STEMS = frozenset({stem(w) for w in _WEAK_WORDS} | _WEAK_WORDS)
_CONCEPT_LABELS = {
    "record-high": "record high", "price-target": "price target", "market-cap": "market cap", "top-pick": "top pick",
    "smart-home": "smart home", "data-center": "data centers", "short-seller": "short seller", "52w-low": "52-week low",
    "stock-split": "stock split", "earnings-beat": "earnings beat", "earnings-miss": "earnings miss",
    "price-cut": "price cuts", "price-hike": "price hikes", "inst-holding": "institutional holdings",
}


def _top_terms(group: list[int], docs: list[_Doc], idf: dict[str, float], common: frozenset[str],
               limit: int = 5) -> list[str]:
    """Readable distinguishing terms: entities, amounts, concepts and
    bigrams shared by the cluster, most specific first."""
    counts: Counter[str] = Counter()
    anchored: set[str] = set()
    surface_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for i in group:
        anchored |= docs[i].anchors
        for f in docs[i].vec:
            counts[f] += 1
            surface = docs[i].surfaces.get(f)
            if surface:
                surface_votes[f][surface] += 1
    min_count = 2 if len(group) >= 3 else 1
    scored: list[tuple[float, str]] = []
    for f, c in counts.items():
        if c < min_count or f.startswith(("ev:", "firm:", "t:")) or _YEAR_RE.match(f) or _PCT_RE.match(f):
            continue
        words = f.split()
        if any(w in _WEAK_STEMS or _NUM_RE.match(w) for w in words):
            continue
        weight = c / len(group) * idf.get(f, 0.0) * (1.3 if f in anchored else 0.7)
        weight *= 1.15 if len(words) > 1 or f in _CONCEPT_LABELS else 1.0
        scored.append((weight, f))
    scored.sort(key=lambda x: (-x[0], x[1]))
    best = {f: w for w, f in scored}
    chosen: list[str] = []
    covered: set[str] = set()
    for w, f in scored:
        parts = f.split()
        if len(parts) == 1 and any(f in b.split() and counts[b] >= 0.6 * counts[f] and best[b] >= 0.5 * w
                                   for b in best if " " in b):
            continue  # its bigram says it better ("morgan stanley" over "morgan")
        if len(parts) > 1 and set(parts) & covered:
            continue
        chosen.append(f)
        covered |= set(parts)
        if len(chosen) >= limit:
            break
    out: list[str] = []
    for f in chosen:
        text = _display(f, surface_votes.get(f), common)
        if text.lower() not in {o.lower() for o in out}:
            out.append(text)
    return out


def _display(feature: str, votes: Counter[str] | None, common: frozenset[str]) -> str:
    """Surface form for a feature: words this batch also writes lower-case
    are shown lower-case; names keep their capitalization."""
    if feature in _CONCEPT_LABELS:
        return _CONCEPT_LABELS[feature]
    if feature.startswith("$"):
        return "$" + feature[1:].upper()
    if not votes:
        return feature
    surface = votes.most_common(1)[0][0]
    words = surface.split()
    shown = []
    for word in words:
        low = word.lower()
        if low in _CONCEPT_LABELS:
            shown.append(_CONCEPT_LABELS[low])
        elif low in common and not _is_entity_shape(word):
            shown.append(low)
        else:
            shown.append(word)
    return " ".join(shown)
