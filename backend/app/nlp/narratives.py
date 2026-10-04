"""Syndication detection and story-level clustering of headlines.

* `find_duplicates(titles)` groups near-identical headlines (wire copies,
  aggregator re-posts, truncations, "... By Investing.com" variants). It
  returns a partition of all indices: every index appears in exactly one
  group, the representative (lowest index — pass titles in priority order)
  first.

* `cluster_narratives(items, company)` groups headlines about the same
  *development* ("Nvidia adds $150B to buyback" / "Nvidia's record repurchase
  shows the stock is too cheap for Huang to resist"):

  1. Syndicated copies are collapsed first (they would fake story cores and
     deflate the idf of a story's own terms) and re-expanded at the end.
  2. Features: TF-IDF weighted content terms with the company's own name
     removed — stemmed words, proper nouns learned from the batch's casing,
     concept tokens that collapse paraphrases ("share repurchase" ==
     "buyback", "third-quarter" == "Q3", "expectations" == "estimates"),
     adjacent-word bigrams, normalized money amounts ("$150 billion" ==
     "$150B"; round "$1B" weighs little), detected events / analyst firms,
     and time-local event features (event group x trading session: "stock
     jumps 5% on delivery beat" and "why is Tesla stock surging today?").
     Hub terms whose documents share little else ("Muse", "AI") are damped.
  3. Exact average-link agglomeration (mean pairwise cosine via cluster
     sum-vectors, time-faded), gated by shared anchors and absolute evidence.
  4. Refinement: merge close cores, move clearly misplaced members, attach
     satellites to the core whose *defining* features they share.

  Measured on hand-labeled real Google News sets in tests/fixtures/nlp
  (pairwise F1 / B-cubed F1): tuning sets NVDA .91/.92, AAPL .83/.89,
  META .50/.81, TGT .82/.87, XYZ .99/.96; validation TSLA .80/.83,
  AMZN .81/.86; test set AMD .89/.89 at its first, untouched scoring
  (.83/.87 after later fixes). The previous version scored NVDA .83/.89,
  TGT .71/.78, META .48/.82, TSLA .75/.77, AMD .90/.91 on the same sets.
  Pure Python, deterministic; ~0.25 s for 500 headlines of one company.
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

# Tuned on the labeled sets in tests/fixtures/nlp (see module docstring and
# tests/nlp/test_narratives.py). Average-link cut-off on mean pairwise cosine:
CLUSTER_THRESHOLD = 0.08
SINGLETON_DAMPING = 0.3
IDF_PRIOR_DOCS = 10
MIN_EVIDENCE = 1.5
HUB_MIN_DF = 4
HUB_REFERENCE = 0.1
HUB_FLOOR = 0.2
TIME_GRACE_H = 24.0
TIME_FADE_H = 144.0
# Clusters sharing no anchor (entity, amount, bigram, concept, firm) need this
# much average-link similarity to merge: plain words link only near-paraphrases.
NONANCHOR_SIM = 0.4
ANCHOR_MAX_DF = 0.35
# Refinement (see _refine): cores, defining features, attach/merge cut-offs.
REFINE = True
CORE_MIN = 3
DEFINING_FRAC = 0.4
DEFINING_FRAC_TIMED = 0.25
ATTACH_SIM = 0.2
CORE_MERGE_SIM = 0.35
REASSIGN_MARGIN = 0.1

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
    longer, in order) — unless each states a number, amount or date the other
    lacks (template headlines about different filings/days). Groups are
    connected components; representative = lowest index."""
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

    # Exact prefix filtering: order each title's tokens rarest-first; a pair
    # reaching the Jaccard/containment cut-offs must share a token from the
    # first |s| - ceil(DUP_JACCARD * |s|) + 1 of either side, so probing that
    # prefix against a full inverted index finds every candidate cheaply.
    df = Counter(tok for s in sets for tok in s)
    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for tok in s:
            index[tok].append(i)
    for a, s in enumerate(sets):
        if not s:
            continue
        prefix = sorted(s, key=lambda tok: (df[tok], tok))[:len(s) - math.ceil(DUP_JACCARD * len(s)) + 1]
        for b in sorted({b for tok in prefix for b in index[tok] if b != a}):
            if find(a) != find(b) and _near_identical(seqs[a], seqs[b], sets[a], sets[b]):
                union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [sorted(g) for _, g in sorted(groups.items())]


_SLOT_RE = re.compile(r"^(?:\$?\d[\d.,]*[%tbmk]?)$")


def _slots(seq: list[str]) -> set[str]:
    """Values that template headlines vary ("Insider Sold Shares Worth
    $1,327,740" / "$1,360,080", "underperforms Monday" / "Friday"). The
    last token is skipped: truncated copies cut it ("...Target of $1")."""
    return {t for t in seq[:-1] if _SLOT_RE.match(t) or t in CALENDAR_WORDS}


def _near_identical(sa: list[str], sb: list[str], a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    slots_a, slots_b = _slots(sa), _slots(sb)
    if slots_a - slots_b and slots_b - slots_a:
        return False  # same template, different facts: separate items
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
    # "best month since 2022" / "best quarter in over 2 years": one rally story.
    (r"\bbest (?:(?:single|full|trading)[- ])?(?:day|week|month|quarter|year|session|stretch|run|start)s?\b", " best-stretch "),
    (r"\bworst (?:(?:single|full|trading)[- ])?(?:day|week|month|quarter|year|session|stretch|run|start)s?\b",
     " worst-stretch "),
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
      r"projections)\b|\bthan (?:(?:wall street|analysts|the street|investors) )?(?:had )?expected\b"), " estimates "),
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
_W_ROUND_MONEY = 0.6
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


def _money_specific(token: str) -> bool:
    """'$8.2b', '$150b', '$486,532' and trillion-scale figures identify a story;
    round '$1b', '$10000', '$100m' recur across unrelated ones (every
    '$1 billion deal')."""
    digits = re.sub(r"[^\d]", "", token).rstrip("0")
    return len(digits) >= 2 or digits not in {"", "1"} or token.endswith("t")  # "$1t": market-cap milestones


def _split_hyphens(tokens: list[str]) -> list[str]:
    """'auto-vote' -> 'auto', 'vote' and 'tesla-spacex' -> 'tesla', 'spacex' so
    hyphenated and spaced spellings share features; concept tokens stay whole."""
    out: list[str] = []
    for tok in tokens:
        if "-" in tok and tok not in _CONCEPTS and not tok[:1].isdigit():
            out.extend(part for part in tok.split("-") if part)
        else:
            out.append(tok)
    return out


# A move attributed to a period ("jumped 30% in September", "has rallied 35%",
# "is down 20% this year") is a recap, not the current session's move.
_PERIOD_MOVE_RE = re.compile(
    r"\b(?:in|during|for|over|since|through)\s+(?:the\s+)?(?:past\s+|last\s+|first\s+)?(?:(?-i:[A-Z])[a-z]+\b|"
    r"(?:19|20)\d\d|week|month|quarter|year|q[1-4]|h[12]|\d+\s+(?:days|weeks|months|years))|"
    r"\b(?:this|last|past)\s+(?:week|month|quarter|year)\b|\b(?:ytd|year[- ]to[- ]date|so far this year)\b",
    re.IGNORECASE,
)
_PERFECT_RE = re.compile(r"\b(?:has|have|had)\b", re.IGNORECASE)


def _period_move(title: str, span: str | None) -> bool:
    """True when a price move in `title` is a multi-day recap, not a session move."""
    text = fold(title)
    idx = text.find(span) if span else -1
    if idx < 0 or not span:
        return False
    after = text[idx + len(span):idx + len(span) + 40].lstrip(" ,")
    return bool(_PERIOD_MOVE_RE.match(after) or _PERFECT_RE.search(text[max(0, idx - 15):idx + len(span)]))


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

    for tok in _split_hyphens(tokenize(text)):
        surface = cased.get(tok, tok)
        if tok in own or tok.lstrip("$") in own:
            content.append(("", "", False))  # the company's own name is not a story feature
        elif tok.startswith("$") and tok[1:2].isdigit():
            specific = _money_specific(tok)
            add(tok, _W_MONEY if specific else _W_ROUND_MONEY, tok.upper(), anchor=specific)
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
        if group and (not group.startswith("move-") or (_own_subject(title, event.span, own)
                                                         and not _period_move(title, event.span))):
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
    # Background pseudo-documents keep idf meaningful in small batches: with
    # 3 headlines all about one buyback, "buyback" must still link them.
    idf = {f: math.log((n + IDF_PRIOR_DOCS) / (c + 0.5)) for f, c in df.items()}
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


def _heavy(vec: dict[str, float], bound: float) -> set[str]:
    """Features left after dropping the lightest ones whose L2 norm < bound."""
    mass = 0.0
    out: set[str] = set()
    for f, w in sorted(vec.items(), key=lambda fw: (fw[1], fw[0])):
        mass += w * w
        if mass >= bound * bound:
            out.add(f)
    return out


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
    # Candidate pairs share a key: an anchor (pairs below NONANCHOR_SIM need
    # one; anchors in a large share of the batch are not story-specific) or a
    # "heavy" feature — the lightest features of a unit vector whose L2 mass
    # stays below NONANCHOR_SIM can't by themselves reach it (Cauchy-Schwarz).
    df = Counter(f for d in docs for f in d.vec)
    anchor_cap = max(3, int(ANCHOR_MAX_DF * len(docs)))
    keys: dict[int, set[str]] = {i: {f for f in d.anchors if df[f] <= anchor_cap} | _heavy(d.vec, NONANCHOR_SIM)
                                 for i, d in enumerate(docs)}
    by_key: dict[str, set[int]] = defaultdict(set)
    for i, feats in keys.items():
        for f in feats:
            by_key[f].add(i)

    heap: list[tuple[float, int, int, int, int]] = []

    def evidence(a: int, b: int) -> float:
        """Shared information in absolute tf-idf units (mean per member), so a
        single ubiquitous word ("Muse" in a quarter of the batch) can't
        glue two short headlines together while a rare shared entity can."""
        ra, rb = raws[a], raws[b]
        na, nb = len(members[a]), len(members[b])
        if len(ra) > len(rb):
            ra, rb, na, nb = rb, ra, nb, na
        return sum(min(w / na, rb[f] / nb) for f, w in ra.items() if f in rb)

    # Cached ΣA·ΣB per cluster pair. Average linkage is reducible:
    # (ΣA + ΣB)·ΣC = ΣA·ΣC + ΣB·ΣC, so a merge never re-scans a big cluster.
    dots: dict[tuple[int, int], float] = {}
    partners: dict[int, set[int]] = defaultdict(set)

    def pair_dot(a: int, b: int) -> float:
        key = (a, b) if a < b else (b, a)
        value = dots.get(key)
        if value is None:
            value = dots[key] = _dot(sums[a], sums[b])
            partners[a].add(b)
            partners[b].add(a)
        return value

    def push(a: int, b: int) -> None:
        a, b = min(a, b), max(a, b)
        sim = pair_dot(a, b) / (len(members[a]) * len(members[b]))
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
        neighbors = {j for f in keys[a] | keys[b] for j in by_key[f]} - {a, b}
        merged_dots = {j: pair_dot(a, j) + pair_dot(b, j) for j in neighbors}  # pre-merge vectors
        for c in (a, b):  # invalidate every cached pair of the two old clusters
            for j in partners.pop(c, set()):
                dots.pop((c, j) if c < j else (j, c), None)
                partners[j].discard(c)
        for j, value in merged_dots.items():
            dots[(a, j) if a < j else (j, a)] = value
            partners[a].add(j)
            partners[j].add(a)
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
        for j in neighbors:
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
    if len(group) == 1:
        return _unit(total), set(counts)
    need = max(2.0, DEFINING_FRAC * len(group))
    # Not every headline about a day's story mentions the price move, so a
    # session feature carried by a quarter of the core already characterizes it.
    need_timed = max(2.0, DEFINING_FRAC_TIMED * len(group))
    return _unit(total), {f for f, c in counts.items() if c >= (need_timed if f.startswith("t:") else need)}


def _refine(groups: list[list[int]], docs: list[_Doc], times: list[float | None]) -> list[list[int]]:
    """Second pass over the average-link result. Average linkage keeps
    stories tight but leaves satellites behind: a big story's own terms are
    frequent (low idf), and a short follow-up headline is never similar to
    *every* member. Cores are clusters of >= CORE_MIN distinct headlines:
    (1) cores with close centroids that share a defining feature merge;
    (2) a core member that is clearly closer to a larger core it shares a
    defining feature with moves there (one nearest-centroid pass);
    (3) every smaller cluster joins the core it shares a defining feature
    with and is most similar to (centroid cosine >= ATTACH_SIM, time-faded)."""
    cores = _merge_cores(sorted((g for g in groups if len(g) >= CORE_MIN), key=lambda g: (-len(g), g[0])),
                         docs, times)
    if not cores:
        return groups
    cores = _reassign(cores, docs, times)
    profiles = [_profile(g, docs) for g in cores]
    out = [list(g) for g in cores]
    for g in (g for g in groups if len(g) < CORE_MIN):
        centroid, _defining = _profile(g, docs)
        k = _best_core(set(centroid), centroid, _span(g, times), cores, profiles, times, ATTACH_SIM)
        if k >= 0:
            out[k].extend(g)
        else:
            out.append(list(g))
    return [sorted(g) for g in out if g]


def _best_core(feats: set[str], vec: dict[str, float], span: tuple[float, float] | None, cores: list[list[int]],
               profiles: list[tuple[dict[str, float], set[str]]], times: list[float | None], floor: float,
               skip: int = -1, min_size: int = 0) -> int:
    """Index of the most similar core sharing a defining feature with `feats`
    (similarity >= floor), or -1."""
    best_k, best_sim = -1, floor
    for k, (centroid, defining) in enumerate(profiles):
        if k == skip or len(cores[k]) < min_size or not feats & defining:
            continue
        sim = _dot(vec, centroid) * _span_factor(span, _span(cores[k], times))
        if sim >= best_sim:
            best_k, best_sim = k, sim
    return best_k


def _merge_cores(cores: list[list[int]], docs: list[_Doc], times: list[float | None]) -> list[list[int]]:
    """Greedily merge the closest pair of cores (centroid cosine >=
    CORE_MERGE_SIM, sharing a defining feature) until none qualifies."""
    while len(cores) > 1:
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
        if best is None:
            break
        _sim, a, b = best
        cores[a] = sorted(cores[a] + cores.pop(b))
    return cores


def _reassign(cores: list[list[int]], docs: list[_Doc], times: list[float | None]) -> list[list[int]]:
    """One nearest-centroid pass: a member moves to a core at least as large
    as its own when it shares a defining feature with it and is closer to it
    (by REASSIGN_MARGIN) than to the rest of its own cluster. Profiles are
    computed once up front, so the result is order-independent."""
    profiles = [_profile(g, docs) for g in cores]
    moves: list[tuple[int, int, int]] = []  # (item, from core, to core)
    for c, group in enumerate(cores):
        total: dict[str, float] = defaultdict(float)
        for i in group:
            for f, w in docs[i].vec.items():
                total[f] += w
        for i in group:
            rest = _unit({f: w - docs[i].vec.get(f, 0.0) for f, w in total.items()})
            own_sim = _dot(docs[i].vec, rest)
            point = (times[i], times[i]) if times[i] is not None else None
            k = _best_core(set(docs[i].vec), docs[i].vec, point, cores, profiles, times,
                           max(ATTACH_SIM, own_sim + REASSIGN_MARGIN), skip=c, min_size=len(group))
            if k >= 0:
                moves.append((i, c, k))
    out = [list(g) for g in cores]
    for i, c, k in moves:
        out[c].remove(i)
        out[k].append(i)
    return [sorted(g) for g in out if g]


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
    # Cluster distinct headlines only: syndicated copies would fake "cores"
    # and inflate the document frequency (lower the idf) of a story's terms.
    copies = find_duplicates(titles)
    rep_times = [times[g[0]] for g in copies]
    rep_docs, idf, common = _vectorize([titles[g[0]] for g in copies], company, rep_times)
    rep_groups = _average_link(rep_docs, CLUSTER_THRESHOLD if threshold is None else threshold, rep_times)
    if REFINE:
        rep_groups = _refine(rep_groups, rep_docs, rep_times)
    docs: list[_Doc] = [rep_docs[0]] * len(items)
    for k, g in enumerate(copies):
        for i in g:
            docs[i] = rep_docs[k]
    groups = [sorted(i for k in rg for i in copies[k]) for rg in rep_groups]

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
    "best-stretch": "best run", "worst-stretch": "worst run",
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
