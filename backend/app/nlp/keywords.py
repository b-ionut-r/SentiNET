"""Keyword chips: the terms people are actually talking about, with tone.

`extract_keywords(texts, scores, company)` -> [(term, count, mean_score)].
Counts are document frequencies (a headline mentioning "buyback" twice
counts once). Meaningful bigrams ("price target", "data center", "Morgan
Stanley") are preferred over their parts; the company's own names/ticker,
publisher names, stopwords, generic finance filler ("stock", "shares",
"investors") and price-move verbs ("jumps", "falls") are excluded — the
chips should say *what* is discussed, not that the stock moved.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import pairwise

from app.nlp.events import canonical_firm, is_known_firm
from app.nlp.relevance import company_terms
from app.nlp.text import (
    CALENDAR_WORDS,
    GENERIC_WORDS,
    HEADLINE_VERBS,
    MOVE_WORDS,
    STOPWORDS,
    clean_text,
    fold,
    is_title_case,
    stem,
    tokenize,
    wordset,
)
from app.sources.base import CompanyRef

# Paraphrases shown as one chip.
_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = tuple((re.compile(p, re.IGNORECASE), c) for p, c in (
    (r"\b(?:share|stock)[- ]repurchases?\b|\bbuy[- ]?backs?\b|\brepurchas\w*\b", " buyback "),
    (r"\b(?:all[- ]time|record)[- ]highs?\b", " record-high "),
    (r"\bprice[- ]targets?\b|\btarget[- ]prices?\b|\bPTs?\b", " price-target "),
    (r"\bmarket (?:cap(?:italization)?|value)\b", " market-cap "),
    (r"\bdata cent(?:er|re)s?\b", " data-center "),
    (r"\blay(?:s|ing)?[- ]?offs?\b|\blaid off\b|\bjob cuts?\b", " layoffs "),
    (r"\bartificial intelligence\b|\ba\.i\.", " AI "),
    (r"\b(?:law)?suits?\b|\bclass[- ]actions?\b", " lawsuit "),
    (r"\bshort[- ]sell(?:er|ers)?\b", " short-seller "),
    (r"\bsmart[- ]home\b", " smart-home "),
    (r"\btop (?:\w+ )?picks?\b", " top-pick "),
    (r"\bearnings (?:call|report)s?\b", " earnings-report "),
    (r"\b(?:cut|cuts|cutting|slash(?:es|ed|ing)?|lower(?:s|ed|ing)?)(?: \w+)? prices?\b|\bprice cuts?\b", " price-cuts "),
    (r"\b(?:raise[sd]?|raising|hike[sd]?|hiking)(?: \w+)? prices?\b|\bprice (?:hikes?|increases?)\b", " price-hikes "),
))
_LABELS = {
    "buyback": "buyback", "record-high": "record high", "price-target": "price target", "market-cap": "market cap",
    "data-center": "data centers", "layoffs": "layoffs", "lawsuit": "lawsuit", "short-seller": "short seller",
    "smart-home": "smart home", "top-pick": "top pick", "earnings-report": "earnings report",
    "price-cuts": "price cuts", "price-hikes": "price hikes",
}
# Extra filler that makes poor chips even when frequent.
_FILLER = wordset("""
says said report reports reported according update updates news today week weekly daily year years month quarter
quarterly time times part amid ahead set sets see sees seen look looks looking way ways need needs thing things
lot lots big bigger biggest high low higher lower record new top stocks stock shares share company companies
largest increase increases increased focus historic implies imply fair upside downside growth think give gives
absolutely happen happens ideas idea future break breaks reason reasons case right left long short best better
worse worst good great strong weak little much many more most less least next last first second third
potential possible likely another other others still just also even only really very well back near
llc lp inc co corp ltd plc sa ag nv com www
sells sell sold selling buys bought buying raises raise raised lowers lower lowered cuts cut reiterates reiterate
maintains maintain keeps keep kept initiates initiate says say sees warns warn hits gets got makes made takes
corp inc ltd plc group holdings firm firms investor investors trader traders analyst analysts market markets
price prices percent pct billion million trillion thousand dollar dollars nasdaq nyse wall street buy sell hold
here's what's why how what who which this that these those it's i'm you're they're we're don't can't won't
""")
# Outlet names that leak into headlines ("... - Yahoo Finance", "Zacks Investment Ideas").
_PUBLISHER_WORDS = wordset("""
reuters bloomberg cnbc wsj barron's barrons marketwatch yahoo benzinga zacks motley fool seekingalpha tipranks
marketbeat gurufocus investorplace thestreet barchart stocktwits tradingview moomoo kalkine trefis tikr finviz
investopedia morningstar globenewswire newswire prnewswire businesswire accesswire
""")
_ACRONYM_RE = re.compile(r"^[A-Z0-9&]{2,6}$")
_WORD_RE = re.compile(r"[A-Za-z][\w&'-]*")


def _swap_if_misordered(scores: object, company: object) -> tuple[Sequence[float] | None, CompanyRef | None]:
    """Accept (texts, company, scores) too — the order in nlp/types.py's
    overview — so a positional mix-up can't silently break keywords."""
    if isinstance(scores, CompanyRef) or (company is not None and not isinstance(company, CompanyRef)):
        scores, company = company, scores
    return scores, company  # type: ignore[return-value]


def _surface_votes(texts: list[str]) -> tuple[dict[str, Counter[str]], set[str]]:
    """Surface spellings per lower-case word, and the names in this batch:
    words written capitalized in some sentence-case headline and never
    lower-case mid-sentence ("Morgan Stanley" keeps its capitals; "shift",
    seen only in Title Case headlines, does not)."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    capital: set[str] = set()
    lower: set[str] = set()
    for text in texts:
        sentence_case = not is_title_case(text)
        for k, word in enumerate(_WORD_RE.findall(text)):
            word = word.removesuffix("'s")
            low = word.lower()
            votes[low][word] += 1
            if sentence_case:
                if word[0].isupper():
                    capital.add(low)
                elif k > 0:
                    lower.add(low)
    return votes, capital - lower


def _display(term: str, votes: dict[str, Counter[str]], names: set[str]) -> str:
    if term in _LABELS:
        return _LABELS[term]
    if is_known_firm(term):
        return canonical_firm(term)
    words = []
    for part in term.split():
        if part in _LABELS:
            words.append(_LABELS[part])
            continue
        surfaces = votes.get(part)
        best = surfaces.most_common(1)[0][0] if surfaces else part
        if _ACRONYM_RE.match(best) or any(c.isupper() for c in best[1:]):
            words.append(best)  # AI, GPU, OpenAI, iPhone
        elif part in names:
            words.append(best[0].upper() + best[1:])
        else:
            words.append(part)
    return " ".join(words)


def extract_keywords(texts: list[str], scores: Sequence[float] | None = None, company: CompanyRef | None = None,
                     top_n: int = 15) -> list[tuple[str, int, float]]:
    """Top `top_n` terms as (term, document count, mean score of the texts
    containing it), most discussed first. Terms seen in only one text are
    dropped once there are 8+ texts."""
    scores, company = _swap_if_misordered(scores, company)
    if not texts:
        return []
    own = company_terms(company)
    cleaned = [fold(clean_text(t or "")) for t in texts]
    votes, names = _surface_votes(cleaned)
    values = list(scores) if scores is not None else [0.0] * len(texts)

    df: Counter[str] = Counter()
    score_sum: dict[str, float] = defaultdict(float)
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)  # stem -> lower-case spellings
    for i, text in enumerate(cleaned):
        for pattern, concept in _PHRASES:
            text = pattern.sub(concept, text)
        seq: list[str] = []  # content stems in order; "" breaks adjacency
        for tok in tokenize(text):
            word = tok.strip("'")
            if (word in own or word.lstrip("$") in own or word in STOPWORDS or word in GENERIC_WORDS
                    or word in _FILLER or word in MOVE_WORDS or word in CALENDAR_WORDS or len(word) < 3
                    or not re.match(r"^[a-z][a-z0-9&_'-]*$", word) or word in _PUBLISHER_WORDS):
                seq.append("")
                continue
            if word in HEADLINE_VERBS:
                seq.append("")
                continue
            key = word if "-" in word else stem(word)
            surfaces[key][word] += 1
            seq.append(key)
        terms = {w for w in seq if w}
        terms |= {f"{a} {b}" for a, b in pairwise(seq) if a and b and a != b}
        for term in terms:
            df[term] += 1
            score_sum[term] += float(values[i]) if i < len(values) else 0.0

    min_count = 2 if len(texts) >= 8 else 1
    candidates = {t: c for t, c in df.items() if c >= min_count}
    ranked: list[tuple[float, str]] = []
    for term, count in candidates.items():
        parts = term.split()
        if len(parts) == 1:
            # Prefer a bigram that carries most of this word's mentions.
            if any(count and candidates.get(b, 0) >= 0.6 * count for b in candidates if " " in b and term in b.split()):
                continue
            weight = count * (1.25 if term in _LABELS else 1.0)
        else:
            weight = count * 1.35
        ranked.append((weight, term))
    ranked.sort(key=lambda x: (-x[0], x[1]))

    out: list[tuple[str, int, float]] = []
    used_parts: set[str] = set()
    for _w, term in ranked:
        parts = term.split()
        if len(parts) > 1 and set(parts) <= used_parts:
            continue
        words = [surfaces[p].most_common(1)[0][0] if surfaces.get(p) else p for p in parts]
        label = _display(" ".join(words), votes, names)
        if any(label.lower() == existing.lower() for existing, _c, _s in out):
            continue
        count = df[term]
        out.append((label, count, round(score_sum[term] / count, 3)))
        used_parts |= set(parts)
        if len(out) >= top_n:
            break
    return out
