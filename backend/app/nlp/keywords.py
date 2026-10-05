"""Keyword chips: the terms people are actually talking about, with tone.

`extract_keywords(texts, scores, company)` -> [(term, count, mean_score)].
Counts are document frequencies over *distinct* headlines: syndicated copies
("... By Investing.com", wire re-posts) count once, so a story repeated by
ten aggregators doesn't drown out everything else, and a headline mentioning
"buyback" twice counts once. Meaningful bigrams ("price target", "data
center", "Morgan Stanley") are preferred over their parts and chips never
repeat a word ("Musk" + "Elon Musk"); the company's own names/ticker,
publisher names, stopwords, generic finance filler ("stock", "shares",
"investors") and price-move verbs ("jumps", "falls") are excluded — the
chips should say *what* is discussed, not that the stock moved.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from app.nlp.events import canonical_firm, is_known_firm
from app.nlp.narratives import find_duplicates
from app.nlp.relevance import company_terms
from app.nlp.text import (
    CALENDAR_WORDS,
    COMMON_HEADLINE_WORDS,
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
    (r"\b(?:analysts'? )?(?:expectations|consensus(?: estimates?)?|estimates?)\b", " estimates "),
    (r"\b(?:first|1st)[- ]quarter\b", " q1 "), (r"\b(?:second|2nd)[- ]quarter\b", " q2 "),
    (r"\b(?:third|3rd)[- ]quarter\b", " q3 "), (r"\b(?:fourth|4th)[- ]quarter\b", " q4 "),
))
_LABELS = {
    "buyback": "buyback", "record-high": "record high", "price-target": "price target", "market-cap": "market cap",
    "data-center": "data centers", "layoffs": "layoffs", "lawsuit": "lawsuit", "short-seller": "short seller",
    "smart-home": "smart home", "top-pick": "top pick", "earnings-report": "earnings report",
    "price-cuts": "price cuts", "price-hikes": "price hikes", "q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4",
}
# Inside a bigram the concept reads in its singular/attributive form.
_PART_LABELS = {**_LABELS, "data-center": "data center", "layoffs": "layoff", "price-target": "price target",
                "record-high": "record-high", "lawsuit": "lawsuit"}
# Extra filler that makes poor chips even when frequent.
_FILLER = wordset("""
says said report reports reported according update updates news today week weekly daily year years month quarter
quarterly time times part amid ahead set sets see sees seen look looks looking way ways need needs thing things
lot lots big bigger biggest high low higher lower record new top stocks stock shares share company companies
largest increase increases increased focus historic implies imply fair upside downside growth think give gives
absolutely happen happens ideas idea future break breaks reason reasons case right left long short best better
worse worst good great strong weak little much many more most less least next last first second third
potential possible likely another other others still just also even only really very well back near
llc lp inc co corp ltd plc sa ag nv com www usd eur gbp jpy cad aud chf sek nok dkk inr cny hkd
sign signs signed signing pay pays paid paying mean means meaning highlight highlights trend trends play plays
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


@dataclass
class _Case:
    """Capitalization evidence gathered from one batch of headlines."""

    votes: dict[str, Counter[str]]  # lower-case word -> surface spellings
    proper: set[str]  # capitalized mid-sentence in a sentence-case headline ("Desai")
    common: set[str]  # written lower-case somewhere ("chips", "holiday")
    phrases: set[tuple[str, str]]  # adjacent capitalized pairs mid-sentence ("World Labs")


def _case_evidence(texts: list[str]) -> _Case:
    """Words seen only in Title Case or sentence-initially are neither proper
    nor common; _display resolves them."""
    case = _Case(defaultdict(Counter), set(), set(), set())
    for text in texts:
        sentence_case = not is_title_case(text)
        for sentence in re.split(r"[:;.!?]\s+|\s[-|]\s", text):
            words = [w.removesuffix("'s") for w in _WORD_RE.findall(sentence)]
            for k, word in enumerate(words):
                low = word.lower()
                case.votes[low][word] += 1
                if not word[0].isupper():
                    case.common.add(low)
                elif sentence_case and k > 0:
                    case.proper.add(low)
                    if words[k - 1][0].isupper() and k > 1:
                        case.phrases.add((words[k - 1].lower(), low))
    case.proper -= case.common
    return case


def _display(term: str, case: _Case) -> str:
    """Readable chip text: acronyms/CamelCase as written, names capitalized,
    common words lower-case. A Title-Case-only word reads as a name unless
    it is an everyday headline word; name phrases seen mid-sentence keep
    their capitals ("World Labs" even though "world model" is lower-case)."""
    if term in _LABELS:
        return _LABELS[term]
    if is_known_firm(term):
        return canonical_firm(term)
    parts = term.split()
    phrase = len(parts) == 2 and (parts[0], parts[1]) in case.phrases

    def ambiguous(part: str) -> bool:
        return part not in case.proper and part not in case.common and part not in COMMON_HEADLINE_WORDS

    def is_name(part: str) -> bool:
        if phrase or part in case.proper:
            return True
        if not ambiguous(part) or part in _LABELS:
            return False
        others = [o for o in parts if o != part]
        return not others or any(o in case.proper or ambiguous(o) for o in others)

    words = []
    for part in parts:
        if part in _LABELS:
            words.append(_LABELS[part] if len(parts) == 1 else _PART_LABELS[part])
            continue
        surfaces = case.votes.get(part)
        best = surfaces.most_common(1)[0][0] if surfaces else part
        if _ACRONYM_RE.match(best) or any(c.isupper() for c in best[1:]):
            words.append(best)  # AI, GPU, OpenAI, iPhone
        elif is_name(part):
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
    raw_values = list(scores) if scores is not None else []
    copies = find_duplicates([t or "" for t in texts])  # syndicated copies count once
    cleaned = [fold(clean_text(texts[g[0]] or "")) for g in copies]
    values = [sum(float(raw_values[i]) if i < len(raw_values) else 0.0 for i in g) / len(g) for g in copies]
    case = _case_evidence(cleaned)

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

    min_count = 2 if len(cleaned) >= 8 else 1
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
    used_plain: set[str] = set()  # plain words already shown
    used_pieces: set[str] = set()  # words inside shown concepts ("price-target" -> price, target)
    for _w, term in ranked:
        parts = term.split()
        plain = {p for p in parts if p not in _LABELS}
        pieces = {piece for p in parts if p in _LABELS for piece in p.split("-")}
        if set(parts) & used_parts or plain & used_pieces or pieces & used_plain:
            continue  # never repeat a word across chips ("Musk" / "Elon Musk", "target" / "price target")
        words = [surfaces[p].most_common(1)[0][0] if surfaces.get(p) else p for p in parts]
        label = _display(" ".join(words), case)
        if any(label.lower() == existing.lower() for existing, _c, _s in out):
            continue
        count = df[term]
        out.append((label, count, round(score_sum[term] / count, 3)))
        used_parts |= set(parts)
        used_plain |= plain
        used_pieces |= pieces
        if len(out) >= top_n:
            break
    return out
