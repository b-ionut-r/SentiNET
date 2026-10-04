"""Evidence extraction for the Sentinel engine: *what* in a text carries sentiment.

`extract(text, social=...)` turns one text into an `Evidence` bundle of signed
`Hit`s (each a meaningful term/phrase with a valence) after applying modifiers.
`app.nlp.engine` turns that evidence into a score. Layers, in priority order:

1. Regex rules for high-precision finance constructions, with numbers:
   analyst rating changes ("cut to Neutral from Buy"), price-target moves
   ("PT raised to $54 from $50" - direction from the numbers), earnings
   beats/misses, guidance raises/cuts, reported-vs-expected figures
   ("EPS $1.66 vs. $1.58"), fund flows, equity offerings, legal relief.
2. Lexicon phrases (longest match first), incl. neutralizers that block
   false friends ("shares outstanding", "in line with", "Best Buy").
3. Composition: movement words take their sign from what moved
   ("costs surge" < 0, "loss narrowed" > 0, "shares tumble 12%" << "dips 1%").
4. Modifiers: negation scope ("not", "fails to", "won't", "avoids"), contrast
   ("but" up-weights the later clause, "despite X" down-weights X), hedges
   ("may", "reportedly"), intensifiers, questions and listicles.

Everything is deterministic, regex/dict based and fast (no models).
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

from app.nlp import lexicon as lx


# --------------------------------------------------------------------------- #
# Normalization + tokenization
# --------------------------------------------------------------------------- #
class Token(NamedTuple):
    text: str  # lowercase surface form (possessive 's stripped, n't -> not)
    start: int  # char offsets into the normalized text
    end: int
    kind: str  # w(ord) num pct cur tag(cashtag) emo sep(hard boundary) soft(comma/colon) skip
    value: float = 0.0  # numeric value for num/pct (signed)
    signed: bool = False  # explicit +/- sign on a number ("+5%", "-3.4%")


_URL = re.compile(r"(?:https?://|www\.)\S+")
_ZW = re.compile(r"[​-‏⁠﻿︎️]")
_ABBREV = re.compile(
    r"\b(vs|inc|corp|co|ltd|plc|jr|sr|st|mr|mrs|ms|dr|est|approx|adj|avg|jan|feb|mar|apr|jun|jul|aug|"
    r"sep|sept|oct|nov|dec|nos|fig|bln|mln|mn|bn|yr|qtr|pts|no)\.(?=\s|$|\d)", re.I)
_COUNTRY = re.compile(r"\b(u)\.(s|k)\.?(?=[\s,;:)]|$)|\b(e)\.(u)\.(?=\s)", re.I)
_DIGIT = re.compile(r"\d")
_GLUED_CCY = re.compile(r"\b(eur|usd|gbp|sek|nok|dkk|chf|cad|aud|jpy|rmb|cny|inr|rs|mln|bln)(?=\d)", re.I)
_TRANSLATE = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "′": "'",
                            "″": '"', " ": " ", "…": "...", "−": "-"})

_TOKEN_RE = re.compile(
    r"""
    (?P<amp>[a-z]{1,2}&[a-z]{1,3}\b)
   |(?P<word>[a-z][a-z0-9]*(?:'[a-z]+)*|n't)
   |(?P<tag>\$[a-z][a-z0-9]{0,5}(?:[.\-][a-z]{1,3})?(?![a-z0-9]))
   |(?P<pct>(?:(?<![\w.%$])[-+])?\d+(?:[.,]\d+)*\s?(?:%|percent\b|per\s?cent\b|pct\b))
   |(?P<cur>[$€£¥])
   |(?P<num>(?:(?<![\w.%$])[-+])?\d+(?:[.,]\d+)*(?:\s?(?:bn|bln|billion|mn|mln|million|tn|trillion|k)\b)?)
   |(?P<hash>[#@][a-z0-9_]+)
   |(?P<emo>[\U0001F000-\U0001FAFF☀-➿⬀-⯿←-⇿])
   |(?P<sep>[.!?;|]+|\s[-–—]+\s|[–—]|\s-$)
   |(?P<soft>[,:()\[\]"])
    """,
    re.X,
)
_SCALE = {"bn": 1e9, "bln": 1e9, "billion": 1e9, "mn": 1e6, "mln": 1e6, "million": 1e6, "tn": 1e12,
          "trillion": 1e12, "k": 1e3}
_CONTRACTIONS = {"n't": "not", "can't": "cant", "won't": "wont", "ain't": "aint"}
# social g-dropping: "goin up", "rippin" -> "going up", "ripping"
_G_DROP = frozenset({"goin", "lookin", "movin", "runnin", "rippin", "pumpin", "dumpin", "sellin", "buyin", "holdin",
                     "gettin", "comin", "nothin", "somethin", "tankin", "flyin", "mooning", "bleedin", "crashin"})


def normalize(text: str) -> str:
    """Unescape HTML, unify quotes/dashes, drop URLs & zero-width chars, collapse spaces."""
    if "&" in text:
        text = html.unescape(text)
    if not text.isascii():
        text = _ZW.sub("", text.translate(_TRANSLATE))
    if "http" in text or "www." in text:
        text = _URL.sub(" ", text)
    if "." in text:
        text = _COUNTRY.sub(lambda m: (m.group(1) or m.group(3)) + (m.group(2) or m.group(4)), text)
        text = _ABBREV.sub(r"\1", text)
    if _DIGIT.search(text):
        text = _GLUED_CCY.sub(r"\1 ", text)
    return " ".join(text.split())


def _number(raw: str) -> float:
    """Parse "1,234.5 mln" / "5,2" (decimal comma) into a float."""
    raw = raw.strip()
    scale = 1.0
    m = re.search(r"([a-z]+)$", raw)
    if m:
        scale = _SCALE.get(m.group(1), 1.0)
        raw = raw[: m.start()].strip()
    sign = -1.0 if raw.startswith("-") else 1.0
    raw = raw.lstrip("+-")
    if "," in raw and "." not in raw and re.fullmatch(r"\d+,\d{1,2}", raw):
        raw = raw.replace(",", ".")  # European decimal comma
    try:
        return sign * float(raw.replace(",", "")) * scale
    except ValueError:
        return 0.0


def tokenize(low: str) -> list[Token]:
    """Tokenize lowercase normalized text. Hyphenated words are split ("all-time" -> all, time)."""
    out: list[Token] = []
    for m in _TOKEN_RE.finditer(low):
        kind = m.lastgroup or "skip"
        s, e = m.span()
        raw = m.group()
        if kind == "word":
            if raw.endswith("'s"):
                raw = raw[:-2]
            elif raw in _CONTRACTIONS:
                raw = _CONTRACTIONS[raw]
            elif raw.endswith("n't"):
                raw = raw.replace("'", "")
            elif raw in _G_DROP:
                raw += "g"
            out.append(Token(raw, s, e, "w"))
        elif kind == "amp":
            out.append(Token(raw, s, e, "w"))
        elif kind == "pct":
            num = re.match(r"[-+]?\d+(?:[.,]\d+)*", raw)
            out.append(Token(raw, s, e, "pct", _number(num.group()) if num else 0.0, raw[0] in "+-"))
        elif kind == "num":
            out.append(Token(raw, s, e, "num", _number(raw), raw[0] in "+-"))
        elif kind == "emo":
            out.append(Token(raw, s, e, "emo"))
        elif kind in ("tag", "cur", "sep", "soft", "hash"):
            out.append(Token(raw.strip(), s, e, kind))
    return out


# --------------------------------------------------------------------------- #
# Evidence types
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Hit:
    """One piece of sentiment evidence over tokens [start, end)."""

    start: int
    end: int
    valence: float
    term: str
    source: str  # "rule:<key>" | "lex" | "move" | "metric" | "social"
    weight: float = 1.0  # product of modifiers
    negated: bool = False
    anchor: int = -1  # token that carries the meaning (the verb of "sales ... rose"); -1 = start
    parts: tuple[tuple[int, int], ...] = ()  # token ranges naming the evidence (metric, verb, %)

    def __post_init__(self) -> None:
        if self.anchor < 0:
            self.anchor = self.start

    @property
    def value(self) -> float:
        return self.valence * self.weight


@dataclass(slots=True)
class Evidence:
    text: str  # normalized text (drivers are substrings of it)
    tokens: list[Token]
    hits: list[Hit] = field(default_factory=list)
    hedges: int = 0  # uncertainty terms seen
    litigious: int = 0
    question: bool = False
    listicle: bool = False
    text_hedge: bool = False


@dataclass(slots=True)
class _Span:
    """A lexicon phrase matched at tokens [start, end) with its roles."""

    start: int
    end: int
    key: str
    lex: Optional[float] = None
    social_only: Optional[float] = None
    direction: Optional[lx.Direction] = None
    metric: Optional[lx.Metric] = None
    negator: bool = False
    hedge: Optional[float] = None
    intens: Optional[float] = None
    contrast: Optional[str] = None  # "shift" | "concessive"
    level_qual: bool = False
    litigious: bool = False
    neutral: bool = False
    used: bool = False  # consumed by composition / negation


# --------------------------------------------------------------------------- #
# Phrase index (all lexicon tables, longest match first)
# --------------------------------------------------------------------------- #
def _norm_key(k: str) -> tuple[str, ...]:
    return tuple(" ".join(k.lower().replace("-", " ").replace("'s ", " ").split()).split())


def _build_index() -> dict[str, list[tuple[tuple[str, ...], dict[str, object]]]]:
    roles: dict[tuple[str, ...], dict[str, object]] = {}

    def put(table, role: str, value: Callable[[object], object] = lambda v: v) -> None:
        items = table.items() if isinstance(table, dict) else ((k, True) for k in table)
        for k, v in items:
            roles.setdefault(_norm_key(k), {})[role] = value(v)

    for table in (lx.POSITIVE, lx.NEGATIVE, lx.SOCIAL):
        put(table, "lex")
    put(lx.LITIGIOUS, "lex")
    put(lx.LITIGIOUS, "litigious", lambda v: True)
    put(lx.SOCIAL_ONLY, "social_only")
    put(lx.DIRECTIONS, "direction")
    put(lx.METRICS, "metric")
    put(lx.NEGATORS, "negator")
    put(lx.NEGATOR_FALLBACK, "negator", lambda v: True)
    put(lx.HEDGES, "hedge")
    put(lx.INTENSIFIERS, "intens")
    put({k: "shift" for k in lx.CONTRAST_SHIFT}, "contrast")
    put({k: "concessive" for k in lx.CONTRAST_CONCESSIVE}, "contrast")
    put(lx.LEVEL_QUALIFIERS, "level_qual")
    put(lx.NEUTRALIZERS, "neutral")
    index: dict[str, list[tuple[tuple[str, ...], dict[str, object]]]] = {}
    for key, r in roles.items():
        if key:
            index.setdefault(key[0], []).append((key, r))
    for cands in index.values():
        cands.sort(key=lambda kr: -len(kr[0]))
    return index


_INDEX = _build_index()
_MAX_PHRASE = max(len(k) for cands in _INDEX.values() for k, _ in cands)


def _match_spans(tokens: list[Token], claimed: bytearray) -> list[Optional[_Span]]:
    """Greedy longest-match of lexicon phrases over unclaimed word/emoji tokens.

    Returns ``at[i]`` = the span starting or covering token i (None if none)."""
    n = len(tokens)
    at: list[Optional[_Span]] = [None] * n
    i = 0
    while i < n:
        tok = tokens[i]
        cands = _INDEX.get(tok.text) if not claimed[i] and tok.kind in ("w", "emo") else None
        if not cands:
            i += 1
            continue
        for key, roles in cands:
            L = len(key)
            if i + L > n:
                continue
            ok = True
            for j in range(1, L):
                t = tokens[i + j]
                if claimed[i + j] or t.text != key[j] or t.kind not in ("w", "emo"):
                    ok = False
                    break
            if not ok:
                continue
            sp = _Span(i, i + L, " ".join(key))
            if roles.get("neutral"):
                sp.neutral = True
            else:
                sp.lex = roles.get("lex")  # type: ignore[assignment]
                sp.social_only = roles.get("social_only")  # type: ignore[assignment]
                sp.direction = roles.get("direction")  # type: ignore[assignment]
                sp.metric = roles.get("metric")  # type: ignore[assignment]
                sp.negator = bool(roles.get("negator"))
                sp.hedge = roles.get("hedge")  # type: ignore[assignment]
                sp.intens = roles.get("intens")  # type: ignore[assignment]
                sp.contrast = roles.get("contrast")  # type: ignore[assignment]
                sp.level_qual = bool(roles.get("level_qual"))
                sp.litigious = bool(roles.get("litigious"))
            for j in range(i, i + L):
                at[j] = sp
            i += L
            break
        else:
            i += 1
    return at


# --------------------------------------------------------------------------- #
# Regex rules (operate on lowercase text with intra-word hyphens -> spaces)
# --------------------------------------------------------------------------- #
class RuleMatch(NamedTuple):
    start: int
    end: int
    valence: float
    term: str
    key: str


_RATING_WORDS = sorted(lx.RATINGS, key=len, reverse=True)
RATING = "(?:" + "|".join(re.escape(r).replace(r"\ ", r"\s+") for r in _RATING_WORDS) + ")"
_AMT = (r"(?:-\s?)?(?:[$€£¥]|usd|eur|gbp|sek|nok|dkk|chf|cad|aud|jpy|rmb|cny|inr|rs\.?)?\s?-?\d[\d,]*(?:\.\d+)?"
        r"(?:\s?(?:bn|bln|billion|mn|mln|million|m|b|k|tn|trillion|cents?|c|%|percent|pct|p|bps))?(?![\w])")
_EXP = (r"(?:(?:analysts?'?|wall\s+street|street|market|consensus|average)\s+)?(?:estimates?|expectations?|"
        r"consensus|forecasts?|views?|projections?|targets?|guidance|est|street|analysts?|whisper\w*|"
        r"predictions?|outlooks?|expected)")
_GUID = (r"(?:full\s+year\s+|annual\s+|fy\s*\d*\s+|\d{4}\s+|q[1-4]\s+|quarterly\s+|sales\s+|revenue\s+|"
         r"profit\s+|earnings\s+|eps\s+)?(?:guidance|outlook|forecasts?|view|guide|projections?|targets?|"
         r"estimates?|expectations?)")
_GAP = r"(?:\s+[^\s.;!?$]+){0,4}?\s+"
_UPV = (r"raise[sd]?|raising|lift(?:s|ed|ing)?|boost(?:s|ed|ing)?|bump(?:s|ed|ing)?|hike[sd]?|hiking|"
        r"increas(?:e|es|ed|ing)|up(?:s|ped|ping)?|nudge[sd]?\s+up|push(?:es|ed)\s+up|improv(?:e|es|ed|ing)|"
        r"strengthen(?:s|ed|ing)?")
_DNV = (r"cut(?:s|ting)?|lower(?:s|ed|ing)?|slash(?:es|ed|ing)?|trim(?:s|med|ming)?|reduc(?:e|es|ed|ing)|"
        r"chop(?:s|ped|ping)?|decreas(?:e|es|ed|ing)|par(?:e|es|ed|ing)|nudge[sd]?\s+down|ratchet(?:s|ed)?\s+down|"
        r"lop(?:s|ped)?|halv(?:e|es|ed|ing)|reel(?:s|ed|ing)?\s+in|rein(?:s|ed|ing)?\s+in")
_PT = r"(?:price\s+targets?|target\s+price|price\s+objective|price\s+tgt|pt|tgt|targets?)"
# "beats by $0.04" / "misses on revenue" (not "Top Executive Calls on Government")
_BY_ON = (r"(?:by\s+(?:[$€£¥]|\d)|on\s+(?:the\s+)?(?:top\s+line|bottom\s+line|revenues?|revs?|sales|eps|earnings|"
          r"profits?|estimates|expectations|both|the\s+top|the\s+bottom|ebitda|margins?|guidance))")


def _rank(name: str) -> int:
    return lx.RATINGS.get(" ".join(name.split()), 0)


def _amount(raw: str) -> Optional[float]:
    """Parse an amount like "$5.2 bln", "EUR 205.5 mn", "-$0.05", "41 cents"."""
    m = re.search(r"(-)?\s?(?:[$€£¥]|[a-z]{2,3}\.?)?\s?(-)?(\d[\d,]*(?:\.\d+)?)\s?([a-z%]+)?", raw.strip())
    if not m:
        return None
    try:
        val = float(m.group(3).replace(",", ""))
    except ValueError:
        return None
    unit = (m.group(4) or "").lower()
    val *= {"bn": 1e9, "bln": 1e9, "billion": 1e9, "b": 1e9, "mn": 1e6, "mln": 1e6, "million": 1e6, "m": 1e6,
            "k": 1e3, "tn": 1e12, "trillion": 1e12, "cents": 0.01, "cent": 0.01, "c": 0.01, "p": 0.01,
            "bps": 0.0001}.get(unit, 1.0)
    return -val if (m.group(1) or m.group(2)) else val


def _is_year(raw: str) -> bool:
    return bool(re.fullmatch(r"\s*(?:19|20)\d\d\s*", raw))


def _analyst_change(m: re.Match[str]) -> Optional[RuleMatch]:
    verb, new, old = m.group("verb"), m.group("new"), m.group("old")
    up_verbs = ("upgrad", "rais", "lift", "boost", "bump", "hike", "ups", "upped")
    down_verbs = ("downgrad", "cut", "lower", "trim", "reduc", "slash")
    delta = _rank(new) - _rank(old) if old else 0
    if delta:
        val = math.copysign(1.0 + 0.15 * (abs(delta) - 1), delta)
    elif verb.startswith(up_verbs):
        val = 1.1 if verb.startswith("upgrad") else 1.0
    elif verb.startswith(down_verbs):
        val = -1.1 if verb.startswith("downgrad") else -1.0
    else:  # "moved to Buy" - stance only
        val = 0.6 * _rank(new)
    label = ("upgrade" if val > 0 else "downgrade" if val < 0 else "rating change") + f" to {' '.join(new.split())}"
    if old:
        label += f" from {' '.join(old.split())}"
    return RuleMatch(m.start(), m.end(), val, label, "analyst")


def _analyst_init(m: re.Match[str]) -> Optional[RuleMatch]:
    rank = _rank(m.group("new"))
    val = {2: 1.0, 1: 0.8, 0: 0.0, -1: -0.8, -2: -1.0}[rank]
    return RuleMatch(m.start(), m.end(), val, f"initiated at {' '.join(m.group('new').split())}", "analyst_init")


def _analyst_keep(m: re.Match[str]) -> Optional[RuleMatch]:
    rank = _rank(m.group("new"))
    return RuleMatch(m.start(), m.end(), 0.6 * rank, f"reiterated {' '.join(m.group('new').split())}",
                     "analyst_reiterate")


_FROM_TO = re.compile(rf"\bto\s+(?P<new>{_AMT})\s*(?:,\s*)?(?:from|vs\.?|versus)\s+(?P<old>{_AMT})")


def _price_target(m: re.Match[str]) -> Optional[RuleMatch]:
    verb = (m.groupdict().get("v") or "").strip()
    sign = 0
    if verb:
        sign = 1 if re.match(_UPV, verb) and not re.match(_DNV, verb) else -1
    tail = m.groupdict().get("tail") or ""
    nums = _FROM_TO.search(tail) or _FROM_TO.search(m.group())
    detail = ""
    mag = 1.0
    if nums:
        new, old = _amount(nums.group("new")), _amount(nums.group("old"))
        if new is not None and old is not None and new != old and old > 0:
            sign = 1 if new > old else -1
            mag = 0.9 + min(0.5, abs(new - old) / old * 2.0)
            detail = f" to {nums.group('new').strip()} from {nums.group('old').strip()}"
    if not sign:
        return None
    term = ("price target raised" if sign > 0 else "price target cut") + detail
    if m.groupdict().get("tail") is None:
        end = m.end()
    elif nums is not None and nums.re is _FROM_TO and tail and nums.string == tail:
        end = m.start("tail") + nums.end()
    else:
        end = m.start("tail")
    return RuleMatch(m.start(), end, sign * mag, term, "price_target")


def _superlative(m: re.Match[str]) -> RuleMatch:
    """"Hedge funds have never been this bullish" (+) / "never been less bullish" (-)."""
    positive = m.group("w") in ("bullish", "optimistic", "confident", "positive", "cheap")
    sign = (1 if positive else -1) * (-1 if m.group("deg") == "less" else 1)
    return RuleMatch(m.start(), m.end(), 1.2 * sign, f"never been {m.group('deg')} {m.group('w')}", "superlative")


def _fixed(val: float, term: str, key: str) -> Callable[[re.Match[str]], RuleMatch]:
    return lambda m: RuleMatch(m.start(), m.end(), val, term, key)


def _guidance(sign: int) -> Callable[[re.Match[str]], RuleMatch]:
    def make(m: re.Match[str]) -> RuleMatch:
        noun = re.search(r"guidance|outlook|forecasts?|view|guide|projections?|targets?|estimates?|expectations?",
                         m.group())
        what = noun.group() if noun else "guidance"
        if what.startswith(("estimate", "expectation")):
            label = f"{'raises' if sign > 0 else 'cuts'} {what}"
        else:
            label = f"{what} {'raised' if sign > 0 else 'cut'}"
        return RuleMatch(m.start(), m.end(), 1.1 * sign, label, "guidance")
    return make


def _flows(m: re.Match[str]) -> RuleMatch:
    act = m.group("act")
    bearish = act.startswith(("sell", "dump", "unload", "bail", "flee", "fled", "exit", "abandon", "short", "bear",
                              "cash", "trim", "sold", "sour", "cool"))
    who = " ".join(m.group("who").split())
    return RuleMatch(m.start(), m.end(), -0.9 if bearish else 0.9, f"{who} {' '.join(act.split())}", "flows")


def _offering(m: re.Match[str]) -> Optional[RuleMatch]:
    span = m.group()
    if re.search(r"\b(?:notes?|bonds?|debt|senior|debentures|credit|loan|term\s+loan)\b", span):
        return RuleMatch(m.start(), m.end(), 0.0, "debt offering", "offering")
    if "convertible" in span:
        return RuleMatch(m.start(), m.end(), -0.35, "convertible offering", "offering")
    if re.search(r"\b(?:ipo|initial\s+public)\b", span):
        return RuleMatch(m.start(), m.end(), 0.0, "ipo", "offering")
    return RuleMatch(m.start(), m.end(), -0.7, "equity offering", "offering")


_RULES: list[tuple[str, re.Pattern[str], Callable[[re.Match[str]], Optional[RuleMatch]]]] = [
    # --- analyst rating changes / initiations / reiterations
    ("analyst", re.compile(
        r"\b(?P<verb>upgrade[sd]?|downgrade[sd]?|raise[sd]?|lift(?:s|ed)?|boost(?:s|ed)?|bump(?:s|ed)?|"
        r"hike[sd]?|mov(?:e|es|ed|ing)|cut(?:s|ting)?|lower(?:s|ed|ing)?|trim(?:s|med)?|reduce[sd]?|"
        r"slash(?:es|ed)?|ups|upped|take[sn]?|took|raising|upgrading|downgrading)\b(?:(?!\bto\b)[^.;!?$\d]){0,50}?\bto\s+(?:an?\s+)?['\"]?(?P<new>" + RATING + r")\b"
        r"(?:[^.;!?$]{0,40}?\bfrom\s+(?:an?\s+)?['\"]?(?P<old>" + RATING + r")\b)?"), _analyst_change),
    ("analyst", re.compile(
        r"(?P<verb>)\b(?P<new>" + RATING + r")\s+from\s+(?P<old>" + RATING + r")\b"), _analyst_change),
    ("analyst_init", re.compile(
        r"\b(?:initiat\w*|start(?:s|ed)?|begins?|began|launch(?:es|ed)?|resume[sd]?|assume[sd]?|reinstate[sd]?|"
        r"pick(?:s|ed)?\s+up|coverage)\b[^.;!?]{0,50}?\b(?:at|with|as)\s+(?:an?\s+)?['\"]?(?P<new>"
        + RATING + r")\b"), _analyst_init),
    ("analyst_reiterate", re.compile(
        r"\b(?:reiterat\w*|maintain\w*|keep(?:s|ing)?|kept|affirm\w*|reaffirm\w*|retain\w*|repeat\w*|"
        r"stick(?:s)?\s+with|stays?\s+at|remains?\s+at)\b[^.;!?$]{0,40}?\b(?P<new>" + RATING + r")\b"
        r"(?!\s+(?:the|a|an|its|their|more|shares|stock|stocks|stake|back|of|in|to)\b)"), _analyst_keep),
    ("analyst_reiterate", re.compile(
        r"\b(?P<new>" + RATING + r")\s+(?:rating\s+)?(?:reiterated|maintained|affirmed|reaffirmed|kept)\b"),
     _analyst_keep),
    # --- price targets (numbers decide direction when present)
    ("price_target", re.compile(
        r"\b(?P<v>" + _UPV + "|" + _DNV + r")\b(?:\s+[^\s.;!?$]+){0,5}?\s+" + _PT + r"\b(?P<tail>[^.;!?]{0,50})"),
     _price_target),
    ("price_target", re.compile(
        r"\b" + _PT + r"\b(?:\s+[^\s.;!?]+){0,3}?\s+(?P<v>raised|lifted|boosted|bumped|hiked|increased|upped|cut|"
        r"lowered|trimmed|reduced|slashed|decreased|pared|chopped)\b(?P<tail>[^.;!?]{0,50})"), _price_target),
    ("price_target", re.compile(r"\b" + _PT + r"\b[^.;!?]{0,25}?\bto\s+" + _AMT + r"\s*(?:,\s*)?(?:from|vs\.?)\s+"
                                + _AMT), _price_target),
    # --- earnings vs. expectations
    ("beat", re.compile(
        r"\b(?:beat(?:s|ing)?|top(?:s|ped|ping)?|exceed(?:s|ed|ing)?|surpass(?:es|ed|ing)?|crush(?:es|ed)?|"
        r"smash(?:es|ed)?|trounce[sd]?|outstrip(?:s|ped)?|blow(?:s)?\s+past|blew\s+past|blow(?:s)?\s+away|"
        r"blew\s+away|best(?:s|ed)?|clear(?:s|ed)?|outpace[sd]?)" + _GAP.replace("{0,4}", "{0,5}") + _EXP + r"\b"),
     _fixed(1.0, "beats estimates", "beat")),
    ("beat", re.compile(r"\b(?:beat(?:s)?|tops|topped|exceed(?:s|ed)?)\s+(?:[^\s.;!?]+\s+){0,2}?" + _BY_ON),
     _fixed(1.0, "beats", "beat")),
    ("beat", re.compile(r"\b(?:eps|earnings|revenues?|sales|profits?|results?|quarter|q[1-4])\s+(?:and\s+\S+\s+)?"
                        r"(?:beat|beats|top|tops|exceed|exceeds|surpass(?:es)?)\b"), _fixed(1.0, "beat", "beat")),
    ("miss", re.compile(
        r"\b(?:miss(?:es|ed|ing)?|undershoot(?:s)?|undershot|fall(?:s|ing)?\s+short\s+of|fell\s+short\s+of|"
        r"c(?:o|a)me[s]?\s+in\s+(?:below|under|short\s+of)|trail(?:s|ed)?|lag(?:s|ged)?)" + _GAP + _EXP + r"\b"),
     _fixed(-1.0, "misses estimates", "miss")),
    ("miss", re.compile(r"\bmiss(?:es|ed)?\s+(?:[^\s.;!?]+\s+){0,2}?" + _BY_ON), _fixed(-1.0, "misses", "miss")),
    ("eps_loss", re.compile(r"\b(?:eps|loss\s+per\s+share)\s+(?:of\s+)?(?:-\s?[$€£¥]|[$€£¥]\s?-|\([$€£¥])\s?\d"),
     _fixed(-0.5, "negative EPS", "eps_loss")),
    ("miss", re.compile(r"\b(?:eps|earnings|revenues?|sales|profits?|results?|quarter|q[1-4])\s+(?:and\s+\S+\s+)?"
                        r"miss(?:es|ed)?\b"), _fixed(-1.0, "miss", "miss")),
    ("miss", re.compile(r"\bfalls?\s+short\b|\bfell\s+short\b"), _fixed(-0.8, "falls short", "miss")),
    ("miss", re.compile(r"\b(?:fail(?:s|ed)?\s+to|did\s+not|didnt|does\s+not|doesnt|do\s+not|dont|not)\s+"
                        r"(?:meet|match|reach|hit|beat|top)\s+(?:[^\s.;!?]+\s+){0,2}?" + _EXP + r"\b"),
     _fixed(-1.0, "misses estimates", "miss")),
    ("job_cuts", re.compile(r"\b(?:cut(?:s|ting)?|slash(?:es|ed|ing)?|eliminat(?:e|es|ed|ing)|ax(?:e|es|ed|ing)?|"
                            r"shed(?:s|ding)?|trim(?:s|med|ming)?|lay(?:s|ing)?\s+off|laid\s+off|reduc(?:e|es|ed|ing))"
                            r"\s+(?:[^\s.;!?]+\s+){0,3}?(?:jobs|positions|workers|employees|staff|roles|headcount|"
                            r"workforce)\b"), _fixed(-1.0, "job cuts", "job_cuts")),
    ("above_exp", re.compile(
        r"\b(?:above|ahead\s+of|better\s+than|exceeding|topping|beating|stronger\s+than|surpassing)\s+(?:the\s+)?"
        r"(?:[^\s.;!?]+\s+){0,2}?" + _EXP + r"\b"), _fixed(0.9, "above expectations", "above_exp")),
    ("below_exp", re.compile(
        r"\b(?:below|under|short\s+of|worse\s+than|weaker\s+than|behind|softer\s+than|missing)\s+(?:the\s+)?"
        r"(?:[^\s.;!?]+\s+){0,2}?" + _EXP + r"\b"), _fixed(-0.9, "below expectations", "below_exp")),
    ("vs_exp", re.compile(r"\b(?:better|stronger)\s+than\s+(?:expected|anticipated|forecast)\b"),
     _fixed(1.0, "better than expected", "above_exp")),
    ("vs_exp", re.compile(r"\b(?:worse|weaker|softer|poorer)\s+than\s+(?:expected|anticipated|forecast)\b"),
     _fixed(-1.0, "worse than expected", "below_exp")),
    ("vs_exp", re.compile(r"\bbetter\s+than\s+feared\b"), _fixed(0.6, "better than feared", "above_exp")),
    ("vs_exp", re.compile(r"\bworse\s+than\s+feared\b"), _fixed(-0.9, "worse than feared", "below_exp")),
    ("inline", re.compile(r"\b(?:in\s+line|inline)\s+with\s+(?:[^\s.;!?]+\s+){0,2}?" + _EXP + r"\b"),
     _fixed(0.0, "in line with estimates", "inline")),
    ("inline", re.compile(r"\b(?:match(?:es|ed|ing)?|meet(?:s|ing)?|met)\s+(?:[^\s.;!?]+\s+){0,3}?" + _EXP + r"\b"),
     _fixed(0.25, "meets estimates", "inline")),
    # --- guidance
    ("guidance", re.compile(r"\b(?:" + _UPV + r")\b(?:\s+[^\s.;!?$]+){0,4}?\s+" + _GUID + r"\b"), _guidance(1)),
    ("guidance", re.compile(
        r"\b(?:" + _DNV + r"|withdraw(?:s|n|ing)?|withdrew|pull(?:s|ed|ing)?|suspend(?:s|ed|ing)?|"
        r"scrap(?:s|ped|ping)?|scal(?:e|es|ed|ing)\s+back|temper(?:s|ed|ing)?|dial(?:s|ed|ing)?\s+back|"
        r"rein(?:s|ed|ing)?\s+in|reel(?:s|ed|ing)?\s+in|walk(?:s|ed|ing)?\s+back)\b(?:\s+[^\s.;!?$]+){0,4}?\s+" + _GUID + r"\b"), _guidance(-1)),
    ("guidance", re.compile(r"\b(?:guidance|outlook|forecast|view)\s+(?:\S+\s+){0,2}?(?:raised|lifted|boosted|"
                            r"increased|hiked|upped|improved)\b"), _guidance(1)),
    ("guidance", re.compile(r"\b(?:guidance|outlook|forecast|view)\s+(?:\S+\s+){0,2}?(?:cut|lowered|slashed|"
                            r"trimmed|reduced|withdrawn|suspended|pulled)\b"), _guidance(-1)),
    ("guidance", re.compile(r"\bguides?\s+(?:\S+\s+){0,3}?(?:above|higher|ahead|up|strong)\b"),
     _fixed(1.0, "guides higher", "guidance")),
    ("guidance", re.compile(r"\bguides?\s+(?:\S+\s+){0,3}?(?:below|lower|down|light|weak|soft)\b"),
     _fixed(-1.0, "guides lower", "guidance")),
    ("guidance", re.compile(r"\b(?:reaffirm\w*|reiterat\w*|maintain\w*|confirm\w*|affirm\w*|stands?\s+by)\b"
                            r"(?:\s+\S+){0,3}?\s+(?:guidance|outlook|forecasts?)\b"),
     _fixed(0.3, "reaffirms guidance", "guidance")),
    # --- flows / positioning
    ("flows", re.compile(
        r"\b(?P<who>hedge\s+funds?|funds|investors|insiders?|institutions|institutional\s+investors|whales|"
        r"traders|smart\s+money|billionaires?|big\s+money|money\s+managers|retail\s+investors|activists?)\s+"
        r"(?:are\s+|were\s+|is\s+|have\s+been\s+|keep\s+|continue\s+to\s+|have\s+|just\s+|still\s+|"
        r"arent\s+done\s+|aren't\s+done\s+|not\s+done\s+)*"
        r"(?P<act>buying|accumulating|piling\s+into|snapping\s+up|loading\s+up|adding|scooping\s+up|pouring\s+into|"
        r"rushing\s+into|flocking\s+to|bought|selling|dumping|unloading|bailing|fleeing|exiting|abandoning|"
        r"shorting|cashing\s+out|trimming|sold|dumped|fled|souring\s+on|soured\s+on|cooling\s+on|warming\s+up)\b"),
     _flows),
    # --- equity issuance
    ("offering", re.compile(
        r"\b(?:prices?|priced|pricing|announces?|announced|launch(?:es|ed)?|proposed|commences?|files?\s+for|"
        r"upsized?|registered\s+direct|at\s+the\s+market|atm|secondary|follow\s+on|underwritten|public|"
        r"common\s+stock|equity|share|stock|overnight|bought\s+deal|private|convertible|notes|ipo)\s+"
        r"(?:[^\s.;!?]+\s+){0,3}?(?:offering|placement)\b"), _offering),
    ("offering", re.compile(r"\b(?:offering|sale)\s+of\s+(?:[^\s.;!?]+\s+){0,3}?(?:shares|common\s+stock|"
                            r"ordinary\s+shares|ads|adss)\b"), _offering),
    # --- legal relief / resolution
    ("legal_relief", re.compile(
        r"\b(?:dismiss(?:es|ed)?|drop(?:s|ped)?|toss(?:es|ed)?|throw(?:s|n)?\s+out|threw\s+out|end(?:s|ed)?|"
        r"close[sd]?|suspend(?:s|ed)?|clear(?:s|ed)?|wins?|won|prevail(?:s|ed)?\s+in)\b(?:\s+[^\s.;!?]+){0,3}?\s+"
        r"(?:lawsuit|suit|case|probe|investigation|charges|complaint|inquiry|patent\s+(?:case|suit|trial))\b"),
     _fixed(0.7, "legal relief", "legal_relief")),
    ("legal_relief", re.compile(
        r"\b(?:lawsuit|suit|case|probe|investigation|charges|claims?|complaint|inquiry)\s+(?:was\s+|is\s+|were\s+|"
        r"has\s+been\s+|have\s+been\s+)?(?:dismissed|dropped|tossed|thrown\s+out|rejected|closed|ended|withdrawn)\b"),
     _fixed(0.7, "legal relief", "legal_relief")),
    ("settlement", re.compile(
        r"\b(?:settle[sd]?|settling|resolve[sd]?|resolving)\b(?:\s+[^\s.;!?]+){0,4}?\s+(?:class\s+action\s+)?"
        r"(?:lawsuits?|suits?|litigation|probes?|cases?|claims?|disputes?|charges|investigations?|allegations|"
        r"arbitration)\b"),
     _fixed(0.45, "settles lawsuit", "settlement")),
    # --- going concern / distress phrasing variants
    ("fine", re.compile(r"(?:[$€£¥]\s?)?\d[\d.,]*\s*(?:million|billion|mln|bln|mn|bn|m|b)?\s+(?:civil\s+)?"
                        r"(?:fine|penalty)\b"), _fixed(-0.7, "fine", "fine")),
    ("going_concern", re.compile(r"\b(?:substantial\s+)?doubt\s+about\s+(?:its\s+|the\s+company\s+)?ability\s+to\s+"
                                 r"continue\s+as\s+a\s+going\s+concern\b"),
     _fixed(-1.5, "going concern doubt", "going_concern")),
    ("bankruptcy", re.compile(r"\b(?:files?|filed|filing)\s+for\s+(?:chapter\s+(?:11|7)\s+)?(?:bankruptcy|"
                              r"insolvency|creditor\s+protection)\b"), _fixed(-1.8, "files for bankruptcy",
                                                                               "bankruptcy")),
    ("bankruptcy", re.compile(r"\bemerg(?:es|ed|ing)\s+from\s+(?:chapter\s+11|bankruptcy)\b"),
     _fixed(0.6, "emerges from bankruptcy", "bankruptcy")),
    ("swing", re.compile(
        r"\b(?:to|into)\s+(?:an?\s+)?(?:[^\s.;!?]+\s+){0,2}?(?:profit|profits|black|surplus|net\s+income)\s+"
        r"(?:of\s+[^,;.!?]{0,25}?\s+)?from\s+(?:an?\s+)?(?:[^\s.;!?]+\s+){0,2}?(?:loss|losses|deficit|red)\b"),
     _fixed(1.0, "swung to profit", "swing")),
    ("swing", re.compile(
        r"\b(?:to|into)\s+(?:an?\s+)?(?:[^\s.;!?]+\s+){0,2}?(?:loss|losses|deficit|red)\s+"
        r"(?:of\s+[^,;.!?]{0,25}?\s+)?from\s+(?:an?\s+)?(?:[^\s.;!?]+\s+){0,2}?(?:profit|profits|black|surplus|"
        r"net\s+income)\b"), _fixed(-1.0, "swung to loss", "swing")),
    ("superlative", re.compile(r"\bnever\s+been\s+(?P<deg>this|so|more|less)\s+(?P<w>bullish|bearish|optimistic|"
                               r"pessimistic|confident|negative|positive|cheap|expensive)\b"), _superlative),
    ("buy_dip", re.compile(r"\b(?:buy(?:s|ing)?|bought|add(?:s|ed|ing)?|load(?:s|ed|ing)?|scoop(?:s|ed|ing)?)\b"
                           r"(?:\s+[^\s.;!?]+){0,3}?\s+(?:the\s+|this\s+|any\s+|every\s+|on\s+)?"
                           r"(?:dip|dips|pullback|pullbacks|weakness|sell\s?off)\b"),
     _fixed(0.7, "buying the dip", "buy_dip")),
]
# Cheap pre-filter: a rule family runs only if one of its trigger substrings occurs.
_TRIGGERS: dict[str, tuple[str, ...]] = {
    "analyst": ("grade", " to ", " from "), "analyst_init": ("initiat", "start", "coverage", "launch", "resum",
                                                          "assum", "reinstat", "pick", "began", "begin"),
    "analyst_reiterate": ("reiterat", "maintain", "keep", "kept", "affirm", "retain", "repeat", "stick", "stay",
                          "remain"),
    "price_target": ("target", "pt", "tgt", "objective"),
    "beat": ("beat", "top", "exceed", "surpass", "crush", "smash", "trounce", "outstrip", "blow", "blew", "best",
             "clear", "outpac"),
    "miss": ("miss", "short", "undersh", "came in", "come in", "comes in", "trail", "lag", "meet", "match",
             "reach", "hit ", "beat", "top"),
    "above_exp": ("above", "ahead", "better", "exceeding", "topping", "beating", "stronger", "surpassing"),
    "below_exp": ("below", "under", "short", "worse", "weaker", "behind", "softer", "missing"),
    "vs_exp": ("than",), "inline": ("line", "match", "meet", "met "),
    "guidance": ("guid", "outlook", "forecast", "view", "projection", "target", "estimate", "expectation"),
    "flows": ("fund", "investor", "insider", "institution", "whale", "trader", "money", "billionaire", "manager",
              "activist"),
    "offering": ("offering", "placement", "sale of"), "legal_relief": ("suit", "case", "probe", "investigation",
                                                                        "charges", "claim", "complaint", "inquiry",
                                                                        "trial"),
    "settlement": ("settl", "resolv"), "going_concern": ("going concern",), "swing": (" from ",),
    "eps_loss": ("eps", "per share"), "job_cuts": ("job", "position", "worker", "employee", "staff", "role",
                                                   "headcount", "workforce"), "fine": ("fine", "penalty"),
    "bankruptcy": ("bankruptcy", "insolvency", "creditor protection", "chapter 11"), "superlative": ("never been",),
    "buy_dip": ("dip", "pullback", "weakness", "sell"),
}


def _trigger_index() -> tuple[re.Pattern[str], dict[str, frozenset[str]]]:
    """One overlapping scan finds every trigger; a match also implies its trigger prefixes."""
    fams: dict[str, set[str]] = {}
    for fam, trigs in _TRIGGERS.items():
        for t in trigs:
            fams.setdefault(t, set()).add(fam)
    closure = {t: frozenset().union(*(fams[p] for p in fams if t.startswith(p))) for t in fams}
    alts = sorted(fams, key=len, reverse=True)
    return re.compile("(?=(" + "|".join(map(re.escape, alts)) + "))"), closure


_TRIGGER_SCAN, _TRIGGER_FAMILIES = _trigger_index()


def find_rule_matches(rtext: str) -> list[RuleMatch]:
    """Run regex rules over hyphen-normalized lowercase text; non-overlapping, priority order."""
    out: list[RuleMatch] = []
    taken: list[tuple[int, int]] = []
    active: set[str] = set()
    for found in set(_TRIGGER_SCAN.findall(rtext)):
        active |= _TRIGGER_FAMILIES[found]
    for key, pattern, make in _RULES:
        if key in _TRIGGERS and key not in active:
            continue
        for m in pattern.finditer(rtext):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            rm = make(m)
            if rm is None:
                continue
            out.append(rm)
            taken.append((rm.start, rm.end))
    return out


# --------------------------------------------------------------------------- #
# Numeric comparisons ("sales $5.17 bln vs. $5.40 bln", "loss narrowed to EUR 1 mn from EUR 2 mn")
# --------------------------------------------------------------------------- #
_CMP = re.compile(
    rf"(?P<new>{_AMT})\s*(?:,\s*)?(?:(?P<dir>up|down|rising|falling|increasing|decreasing)\s+)?"
    rf"(?:vs\.?|versus|compared\s+(?:to|with)|against|from|as\s+against)\s+(?:the\s+|an?\s+|a\s+year\s+ago\s+)?"
    rf"(?:(?P<oldmetric>(?:[a-z-]+\s+){{0,2}}?(?:loss|losses|profit|profits|income|deficit))\s+(?:of\s+)?)?"
    rf"(?P<old>{_AMT})")
_CMP_FROM = re.compile(rf"\bfrom\s+(?P<old>{_AMT})\s+to\s+(?P<new>{_AMT})")
_CONSENSUS = re.compile(rf"(?:consensus|estimates?|expectations?|expected|est\.?|forecast|view)\s+(?:of\s+|was\s+|"
                        rf"for\s+|at\s+|is\s+)?(?P<est>{_AMT})|(?P<est2>{_AMT})\s+(?:consensus|est\b|estimate|"
                        rf"expected)")


def _numeric_hits(rtext: str, tokens: list[Token], at: list[Optional[_Span]], claimed: bytearray,
                  tok_of: Callable[[int], int]) -> list[Hit]:
    hits: list[Hit] = []
    if not any(t.kind in ("num", "pct") for t in tokens):
        return hits
    matches = [(m, "cmp") for m in _CMP.finditer(rtext)] + [(m, "from") for m in _CMP_FROM.finditer(rtext)]
    used_until = -1
    for m, kind in sorted(matches, key=lambda mk: mk[0].start()):
        if m.start() < used_until:
            continue
        new_raw, old_raw = m.group("new"), m.group("old")
        if _is_year(new_raw) or _is_year(old_raw) or not re.search(r"\d", new_raw) or not re.search(r"\d", old_raw):
            continue
        new, old = _amount(new_raw), _amount(old_raw)
        if new is None or old is None or new == old:
            continue
        s_tok, e_tok = tok_of(m.start()), tok_of(m.end() - 1) + 1
        if any(claimed[s_tok:e_tok]):
            continue
        # Skip when an explicit movement word already governs this comparison
        # ("rose to EUR 5 mn from EUR 4 mn", "down from") - composition handles it.
        if m.groupdict().get("dir") or _direction_before(at, s_tok, 5):
            continue
        if re.search(r"\b(?:range[sd]?|ranging|between|vary|varies|varying|spanning)\s*$", rtext[max(0, m.start() - 14):
                                                                                                m.start()]):
            continue  # "prices range from $799 to $1,099" is a span, not a change
        pol_new, msp = _metric_before(tokens, at, s_tok, 10)
        pol_old = pol_new
        oldmetric = m.groupdict().get("oldmetric")
        if oldmetric:
            pol_old = -1.0 if re.search(r"loss|deficit", oldmetric) else 1.0
        if pol_new == 0:
            pol_new = pol_old = 1.0
        diff = new * pol_new - old * pol_old
        if diff == 0:
            continue
        rel = abs(diff) / max(abs(old), 1e-9)
        val = math.copysign(0.55 + 0.45 * min(1.0, rel / 0.2), diff)
        term = f"{new_raw.strip()} vs {old_raw.strip()}"
        if msp is not None:  # the metric's own valence ("pre-tax loss") is now accounted for
            msp.used = True
            for j in range(msp.start, msp.end):
                claimed[j] = 1
        hits.append(Hit(s_tok, e_tok, val * 0.85, term, "rule:numbers"))
        used_until = m.end()
        # reported vs. consensus in the same sentence ("...; FactSet consensus $1.63")
        rest = re.split(r"[.!?](?:\s|$)", rtext[m.end(): m.end() + 70])[0]
        cm = _CONSENSUS.search(rest)
        if cm:
            est = _amount(cm.group("est") or cm.group("est2"))
            if est is not None and est != 0 and new != est:
                d2 = new * pol_new - est * pol_new
                v2 = math.copysign(0.5 + 0.4 * min(1.0, abs(d2) / abs(est) / 0.05), d2)
                cs = tok_of(m.end() + cm.start())
                ce = tok_of(m.end() + cm.end() - 1) + 1
                hits.append(Hit(cs, ce, v2 * 0.7, f"vs consensus {(cm.group('est') or cm.group('est2')).strip()}",
                                "rule:numbers"))
                used_until = m.end() + cm.end()
    return hits


def _direction_before(at: list[Optional[_Span]], i: int, window: int) -> bool:
    for j in range(max(0, i - window), i):
        sp = at[j]
        if sp is not None and sp.direction is not None:
            return True
    return False


def _metric_before(tokens: list[Token], at: list[Optional[_Span]], i: int,
                   window: int) -> tuple[float, Optional[_Span]]:
    """Polarity (+1/-1, 0 if none) and span of the nearest metric before token ``i``."""
    for j in range(i - 1, max(-1, i - window - 1), -1):
        if tokens[j].kind == "sep":
            break
        sp = at[j]
        if sp is not None and sp.metric is not None:
            return (1.0 if sp.metric.polarity > 0 else -1.0), sp
    return 0.0, None


# --------------------------------------------------------------------------- #
# Composition: direction x metric
# --------------------------------------------------------------------------- #
_BE = frozenset({"is", "are", "was", "were", "be", "been", "now", "still", "already", "currently", "closed",
                 "close", "closes", "ends", "ended", "trading", "trades", "traded", "opened", "opens", "finished",
                 "stock", "shares", "futures"})
_SECOND_ROUND = re.compile(r"(?:a\s+|the\s+|another\s+)?(?:second|third|fourth|fifth|sixth|\d+(?:st|nd|rd|th)|"
                           r"straight|consecutive|another|\d+)")
_UPDOWN_NEXT = frozenset({"premarket", "pre", "pre-market", "after", "on", "as", "since", "sharply", "slightly",
                          "big", "nearly", "almost", "more", "over", "from", "today", "again", "significantly",
                          "yoy", "ytd", "points", "pts", "bps", "strongly", "modestly", "double", "triple",
                          "in", "for", "this", "so", "about", "around", "roughly", "following", "despite", "amid",
                          "ahead", "versus", "vs", "year", "week", "month", "quarter", "overnight", "early", "late",
                          "midday", "afternoon", "intraday", "big", "hard", "considerably"})
_LEVEL_TRIGGERS = frozenset({"hit", "hits", "hitting", "touch", "touches", "touched", "reach", "reaches", "reached",
                             "at", "near", "to", "new", "fresh", "set", "sets", "notch", "notches", "notched",
                             "record", "all", "time", "week", "year", "years", "month", "months", "decade",
                             "session", "multi", "lifetime", "since"})
_FROMISH = frozenset({"from", "off", "after"})
_RECORD_NEXT = frozenset({"year", "quarter", "results", "month", "week", "day", "run", "close", "closing", "levels",
                          "level", "territory", "sales", "numbers", "performance"})
_CONNECTORS = frozenset({"in", "of", "for", "at", "to"})
_FILLER = frozenset({"the", "a", "an", "its", "their", "his", "her", "our", "this", "that", "these", "those",
                     "full", "year", "annual", "quarterly", "fiscal", "first", "second", "third", "fourth",
                     "quarter", "q1", "q2", "q3", "q4", "h1", "h2", "fy", "company", "group", "total", "overall",
                     "adjusted", "adj", "comparable", "global", "us", "domestic", "international", "core",
                     "organic", "underlying", "reported", "consolidated", "its", "own", "per", "share", "and"})


def _pct_magnitude(p: float) -> float:
    """Map |percent move| to a magnitude multiplier (0.3% ~ 0.65, 1% ~ 0.75, 5% ~ 1.05, 12% ~ 1.3, 40% ~ 1.6)."""
    p = abs(p)
    return min(1.7, (0.45 + 0.55 * math.log1p(p) / math.log1p(10)) / 0.8)


def _find_metric(tokens: list[Token], at: list[Optional[_Span]], i0: int, step: int, limit: float, *,
                 soft_ok: bool = False, stop_words: frozenset[str] = frozenset(),
                 through_moves: bool = False) -> Optional[_Span]:
    """Scan from token ``i0`` in direction ``step`` for a metric span within ``limit`` content words.

    Stops at hard boundaries, other movement words, contrast markers and (unless
    ``soft_ok``) commas/colons, so "costs surge, shares tumble" pairs correctly."""
    seen = 0.0
    j = i0
    n = len(tokens)
    while 0 <= j < n and seen < limit:
        t = tokens[j]
        if t.kind == "sep" or (t.kind == "soft" and not soft_ok):
            return None
        if t.kind == "soft":
            seen += 1
            j += step
            continue
        sp = at[j]
        if sp is not None:
            if sp.metric is not None and not sp.neutral:
                if step > 0:  # the head of a compound moves: "(slower) revenue growth"
                    head = at[sp.end] if sp.end < n else None
                    if head is not None and head.key in lx.METRIC_DIRECTIONS:
                        sp = head
                return sp
            if sp.direction is not None and sp.metric is None and sp.key not in lx.FOOTPRINT_VERBS \
                    and not sp.level_qual and sp.key not in ("record", "records") and not through_moves:
                return None
            if sp.contrast is not None:
                return None
            j = sp.end if step > 0 else sp.start - 1
            seen += 1
            continue
        if t.text in stop_words:
            return None
        if t.kind == "w" and t.text not in _FILLER:
            seen += 1
        elif t.kind in ("num", "pct", "cur", "tag"):
            seen += 0.5  # amounts sit between metric and verb ("sales of EUR 5 mn rose")
        j += step
    return None


def _nearby_pct(tokens: list[Token], start: int, end: int) -> Optional[int]:
    """Index of a percent token right after (preferred) or just before a movement word, same clause."""
    n = len(tokens)
    for j in range(end, min(n, end + 4)):
        t = tokens[j]
        if t.kind == "pct":
            return j
        if t.kind == "sep" or (t.kind == "w" and t.text not in ("by", "of", "nearly", "almost", "about", "over",
                                                                  "more", "than", "some", "around", "roughly",
                                                                  "a", "an", "another", "further", "up", "down",
                                                                  "to", "as", "much", "least", "at")):
            break
    for j in range(start - 1, max(-1, start - 3), -1):
        t = tokens[j]
        if t.kind == "pct":
            return j
        if t.kind == "sep":
            break
    return None


def _level_ok(tokens: list[Token], at: list[Optional[_Span]], sp: _Span) -> tuple[bool, float]:
    """Is "high"/"low" a market extreme here? Returns (valid, strength multiplier).

    "hits 52-week low" / "record highs" / "lowest since 2009" count; "from record
    highs" / "after hitting a record" are the starting point, not the news."""
    qualified = 1.0
    for j in range(max(0, sp.start - 5), sp.start):
        if tokens[j].text in _FROMISH:
            return False, 0.0
    for j in range(max(0, sp.start - 3), sp.start):
        t = tokens[j]
        other = at[j]
        if other is not None and other.level_qual:
            qualified = max(qualified, 2.2 if other.key in ("record", "all time", "alltime", "52 week", "lifetime",
                                                            "multi year", "historic") else 1.8)
        elif t.kind == "num" and j + 1 < len(tokens) and tokens[j + 1].text in ("week", "year", "month", "day"):
            qualified = max(qualified, 2.0)
    for j in range(sp.end, min(len(tokens), sp.end + 3)):
        t = tokens[j]
        if t.text in ("since", "in", "of") and j + 1 < len(tokens) and (
                tokens[j + 1].kind == "num" or tokens[j + 1].text in ("years", "decades", "a", "more", "over",
                                                                      "nearly", "almost", "two", "three", "five",
                                                                      "ten", "months", "weeks", "the")):
            qualified = max(qualified, 1.8)
            break
    if sp.key in ("high", "low") and qualified == 1.0:
        prev = tokens[sp.start - 1].text if sp.start > 0 else ""
        if prev not in _LEVEL_TRIGGERS:
            return False, 0.0
    return True, qualified


_NOUN_STOPS = frozenset({"of", "about", "over", "on", "for", "against", "after", "since", "following", "amid",
                         "before", "despite", "as", "from"})
_PREPS = frozenset({"on", "in", "for", "at", "after", "as", "amid", "to", "from", "with", "by", "of", "over",
                    "since", "into", "despite", "than"})
# "show up", "clean up", "break down", "end up": particles of phrasal verbs, not price moves
_PHRASAL_PREV = frozenset("""
show shows showed showing clean cleans cleaned cleaning prop props propped propping team teams teamed sign
signs signed set sets setting end ends ended ending wind winds wound back backs backed follow follows followed
line lines lined shake shakes shook open opens opened opening make makes made making take takes took taking
catch catches caught catching keep keeps kept keeping look looks looked looking sum sums summed wrap wraps
wrapped bring brings brought come comes came coming turn turns turned turning put puts call calls called beef
beefs beefed gear gears geared gearing dig digs dug mix mixes mixed pile piles piled rack racks racked rake
rakes raked rev revs revved scoop scoops scooped snap snaps snapped stock stocks stocked hold holds held
holding break breaks broke broken breaking water watered track tracked narrow calm boil hunker lay lays laid
trickle tone settle settles settled bog bogged shut crack write writes wrote mark marks marked double doubles
doubled doubling tie ties tied sit sits sat stand stands stood live lives lived sell sells sold buy buys
bought use uses used wake wakes woke waking speak speaks spoke cozy gobble eat eats ate dress dressed pay
pays paid chalk chalked cough coughed add adds added adding step steps stepped stepping shore shores shored
buck bucks bucked hike hikes hiked ramp ramps ramped boot boots booted give gives gave giving
""".split())
_RECORD_PREV = frozenset({"new", "fresh", "hit", "hits", "hitting", "at", "to", "set", "sets", "reach", "reaches",
                          "reached", "notch", "notches", "notched", "close", "closes", "closed", "another", "all"})


def _attach(sp: _Span, tokens: list[Token], at: list[Optional[_Span]]) -> Optional[tuple[Optional[_Span], float]]:
    """Find the metric a movement word moves. None = not a movement here (skip)."""
    d = sp.direction
    assert d is not None
    n = len(tokens)
    nxt = tokens[sp.end] if sp.end < n else None
    prev = tokens[sp.start - 1] if sp.start > 0 else None
    right = at[sp.end] if sp.end < n else None
    if d.pos == "l":  # high / low / highs / lowest ...
        ok, mult = _level_ok(tokens, at, sp)
        if not ok:
            if sp.key in ("high", "low") and right is not None and right.metric is not None:
                return right, 1.0  # adjective: "high debt", "low costs"
            if sp.key in ("high", "low") and (nxt is None or nxt.kind in ("sep", "soft")):
                metric = _find_metric(tokens, at, sp.start - 1, -1, 3)  # predicate: "recession risk is low"
                return (metric, 1.6) if metric is not None else None
            return None
        # "claims fall to lowest since April": the extreme belongs to the verb's subject
        return _find_metric(tokens, at, sp.start - 1, -1, 7, through_moves=True), mult
    if d.pos == "q":  # needs a quantity right after: "production contracts 3.8%"
        nums = [t for t in tokens[sp.end: sp.end + 2] if t.kind in ("pct", "num")]
        return (_find_metric(tokens, at, sp.start - 1, -1, 4), 1.0) if nums else None
    if sp.key in ("record", "records"):
        if right is not None and right.direction is not None and right.direction.pos == "l":
            return None  # "record high": qualifier only
        metric = _find_metric(tokens, at, sp.end, 1, 2) if sp.key == "record" else None
        if metric is not None:
            return metric, 1.0
        before = {t.text for t in tokens[max(0, sp.start - 3): sp.start]}
        if before & _RECORD_PREV and not any(t.text in _FROMISH for t in tokens[max(0, sp.start - 5): sp.start]):
            return _find_metric(tokens, at, sp.start - 1, -1, 4), 1.1
        if nxt is not None and nxt.text in _RECORD_NEXT:
            return None if sp.key == "records" else (None, 0.75)
        return None
    if d.pos == "p":  # up / down
        if prev is not None and prev.text in _PHRASAL_PREV:
            return None
        metric = _find_metric(tokens, at, sp.start - 1, -1, 3, soft_ok=True)
        if nxt is not None and nxt.text == "for":
            tail = " ".join(t.text for t in tokens[sp.end + 1: sp.end + 4])
            return (metric, 1.0) if _SECOND_ROUND.match(tail) else None
        ctx = (nxt is not None and (nxt.kind in ("pct", "num", "cur") or nxt.text in _UPDOWN_NEXT)) or (
            prev is not None and (prev.text in _BE or prev.kind in ("tag", "soft", "sep")))
        if metric is None and (not ctx or (nxt is not None and nxt.text in ("to", "with", "and"))):
            return None
        return metric, 1.0
    if d.pos == "n":
        metric = None
        if nxt is not None and nxt.text in _CONNECTORS:
            after = tokens[sp.end + 1] if sp.end + 1 < n else None
            if after is not None and after.kind not in ("pct", "num", "cur"):  # not "decrease of 25.7%"
                metric = _find_metric(tokens, at, sp.end + 1, 1, 3)
        if metric is None:  # compound: "dividend cut", "sales growth"
            metric = _find_metric(tokens, at, sp.start - 1, -1, 1)
        if metric is None and sp.key in lx.HOMOGRAPHS:  # verb reading: "cut existing tariffs", "sales fall"
            return _attach_verb(sp, tokens, at, nxt, prev)
        if metric is None:  # subject: "sales posted a 5% increase" (not "fears of a selloff")
            metric = _find_metric(tokens, at, sp.start - 1, -1, 5, stop_words=_NOUN_STOPS)
        return metric, 1.0
    if d.pos == "a":
        metric = None
        if nxt is not None and nxt.text not in _PREPS:  # "higher raw material costs", not "lower on concerns"
            metric = _find_metric(tokens, at, sp.end, 1, 3, stop_words=_PREPS)
        if metric is None and d.default > 0:
            metric = _find_metric(tokens, at, sp.start - 1, -1, 4)
        return metric, 1.0
    return _attach_verb(sp, tokens, at, nxt, prev)


def _attach_verb(sp: _Span, tokens: list[Token], at: list[Optional[_Span]], nxt: Optional[Token],
                 prev: Optional[Token]) -> Optional[tuple[Optional[_Span], float]]:
    """Verb attachment: object first for transitive verbs, else the subject (across appositives)."""
    key = sp.key
    if key.endswith("ed") and prev is not None and prev.text in _PREPS | {"a", "an", "the", "its", "their"} \
            and nxt is not None and nxt.kind == "w" and at[sp.end] is None and nxt.text not in _PREPS:
        return None  # participle used as adjective: "from narrowed focus", "a reduced stake"
    if key in lx.FOOTPRINT_VERBS:
        metric = _find_metric(tokens, at, sp.end, 1, 3)
        return (metric, 1.0) if metric is not None and metric.key in lx.FOOTPRINT_METRICS else None
    if key in lx.TREND_ONLY:
        metric = _find_metric(tokens, at, sp.end, 1, 3)
        return (metric, 1.0) if metric is not None and metric.key in lx.TREND_METRICS else None
    metric = None
    if key in lx.TRANSITIVE or key.split(" ")[0] in lx.TRANSITIVE:  # object: "cut existing tariffs"
        metric = _find_metric(tokens, at, sp.end, 1, 3, stop_words=_PREPS)
    gerund_complement = key.endswith("ing") and prev is not None and prev.text in ("of", "to", "for", "about")
    if metric is None and not gerund_complement:
        # subject, possibly across an appositive: "Operating profit, excluding X, rose"
        metric = _find_metric(tokens, at, sp.start - 1, -1, 6, soft_ok=True)
    if metric is None and nxt is not None and nxt.text in ("in", "of") and sp.end + 1 < len(tokens) \
            and tokens[sp.end + 1].kind not in ("pct", "num", "cur"):  # "rise in sales", not "decrease of 25%"
        metric = _find_metric(tokens, at, sp.end + 1, 1, 3)
    return metric, 1.0


def _numeric_change(tokens: list[Token], i: int) -> Optional[float]:
    """Percent change from "to EUR 13.1 mn from EUR 8.7 mn" / "from 5 to 7" after a movement word."""
    vals: dict[str, float] = {}
    j, n = i, len(tokens)
    while j < n and j < i + 12 and tokens[j].kind != "sep":
        t = tokens[j]
        if t.text in ("to", "from") and t.text not in vals:
            for k in range(j + 1, min(n, j + 4)):
                if tokens[k].kind == "num":
                    if not (1900 <= tokens[k].value <= 2100 and float(tokens[k].value).is_integer()):
                        vals[t.text] = tokens[k].value
                    break
                if tokens[k].kind not in ("cur", "w") or tokens[k].text in ("to", "from"):
                    break
        j += 1
    new, old = vals.get("to"), vals.get("from")
    if new is None or old is None or old == 0:
        return None
    return abs(new - old) / abs(old) * 100.0


def _compose(tokens: list[Token], at: list[Optional[_Span]], spans: list[_Span]) -> list[Hit]:
    """Pair movement words with the metric they move; emit signed hits."""
    pairs: list[tuple[_Span, Optional[_Span], float]] = []
    consumed: set[int] = set()
    for sp in spans:
        if sp.direction is None or sp.neutral:
            continue
        found = _attach(sp, tokens, at)
        if found is None:
            continue
        metric, mult = found
        if metric is sp:
            metric = None
        if metric is not None:
            consumed.add(id(metric))
        pairs.append((sp, metric, mult))

    hits: list[Hit] = []
    for sp, metric, mult in pairs:
        d = sp.direction
        assert d is not None
        if sp.metric is not None and id(sp) in consumed and sp.key in lx.METRIC_DIRECTIONS:
            continue  # "growth" moved by "slowed": the pair carries it
        sign = d.sign
        if metric is None:
            if d.default == 0.0:
                continue
            pol, base = 1.0, d.strength * d.default
        else:
            pol, base = (1.0 if d.fixed else metric.metric.polarity), d.strength  # type: ignore[union-attr]
            if sign > 0 and sp.key.startswith("lift") and metric.key in lx.LIFTABLE:
                sign = -1  # "lifts tariffs" removes them
        pi = _nearby_pct(tokens, sp.start, sp.end)
        change = None if pi is not None else _numeric_change(tokens, sp.end)
        if pi is not None:
            base *= _pct_magnitude(tokens[pi].value)
        elif change is not None:
            base *= _pct_magnitude(change)
        elif d.pos == "p":
            base *= 1.0 if (sp.end < len(tokens) and tokens[sp.end].kind in ("num", "cur")) else 0.85
        val = sign * pol * base * mult
        if abs(val) < 1e-6:
            continue
        parts = [(sp.start, sp.end)]
        if metric is not None:
            parts.append((metric.start, metric.end))
        if pi is not None:
            parts.append((pi, pi + 1))
        parts.sort()
        lo, hi = parts[0][0], max(e for _, e in parts)
        hits.append(Hit(lo, hi, max(-2.0, min(2.0, val)), "", "move", anchor=sp.start, parts=tuple(parts)))
        sp.used = True
        if metric is not None:
            metric.used = True
    return hits


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
# Modifier strengths (tuned on the Twitter-financial train split).
NEGATION_FLIP = 0.65  # negated positive flips and shrinks ("not strong")
NEGATION_FLIP_NEG = 0.8  # negated negative = relief ("won't hinder growth"), capped at NEGATED_CAP
NEGATED_CAP = 1.0  # "avoids bankruptcy" is relief, not euphoria
SHIFT_BEFORE, SHIFT_AFTER = 0.55, 1.2  # "A but B": B is what matters
CONCESSIVE_FACTOR = 0.3  # "despite A, B": A is background
BACKGROUND_FACTOR = 0.4  # "B after A" / "B amid A"
QUESTION_FACTOR = 0.25  # "Is X a buy?" asks, it does not assert
LISTICLE_FACTOR = 0.6  # "5 stocks to buy now" is marketing, not news
TEXT_HEDGE_FACTOR = 0.85  # "- report", "sources say"

_HYPHEN_IN_WORD = re.compile(r"(?<=[a-z0-9])-(?=[a-z])|(?<=[a-z])-(?=[0-9])")
_LISTICLE = re.compile(r"^\W*(?:the\s+|these\s+)?(?:top\s+)?\d{1,2}\s+(?:[a-z'-]+\s+){0,3}?(?:stocks?|reasons?|"
                       r"things|ways|picks|names|companies|etfs?|funds|charts|shares|tips|signs|questions|lessons|"
                       r"takeaways|mistakes|risks|myths|facts|secrets|rules|habits|trends|ideas|plays|bets|winners|"
                       r"losers|buys|dividend\w*|growth\s+stocks|tech\s+stocks)\b|"
                       r"\bstocks?\s+to\s+(?:buy|watch|sell|avoid|own|consider)\b|\bbest\s+[a-z-]*\s*stocks\b")


def extract(text: str, *, social: bool = False) -> Evidence:
    """Find every piece of sentiment evidence in ``text`` and apply modifiers."""
    norm = normalize(text)
    low = norm.lower()
    if len(low) != len(norm):  # exotic casing changed length; keep offsets consistent
        norm = low
    tokens = tokenize(low)
    ev = Evidence(text=norm, tokens=tokens)
    n = len(tokens)
    if not n:
        return ev
    rtext = _HYPHEN_IN_WORD.sub(" ", low) if "-" in low else low
    starts = [t.start for t in tokens]

    def tok_of(char: int) -> int:
        lo, hi = 0, n - 1
        while lo < hi:  # last token starting at or before char
            mid = (lo + hi + 1) // 2
            if starts[mid] <= char:
                lo = mid
            else:
                hi = mid - 1
        return lo

    claimed = bytearray(n)
    hits: list[Hit] = []
    for rm in find_rule_matches(rtext):
        s, e = tok_of(rm.start), tok_of(max(rm.start, rm.end - 1)) + 1
        for j in range(s, e):
            claimed[j] = 1
        if rm.valence:
            hits.append(Hit(s, e, rm.valence, rm.term, f"rule:{rm.key}"))

    at = _match_spans(tokens, claimed)
    hits.extend(_numeric_hits(rtext, tokens, at, claimed, tok_of))
    for h in hits:
        for j in range(h.start, h.end):
            claimed[j] = 1
    # spans may overlap rule claims only partially; drop those
    spans: list[_Span] = []
    seen: set[int] = set()
    for sp in at:
        if sp is not None and id(sp) not in seen:
            seen.add(id(sp))
            if not any(claimed[sp.start:sp.end]):
                spans.append(sp)
            else:
                for j in range(sp.start, sp.end):
                    at[j] = None

    hits.extend(_compose(tokens, at, spans))
    for sp in spans:
        if sp.used or sp.neutral:
            continue
        if social and sp.social_only is not None:  # social register overrides ("bulls" = the other camp)
            if sp.social_only:
                hits.append(Hit(sp.start, sp.end, sp.social_only, "", "social"))
        elif sp.lex is not None and sp.lex != 0.0:
            hits.append(Hit(sp.start, sp.end, sp.lex, "", "lex"))
        elif sp.metric is not None and sp.metric.intrinsic and sp.direction is None:
            hits.append(Hit(sp.start, sp.end, sp.metric.intrinsic, "", "metric"))
        if sp.hedge is not None:
            ev.hedges += 1
        if sp.litigious:
            ev.litigious += 1
    # signed percent tokens not attached to any movement word: "$AQST (+13.1% pre)", "Varonis -6%"
    covered = bytearray(n)
    for h in hits:
        for j in range(h.start, h.end):
            covered[j] = 1
    for i, t in enumerate(tokens):
        if t.kind == "pct" and t.signed and not covered[i] and not claimed[i] and t.value:
            hits.append(Hit(i, i + 1, math.copysign(0.75 * _pct_magnitude(t.value), t.value), "", "move"))

    _apply_modifiers(ev, tokens, at, spans, hits, low)
    for h in hits:
        if not h.term:
            h.term = _term(norm, tokens, h)
    ev.hits = hits
    return ev


def _term(norm: str, tokens: list[Token], h: Hit) -> str:
    """Driver label: the exact text when short (so UIs can highlight it), else "metric verb [pct]"."""
    if h.end - h.start <= 5 or not h.parts:
        return norm[tokens[h.start].start: tokens[h.end - 1].end]
    words = [norm[tokens[a].start: tokens[b - 1].end] for a, b in h.parts]
    if h.negated and h.start < h.parts[0][0]:
        words.insert(0, norm[tokens[h.start].start: tokens[h.start].end])
    return " ".join(words).lower()


def _apply_modifiers(ev: Evidence, tokens: list[Token], at: list[Optional[_Span]], spans: list[_Span],
                     hits: list[Hit], low: str) -> None:
    n = len(tokens)
    # sentence ids (hard boundaries) and clause ids (soft boundaries too)
    sent = [0] * n
    sid = 0
    question_sents: set[int] = set()
    for i, t in enumerate(tokens):
        sent[i] = sid
        if t.kind == "sep":
            if "?" in t.text:
                question_sents.add(sid)
            sid += 1
    # a trailing question without "?" separators: "Is X a buy" (headline style, no punctuation)
    first = tokens[0].text
    if first in lx.QUESTION_STARTERS and first not in ("why", "how", "what", "who", "when", "where", "which") \
            and sent[-1] == 0 and n > 2 and tokens[1].kind in ("w", "tag"):
        question_sents.add(0)
    ev.question = bool(question_sents)
    ev.listicle = bool(_LISTICLE.search(low))
    ev.text_hedge = any(sp.key in lx.TEXT_HEDGES for sp in spans)
    if _ATTRIBUTION.search(low):
        ev.text_hedge = True

    negators = [sp for sp in spans if sp.negator and not sp.neutral]
    _absorb_intensifying_adjectives(tokens, spans, hits)
    background = _background_clauses(tokens)
    hedge_spans = [sp for sp in spans if sp.hedge is not None]
    intens_spans = [sp for sp in spans if sp.intens is not None]
    contrast_spans = [sp for sp in spans if sp.contrast is not None]

    for h in hits:
        w = 1.0
        # --- negation: a negator up to 3 word tokens before the hit, same clause
        neg = _negator_for(tokens, at, negators, h, hits)
        if neg is not None:
            flipped = -(NEGATION_FLIP if h.valence > 0 else NEGATION_FLIP_NEG) * h.valence
            h.valence = max(-NEGATED_CAP, min(NEGATED_CAP, flipped))
            h.negated = True
            neg.used = True
            h.start = min(h.start, neg.start)
        # --- intensifiers / diminishers adjacent (2 before, 2 after)
        a = h.anchor
        for sp in intens_spans:
            if (h.start - 2 <= sp.start < h.start or h.end <= sp.start < h.end + 2) and sent[sp.start] == sent[a]:
                if not any(tokens[j].kind in ("sep", "soft") for j in range(min(sp.start, h.end), max(sp.start, h.end))):
                    w *= sp.intens  # type: ignore[operator]
        # --- hedges up to 6 tokens before, same sentence
        for sp in hedge_spans:
            if a - 6 <= sp.start < a and sent[sp.start] == sent[a]:
                w *= sp.hedge  # type: ignore[operator]
                break
        # --- contrast
        for sp in contrast_spans:
            if sent[sp.start] != sent[a]:
                continue
            if sp.contrast == "shift":
                if sp.key == "yet" and sp.start > 0 and tokens[sp.start - 1].text in ("not", "no", "nor"):
                    continue
                if sp.key == "though" and sp.start > 0 and tokens[sp.start - 1].text == "even":
                    continue
                w *= SHIFT_BEFORE if a < sp.start else SHIFT_AFTER
            else:  # concessive: the clause it introduces is background
                end = _clause_end(tokens, sp.end)
                if sp.end <= a < end:
                    w *= CONCESSIVE_FACTOR
        if background[a]:
            w *= BACKGROUND_FACTOR
        if sent[a] in question_sents:
            w *= QUESTION_FACTOR
        if ev.listicle:
            w *= LISTICLE_FACTOR
        if ev.text_hedge:
            w *= TEXT_HEDGE_FACTOR
        h.weight = min(1.8, w)

    # negators that negated nothing but carry meaning themselves ("fails to meet")
    for sp in negators:
        if not sp.used and sp.key in lx.NEGATOR_FALLBACK:
            hits.append(Hit(sp.start, sp.end, lx.NEGATOR_FALLBACK[sp.key], "", "lex"))


def _background_clauses(tokens: list[Token]) -> list[bool]:
    """Tokens inside "after ..." / "following ..." / "amid ..." clauses: context for the main
    move ("Gold falls after strong jobs report"), so they count less than the headline verb."""
    n = len(tokens)
    flags = [False] * n
    for i, t in enumerate(tokens):
        if t.text == "from" and i > 0 and tokens[i - 1].text in _RECOVERY_WORDS:
            for j in range(i + 1, _clause_end(tokens, i + 1)):  # "bounces back from a steep sell-off"
                flags[j] = True
            continue
        if t.text in ("after", "following", "amid", "amidst") and i > 0:
            nxt = tokens[i + 1].text if i + 1 < n else ""
            if nxt in ("hours", "market", "the", "close") and (i + 2 >= n or tokens[i + 2].text in ("bell", "close")
                                                                or nxt in ("hours", "market", "close")):
                continue  # "after hours", "after the bell"
            for j in range(i + 1, _clause_end(tokens, i + 1)):
                flags[j] = True
    return flags


_RECOVERY_WORDS = frozenset({"back", "rebounds", "rebound", "rebounded", "rebounding", "recovers", "recovered",
                             "recover", "recovering", "emerges", "emerged", "bounces", "bounced", "bouncing",
                             "rallies", "rallied", "rises", "rose", "climbs", "climbed", "up", "higher", "lower",
                             "down", "falls", "fell", "slips", "slipped", "retreats", "retreated", "pulls",
                             "away"})
_ATTRIBUTION = re.compile(r"^(?:report|sources|rumou?r)s?\s*:|[-\u2013\u2014(]\s*(?:report|sources|rumou?r)s?\s*\)?\s*$|"
                          r"\b(?:report|reports)\s+say|\bsources\s+(?:say|said|tell|told)\b")
_INTENSIFYING_ADJ = frozenset({"strong", "stronger", "robust", "solid", "steady"})


def _absorb_intensifying_adjectives(tokens: list[Token], spans: list[_Span], hits: list[Hit]) -> None:
    """"strong headwinds" / "solid losses": the adjective intensifies the negative noun."""
    by_start = {h.start: h for h in hits}
    drop: list[Hit] = []
    for h in hits:
        if h.source != "lex" or h.valence <= 0 or h.end - h.start != 1:
            continue
        if tokens[h.start].text not in _INTENSIFYING_ADJ:
            continue
        nxt = by_start.get(h.end)
        if nxt is not None and nxt.valence < 0:
            nxt.valence *= 1.25
            drop.append(h)
    for h in drop:
        hits.remove(h)


def _clause_end(tokens: list[Token], i: int) -> int:
    for j in range(i, len(tokens)):
        if tokens[j].kind in ("sep", "soft"):
            return j
    return len(tokens)


def _negator_for(tokens: list[Token], at: list[Optional[_Span]], negators: list[_Span], h: Hit,
                 hits: list[Hit]) -> Optional[_Span]:
    """The negator scoping over ``h``: the nearest one up to 3-4 words before, same clause, that has
    not negated an earlier hit - unless ``h`` is that hit's "of" complement ("not guilty of fraud")."""
    best: Optional[_Span] = None
    target = h.anchor if h.start <= h.anchor < h.end else h.start
    for sp in negators:
        if sp.end > target or (sp.start >= h.start and sp.end > h.end):
            continue
        if sp.used and not any(o.negated and o.start == sp.start and o.end + 1 == h.start
                               and tokens[o.end].text in ("of", "in") for o in hits):
            continue
        gap = 0
        ok = True
        for j in range(sp.end, target):
            t = tokens[j]
            if t.kind in ("sep", "soft") or (at[j] is not None and at[j].contrast is not None):  # type: ignore[union-attr]
                ok = False
                break
            if t.kind == "w":
                gap += 1
        limit = 4 if sp.key in ("unlikely", "no longer", "without", "never") else 3
        if ok and gap <= limit and (best is None or sp.start > best.start):
            best = sp
    return best
