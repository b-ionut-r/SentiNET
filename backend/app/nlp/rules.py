"""Evidence extraction for the Sentinel engine: *what* in a text carries sentiment.

`extract(text, social=...)` turns one text into an `Evidence` bundle of signed
`Hit`s (each a meaningful term/phrase with a valence) after applying modifiers.
`app.nlp.engine` turns that evidence into a score. Layers, in priority order:

1. Regex rules for high-precision finance constructions, with numbers:
   analyst rating changes ("cut to Neutral from Buy"), price-target moves
   ("PT raised to $54 from $50" - direction from the numbers), earnings
   beats/misses, guidance raises/cuts (signed by *what* is forecast: a higher
   loss or inflation forecast is bad news), reported-vs-expected figures
   ("EPS $1.66 vs. $1.58", "EPS 74 cents; consensus 86 cents"), fund flows,
   options flow, insider trades (Form-4 pay and plan transactions are not
   trades), regulatory approvals, rejections and enforcement, charges, equity
   offerings, legal relief, contract wins, index changes, trend endings, the
   other camp's pain ("bears getting destroyed"); 13F bot headlines ("Vanguard
   Group Inc. Raises Stake in X") are recognized and neutral.
   A rule's word gap never crosses a clause, negator or second verb
   ("Lower Despite Strong Growth Outlook", "FDA did not approve"), and a
   negator inside a rule's span still negates it.
2. Lexicon phrases (longest match first), incl. neutralizers that block
   false friends ("shares outstanding", "Best Buy", "small-cap", "of record").
3. Composition: movement words take their sign from what moved
   ("costs surge" < 0, "loss narrowed" > 0, "shares tumble 12%" << "dips 1%",
   "inflation lighter than expected" > 0, "wipe $5T from GDP" < 0, "rally
   wiped out" < 0, "VIX +15%" < 0). Transitive uses with a non-metric object
   ("Google drops plan", "opens up a way") are not price moves, and a subject
   search never crosses a relative clause or a comma clause with its own
   subject ("Rate Fears, NVIDIA Rises 3%").
4. Modifiers: negation scope ("not", "fails to", "won't", "avoids"), contrast
   ("but" up-weights the later clause, "despite X" down-weights X), context
   clauses ("after/amid/as ...", "while the market ...", purpose "to ..." and
   ", which ..." asides count less and never overturn the headline verb),
   hedges ("may", "reportedly"), intensifiers (adjectives only forward:
   "PT cut on limited upside"), questions ("Is X a buy?" asks; "Why is X
   down?" presupposes the drop) and listicles.
5. Optional subject attribution (``target``): a move or results event whose
   subject is another company ("SoFi falls 3%; Affirm drops 4%") counts 0.3x.

In the social register, trader words ("long", "calls", "buying") only count
in trading talk (a cashtag, an amount, trading vocabulary) and in a position
reading ("long $TSLA", not "for a long time").

Everything is deterministic, regex/dict based and fast (no models).
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from collections.abc import Callable, Sequence

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
_ZW = re.compile("[\u200b-\u200f\u2060\ufeff\ufe0e\ufe0f]")
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
   |(?P<xtick>[a-z]{1,6}\.(?:ax|to|hk|ss|sz|pa|de|sw|mi|as|br|st|ol|he|ks|kq|ns|bo|sa|mx|nz|si|v|l|t)\b(?!\.))
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
        elif kind == "xtick":  # exchange-suffixed ticker ("NCM.AX", "SHOP.TO"): a name, never a word
            out.append(Token(raw, s, e, "tag"))
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
    display: str = ""  # driver text shown to users (verbatim span when short; else ``term``)

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
    neutral_cues: int = 0  # explicit "no news" evidence: "in line with estimates", "unchanged", debt offerings


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
        r"consensus|forecasts?|views?|projections?|(?<!\bto\s)targets?(?!'|\s+(?:co|corp|corporation|inc|stores?|brands?)\b)|"
        r"guidance|est|street|analysts?|whisper\w*|"
        r"predictions?|outlooks?|expected)")
_GUID_PREFIX = (r"(?:full\s+year\s+|annual\s+|fy\s*\d*\s+|\d{4}\s+|q[1-4]\s+|quarterly\s+|sales\s+|revenue\s+|"
                r"profit\s+|earnings\s+|eps\s+)")
# a bare "target(s)" is a company name as often as a goal ("compete with Target"): only with a metric
_GUID = (rf"(?:{_GUID_PREFIX}?(?:guidance|outlook|forecasts?|view|guide|projections?|estimates?|expectations?)|"
         rf"(?:{_GUID_PREFIX}|growth\s+|margin\s+|production\s+|delivery\s+|financial\s+|medium\s+term\s+|"
         rf"long\s+term\s+)targets?)")
_UPV = (r"raise[sd]?|raising|lift(?:s|ed|ing)?|boost(?:s|ed|ing)?|bump(?:s|ed|ing)?|hike[sd]?|hiking|"
        r"increas(?:e|es|ed|ing)|ups|upped|upping|nudge[sd]?\s+up|push(?:es|ed)\s+up|improv(?:e|es|ed|ing)|"
        r"strengthen(?:s|ed|ing)?")
_DNV = (r"cut(?:s|ting)?|lower(?:s|ed|ing)?|slash(?:es|ed|ing)?|trim(?:s|med|ming)?|reduc(?:e|es|ed|ing)|"
        r"chop(?:s|ped|ping)?|decreas(?:e|es|ed|ing)|par(?:e|es|ed|ing)|nudge[sd]?\s+down|ratchet(?:s|ed)?\s+down|"
        r"lop(?:s|ped)?|halv(?:e|es|ed|ing)|reel(?:s|ed|ing)?\s+in|rein(?:s|ed|ing)?\s+in")
# Words a rule's gap may not cross: negators ("FDA did not approve"), contrast and subordinate
# clauses ("Lower Despite Strong Growth Outlook"), other clause verbs ("cuts shipments, holds guidance")
# and other up/down verbs ("cuts prices and boosts outlook" is two events).
_GAP_STOP = (r"(?:not|no|never|did|didnt|does|doesnt|do|dont|wont|cant|cannot|isnt|wasnt|arent|werent|wouldnt|"
             r"declines?|declined|refuses?|refused|fails?|failed|despite|but|although|while|though|whereas|yet|"
             r"however|holds?|held|keeps?|kept|maintains?|maintained|reaffirms?|reaffirmed|reiterates?|reiterated|"
             r"affirms?|affirmed|says|said|after|amid|because|since|when|"
             + _UPV + "|" + _DNV + r")")
# a comma inside a gap only lists metrics: "cuts profit, sales forecasts"
_GAP_METRIC = (r"(?:sales|revenues?|profits?|earnings|eps|margins?|ebitda|income|production|deliveries|shipments|"
               r"output|capex|growth|ffo|cash\s+flow|guidance|outlook|dividend)")
_GAP_WORD = rf"\s+(?!{_GAP_STOP}\b)[^\s.;!?$,]+(?:,(?=\s+(?:{_GAP_METRIC}|inc|corp|ltd|plc|co|llc|lp)\b))?"


def _gap(n: int) -> str:
    """Up to ``n`` words between a rule's verb and its noun, never across a clause (see ``_GAP_STOP``)."""
    return rf"(?:{_GAP_WORD}){{0,{n}}}?\s+"


_GAP = _gap(4)
# "... outlook to up 1.3%-1.5% from up 1.5%-2.5%": the figures detail the revision, they are not moves
_RANGE_TAIL = (r"(?:\s+(?:to|at|of)\s+(?:a\s+range\s+of\s+)?(?:up\s+|down\s+)?[^\s.;!?]*\d[^\s;!?]*"
               r"(?:\s+(?:to|-)\s+[^\s;!?]*\d[^\s;!?]*)?(?:\s+(?:from|vs\.?)\s+(?:up\s+|down\s+)?"
               r"[^\s;!?]*\d[^\s;!?]*(?:\s+(?:to|-)\s+[^\s;!?]*\d[^\s;!?]*)?)?)?")
# Analyst shorthand names the stock between verb and "target" ("trims Nielsen target"); a preposition,
# article or another metric there means a different target ("raises wages to compete with Target",
# "raises its inflation target").
_NAME_GAP = (r"(?:\s+(?!(?:" + _GAP_STOP + r"|to|on|in|with|for|of|at|by|from|into|ahead|against|over|as|and|than|the|a|an|"
             r"following|before|prices?|wages|"
             r"costs?|jobs|stake|position|holdings|shares|stock|inflation|rates?|emissions?|carbon|sales|revenues?|"
             r"growth|margins?|production|deliver(?:y|ies)|financial|deficit|debt|savings|profits?|earnings|eps|capex|"
             r"spending|dividend|buyback|net|zero|climate|annual|full|year|\d+)\b)[^\s.;!?$,]+"
             r"(?:,(?=\s+(?:inc|corp|ltd|plc|co|llc|lp)\b))?){0,2}?\s+")
_BARE_TARGET = (r"targets?(?!'|\s+(?:co|corp|corporation|inc|stores?|brands?|stake|shares|stock|position|holdings|"
                r"date|rate|range|audience|market|customers?)\b)")
# A bare "target" is a price target only in analyst phrasing ("its target on X", "target to $150
# from $130"); otherwise it is as likely the retailer ("Walmart raises wages to compete with Target").
_PT = (r"(?:price\s+targets?|target\s+price|price\s+objective|price\s+tgt|pt|tgt|"
       r"(?:(?:its|their|his|her)\s+)?targets?(?=\s+on\s+\S|\s+(?:for\s+(?:\S+\s+){1,3}?)?(?:to|from)\s+"
       r"(?:[$€£¥]|usd|eur|gbp|sek|nok|chf|cad|aud)?\s?\d|\s+of\s+[$€£¥]))")
# "beats by $0.04" / "misses on revenue" (not "Top Executive Calls on Government")
_BY_ON = (r"(?:by\s+(?:[$€£¥]\s?)?\d[\d,]*(?:\.\d+)?(?:\s?(?:cents?|c|%|percent)\b)?|"
          r"on\s+(?:the\s+)?(?:top\s+line|bottom\s+line|revenues?|revs?|sales|eps|earnings|"
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


_GUIDANCE_NOUN = re.compile(r"\b(?:guidance|outlook|forecasts?|view|guide|projections?|targets?|estimates?|"
                            r"expectations?)\b")
_METRIC_PHRASES = sorted({" ".join(k.lower().replace("-", " ").split()) for k in lx.METRICS}, key=len, reverse=True)


def _forecast_metric(words: list[str]) -> tuple[Optional[float], str]:
    """Polarity of what a forecast is about, read from the words right before the guidance noun
    (nearest first): "net loss outlook" -1, "capex forecast" 0, "sales forecast" +1, else None."""
    words = [w.strip(",'\"") for w in words]
    for end in range(len(words), 0, -1):
        for size in (3, 2, 1):
            if end - size < 0:
                continue
            key = " ".join(words[end - size: end])
            if key in lx.FORECAST_NEUTRAL:
                return 0.0, key
            metric = lx.METRICS.get(key)
            if metric is not None and key not in _META_HEADS:
                return (1.0 if metric.polarity > 0 else -1.0), key
    return None, ""


def _guidance(sign: int) -> Callable[[re.Match[str]], RuleMatch]:
    """Guidance raised/cut. The sign is the verb's times the forecast metric's polarity: a higher
    *loss* or *inflation* forecast is bad news, a capex forecast is neither."""
    def make(m: re.Match[str]) -> RuleMatch:
        noun = _GUIDANCE_NOUN.search(m.group())
        what = noun.group() if noun else "guidance"
        if m.groupdict().get("v") is not None:  # "raises its net loss outlook": words between verb and noun
            noun_at = m.start() + noun.start() if noun else m.end()
            words = m.string[m.end("v"): noun_at].split()
        else:  # "loss forecast raised": the metric precedes the match
            words = m.string[max(0, m.start() - 40): m.start() + (noun.start() if noun else 0)].split()[-3:]
        polarity, metric = _forecast_metric(words)
        if polarity is None:  # "cuts its forecast for inflation", "raises outlook on margins"
            after = re.match(r"\s+(?:for|of|on)\s+((?:[a-z&'-]+\s*){1,4})", m.string[m.end():])
            following = after.group(1).split() if after else []
            for k in range(1, len(following) + 1):  # the first metric named wins
                polarity, metric = _forecast_metric(following[:k])
                if polarity is not None:
                    break
        verb =("raises" if sign > 0 else "cuts") if what.startswith(("estimate", "expectation")) else ""
        label = f"{verb} {metric} {what}" if verb else f"{metric} {what} {'raised' if sign > 0 else 'cut'}"
        value = 1.1 * sign * (1.0 if polarity is None else polarity)
        return RuleMatch(m.start(), m.end(), value, " ".join(label.split()), "guidance")
    return make


def _flows(m: re.Match[str]) -> RuleMatch:
    act = m.group("act")
    bearish = act.startswith(("sell", "dump", "unload", "bail", "flee", "fled", "exit", "abandon", "short", "bear",
                              "cash", "trim", "sold", "sour", "cool"))
    who = " ".join(m.group("who").split())
    return RuleMatch(m.start(), m.end(), -0.9 if bearish else 0.9, f"{who} {' '.join(act.split())}", "flows")


def _imperative(m: re.Match[str]) -> RuleMatch:
    """A headline that tells you what to do ("Avalara: Buy This Leader", "Sell Nike")."""
    v = m.group("v")
    return RuleMatch(m.start("v"), m.end(), 0.7 if v == "buy" else -0.7, v, "imperative")


def _pct_outcome(m: re.Match[str]) -> RuleMatch:
    """"have a 33% loss to show for it" / "a 159% gain"."""
    return RuleMatch(m.start(), m.end(), -0.9 if m.group("w").startswith("loss") else 0.8, m.group(), "pct_loss")


def _fast_enough(m: re.Match[str]) -> RuleMatch:
    """"Hedge funds couldn't dump X fast enough": eager selling, not a negated sale."""
    selling = m.group("v") in ("dump", "sell", "unload", "ditch")
    return RuleMatch(m.start(), m.end(), -0.8 if selling else 0.8, f"couldn't {m.group('v')} fast enough",
                     "fast_enough")


_BEARISH_TRENDS = ("selloff", "sell", "slump", "decline", "bear", "downturn", "recession", "crisis", "slide", "rout",
                   "downtrend", "bears")


def _trend_end(m: re.Match[str]) -> RuleMatch:
    """A trend ending: the end of a boom is bad news, the end of a slump good news."""
    bearish_trend = m.group("what").startswith(_BEARISH_TRENDS)
    return RuleMatch(m.start(), m.end(), 0.7 if bearish_trend else -0.8,
                     f"{' '.join(m.group('what').split())} ending", "trend_end")


def _streak(m: re.Match[str]) -> RuleMatch:
    """A streak ending flips its meaning: a losing streak snapped is good news."""
    losing = (m.group("kind") or m.group("kind2")) == "losing"
    return RuleMatch(m.start(), m.end(), 0.6 if losing else -0.6,
                     "losing streak ends" if losing else "winning streak ends", "streak")


def _vs_consensus(m: re.Match[str]) -> Optional[RuleMatch]:
    """"Q3 adj. EPS 74 cents; FactSet consensus 86 cents": reported vs. expected decides the sign."""
    new, est = _amount(m.group("new")), _amount(m.group("est"))
    if new is None or est is None or new == est or not est:
        return None
    sign = 1 if new > est else -1
    mag = 0.8 + 0.4 * min(1.0, abs(new - est) / abs(est) / 0.1)
    return RuleMatch(m.start(), m.end(), sign * mag, f"{m.group('new').strip()} vs consensus {m.group('est').strip()}",
                     "vs_consensus")


def _options(m: re.Match[str]) -> RuleMatch:
    """Options flow ("2000 July $1900 calls opening"): bought calls/sold puts lean bullish."""
    sign = 1 if m.group("side") == "calls" else -1
    tail = m.group("tail") or ""
    sold = re.search(r"\b(?:sold|selling|sell|sells|written|writing)\b", tail)
    if sold:
        sign = -sign
    label = f"{m.group('side')} {'sold' if sold else 'bought'}"
    end = m.start("tail") + sold.end() if sold else m.end("side")
    return RuleMatch(m.start(), end, 0.6 * sign * (0.6 if sold else 1.0), label, "options_flow")


# Form-4 mechanics that are not a trading decision: pay, plan purchases, tax withholding, option exercises
_INSIDER_ROUTINE = re.compile(
    r"\b(?:units?|rsus?|restricted|benefit\w*|payroll|award\w*|grant\w*|vest\w*|withh[oe]ld\w*|withholding|"
    r"exercis\w*|options?|gift\w*|reinvest\w*|drip|espp|401\s?k|compensation|in\s+lieu|tax(?:es)?|"
    r"deferred|conversion|converted|settle\w*)\b")
_INSIDER_PURCHASE = re.compile(r"\b(?:open\s+market|purchas\w*|bought|buys|buying)\b")


def _insider(m: re.Match[str]) -> RuleMatch:
    """Executives trading their own stock. Open-market purchases are a strong vote of confidence;
    sales are often routine (half weight under a 10b5-1 plan); compensation, plan and tax-withholding
    transactions ("acquires 512 stock units through payroll deductions") say nothing."""
    act = m.group("act")
    label = f"{' '.join(m.group('who').split())} {act}"  # "ceo buys", "co founder sold"
    context = m.group() + " " + m.string[m.end(): m.end() + 60]
    buy = act.startswith(("buy", "bought", "purchas", "acquir", "add"))
    if _INSIDER_ROUTINE.search(context) and not (buy and re.search(r"\bopen\s+market\b", context)):
        return RuleMatch(m.start(), m.end(), 0.0, f"{label} (compensation/plan)", "insider")
    if buy:  # "acquires/adds" without a purchase is weaker evidence than "buys"
        return RuleMatch(m.start(), m.end(), 0.8 if _INSIDER_PURCHASE.search(context) else 0.45, label, "insider")
    planned = re.search(r"\b10b5[\s-]?1\b|\btrading\s+plan\b|\bpre\s?arranged\b", context)
    return RuleMatch(m.start(), m.end(), -0.25 if planned else -0.5, label, "insider")


_BIG_AMOUNT = re.compile(r"(?:[$€£¥]\s?\d[\d,.]*\s*(?:billion|bln|bn|b|trillion|tn)\b|"
                         r"[$€£¥]\s?(?:[1-9]\d{2,}|[1-9]\d{0,2},\d{3})(?:\.\d+)?\s*(?:million|mln|mn|m)\b|"
                         r"\b\d[\d,.]*\s*(?:billion|bln|bn)\b)")


def _contract_win(m: re.Match[str]) -> RuleMatch:
    """A contract win is routine news unless it is large ("wins $10 billion Pentagon contract")."""
    near = m.string[m.start(): min(len(m.string), m.end() + 40)]
    return RuleMatch(m.start(), m.end(), 0.8 if _BIG_AMOUNT.search(near) else 0.3, "wins contract", "contract_win")


def _policy_path(m: re.Match[str]) -> RuleMatch:
    """"Fed signals fewer rate cuts" is hawkish (-), "more rate cuts" dovish (+); hikes the other way."""
    more = m.group("q") in ("more", "additional", "further", "extra", "deeper", "bigger", "faster", "aggressive")
    easing = m.group("w").startswith(("cut", "reduction"))
    sign = 1 if more == easing else -1
    return RuleMatch(m.start(), m.end(), 0.8 * sign, f"{m.group('q')} rate {m.group('w')}", "policy_path")


_INDEX_NAME_BEFORE = re.compile(r"(?:s&p(?:\s*\d+)?|nasdaq(?:\s*\d+)?|dow(?:\s+jones)?|russell\s+\d+|ftse\s+\d+)\s*$")


def _index_change(m: re.Match[str]) -> Optional[RuleMatch]:
    """Index inclusion (+) / deletion (-); "S&P 500 joins Dow in the red" is two indexes, not a change."""
    if _INDEX_NAME_BEFORE.search(m.string[max(0, m.start() - 20): m.start()]):
        return None
    out = bool(re.match(r"removed|dropped|deleted|kicked|booted|exit", m.group("act")))
    return RuleMatch(m.start(), m.end(), -0.8 if out else 0.8, "index deletion" if out else "index inclusion",
                     "index_change")


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
        r"slash(?:es|ed)?|ups|upped|take[sn]?|took|raising|upgrading|downgrading)\b"
        r"(?:(?!\bto\b)(?:\$[a-z]{1,6}\b|[^.;!?$\d])){0,50}?\bto\s+(?:an?\s+)?['\"]?(?P<new>" + RATING + r")\b"
        # a rating ends its phrase; "to reduce debt", "to add market share", "offers to buy X" are infinitives
        r"(?=\s*(?:$|[,.;:!?)('\"|\-\u2013\u2014])|\s+(?:at|from|by|on|with|in|after|amid|as|and|ahead|rating|"
        r"vs|versus|following|citing|over|due|for|post|into|says|while|despite|but|pt|price|target|stance|"
        r"recommendation|equivalent|territory)\b)"
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
    ("pt_extreme", re.compile(r"\b(?:street|wall\s+street)\s+(?P<w>low|high)\s+(?:price\s+)?(?:targets?|pt)\b"),
     lambda m: RuleMatch(m.start(), m.end(), 0.9 if m.group("w") == "high" else -0.9, f"street-{m.group('w')} target",
                         "price_target")),
    ("price_target", re.compile(
        r"\b(?P<v>" + _UPV + "|" + _DNV + r")\b" + _gap(5) + _PT + r"\b(?P<tail>[^.;!?]{0,50})"),
     _price_target),
    ("price_target", re.compile(
        r"\b" + _PT + r"\b(?:\s+[^\s.;!?]+){0,3}?\s+(?P<v>raised|lifted|boosted|bumped|hiked|increased|upped|cut|"
        r"lowered|trimmed|reduced|slashed|decreased|pared|chopped)\b(?P<tail>[^.;!?]{0,50})"), _price_target),
    ("price_target", re.compile(r"\b" + _PT + r"\b[^.;!?]{0,25}?\bto\s+" + _AMT + r"\s*(?:,\s*)?(?:from|vs\.?)\s+"
                                + _AMT), _price_target),
    # bare "target" in analyst shorthand: "SunTrust trims Nielsen target", "NLight target raised on deal"
    ("price_target", re.compile(r"\b(?P<v>" + _UPV + "|" + _DNV + r")\b" + _NAME_GAP + _BARE_TARGET
                                + r"\b(?P<tail>[^.;!?]{0,50})"), _price_target),
    ("price_target", re.compile(r"\b" + _BARE_TARGET + r"\s+(?P<v>raised|lifted|boosted|bumped|hiked|increased|"
                                r"upped|cut|lowered|trimmed|reduced|slashed|decreased|pared)\b(?=\s+(?:to|at|by|on|"
                                r"from|after|following|amid|as)\b|\s*[.;,:!?)]|\s*$)(?P<tail>[^.;!?]{0,50})"),
     _price_target),
    # --- earnings vs. expectations
    ("beat", re.compile(
        r"\b(?:beat(?:s|ing)?|top(?:s|ped|ping)?|exceed(?:s|ed|ing)?|surpass(?:es|ed|ing)?|crush(?:es|ed)?|"
        r"smash(?:es|ed)?|trounce[sd]?|outstrip(?:s|ped)?|blow(?:s)?\s+past|blew\s+past|blow(?:s)?\s+away|"
        r"blew\s+away|best(?:s|ed)?|clear(?:s|ed)?|outpace[sd]?|(?:squeak|edge|sail|breeze|cruise|race|zoom)"
        r"(?:s|d|ed)?\s+(?:past|by|over))" + _gap(5) + _EXP + r"\b"),
     _fixed(1.0, "beats estimates", "beat")),
    ("beat", re.compile(r"\b(?:beat(?:s)?|tops|topped|exceed(?:s|ed)?)\s+(?:[^\s.;!?]+\s+){0,2}?" + _BY_ON),
     _fixed(1.0, "beats", "beat")),
    ("beat", re.compile(r"\b(?:eps|earnings|revenues?|sales|profits?|results?|quarter|q[1-4])\s+(?:and\s+\S+\s+)?"
                        r"(?:beat|beats|top|tops|exceed|exceeds|surpass(?:es)?)\b"), _fixed(1.0, "beat", "beat")),
    ("miss", re.compile(
        r"\b(?:miss(?:es|ed|ing)?|undershoot(?:s)?|undershot|fall(?:s|ing)?\s+short\s+of|fell\s+short\s+of|"
        r"c(?:o|a)m(?:e|es|ing)\s+(?:in|up)\s+(?:below|under|short\s+of|shy\s+of|light\s+of)|fall(?:s|ing)?\s+shy\s+of|"
        r"fell\s+shy\s+of|trail(?:s|ed)?|lag(?:s|ged)?)" + _GAP + _EXP + r"\b"),
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
    ("vs_exp", re.compile(r"\bbetter\s+than\s+(?:expected|anticipated|forecast)\b"),
     _fixed(1.0, "better than expected", "above_exp")),
    ("vs_exp", re.compile(r"\b(?:worse|poorer)\s+than\s+(?:expected|anticipated|forecast)\b"),
     _fixed(-1.0, "worse than expected", "below_exp")),
    ("vs_exp", re.compile(r"\bbetter\s+than\s+feared\b"), _fixed(0.6, "better than feared", "above_exp")),
    ("vs_exp", re.compile(r"\bworse\s+than\s+feared\b"), _fixed(-0.9, "worse than feared", "below_exp")),
    ("inline", re.compile(r"\b(?:in\s+line|inline)\s+with\s+(?:[^\s.;!?]+\s+){0,2}?" + _EXP + r"\b"),
     _fixed(0.0, "in line with estimates", "inline")),
    ("inline", re.compile(r"\b(?:match(?:es|ed|ing)?|meet(?:s|ing)?|met)\s+(?:[^\s.;!?]+\s+){0,3}?" + _EXP + r"\b"),
     _fixed(0.25, "meets estimates", "inline")),
    # --- guidance
    ("guidance", re.compile(r"\b(?P<v>" + _UPV + r")\b" + _gap(5) + _GUID + r"\b" + _RANGE_TAIL), _guidance(1)),
    ("guidance", re.compile(
        r"\b(?P<v>" + _DNV + r"|withdraw(?:s|n|ing)?|withdrew|pull(?:s|ed|ing)?|suspend(?:s|ed|ing)?|"
        r"scrap(?:s|ped|ping)?|scal(?:e|es|ed|ing)\s+back|temper(?:s|ed|ing)?|dial(?:s|ed|ing)?\s+back|"
        r"rein(?:s|ed|ing)?\s+in|reel(?:s|ed|ing)?\s+in|walk(?:s|ed|ing)?\s+back)\b" + _gap(5)
        + _GUID + r"\b" + _RANGE_TAIL),
     _guidance(-1)),
    ("guidance", re.compile(r"\b(?:guidance|outlook|forecast|view)\s+(?:\S+\s+){0,2}?(?:raised|lifted|boosted|"
                            r"increased|hiked|upped|improved)\b"), _guidance(1)),
    ("guidance", re.compile(r"\b(?:guidance|outlook|forecast|view)\s+(?:\S+\s+){0,2}?(?:cut|lowered|slashed|"
                            r"trimmed|reduced|withdrawn|suspended|pulled)\b"), _guidance(-1)),
    ("guidance", re.compile(r"\b(?:guidance|outlook|forecasts?|view)\s+(?:also\s+|again\s+)?(?:disappoints?|"
                            r"disappointed|underwhelms?|underwhelmed|misses|missed|falls?\s+short|fell\s+short|"
                            r"lags?|lagged|sours?|soured)\b"), _fixed(-1.0, "guidance disappoints", "guidance")),
    ("guidance", re.compile(r"\b(?:weak|weaker|soft|softer|light|disappointing|downbeat|gloomy|cautious|tepid|"
                            r"lackluster|lacklustre|bleak|dismal|poor|grim|sluggish)\s+(?:full\s+year\s+|annual\s+|"
                            r"quarterly\s+|q[1-4]\s+|\d{4}\s+|sales\s+|revenue\s+|profit\s+|earnings\s+)?"
                            r"(?:guidance|outlook|forecast)\b"), _fixed(-1.0, "weak guidance", "guidance")),
    ("guidance", re.compile(r"\b(?:strong|stronger|upbeat|bullish|robust|rosy|solid|optimistic|bright|brighter|"
                            r"blowout)\s+(?:full\s+year\s+|annual\s+|quarterly\s+|q[1-4]\s+|\d{4}\s+|sales\s+|"
                            r"revenue\s+|profit\s+|earnings\s+)?(?:guidance|outlook|forecast)\b"),
     _fixed(0.9, "strong guidance", "guidance")),
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
        r"rushing\s+into|flocking\s+to|betting\s+(?:big\s+)?on|bet\s+on|bullish\s+on|bought|selling|dumping|"
        r"unloading|bailing|fleeing|exiting|abandoning|"
        r"shorting|cashing\s+out|trimming|sold|dumped|fled|souring\s+on|soured\s+on|cooling\s+on|warming\s+up)\b"),
     _flows),
    # 13F holding changes by institutions (MarketBeat-style bot headlines): stale, routine, no stance
    ("holdings_13f", re.compile(
        r"\b(?:llc|llp|lp|inc|corp|corporation|co|ltd|plc|n\.?a|group|management|mgmt|advisors?|advisers?|"
        r"capital|partners|bank|trust|fund|funds|investments?|associates|wealth|securities|holdings|financial|"
        r"company|counsel|ag|sa|se|nv|bv|gmbh)(?:\s+[a-z]{2})?\s+(?:has\s+)?"
        r"(?:raises?|raised|increases?|increased|boosts?|boosted|grows?|grew|lifts?|lifted|cuts?|reduces?|reduced|"
        r"trims?|trimmed|lowers?|lowered|decreases?|decreased|sells?|sold|buys?|bought|acquires?|acquired|"
        r"purchases?|purchased|takes?|took|initiates?|opens?|establishes?|invests?|invested|makes?|adds?\s+to|"
        r"added\s+to|has|holds?|expands?|expanded|scales?\s+back|pares?)\s+"
        # MarketBeat phrasing has no possessive: "BW Group trims its stake in DHT" is a strategic holder's move
        r"(?:[$]?\d[\d,.]*\s*(?:million|billion|thousand|k|m|mln|bn)?\s+)?(?:(?:a|an|new)\s+)*"
        r"(?:stock\s+|share\s+)?(?:stake|position|positions|holdings?|shares|investment)\s+(?:in|of)\b"),
     _fixed(0.0, "institutional holding change", "holdings_13f")),
    # the other camp's pain: "bears getting destroyed" is bullish, "bulls trapped" bearish
    ("camp_pain", re.compile(
        r"\b(?P<camp>bears|shorts|short\s+sellers|bulls|longs|dip\s+buyers)\s+(?:(?:are|were|is|getting|got|"
        r"being|been|just|absolutely|totally|so|all|now|still)\s+){0,3}(?:destroyed|wrecked|rekt|crushed|burned|"
        r"burnt|obliterated|annihilated|demolished|slaughtered|massacred|in\s+shambles|trapped|liquidated|"
        r"roasted|toast|cooked|in\s+pain|panicking|crying|hurting|squeezed|wiped\s+out|blown\s+out|"
        r"taken\s+out|stopped\s+out|run\s+over|steamrolled)\b"),
     lambda m: RuleMatch(m.start(), m.end(), 0.8 if m.group("camp").startswith(("bear", "short")) else -0.8,
                         m.group(), "camp_pain")),
    # --- results & outlook idioms
    ("inline", re.compile(r"\b(?:eps|earnings|revenues?|sales|results?|ffo|affo|nii|ebitda|q[1-4])\s+(?:and\s+\S+\s+)?"
                          r"(?:in\s+line|inline)\b(?!\s+with)"), _fixed(0.5, "in-line", "inline")),
    ("trend_end", re.compile(
        r"\b(?:so\s+long|goodbye|good\s+bye|bye\s+bye|farewell)\s+(?:to\s+)?(?:the\s+)?(?P<what>bull\s+market|bull\s+run|"
        r"rally|boom|bear\s+market|bears|selloff|sell\s+off|slump|downturn|recession|downtrend|uptrend)\b"), _trend_end),
    ("trend_end", re.compile(
        r"\b(?P<what>boom|rally|run|bull\s+market|bull\s+run|growth|expansion|recovery|upswing|selloff|sell\s+off|"
        r"slump|decline|bear\s+market|downturn|recession|crisis|slide|rout|downtrend|uptrend)\s+(?:is\s+|was\s+|may\s+be\s+|"
        r"could\s+be\s+|appears\s+)?(?:coming\s+to\s+an\s+end|over|ends|ended|is\s+ending|fizzles|fizzled|fades|faded|"
        r"runs\s+out\s+of\s+steam|ran\s+out\s+of\s+steam|stalls|stalled)\b"), _trend_end),
    ("metric_hit", re.compile(r"\b(?:revenues?|sales|earnings|profits?|margins?|results|demand|growth|eps)\s+hit\b"
                              r"(?!\s+(?:a\s+|an\s+|the\s+|new\s+|fresh\s+)?(?:record|all\s+time|high|highs|peak|"
                              r"milestone|target|\$|\d))"), _fixed(-0.7, "revenue hit", "metric_hit")),
    ("pct_loss", re.compile(r"\b\d[\d.,]*\s?%\s+(?P<w>loss|losses|gain|gains|return)\b(?!\s+on\b)"), _pct_outcome),
    # --- positioning, ratings and trading idioms
    ("fast_enough", re.compile(r"\b(?:couldn'?t|could\s+not|can'?t|cannot|can\s+not)\s+(?P<v>dump|sell|unload|ditch|buy|"
                               r"own|get|grab|scoop\s+up|add)\b[^.;!?]{0,40}?\bfast\s+enough\b"), _fast_enough),
    ("out_of_slump", re.compile(
        r"\b(?:snap\w*|climb\w*|pull\w*|bounc\w*|break\w*|broke|emerg\w*|recover\w*|come|comes|came|coming|"
        r"crawl\w*|dig\w*|dug)\s+out\s+of\s+(?:a\s+|the\s+|its\s+|their\s+)?(?:[a-z-]+\s+)?(?:slump|rut|funk|downturn|"
        r"recession|slide|decline|bear\s+market|hole|selloff|sell\s+off|losing\s+streak|crisis|bankruptcy)\b"),
     _fixed(0.8, "out of a slump", "out_of_slump")),
    ("good_buy", re.compile(r"\b(?:an?|is\s+a|still\s+a)\s+(?:good|great|decent|solid|screaming|clear|strong|compelling|"
                            r"smart|top|bargain|long\s+term)\s+buy\b(?!\s+(?:back|out))"),
     _fixed(0.9, "a good buy", "good_buy")),
    ("streak", re.compile(r"\b(?:snap(?:s|ped)?|end(?:s|ed)?|break(?:s)?|broke|halt(?:s|ed)?)\s+(?:[^\s.;!?]+\s+){0,3}?"
                          r"(?P<kind>win(?:ning)?|losing)\s+streak\b|\b(?P<kind2>win(?:ning)?|losing)\s+streak\s+"
                          r"(?:ends|ended|snapped|is\s+over|comes\s+to\s+an\s+end)\b"), _streak),
    ("imperative", re.compile(r"(?:^|[:;]\s+|\s[-\u2013\u2014]\s+)(?P<v>buy|sell)\s+"
                              r"(?!back\b|out\b|in\b|into\b|side\b|on\b|off\b|to\b|or\b|and\b|now,?\s+pay)"
                              r"(?:this|these|the|now|shares|stock|\$?[a-z])"), _imperative),
    ("top_pick", re.compile(r"\btop\s+(?:[a-z&]+\s+){1,2}?picks?\b"), _fixed(0.9, "top pick", "top_pick")),
    ("options_flow", re.compile(
        r"(?:[$]\s?\d[\d.,]*|\b\d[\d.,]*)\s+(?P<side>calls|puts)\b(?P<tail>(?:\s+[^\s.;!?]+){0,3})"), _options),
    # --- reported numbers, charges, insiders, approvals
    ("vs_consensus", re.compile(
        rf"\b(?P<m>eps|earnings|revenues?|sales|ffo|ebitda|net\s+income)\s+(?:of\s+|was\s+|at\s+)?(?P<new>{_AMT})\s*[;,]?\s+"
        rf"(?:[a-z]+\s+){{0,2}}?(?:consensus|estimates?|expectations?|est\.?)\s+(?:of\s+|was\s+|at\s+|is\s+)?(?P<est>{_AMT})"),
     _vs_consensus),
    ("charge", re.compile(
        r"\b(?:incur\w*|take[sn]?|taking|took|book\w*|record\w*|post(?:s|ed|ing)?|flag\w*|expects?)\s+(?:a\s+|an\s+)?"
        r"(?:[$€£¥]?\s?\d[\d.,]*\s*(?:million|billion|mln|bln|mn|bn|m|b)?\s+)?(?:(?:pre\s?tax|after\s?tax|one\s?time|"
        r"non\s?cash|impairment|restructuring|write\s?down|goodwill|special|quarterly)\s+)*charges?\b"
        r"(?<!\btake\scharge)(?<!\btakes\scharge)(?<!\btook\scharge)(?<!\btaking\scharge)(?<!\btaken\scharge)"),
     _fixed(-0.6, "charge", "charge")),  # "takes charge" (of a company) is an idiom, not an accounting charge
    ("insider", re.compile(
        r"\b(?P<who>ceo|cfo|coo|chairman|chairwoman|chair|founder|co\s?founder|directors?|insiders?|executives?|"
        r"execs?|president|chief\s+executive|board\s+members?)\b(?:[^.;!?]|\.\d){0,40}?\b(?P<act>buys|bought|buying|"
        r"purchases|purchased|acquires|acquired|adds|added|sells|sold|selling|dumps|dumped|unloads|unloaded)\b"
        r"(?:[^.;!?]|\.\d){0,40}?\b"
        r"(?:shares|stock|stake|options|calls)\b"), _insider),
    ("license", re.compile(
        r"\b(?:obtain(?:s|ed|ing)?|receiv(?:e|es|ed|ing)|secur(?:e|es|ed|ing)|wins?|won|gets?|got|gain(?:s|ed)?|"
        r"grant(?:s|ed)|award(?:s|ed)|earn(?:s|ed)?|land(?:s|ed)?)" + _GAP + r"(?P<what>licen[cs]es?|"
        r"approvals?|clearance|permits?|patents?|authori[sz]ation|certification|designation|orphan\s+drug)\b"),
     lambda m: RuleMatch(m.start(), m.end(), 0.7, f"{' '.join(m.group('what').split())} secured", "license")),
    ("regulatory_no", re.compile(
        r"\b(?:fda|ema|chmp|mhra|pmda|nmpa|health\s+canada|regulators?|antitrust\s+(?:regulators?|authorit\w+)|"
        r"european\s+commission|cfius|watchdog|ftc|doj|sec|cma)\b" + _gap(3)
        + r"(?:reject(?:s|ed|ing)?|(?:declines?|declined|refuses?|refused|fails?|failed|did\s+not|didnt|does\s+not|"
        r"doesnt|will\s+not|wont|would\s+not|wouldnt)\s+(?:to\s+)?(?:approve|clear|authori[sz]e|grant|accept)|"
        r"block(?:s|ed|ing)?|votes?\s+against|voted\s+against|turn(?:s|ed)?\s+down|denie[sd]|"
        r"sues?\s+to\s+block)\b"), _fixed(-1.0, "regulatory rejection", "regulatory_no")),
    ("regulatory_no", re.compile(
        r"\b(?:approval|application|nda|bla|merger|takeover|acquisition|deal|bid)\s+(?:was\s+|is\s+|has\s+been\s+)?"
        r"(?:rejected|denied|blocked|not\s+approved)\b"), _fixed(-0.9, "regulatory rejection", "regulatory_no")),
    ("regulatory_ok", re.compile(
        r"\b(?:fda|ema|chmp|mhra|pmda|nmpa|health\s+canada|regulators?|antitrust\s+(?:regulators?|authorit\w+)|"
        r"european\s+commission|cfius|watchdog|sec|cftc|ftc|doj|finra|occ|cma|fcc|faa|nhtsa|ferc|"
        r"federal\s+reserve)\b(?:" + _GAP + r"(?:approv(?:e|es|ed|ing)|clear(?:s|ed)?|ok'?d|okays|"
        r"green\s?light\w*|authori[sz](?:e|es|ed|ing)|accept(?:s|ed)?|grant(?:s|ed)?|oks)|\s+approvals?)\b"
        # approving its own rulebook is process, not news for a company
        r"(?!\s+(?:\S+\s+){0,5}?(?:rules?|rulemaking|regulations?|proposals?|amendments?|framework|guidelines?|"
        r"standards?|changes|budget|minutes)\b)"), _fixed(0.9, "regulatory approval", "regulatory_ok")),
    ("returns", re.compile(r"\b(?:made|gained|earned|returned)\s+(?:a\s+|over\s+|more\s+than\s+)?\d[\d.,]*\s?%"),
     _fixed(0.8, "gained", "returns")),
    # --- equity issuance
    # a securities word, an amount or a pricing verb must mark it: "launches enterprise offering" is a product
    ("offering", re.compile(
        r"\b(?:(?:registered\s+direct|at\s+the\s+market|atm|secondary|follow\s+on|underwritten|initial\s+public|public|"
        r"common\s+stock|equity|share|stock|overnight|bought\s+deal|private|convertible|notes|ipo)\s+"
        r"(?:[^\s.;!?]+\s+){0,3}?|"
        r"(?:prices?|priced|pricing|announces?|announced|launch(?:es|ed)?|proposed|commences?|files?\s+for|"
        r"upsizes?|upsized)\s+(?:(?:a|an|its|the|proposed|upsized)\s+)?(?:[$€£]\s?)?\d[\d.,]*\s*"
        r"(?:million|billion|mln|bln|mn|bn|m|b)?\s+(?:[^\s.;!?]+\s+){0,2}?|"
        r"(?:prices?|priced|pricing|upsizes?|upsized)\s+(?:(?:its|an?|the|proposed|upsized)\s+)?)"
        r"(?:offering|placement)\b"), _offering),
    ("offering", re.compile(r"\b(?:offering|sale)\s+of\s+(?:[^\s.;!?]+\s+){0,3}?(?:shares|common\s+stock|"
                            r"ordinary\s+shares|ads|adss)\b"), _offering),
    # --- legal relief / resolution
    ("reported_loss", re.compile(
        r"\b(?:reports?|reported|posts?|posted|books?|booked|logs?|logged|records?|recorded|swings?\s+to|swung\s+to|"
        r"sinks?\s+to|sank\s+to|slips?\s+to|slipped\s+to)\s+(?:a\s+|an\s+|its\s+)?(?:(?!(?:narrower|smaller|"
        r"lower|reduced|wider|bigger|larger|higher|increased|deeper)\b)[a-z0-9-]+\s+){0,3}?(?:net\s+)?loss(?:es)?\b"
        r"(?![^.;!?]{0,60}\b(?:vs\.?|versus|compared|from|narrow\w*|widen\w*)\b)"),
     _fixed(-0.8, "reports loss", "reported_loss")),
    ("contract_win", re.compile(
        r"\b(?:wins|won|winning|secures?|secured|securing|lands|landed|landing|awarded|clinch(?:es|ed)?|bags|bagged|"
        r"snags|snagged)\s+(?:a\s+|an\s+)?(?:(?!(?:not|no|but|despite|in|at|for|of|on|from|with|after|as)\b)"
        r"[^\s.;!?,]+\s+){0,4}?(?:contracts?|orders?|tenders?|deals?)\b"), _contract_win),
    ("legal_relief", re.compile(r"\b(?:cleared|acquitted|exonerated|absolved)\s+(?:of|in)\b(?:\s+(?:all\s+|any\s+)?"
                                r"(?:[a-z-]+\s+){0,2}?(?:wrongdoing|charges?|fraud|allegations?|misconduct|claims?|"
                                r"liability|negligence|violations?|probe|investigation|case|lawsuit|suit)\b)?"),
     _fixed(0.8, "cleared", "legal_relief")),
    ("policy_path", re.compile(
        r"\b(?P<q>fewer|more|less|additional|further|extra|deeper|bigger|smaller|faster|slower|aggressive)\s+"
        r"(?:interest\s+)?rate\s+(?P<w>cuts?|hikes?|increases?|reductions?)\b"), _policy_path),
    ("index_change", re.compile(
        r"\b(?P<act>added\s+to|join(?:s|ed|ing)?|to\s+join|inclusion\s+in|included\s+in|set\s+to\s+join|removed\s+from|"
        r"dropped\s+from|deleted\s+from|kicked\s+out\s+of|booted\s+from|exits?|exiting)\s+(?:the\s+)?(?:s&p\s*(?:500|400|"
        r"600)?|nasdaq\s*100|dow(?:\s+jones)?(?:\s+industrial\s+average)?|russell\s+\d+|ftse\s+100|msci\s+[a-z]+)\b"),
     _index_change),
    ("legal_relief", re.compile(
        r"\b(?:dismiss(?:es|ed)?|drop(?:s|ped)?|toss(?:es|ed)?|throw(?:s|n)?\s+out|threw\s+out|end(?:s|ed)?|"
        r"close[sd]?|suspend(?:s|ed)?|clear(?:s|ed)?|wins?|won|prevail(?:s|ed)?\s+in)\b" + _GAP + r""
        r"(?:lawsuits?|suits?|cases?|probes?|investigations?|charges|complaints?|inquir(?:y|ies)|"
        r"patent\s+(?:case|suit|trial))\b"),
     _fixed(0.7, "legal relief", "legal_relief")),
    ("legal_relief", re.compile(
        r"\b(?:lawsuit|suit|case|probe|investigation|charges|claims?|complaint|inquiry)\s+(?:was\s+|is\s+|were\s+|"
        r"has\s+been\s+|have\s+been\s+)?(?:dismissed|dropped|tossed|thrown\s+out|rejected|closed|ended|withdrawn)\b"),
     _fixed(0.7, "legal relief", "legal_relief")),
    ("settlement", re.compile(
        r"\b(?:settle[sd]?|settling|resolve[sd]?|resolving)\b" + _GAP + r"(?:class\s+action\s+)?"
        r"(?:lawsuits?|suits?|litigation|probes?|cases?|claims?|disputes?|charges|investigations?|allegations|"
        r"arbitration)\b"),
     _fixed(0.45, "settles lawsuit", "settlement")),
    # enforcement actions ("SEC charges Ripple executives"); after legal relief so "SEC drops charges" wins
    ("enforcement", re.compile(
        r"\b(?:sec|cftc|ftc|doj|finra|justice\s+department|department\s+of\s+justice|prosecutors?|"
        r"attorneys?\s+general|regulators?|watchdog|eu|european\s+commission)\s+(?:\S+\s+)?"
        r"(?P<v>charges|charged|sues|sued|fines|fined|indicts|indicted|accuses|accused|probes|probing|"
        r"subpoenas|subpoenaed|raids|raided|cracks\s+down\s+on)\b"
        r"(?!\s+(?:\S+\s+){0,2}?(?:dropped|dismissed|withdrawn|tossed))"),
     lambda m: RuleMatch(m.start(), m.end(), -0.9, f"enforcement: {' '.join(m.group('v').split())}",
                         "enforcement")),
    # --- going concern / distress phrasing variants
    ("fine", re.compile(r"(?:[$€£¥]\s?)?\d[\d.,]*\s*(?:million|billion|mln|bln|mn|bn|m|b)?\s+"
                        r"(?:(?:eu|us|u\.s\.|uk|sec|ftc|doj|civil|antitrust|criminal|privacy|gdpr|record)\s+){0,2}"
                        r"(?:fine|penalty|penalties)\b|\bto\s+pay\s+(?:a\s+)?(?:record\s+)?(?:[$€£¥]\s?)?\d[\d.,]*\s*"
                        r"(?:million|billion|mln|bln|mn|bn|m|b)?\s+(?:in\s+)?(?:fines?|penalt(?:y|ies)|damages)\b"),
     _fixed(-0.7, "fine", "fine")),
    ("ordered_to_pay", re.compile(r"\b(?:order(?:s|ed)?|told|forced|forces)\s+(?:[^\s.;!?]+\s+){0,3}?to\s+pay\b"),
     _fixed(-0.8, "ordered to pay", "ordered_to_pay")),
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
    "analyst": ("grade", " to ", " from "),
    "analyst_init": ("initiat", "start", "coverage", "launch", "resum", "assum", "reinstat", "pick", "began",
                     "begin"),
    "analyst_reiterate": ("reiterat", "maintain", "keep", "kept", "affirm", "retain", "repeat", "stick", "stay",
                          "remain"),
    "price_target": ("target", "pt", "tgt", "objective"),
    "beat": ("beat", "top", "exceed", "surpass", "crush", "smash", "trounce", "outstrip", "blow", "blew", "best",
             "clear", "outpac", "squeak", "edge", "sail", "breez", "cruis", "race", "zoom"),
    "miss": ("miss", "short", "shy", "undersh", "came in", "come in", "comes in", "trail", "lag", "meet", "match",
             "reach", "hit ", "beat", "top"),
    "above_exp": ("above", "ahead", "better", "exceeding", "topping", "beating", "stronger", "surpassing"),
    "below_exp": ("below", "under", "short", "worse", "weaker", "behind", "softer", "missing"),
    "vs_exp": ("than",),
    "inline": ("line", "match", "meet", "met "),
    "guidance": ("guid", "outlook", "forecast", "view", "projection", "target", "estimate", "expectation"),
    "flows": ("fund", "investor", "insider", "institution", "whale", "trader", "money", "billionaire", "manager",
              "activist"),
    "offering": ("offering", "placement", "sale of"),
    "legal_relief": ("suit", "case", "probe", "investigation", "charges", "claim", "complaint", "inquiry", "trial",
                     "cleared", "acquitted", "exonerated", "absolved"),
    "settlement": ("settl", "resolv"),
    "going_concern": ("going concern",),
    "swing": (" from ",),
    "eps_loss": ("eps", "per share"),
    "job_cuts": ("job", "position", "worker", "employee", "staff", "role", "headcount", "workforce"),
    "fine": ("fine", "penalty", "penalties", "damages"),
    "reported_loss": ("loss",),
    "contract_win": ("contract", "order", "tender", "deal"),
    "index_change": ("s&p", "nasdaq", "dow", "russell", "ftse", "msci"),
    "policy_path": ("rate cut", "rate hike", "rate increase", "rate reduction"),
    "pt_extreme": ("street",),
    "ordered_to_pay": ("to pay",),
    "bankruptcy": ("bankruptcy", "insolvency", "creditor protection", "chapter 11"),
    "superlative": ("never been",),
    "fast_enough": ("fast enough",),
    "out_of_slump": ("out of",),
    "good_buy": (" buy",),
    "imperative": ("buy", "sell"),
    "vs_consensus": ("consensus", "estimate", "expectation", "est"),
    "charge": ("charge",),
    "top_pick": ("pick",),
    "streak": ("streak",),
    "metric_hit": (" hit",),
    "pct_loss": ("% loss", "% gain", "% return", "%loss", "%gain"),
    "trend_end": ("end", "over", "fizzle", "fade", "steam", "stall", "so long", "goodbye", "good bye", "bye bye",
                  "farewell"),
    "options_flow": ("calls", "puts"),
    "insider": ("ceo", "cfo", "coo", "chair", "founder", "director", "insider", "exec", "president", "chief",
                "board"),
    "license": ("licen", "approval", "clearance", "permit", "patent", "authori", "certification", "designation",
                "orphan"),
    "returns": ("made", "gained", "earned", "returned"),
    "regulatory_ok": ("fda", "ema", "chmp", "mhra", "pmda", "nmpa", "health canada", "regulator", "antitrust",
                      "european commission", "cfius", "watchdog", "sec", "cftc", "ftc", "doj", "finra", "occ",
                      "cma", "fcc", "faa", "nhtsa", "ferc", "federal reserve"),
    "enforcement": ("sec", "cftc", "ftc", "doj", "finra", "justice", "prosecutor", "attorney", "regulator",
                    "watchdog", "eu", "european commission"),
    "regulatory_no": ("fda", "ema", "chmp", "mhra", "pmda", "nmpa", "health canada", "regulator", "antitrust",
                      "european commission", "cfius", "watchdog", "ftc", "doj", "sec", "cma", "rejected", "denied",
                      "blocked", "not approved"),
    "buy_dip": ("dip", "pullback", "weakness", "sell"),
    "holdings_13f": ("stake", "position", "holding", "shares", "investment"),
    "camp_pain": ("bears", "shorts", "short sellers", "bulls", "longs", "dip buyers"),
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
    for m, _kind in sorted(matches, key=lambda mk: mk[0].start()):
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
                          "midday", "afternoon", "intraday", "hard", "considerably"})
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
                     "organic", "underlying", "reported", "consolidated", "own", "per", "share", "and", "much",
                     "far", "slightly", "somewhat", "just", "only", "even", "still"})


def _pct_magnitude(p: float) -> float:
    """Map |percent move| to a magnitude multiplier (0.3% ~ 0.65, 1% ~ 0.75, 5% ~ 1.05, 12% ~ 1.3, 40% ~ 1.6)."""
    p = abs(p)
    return min(1.7, (0.45 + 0.55 * math.log1p(p) / math.log1p(10)) / 0.8)


def _find_metric(tokens: list[Token], at: list[Optional[_Span]], i0: int, step: int, limit: float, *,
                 soft_ok: bool = False, stop_words: frozenset[str] = frozenset(),
                 through_moves: bool = False, stop_at_valence: bool = False) -> Optional[_Span]:
    """Scan from token ``i0`` in direction ``step`` for a metric span within ``limit`` content words.

    Stops at hard boundaries, other movement words, contrast markers and (unless
    ``soft_ok``) commas/colons, so "costs surge, shares tumble" pairs correctly.
    ``stop_at_valence`` also stops at evaluative words ("more weak demand" is weak demand)."""
    seen = 0.0
    j = i0
    n = len(tokens)
    while 0 <= j < n and seen < limit:
        t = tokens[j]
        if t.kind == "sep" or (t.kind == "soft" and (not soft_ok or t.text != ",")):
            return None
        if t.kind == "soft":
            seen += 1
            opener = _relative_clause_opener(tokens, j) if step < 0 else None
            if opener is None and step < 0 and t.text == "," and j < i0 and tokens[j + 1].kind == "w" \
                    and tokens[j + 1].text not in _APPOSITIVE_OPENERS:
                return None  # "Rate Fears, NVIDIA Rises 3%": the clause after the comma has its own subject
            j = (opener if opener is not None else j) + step  # "Oil prices, which surged on fears, slide"
            continue
        sp = at[j]
        if sp is not None:
            if sp.metric is not None and not sp.neutral:
                if step > 0:  # the head of a compound moves: "(slower) revenue growth", "dollar funding squeeze"
                    head = at[sp.end] if sp.end < n else None
                    while head is not None and head.metric is not None and not head.neutral and head is not sp \
                            and head.key not in _META_HEADS:
                        sp = head
                        head = at[sp.end] if sp.end < n else None
                return sp
            if sp.direction is not None and sp.metric is None and sp.key not in lx.FOOTPRINT_VERBS \
                    and not sp.level_qual and sp.key not in ("record", "records") and not through_moves:
                return None
            if sp.contrast is not None or (stop_at_valence and sp.lex and sp.metric is None):
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


_RELATIVE = frozenset({"which", "who", "whose"})
# words that open an appositive or aside, so a subject before the comma still governs the verb after it:
# "Operating profit, excluding one-offs, rose", "Sales, up 5% in the quarter, beat"
_APPOSITIVE_OPENERS = frozenset({"excluding", "including", "ex", "excl", "incl", "adjusted", "net", "as", "at", "in",
                                 "of", "from", "after", "before", "a", "an", "the", "its", "their", "up", "down",
                                 "while", "though", "although", "not", "even", "both", "according", "said", "says",
                                 "based", "driven", "led", "compared", "versus", "vs", "mainly", "mostly", "largely",
                                 "partly", "also", "meanwhile", "however", "which", "who", "whose", "and", "or",
                                 "but", "on", "for", "by", "with", "particularly", "especially", "notably", "too",
                                 "again", "still", "already", "now", "then", "much", "far", "slightly", "well",
                                 "way", "significantly", "somewhat", "considerably", "sharply", "only", "just"})


def _relative_clause_opener(tokens: list[Token], close: int) -> Optional[int]:
    """If the comma at ``close`` ends a relative clause (", which surged last week,"), the index of
    the comma that opens it; the clause describes the subject, it is not the subject."""
    for k in range(close - 1, max(-1, close - 16), -1):
        t = tokens[k]
        if t.kind == "sep":
            return None
        if t.kind == "soft":
            return k if t.text == "," and k + 1 < close and tokens[k + 1].text in _RELATIVE else None
    return None


_PCT_QUALIFIERS = frozenset({"a", "an", "ytd", "yoy", "annual", "annualized", "weekly", "monthly", "daily",
                             "intraday", "overnight", "quarterly", "one", "two", "three", "day", "week", "month",
                             "year", "session"})


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
    for j in range(start - 1, max(-1, start - 4), -1):
        t = tokens[j]
        if t.kind == "pct":
            return j
        if t.kind == "sep" or (t.kind == "w" and t.text not in _PCT_QUALIFIERS):
            break  # "a 5% rise", "60% YTD rally" (not "falls 3% as rising yields ...": another clause's)
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
    if sp.key in ("peak", "peaks") and qualified == 1.0:
        return False, 0.0  # "peak" is a level only when qualified ("3-week peak"); "peak season" is not
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
rakes raked rev revs revved scoop scoops scooped snap snaps snapped stocked stocking hold holds held
holding break breaks broke broken breaking water watered track tracked narrow calm boil hunker lay lays laid
trickle tone settle settles settled bog bogged shut crack write writes wrote mark marks marked double doubles
doubled doubling tie ties tied sit sits sat stand stands stood live lives lived sell sells sold buy buys
bought use uses used wake wakes woke waking speak speaks spoke cozy gobble eat eats ate dress dressed pay
pays paid chalk chalked cough coughed add adds added adding step steps stepped stepping shore shores shored
buck bucks bucked hike hikes hiked ramp ramps ramped boot boots booted give gives gave giving
""".split())
_SUBORDINATORS = frozenset({"after", "before", "amid", "amidst", "following", "despite", "while", "since", "because",
                            "when", "although", "though", "whereas", "unless", "as", "due", "thanks", "owing"})
# losing/gaining *anything* is evaluative: "loses license", "gains ground"
_EVALUATIVE_TRANSITIVE = frozenset(lx.verb_forms("lose", extra=("lost",)) + lx.verb_forms("gain"))
# a following word that is itself a verb means no object: "rally started", "stocks rally continues"
_VERBISH = frozenset({"is", "are", "was", "were", "has", "have", "had", "will", "would", "could", "can", "may",
                      "might", "should", "must", "continues", "continue", "begins", "began", "starts", "seems",
                      "looks", "appears", "remains", "stays", "keeps", "kept"})
# words that may follow a bare (intransitive) price move: "Tesla tumbles Monday", "slides below $200"
_MOVE_NEXT = _UPDOWN_NEXT | _PREPS | frozenset({
    "below", "above", "under", "toward", "towards", "past", "near", "back", "further", "anew", "again", "more",
    "most", "nearly", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today",
    "tonight", "yesterday", "tomorrow", "now", "then", "too", "also", "still", "but", "and", "or", "while",
    "despite", "amid", "ahead", "into", "through", "fast", "quickly", "sharply", "steeply", "lower", "higher",
    "down", "up", "off", "out", "away", "even", "following", "before", "during", "until", "when",
    "where", "which", "who", "that", "if", "because", "pre", "post", "intraday", "late", "early", "big",
    "hard", "afterhours", "premarket", "midday", "slightly", "modestly", "significantly", "dramatically",
    "almost", "over", "some", "rs", "x", "vs", "versus", "soon", "later", "next", "week", "month", "year", "already",
    "session", "trading", "hours", "day", "days", "weeks", "months", "years"})
_RECORD_PREV = frozenset({"new", "fresh", "hit", "hits", "hitting", "at", "to", "set", "sets", "reach", "reaches",
                          "reached", "notch", "notches", "notched", "close", "closes", "closed", "another", "all"})
# "record fine": a size, not a high (negative *metrics* such as "record loss" compose normally)
_RECORD_BAD_NEXT = frozenset({"fine", "fines", "penalty", "penalties", "verdict", "damages", "judgment", "judgement",
                              "settlement", "sanction", "sanctions", "amount", "sum"})


# heads that describe a metric rather than replace it: "loss outlook" is still about the loss
_META_HEADS = frozenset({"outlook", "forecast", "forecasts", "guidance", "estimate", "estimates", "expectations",
                         "target", "targets", "view", "rating", "ratings"})
_PRICE_NOUNS = frozenset({"shares", "stock", "stocks", "share price", "stock price", "share prices", "stock prices"})
_DETERMINERS = frozenset({"the", "a", "an", "its", "their", "his", "her", "our", "your", "my", "these", "those",
                          "of", "for", "with", "by", "about"})
_QTY_QUALIFIERS = frozenset({"over", "nearly", "almost", "about", "around", "roughly", "more", "than", "by", "some",
                             "another", "a", "further", "as", "much", "at", "least", "just", "only"})


# subjects that make "stock up on" the verb (hoarding), not a share price ("Tesla stock up on deliveries")
_STOCKPILERS = frozenset({"to", "will", "we", "they", "i", "you", "investors", "consumers", "shoppers", "people",
                          "traders", "buyers", "retailers", "companies", "households", "funds", "should", "could",
                          "must", "may", "might", "would", "and", "time", "firms", "banks", "families", "americans",
                          "customers", "stores", "farmers", "manufacturers", "countries", "nations", "hoarders",
                          "importers", "refiners", "governments", "consumer", "parents"})


def _stock_as_verb(tokens: list[Token], i: int) -> bool:
    """Is token ``i`` the verb "stock" ("Investors stock up on canned goods", "Stock up on ...")?"""
    if i < 0 or tokens[i].text not in ("stock", "stocks"):
        return False
    if i == 0:
        return tokens[i].text == "stock"  # imperative; a headline's leading "Stocks up ..." is the market
    return tokens[i - 1].text in _STOCKPILERS


def _quantity_after(tokens: list[Token], i: int) -> bool:
    """A number/percent follows within a few qualifier words ("down over 9%", "up by $2")."""
    for j in range(i, min(len(tokens), i + 4)):
        t = tokens[j]
        if t.kind in ("pct", "num", "cur"):
            return True
        if t.text not in _QTY_QUALIFIERS:
            return False
    return False


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
    if d.pos == "o":  # needs a percent right after: "Euro off 0.1%"
        if nxt is None or nxt.kind != "pct":
            return None
        return _find_metric(tokens, at, sp.start - 1, -1, 3), 1.0
    if d.pos == "q":  # needs a quantity right after: "production contracts 3.8%"
        nums = [t for t in tokens[sp.end: sp.end + 2] if t.kind in ("pct", "num")]
        return (_find_metric(tokens, at, sp.start - 1, -1, 4), 1.0) if nums else None
    if sp.key in ("record", "records"):
        if right is not None and right.direction is not None and right.direction.pos == "l":
            return None  # "record high": qualifier only
        if nxt is not None and nxt.text in _RECORD_BAD_NEXT:
            return None  # "record fine": the size of a penalty, not a high
        if nxt is not None and nxt.kind in ("cur", "num"):
            # "exports to hit record 5 million tonnes" is a high; "pay a record $5.7 billion" is not
            metric = _find_metric(tokens, at, sp.start - 1, -1, 5)
            return (metric, 1.1) if metric is not None and metric.metric.polarity > 0 else None  # type: ignore[union-attr]
        metric = _find_metric(tokens, at, sp.end, 1, 2) if sp.key == "record" else None
        if metric is not None:
            return metric, 1.0
        before = {t.text for t in tokens[max(0, sp.start - 3): sp.start]}
        if before & _RECORD_PREV and not any(t.text in _FROMISH for t in tokens[max(0, sp.start - 5): sp.start]):
            # "stock hits record" / "rises to record": a market reaching a high. Without a subject
            # metric or a move verb ("hits new record of 300,000 robots", "debt, a new record") it
            # says nothing about sentiment.
            metric = _find_metric(tokens, at, sp.start - 1, -1, 5)
            if metric is not None or _direction_before(at, sp.start, 4):
                return metric, 1.1
            return None
        if nxt is not None and nxt.text in _RECORD_NEXT:
            return None if sp.key == "records" else (None, 0.75)
        return None
    if d.pos == "p":  # up / down
        quantified = _quantity_after(tokens, sp.end)  # "stock up 3.5%", "down over 9%" are moves
        if prev is not None and not quantified and (prev.text in _PHRASAL_PREV or _stock_as_verb(tokens, sp.start - 1)):
            return None
        metric = _find_metric(tokens, at, sp.start - 1, -1, 3, soft_ok=True)
        if metric is not None and quantified and metric.key in lx.TREND_METRICS:
            metric = None  # "extends losses, now down 5%": the stock is down, not the losses
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
            if after is not None and (after.kind not in ("pct", "num", "cur") or _is_year_token(after)):
                # "rise in sales", "drop in 2020 revenue" (not "decrease of 25.7%")
                metric = _find_metric(tokens, at, sp.end + 1, 1, 3, stop_words=_SUBORDINATORS)
        if metric is None:  # compound: "dividend cut", "sales growth"
            metric = _find_metric(tokens, at, sp.start - 1, -1, 1)
        if metric is None and sp.key in lx.HOMOGRAPHS and prev is not None and prev.text in _DETERMINERS:
            return None, 1.0  # noun reading: "a rally prior to earnings", "the drop"
        if metric is None and sp.key in lx.HOMOGRAPHS:  # verb reading: "cut existing tariffs", "sales fall"
            return _attach_verb(sp, tokens, at, nxt, prev)
        if metric is None:  # subject: "sales posted a 5% increase" (not "fears of a selloff")
            metric = _find_metric(tokens, at, sp.start - 1, -1, 5, stop_words=_NOUN_STOPS)
        return metric, 1.0
    if d.pos == "a":
        metric = None
        if nxt is not None and nxt.text not in _PREPS:  # "higher raw material costs", not "lower on concerns"
            metric = _find_metric(tokens, at, sp.end, 1, 3, stop_words=_PREPS, stop_at_valence=True)
        if metric is None and d.default > 0:  # predicate: "inflation at 3.0%, much lighter than expected"
            metric = _find_metric(tokens, at, sp.start - 1, -1, 6 if " than " in sp.key else 4,
                                  soft_ok=" than " in sp.key)
        return metric, 1.0
    return _attach_verb(sp, tokens, at, nxt, prev)


def _attach_verb(sp: _Span, tokens: list[Token], at: list[Optional[_Span]], nxt: Optional[Token],
                 prev: Optional[Token]) -> Optional[tuple[Optional[_Span], float]]:
    """Verb attachment: object first for transitive verbs, else the subject (across appositives)."""
    key = sp.key
    if key.endswith("ed") and prev is not None and prev.text in _PREPS | {"a", "an", "the", "its", "their"} \
            and nxt is not None and nxt.kind == "w" and at[sp.end] is None and nxt.text not in _PREPS:
        return None  # participle used as adjective: "from narrowed focus", "a reduced stake"
    if prev is not None and prev.text in _DETERMINERS and key not in lx.HOMOGRAPHS \
            and not _quantity_after(tokens, sp.end):
        return None  # noun reading: "stick to the rockets", "a slip of the tongue"
    if key == "advanced" and nxt is not None and nxt.kind == "w" and nxt.text not in _PREPS:
        return None  # adjective: "Advanced Drainage Systems", "advanced chips"
    right_sp = at[sp.end] if sp.end < len(tokens) else None
    if right_sp is not None and right_sp.key in _PRICE_NOUNS and key not in lx.TRANSITIVE:
        return None  # a name, not a move: "Zoom shares get whacked", "Rocket stock"
    if key in lx.FOOTPRINT_VERBS:
        metric = _find_metric(tokens, at, sp.end, 1, 3)
        if metric is None and prev is not None and prev.text in ("are", "were", "be", "been", "being", "is", "was",
                                                                 "remain", "remains", "stay", "stays"):
            metric = _find_metric(tokens, at, sp.start - 1, -1, 4)  # passive: "stores are closed"
        return (metric, 1.0) if metric is not None and metric.key in lx.FOOTPRINT_METRICS else None
    if key in lx.TREND_ONLY:
        metric = _find_metric(tokens, at, sp.end, 1, 3)
        return (metric, 1.0) if metric is not None and metric.key in lx.TREND_METRICS else None
    right = at[sp.end] if sp.end < len(tokens) else None
    if key.endswith("ing") and right is not None and right.metric is not None and not right.neutral \
            and (prev is None or prev.text not in _BE):
        return right, 1.0  # participle as adjective: "decelerating growth" (not "is cutting costs")
    metric = None
    if key in lx.TRANSITIVE or key.split(" ")[0] in lx.TRANSITIVE:  # object: "cut existing tariffs"
        metric = _find_metric(tokens, at, sp.end, 1, 3, stop_words=_PREPS)
        if metric is None:
            metric = _removed_from(tokens, at, sp.end, removal=sp.direction is not None and sp.direction.sign < 0)
        if metric is None and key.split(" ")[0] in _DESTROY_VERBS and _quantity_after(tokens, sp.end):
            return None, 1.0  # "Selloff wipes out $1 trillion": the subject is the agent, not the victim
    gerund_complement = key.endswith("ing") and prev is not None and prev.text in ("of", "to", "for", "about")
    if metric is None and not gerund_complement:
        # subject, possibly across an appositive: "Operating profit, excluding X, rose" - but not
        # out of a subordinate clause ("pared gains after climbing to records")
        metric = _find_metric(tokens, at, sp.start - 1, -1, 6, soft_ok=True, stop_words=_SUBORDINATORS)
    if metric is None and nxt is not None and nxt.text in ("in", "of") and sp.end + 1 < len(tokens) \
            and tokens[sp.end + 1].kind not in ("pct", "num", "cur"):  # "rise in sales", not "decrease of 25%"
        metric = _find_metric(tokens, at, sp.end + 1, 1, 3)
    phrasal_obj = " " in key and key.rsplit(" ", 1)[-1] in ("up", "down", "back", "off", "out")  # "open up a way"
    if metric is None and nxt is not None and nxt.kind == "w" and nxt.text not in _MOVE_NEXT \
            and (" " not in key or phrasal_obj) \
            and not nxt.text.endswith("ed") and nxt.text not in _VERBISH \
            and at[sp.end] is None and not _quantity_after(tokens, sp.end) and key not in _EVALUATIVE_TRANSITIVE:
        return None  # transitive with a non-metric object: "Google drops plan", "Musk drops song"
    return metric, 1.0


def _is_year_token(t: Token) -> bool:
    return t.kind == "num" and 1900 <= t.value <= 2100 and float(t.value).is_integer() and not t.signed


def _boundary_between(tokens: list[Token], i: int, j: int) -> bool:
    """A comma/colon/sentence break, or a word, between token positions ``i`` and ``j``."""
    return any(t.kind in ("soft", "sep") or t.kind == "w" for t in tokens[i:j])


def _conjoined_conflict(tokens: list[Token], at: list[Optional[_Span]], metric: _Span) -> bool:
    """Is ``metric`` the second of two conjoined subjects with opposite polarity ("stocks and yields")?"""
    assert metric.metric is not None
    j = metric.start - 1
    for _ in range(2):  # "stock futures and [Treasury] yields"
        if j >= 1 and tokens[j].kind == "w" and tokens[j].text not in ("and", "&") and at[j] is None:
            j -= 1
    if j < 1 or tokens[j].text not in ("and", "&", ","):
        return False
    for k in range(j - 1, max(-1, j - 4), -1):
        other = at[k]
        if other is not None and other.metric is not None and other is not metric:
            return (other.metric.polarity > 0) != (metric.metric.polarity > 0)
        if tokens[k].kind in ("sep", "soft"):
            break
    return False


_DESTROY_VERBS = frozenset(lx.verb_forms("wipe") + lx.verb_forms("erase") + lx.verb_forms("obliterate"))


def _removed_from(tokens: list[Token], at: list[Optional[_Span]], i: int, removal: bool = False) -> Optional[_Span]:
    """"wipe $5 trillion from GDP" / "shave 2% off sales": the metric after from/off when an amount
    comes first; for ``removal`` verbs also "cuts $2 billion in costs" ("adds $90 bln in cash" is not)."""
    j, n, seen_amount = i, len(tokens), False
    while j < n and j < i + 5:
        t = tokens[j]
        if t.kind in ("num", "pct", "cur"):
            seen_amount = True
        elif seen_amount and (t.text in ("from", "off") or (t.text in ("in", "of") and removal)):
            return _find_metric(tokens, at, j + 1, 1, 3)
        elif t.kind != "w" or t.text not in ("a", "an", "about", "nearly", "almost", "over", "up", "to", "some"):
            if not (t.kind == "w" and t.text in ("billion", "million", "trillion")):
                return None
        j += 1
    return None


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


def _level_qualifiers_start(tokens: list[Token], at: list[Optional[_Span]], sp: _Span) -> int:
    """Start of a level's qualifiers, so the driver reads "52-week highs", not "highs"."""
    j = sp.start
    while j > 0 and sp.start - j < 4:
        t, other = tokens[j - 1], at[j - 1]
        if not (t.kind == "num" or (other is not None and other.level_qual) or t.text in ("week", "year", "month",
                                                                                           "day", "time", "all")):
            break
        j -= 1
    return j


def _compose(tokens: list[Token], at: list[Optional[_Span]], spans: list[_Span]) -> list[Hit]:
    """Pair movement words with the metric they move; emit signed hits."""
    pairs: list[tuple[_Span, Optional[_Span], float]] = []
    consumed: set[int] = set()
    directed: set[int] = set()
    for sp in spans:
        if sp.direction is None or sp.neutral:
            continue
        found = _attach(sp, tokens, at)
        if found is None:
            continue
        metric, mult = found
        if metric is sp:
            metric = None
        if metric is not None and metric.end <= sp.start and _conjoined_conflict(tokens, at, metric):
            metric = None  # "stock futures and bond yields drop": opposite readings, keep the plain move
        if metric is not None and id(metric) in directed and (
                sp.direction.pos == "l" or _boundary_between(tokens, metric.end, sp.start)):
            # "sales slump, squeezing automakers" / "decline to an 18-year low": the earlier word is
            # itself the move, not the thing being moved
            metric = None
        if sp.key in lx.CONTAIN_VERBS and (metric is None or metric.metric.polarity > 0):  # type: ignore[union-attr]
            continue  # only bad things are contained ("costs contained"), not "contained within guidance"
        if sp.key in lx.PAUSE_VERBS and (metric is None or metric.key not in lx.TREND_METRICS):
            if metric is None:
                continue
            if metric.end <= sp.start:  # "rally pauses" is news; "stocks and bonds pause" barely is
                mult *= 0.35  # (a company pausing production or its buyback still is)
        if metric is not None and sp.key in lx.PERSIST_VERBS and metric.metric.polarity > 0 \
                and metric.key not in lx.TREND_METRICS:  # type: ignore[union-attr]
            continue  # "Silver lingering at $17": a level holding, not news
        if metric is not None and metric.key in lx.OBJECT_AMBIGUOUS and metric.start >= sp.end:
            sp.used = metric.used = True  # "Walmart raises wages": a cost and a perk, no market sign
            continue
        if metric is not None:
            consumed.add(id(metric))
        directed.add(id(sp))
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
        parts = [(_level_qualifiers_start(tokens, at, sp) if d.pos == "l" else sp.start, sp.end)]
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
# Social register: trader words only count when a post is about trading
# --------------------------------------------------------------------------- #
_TRADING_WORDS = frozenset({"shares", "stock", "stocks", "options", "position", "positions", "chart", "ticker",
                            "portfolio", "dca", "premarket", "afterhours", "eod", "eow", "ath", "pt", "bagholder",
                            "bagholders", "shorts", "longs", "squeeze", "calls", "puts", "contracts", "strike",
                            "expiry", "expiration", "earnings", "er", "dip", "breakout", "support", "resistance"})
_POSITION_PREV = frozenset({"im", "i'm", "am", "went", "go", "going", "get", "getting", "got", "stay", "staying",
                            "stayed", "still", "remain", "remains", "be", "been", "being", "we're", "were", "are",
                            "was", "is", "added", "very", "net", "flipped", "flip", "always", "and", "or"})
# "too long", "story short", "falls short": English, even at the end of a sentence
_PROSE_BEFORE_POSITION = frozenset({"too", "so", "how", "as", "story", "fall", "falls", "fell", "falling", "come",
                                    "comes", "came", "coming", "cut", "cuts", "run", "list", "the", "a", "in", "for",
                                    "that", "this", "not", "very", "life", "time", "way", "stop", "stops", "sell",
                                    "selling", "sold", "covered"})
_POSITION_NEXT = frozenset({"here", "now", "again", "position", "positions", "calls", "puts", "shares", "from", "since",
                            "at", "and", "&", "until", "till", "into", "more", "everything", "it", "this", "them"})
_OPTION_PREV = frozenset({"buy", "buying", "bought", "buys", "grab", "grabbed", "grabbing", "load", "loaded", "loading",
                          "my", "more", "some", "cheap", "weekly", "weeklies", "otm", "itm", "holding", "those", "these",
                          "sold", "selling", "sell", "added", "adding", "add", "exercised", "rolled", "roll", "few",
                          "long", "short", "bullish", "bearish", "dated", "leap", "leaps", "monthly", "monthlies",
                          "nov", "dec", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                          "term", "day", "dte", "week", "month", "year", "expiry"})
_OPTION_NEXT = frozenset({"printing", "print", "printed", "prints", "expire", "expiring", "expired", "are", "will",
                          "up", "down", "paid", "itm", "otm", "bought", "sold", "loaded", "here", "now"})


def _trading_context(tokens: list[Token]) -> bool:
    """Is this post about trading (a cashtag, an amount, or trading vocabulary)? Prose from forums
    ("for a long time", "it calls the API") must not read as positions."""
    return any(t.kind in ("tag", "cur") or (t.kind == "w" and t.text in _TRADING_WORDS) for t in tokens)


def _position_reading(tokens: list[Token], sp: _Span) -> bool:
    """Do "long"/"short"/"calls"/"puts" name a position here ("long $INTC", "bought more calls")
    rather than ordinary English ("as long as", "long story short", "calls for a vote")?"""
    if sp.key not in ("long", "short", "calls", "puts"):
        return True
    prev = tokens[sp.start - 1] if sp.start > 0 else None
    nxt = tokens[sp.end] if sp.end < len(tokens) else None
    if (prev is not None and prev.kind == "tag") or (nxt is not None and nxt.kind == "tag"):
        return True
    if sp.key in ("long", "short"):
        if prev is not None and (prev.text in _POSITION_PREV or (sp.key == "short" and prev.text == "to")):
            return True
        if nxt is None or nxt.kind == "sep":  # "doubling down short", "WARREN IS SHORT!!!"
            return prev is not None and prev.text not in _PROSE_BEFORE_POSITION
        return nxt.text in _POSITION_NEXT
    if prev is not None and (prev.kind in ("num", "cur", "pct") or prev.text in _OPTION_PREV):
        return True
    if nxt is not None and nxt.text == "on":
        return sp.end + 1 < len(tokens) and tokens[sp.end + 1].kind == "tag"  # "calls on $NVDA"
    return nxt is not None and nxt.text in _OPTION_NEXT


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
# Modifier strengths (tuned on the Twitter-financial train split).
NEGATION_FLIP = 0.8  # negated positive flips and shrinks ("not strong")
NEGATION_FLIP_NEG = 0.8  # negated negative = relief ("won't hinder growth"), capped at NEGATED_CAP
NEGATED_CAP = 1.0  # "avoids bankruptcy" is relief, not euphoria
SHIFT_BEFORE, SHIFT_AFTER = 0.55, 1.2  # "A but B": B is what matters
CONCESSIVE_FACTOR = 0.3  # "despite A, B": A is background
BACKGROUND_FACTOR = 0.4  # "B after A" / "B amid A"
QUESTION_FACTOR = 0.25  # "Is X a buy?" asks, it does not assert
WHY_FACTOR = 0.7  # "Why Is Nvidia Stock Down Today?" presupposes the drop; it asks only for the reason
LISTICLE_FACTOR = 0.6  # "5 stocks to buy now" is marketing, not news
TEXT_HEDGE_FACTOR = 0.85  # "- report", "sources say"
# rules whose own pattern already reads the negation ("did not meet estimates", "regulators refuse to clear")
_SELF_NEGATING = frozenset({"miss", "fast_enough", "flows", "superlative", "regulatory_no"})
_SPAN_NEGATORS = frozenset({"not", "no", "never", "didnt", "doesnt", "dont", "wont", "isnt", "wasnt", "cant",
                            "cannot", "arent", "werent", "wouldnt"})

_HYPHEN_IN_WORD = re.compile(r"(?<=[a-z0-9])-(?=[a-z])|(?<=[a-z])-(?=[0-9])")
_LISTICLE = re.compile(r"^\W*(?:the\s+|these\s+)?(?:top\s+)?\d{1,2}\s+(?:[a-z'-]+\s+){0,3}?(?:stocks?|reasons?|"
                       r"things|ways|picks|names|companies|etfs?|funds|charts|shares|tips|signs|questions|lessons|"
                       r"takeaways|mistakes|risks|myths|facts|secrets|rules|habits|trends|ideas|plays|bets|winners|"
                       r"losers|buys|dividend\w*|growth\s+stocks|tech\s+stocks)\b|"
                       r"\bstocks?\s+to\s+(?:buy|watch|sell|avoid|own|consider)\b|\bbest\s+[a-z-]*\s*stocks\b")


def extract(text: str, *, social: bool = False, target: Optional[Sequence[str]] = None) -> Evidence:
    """Find every piece of sentiment evidence in ``text`` and apply modifiers.

    ``target`` (the analysed company's ticker and names) enables subject attribution: a price move
    or results event whose subject is another company ("...; Affirm Drops 4%") counts less."""
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
        if not rm.valence:
            ev.neutral_cues += 1  # recognized and neutral: "in line with estimates", "debt offering"
        else:
            hit = Hit(s, e, rm.valence, rm.term, f"rule:{rm.key}")
            if rm.key not in _SELF_NEGATING and any(tokens[j].text in _SPAN_NEGATORS for j in range(s, e)):
                # a negator the rule's span swallowed still negates it (claimed tokens never become
                # negator spans): flip and shrink like any other negated evidence
                hit.valence = -(NEGATION_FLIP if rm.valence > 0 else NEGATION_FLIP_NEG) * rm.valence
                hit.valence = max(-NEGATED_CAP, min(NEGATED_CAP, hit.valence))
                hit.negated = True
                hit.term = f"not {rm.term}"
            hits.append(hit)

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
    hits.extend(_signed_pct_hits(tokens, at, hits, claimed))
    trading = social and _trading_context(tokens)
    for sp in spans:
        if sp.neutral and sp.key in lx.NEUTRAL_CUES:
            ev.neutral_cues += 1
        if sp.used or sp.neutral:
            continue
        if social and sp.social_only is not None:  # social register overrides ("bulls" = the other camp)
            if sp.social_only and trading and _position_reading(tokens, sp):
                hits.append(Hit(sp.start, sp.end, sp.social_only, "", "social"))
        elif sp.lex is not None and sp.lex != 0.0:
            hits.append(Hit(sp.start, sp.end, sp.lex, "", "lex"))
        elif sp.metric is not None and sp.metric.intrinsic and sp.direction is None:
            hits.append(Hit(sp.start, sp.end, sp.metric.intrinsic, "", "metric"))
        if sp.hedge is not None:
            ev.hedges += 1
        if sp.litigious:
            ev.litigious += 1

    _apply_modifiers(ev, tokens, at, spans, hits, low)
    if target:
        _attribute(norm, tokens, at, claimed, hits, target)
    for h in hits:
        if not h.term:
            h.term = _term(norm, tokens, h)
        h.display = _display(norm, tokens, h)
    ev.hits = hits
    return ev


# a percent after these is a reference figure ("vs +3.0%"), before these it is one ("+3.0% expected")
_REFERENCE_BEFORE = frozenset({"vs", "versus", "expected", "consensus", "est", "estimate", "estimates", "forecast",
                               "exp", "prior", "previous", "prev", "eyed", "seen", "from"})
_REFERENCE_AFTER = frozenset({"expected", "consensus", "est", "estimate", "estimates", "forecast", "exp", "prior",
                              "previous", "prev", "eyed", "seen"})
# qualifiers between a metric and its print ("CPI +0.4% m/m, core +0.3%")
_PRINT_QUALIFIERS = frozenset({"core", "headline", "m", "y", "q", "mom", "yoy", "qoq", "ytd", "annual", "annualized",
                               "monthly", "weekly", "quarterly", "adj", "adjusted"})


def _signed_pct_hits(tokens: list[Token], at: list[Optional[_Span]], hits: list[Hit],
                     claimed: bytearray) -> list[Hit]:
    """Signed percents no movement word took: "$AQST (+13.1% pre)", "Varonis -6%", "Costs +12% y/y".

    The sign is the change; what changed decides the sentiment: "+12%" on costs, CPI or the VIX
    is bad news. Reference figures ("vs +3.0% expected") are not moves."""
    covered = bytearray(len(tokens))
    for h in hits:
        for j in range(h.start, h.end):
            covered[j] = 1
    out: list[Hit] = []
    for i, t in enumerate(tokens):
        if t.kind != "pct" or not t.signed or covered[i] or claimed[i] or not t.value:
            continue
        prev = tokens[i - 1].text if i > 0 else ""
        nxt = tokens[i + 1].text if i + 1 < len(tokens) else ""
        if prev in _REFERENCE_BEFORE or nxt in _REFERENCE_AFTER:
            continue
        val = math.copysign(0.75 * _pct_magnitude(t.value), t.value)
        j = i - 1
        while j >= 0 and tokens[j].kind == "soft" and tokens[j].text in "()[]":
            j -= 1
        named = j >= 0 and tokens[j].kind in ("w", "tag") and (at[j] is None or at[j].metric is None) \
            and tokens[j].text not in _PRINT_QUALIFIERS  # "Shanghai +0.5%", "$AQST (+13.1% pre)"
        pol, msp = (0.0, None) if named else _metric_before(tokens, at, i, 6)
        if msp is None:  # a ticker, index or company moved
            out.append(Hit(i, i + 1, val, "", "move"))
            continue
        msp.used = True  # its intrinsic valence is part of this move now
        out.append(Hit(msp.start, i + 1, val * pol, "", "move", anchor=i, parts=((msp.start, msp.end), (i, i + 1))))
    return out


MAX_VERBATIM_WORDS = 6


def _display(norm: str, tokens: list[Token], h: Hit) -> str:
    """User-facing driver: rule evidence shows its exact wording when short ("tops expectations",
    "cut to Neutral from Buy") so UIs can highlight it; long spans keep the canonical label
    ("price target cut to $14 from $18"), which also states what the numbers imply."""
    if not h.source.startswith("rule:") or h.source == "rule:numbers":
        return h.term
    span = norm[tokens[h.start].start: tokens[h.end - 1].end].rstrip(",;: ")  # "to $23 From $28," -> "$28"
    return span if len(span.split()) <= MAX_VERBATIM_WORDS else h.term


def _term(norm: str, tokens: list[Token], h: Hit) -> str:
    """Driver label: the exact text when short (so UIs can highlight it), else "metric verb [pct]"."""
    span = norm[tokens[h.start].start: tokens[h.end - 1].end]
    if not h.parts or (h.end - h.start <= 5 and ("," not in span or len(h.parts) < 2)):
        return span  # verbatim, unless a comma would glue two clauses ("Buyback, Lifting")
    words = [norm[tokens[a].start: tokens[b - 1].end] for a, b in h.parts]
    if len(words) > 1:  # "stocks ... 52-week highs": the generic price noun explains nothing
        words = [w for w in words if w.lower() not in _PRICE_NOUNS] or words
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
            and sent[-1] == 0 and n > 2 and tokens[1].kind in ("w", "tag") \
            and low[tokens[0].end: tokens[1].start].isspace():  # "Can-Fite +17%" is a name
        question_sents.add(0)
    ev.question = bool(question_sents)
    asking = _question_tokens(tokens, sent, question_sents)
    ev.listicle = bool(_LISTICLE.search(low))
    ev.text_hedge = any(sp.key in lx.TEXT_HEDGES for sp in spans)
    if _ATTRIBUTION.search(low):
        ev.text_hedge = True

    negators = [sp for sp in spans if sp.negator and not sp.neutral]
    _absorb_intensifying_adjectives(tokens, spans, hits)
    background = _background_clauses(tokens, at, hits, sent)
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
        # --- intensifiers / diminishers: up to 2 tokens before; after the hit only adverbs ("fell
        # sharply"), since a following adjective modifies the next noun ("PT cut on limited upside")
        a = h.anchor
        for sp in intens_spans:
            before = h.start - 2 <= sp.start < h.start
            after = h.end <= sp.start < h.end + 2 and _postposable(sp.key)
            if (before or after) and sent[sp.start] == sent[a]:
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
        w *= asking[a]
        if ev.listicle:
            w *= LISTICLE_FACTOR
        if ev.text_hedge:
            w *= TEXT_HEDGE_FACTOR
        h.weight = min(1.8, w)
    _cap_background(hits, background)

    # negators that negated nothing but carry meaning themselves ("fails to meet")
    for sp in negators:
        if not sp.used and sp.key in lx.NEGATOR_FALLBACK:
            hits.append(Hit(sp.start, sp.end, lx.NEGATOR_FALLBACK[sp.key], "", "lex"))


_POSTPOSED_INTENSIFIERS = frozenset({"a bit", "a little", "further", "off the cliff", "off a cliff",
                                     "for a second straight", "for a third straight"})


def _postposable(key: str) -> bool:
    """Can this intensifier modify the evidence *before* it? Adverbs can ("shares fell sharply");
    adjectives modify what follows ("cut on limited upside", "beat; big miss on margins")."""
    return key.endswith("ly") or key in _POSTPOSED_INTENSIFIERS


def _question_tokens(tokens: list[Token], sent: list[int], question_sents: set[int]) -> list[float]:
    """Evidence weight per token from questions (1.0 = asserted).

    Only the clause that actually asks is dampened: "Earnings dropped 16%, how did it fare?" states
    the drop. The question starts at the last comma/colon followed by a question word; without one
    the whole sentence asks. A "why" question presupposes its content ("Why is X stock down?"), so
    it is dampened far less than a yes/no question ("Is X a buy?")."""
    n = len(tokens)
    factors = [1.0] * n
    for sid in question_sents:
        idx = [i for i in range(n) if sent[i] == sid]
        start = idx[0]
        for i in idx[:-1]:
            if tokens[i].kind == "soft" and (tokens[i + 1].text in lx.QUESTION_STARTERS or (
                    tokens[i + 1].text in ("so", "but", "and") and i + 2 < len(tokens)
                    and tokens[i + 2].text in lx.QUESTION_STARTERS)):  # "..., so why not ...?"
                start = i + 1
        lead = tokens[start].text if tokens[start].text not in ("so", "but", "and") or start + 1 >= n \
            else tokens[start + 1].text
        why = lead == "why" and not (start + 1 < n and tokens[start + 1].text in ("not", "wouldnt", "would"))
        for i in idx:
            if i >= start:
                factors[i] = WHY_FACTOR if why else QUESTION_FACTOR
    return factors


BACKGROUND_CAP = 0.2  # opposing context may offset at most a fifth of the main clause


def _cap_background(hits: list[Hit], background: list[bool]) -> None:
    """Context never overturns the headline: "bounces 1.7% after plunging 14%" stays a bounce."""
    main = sum(h.value for h in hits if not background[h.anchor])
    ctx = sum(h.value for h in hits if background[h.anchor])
    if main and ctx and (main > 0) != (ctx > 0) and abs(ctx) > BACKGROUND_CAP * abs(main):
        scale = BACKGROUND_CAP * abs(main) / abs(ctx)
        for h in hits:
            if background[h.anchor]:
                h.weight *= scale


_AS_NOT_CAUSAL = frozenset({"well", "such", "much", "many", "long", "soon", "far", "of", "part", "a", "an", "the",
                            "expected", "anticipated", "forecast", "planned", "usual", "per", "if", "though", "to",
                            "high", "low", "little", "few", "good", "big", "always", "ever", "is", "it"})


def _background_clauses(tokens: list[Token], at: list[Optional[_Span]], hits: list[Hit],
                        sent: list[int]) -> list[bool]:
    """Tokens inside "after ..." / "following ..." / "amid ..." clauses - and, once the main clause
    already carries evidence, "... as <clause>" ("Dollar rises as relations worsen"), "... while the
    market <clause>" and purpose clauses ("cuts jobs to reduce costs") - are context for the main
    move, so they count less than the headline verb."""
    n = len(tokens)
    flags = [False] * n
    evidence_at = sorted(h.anchor for h in hits)
    for i, t in enumerate(tokens):
        if t.text in _RELATIVE and i > 0 and tokens[i - 1].text == ",":
            end = _clause_end(tokens, i + 1)
            if end < n and tokens[end].text == ",":  # ", which surged last week on supply fears,"
                for j in range(i, end):
                    flags[j] = True
            continue
        if t.text == "as" and 0 < i < n - 1 and tokens[i + 1].text not in _AS_NOT_CAUSAL \
                and tokens[i - 1].text not in _AS_NOT_CAUSAL and tokens[i + 1].kind == "w" \
                and any(a < i and sent[a] == sent[i] for a in evidence_at):
            for j in range(i + 1, _clause_end(tokens, i + 1)):
                flags[j] = True
            continue
        if t.text == "from" and i > 0 and tokens[i - 1].text in _RECOVERY_WORDS:
            for j in range(i + 1, _clause_end(tokens, i + 1)):  # "bounces back from a steep sell-off"
                flags[j] = True
            continue
        if t.text == "to" and 0 < i < n - 1 and _is_verb(at, i + 1) \
                and any(h.anchor == i + 1 and h.source == "move" for h in hits) \
                and any(h.end == i and sent[h.anchor] == sent[i] and (h.source.startswith("rule:") or _is_verb(at, h.anchor))
                        for h in hits):
            for j in range(i + 1, _clause_end(tokens, i + 1)):  # purpose: "cuts jobs to reduce costs"
                flags[j] = True
            continue
        if t.text in ("while", "whereas") and 0 < i < n - 1 and _market_subject(tokens, i + 1) \
                and any(a < i and sent[a] == sent[i] for a in evidence_at):
            for j in range(i + 1, _clause_end(tokens, i + 1)):  # "X Stock Dips While Market Gains"
                flags[j] = True
            continue
        if t.text in ("after", "following", "amid", "amidst"):
            if i == 0 and _clause_end(tokens, 1) == n:
                continue  # "After X, Y" needs the comma to know where the context ends
            nxt = tokens[i + 1].text if i + 1 < n else ""
            if nxt in ("hours", "market", "the", "close") and (i + 2 >= n or tokens[i + 2].text in ("bell", "close")
                                                                or nxt in ("hours", "market", "close")):
                continue  # "after hours", "after the bell"
            for j in range(i + 1, _clause_end(tokens, i + 1)):
                flags[j] = True
    return flags


def _is_verb(at: list[Optional[_Span]], i: int) -> bool:
    """Is token ``i`` a movement *verb* ("reduce", "cuts"), not a noun or adjective direction?"""
    sp = at[i] if 0 <= i < len(at) else None
    return sp is not None and sp.direction is not None and sp.direction.pos == "v"


_MARKET_SUBJECTS = frozenset({"market", "markets", "stocks", "s&p", "dow", "nasdaq", "index", "indexes", "indices",
                              "wall", "broader", "sector", "peers", "rivals", "benchmark", "futures", "equities"})


def _market_subject(tokens: list[Token], i: int) -> bool:
    """Is the clause starting at ``i`` about the market rather than the stock ("while the S&P 500 gains")?"""
    for t in tokens[i: i + 3]:
        if t.text in _MARKET_SUBJECTS:
            return True
        if t.kind != "w" or t.text not in ("the", "a", "broad", "overall", "wider", "its", "u", "us"):
            return False
    return False


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
    if all(tokens[j].kind == "emo" for j in range(h.start, h.end)):
        return None  # "can't stop won't stop 🚀": an emoji states the mood, it is not negated
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
            if t.kind in ("sep", "soft") or t.text in _SUBORDINATORS or (
                    at[j] is not None and at[j].contrast is not None):  # type: ignore[union-attr]
                ok = False  # "not concluded due to probe": the probe is not negated
                break
            if t.kind == "w":
                gap += 1
        limit = 4 if sp.key in ("unlikely", "no longer", "without", "never", "end to") else 3
        if ok and gap <= limit and (best is None or sp.start > best.start):
            best = sp
    return best


# --------------------------------------------------------------------------- #
# Subject attribution (optional): whose move is it?
# --------------------------------------------------------------------------- #
OFF_TARGET_FACTOR = 0.3  # a peer's move in the analysed company's headline is context, not its news
# Evidence whose grammatical subject is the company it is about. Analyst, regulatory and legal rules
# are not here: their subject is the actor ("Morgan Stanley raises ...") and the company the object.
_SUBJECT_RULES = frozenset({"rule:beat", "rule:miss", "rule:guidance", "rule:job_cuts", "rule:reported_loss",
                            "rule:swing", "rule:charge", "rule:vs_consensus", "rule:eps_loss", "rule:above_exp",
                            "rule:below_exp", "rule:offering", "rule:bankruptcy", "rule:going_concern",
                            "rule:contract_win", "rule:license", "rule:returns", "rule:streak", "rule:pct_loss",
                            "rule:metric_hit", "rule:numbers"})
# capitalized words that are not company names (title-case headlines capitalize everything)
_NAME_STOP = frozenset("""
the a an is are was were be been being has have had will would can could may might should must do does did
its their his her our your my this that these those and or nor but of in on at to for from by with as after
before amid why how what who when where which here heres there it its i we you they he she inc corp corporation
co ltd plc llc lp group stock stocks shares share price prices today tonight yesterday monday tuesday wednesday
thursday friday saturday sunday week weeks year years month months day days quarter q1 q2 q3 q4 fy premarket
trading report reports reported update news analyst analysts says said say ceo cfo coo chief executive
executives president chairman founder new top big u us usa uk eu ai etf etfs nyse nasdaq amex otc ipo sec fed
inside why's what's here's watch earnings call calls results result posts posted announces announced unveils
unveiled launches launched plans plan gets got makes made takes took sets set files filed signs signed sees saw
expects tells told notes adds just now still also more most than then so not no all over under into out up down
should vs via per about against amid stock's company firm firms maker makers chipmaker giant giants rival rivals
peer peers sector industry market markets investors traders wall street dow s&p while during because despite
through within without between among across around toward towards since until unless whether if though although
yet even only also again ever never every each both either neither other another such same own off back away
fda ftc doj cftc irs ecb imf opec gdp cpi ppi pce eps ev evs ceo cto cio otc faa fcc nhtsa epa finra occ cma
ferc jv max pro gpu gpus cpu cpus ipo ytd ath january february march april may june july august september
october november december jan feb mar apr jun jul aug sep sept oct nov dec morgan stanley goldman sachs jpmorgan
citi citigroup barclays ubs hsbc rbc bmo jefferies bernstein evercore mizuho nomura wedbush piper sandler
oppenheimer needham keybanc truist stifel baird wells fargo deutsche macquarie cantor rosenblatt bofa bank
america analysts' wolfe redburn loop benchmark td cowen raymond james daiwa
""".split())
_SPEECH = frozenset({"says", "said", "say", "sees", "warns", "warned", "tells", "told", "predicts", "predicted",
                     "thinks", "believes", "notes", "noted", "argues", "argued", "claims", "claimed", "touts",
                     "touted", "explains", "according", "writes", "wrote", "insists", "insisted"})


def _target_positions(tokens: list[Token], norm: str, target: Sequence[str]) -> set[int]:
    """Token positions that mention the target (any alias as a token sequence, its cashtag, or its
    ticker written in capitals)."""
    pos: set[int] = set()
    texts = [t.text for t in tokens]
    for alias in target:
        alias = (alias or "").strip()
        if not alias:
            continue
        low = alias.lower().lstrip("$")
        base = low.split(".")[0].split("-")[0]  # "BTC-USD" -> "btc", "SHOP.TO" -> "shop"
        seq = [t.text for t in tokenize(normalize(low)) if t.kind in ("w", "tag", "num")]
        ticker_like = alias.lstrip("$").isupper() and " " not in alias and len(alias.lstrip("$")) <= 6
        for i, t in enumerate(tokens):
            if t.kind == "tag" and t.text.lstrip("$").split(".")[0].split("-")[0] == base:
                pos.add(i)  # "$NVDA", "NCM.AX"
            elif ticker_like and t.kind == "w" and t.text == base and (
                    len(base) >= 4 or (norm[t.start: t.end].isupper() and (len(base) >= 2 or _listed(tokens, i)))):
                pos.add(i)  # "nvda"; short tickers in capitals: "TGT", "VZ, T, TMUS" (not "on", "all", "a")
            elif seq and not ticker_like and texts[i: i + len(seq)] == seq:
                pos.update(range(i, i + len(seq)))
    return pos


def _listed(tokens: list[Token], i: int) -> bool:
    """A one-letter token inside a ticker list or exchange tag: "VZ, T, TMUS", "(NYSE:T)"."""
    prev = tokens[i - 1].text if i else ""
    nxt = tokens[i + 1].text if i + 1 < len(tokens) else ""
    return prev in (",", ":", "(") or nxt in (",", ")")


def _is_name(norm: str, tokens: list[Token], at: list[Optional[_Span]], claimed: bytearray, j: int) -> bool:
    """A capitalized word with no lexicon role (or a cashtag): a candidate name."""
    t = tokens[j]
    if t.kind == "tag":
        return True
    if t.kind != "w" or len(t.text) < 2 or claimed[j] or at[j] is not None or t.text in _NAME_STOP:
        return False
    if j and tokens[j - 1].kind in ("num", "cur") and tokens[j - 1].end == t.start:
        return False  # "$150B": a unit, not a name
    return norm[t.start: t.start + 1].isupper()


_AUX = frozenset({"is", "are", "was", "were", "has", "have", "had", "will", "would", "could", "may", "might",
                  "just", "also", "now", "still", "stock", "stocks", "shares", "share", "price"})


def _title_case(norm: str, tokens: list[Token]) -> bool:
    """Headline Title Case capitalizes common words too, so a capital letter says little there."""
    words = [t for t in tokens[1:] if t.kind == "w" and len(t.text) >= 4]
    return len(words) >= 3 and sum(norm[t.start: t.start + 1].isupper() for t in words) >= 0.6 * len(words)


def _company_like(norm: str, tokens: list[Token], lo: int, hi: int, h: Hit, title: bool) -> bool:
    """Is the name run tokens[lo:hi] a company acting as the subject of ``h``? Without NER we ask for
    a company signal: a cashtag; a possessive or price noun after it ("BYD's", "Cerebras stock"); or
    the run right before the verb ("Qualcomm Is Losing") - which in Title Case also needs a number
    in the evidence or the run to open the sentence ("Affirm Drops 4%", not "Using Less Leverage")."""
    if any(t.kind == "tag" for t in tokens[lo:hi]):
        return True
    last = tokens[hi - 1]
    if norm[last.start: last.end].lower().endswith("'s") or (
            hi < len(tokens) and tokens[hi].text in ("stock", "stocks", "shares")):
        return True
    if not all(t.text in _AUX for t in tokens[hi:h.anchor]):
        return False
    if not title:
        return True
    numbered = any(tokens[j].kind in ("pct", "num") for j in range(h.start, h.end))
    opens = lo == 0 or tokens[lo - 1].kind == "sep" or tokens[lo - 1].text == ","  # "...; NVIDIA Ticks Up, AMD Slips"
    return numbered or opens


def _subject_is_other(norm: str, tokens: list[Token], at: list[Optional[_Span]], claimed: bytearray,
                      targets: set[int], h: Hit, title: bool) -> bool:
    """Is the nearest company-like subject before ``h`` (same sentence) a company other than the
    target? Coordinated subjects ("Ford, GM and Stellantis") and speakers ("Huang says") are not."""
    if any(h.start <= j < h.end for j in targets):
        return False  # the evidence names the target itself ("Microsoft boosts Nvidia orders")

    def run_start(j: int) -> tuple[int, bool]:  # (index before the name run, run contains the target)
        hit_target = False
        while j >= 0 and (j in targets or _is_name(norm, tokens, at, claimed, j)):
            hit_target = hit_target or j in targets
            j -= 1
        return j, hit_target

    j = h.anchor - 1
    while j >= 0 and tokens[j].kind != "sep":
        if tokens[j].text in _SPEECH or j in targets:
            return False
        if _is_name(norm, tokens, at, claimed, j):
            hi = j + 1
            j, is_target = run_start(j)
            if is_target:
                return False  # "Nvidia Blackwell sales surge": the run is the target's
            if not _company_like(norm, tokens, j + 1, hi, h, title):
                continue  # a capitalized common word in a title-case headline: keep looking
            while j >= 0 and tokens[j].text in ("and", "&", ",", "or"):  # "Ford, GM and Stellantis"
                j -= 1
                if j >= 0 and (j in targets or _is_name(norm, tokens, at, claimed, j)):
                    j, is_target = run_start(j)
                    if is_target:
                        return False
            return True
        j -= 1
    return False


def _attribute(norm: str, tokens: list[Token], at: list[Optional[_Span]], claimed: bytearray, hits: list[Hit],
               target: Sequence[str]) -> None:
    """Down-weight moves and results events whose subject is another named company - only when the
    target is mentioned at all (otherwise the caller's relevance filter decides)."""
    targets = _target_positions(tokens, norm, target)
    if not targets:
        return
    title = _title_case(norm, tokens)
    for h in hits:
        if (h.source == "move" or h.source in _SUBJECT_RULES) and _subject_is_other(norm, tokens, at, claimed,
                                                                                   targets, h, title):
            h.weight *= OFF_TARGET_FACTOR
