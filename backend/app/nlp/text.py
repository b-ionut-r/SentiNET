"""Text hygiene shared by the NLP layer: cleaning, publisher-suffix removal,
junk detection, dedup normalization, tokenization and light stemming.

Everything here is pure, deterministic and cheap (regex + dict lookups), so it
can run on every headline/post of an analysis without measurable cost.
"""
from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]{0,400}>")
# Inline formatting tags vanish without a gap ("<b>Nvidia</b>'s" -> "Nvidia's").
_INLINE_TAG_RE = re.compile(r"</?(?:a|b|i|u|em|strong|span|font|small|sup|sub|mark)\b[^>]{0,400}>", re.IGNORECASE)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# Zero-width/invisible chars, BOM, object-replacement char (StockTwits embeds U+FFFC).
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\ufffc\u00ad]")
_WS_RE = re.compile(r"\s+")

_ASCII_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", " ": " ", " ": " ", " ": " ", "…": "...",
})


def clean_text(text: str | None) -> str:
    """Strip HTML tags/entities, URLs, invisible characters and redundant
    whitespace. Emojis and punctuation are kept (they carry sentiment)."""
    if not text:
        return ""
    out = text
    for _ in range(2):  # feeds are often double-escaped ("&amp;#39;")
        unescaped = html.unescape(out)
        if unescaped == out:
            break
        out = unescaped
    out = _INLINE_TAG_RE.sub("", out)
    out = _TAG_RE.sub(" ", out)
    out = _URL_RE.sub(" ", out)
    out = _INVISIBLE_RE.sub("", out)
    out = unicodedata.normalize("NFC", out).replace(" ", " ")
    return _WS_RE.sub(" ", out).strip()


def fold(text: str) -> str:
    """Map typographic quotes/dashes/ellipses to ASCII so regexes match one form."""
    return text.translate(_ASCII_FOLD)


# --------------------------------------------------------------------------- #
# Publisher suffixes (" - Reuters", " | Fortune")
# --------------------------------------------------------------------------- #
# The last spaced separator: " - ", " | ", " — " (hyphens inside words like
# "Review-Journal" don't count).
_SUFFIX_SEP_RE = re.compile(r"\s+(?:-|\||–|—|―|::)\s+(?!.*\s(?:-|\||–|—|―|::)\s)")
_DOMAIN_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$", re.IGNORECASE)
_NOT_PUBLISHER_CHARS = re.compile(r"[?!%$\"]|\d{4,}")


def _looks_like_publisher(segment: str, publisher: str | None) -> bool:
    seg = segment.strip().rstrip(".")
    if not seg or len(seg) > 48:
        return False
    if publisher and seg.lower() == publisher.strip().lower():
        return True
    if _DOMAIN_RE.match(seg):
        return True
    from app.nlp.publishers import is_known_publisher  # local: publishers imports text

    if is_known_publisher(seg):
        return True
    if _NOT_PUBLISHER_CHARS.search(seg):
        return False
    words = seg.split()
    # Short Title-Case phrase ("Key Context by Tae Kim", "WGAU Radio", "FOX 5 Atlanta").
    return 1 <= len(words) <= 5 and all(
        w[:1].isupper() or w[:1].isdigit() or w.lower() in {"by", "of", "the", "and", "on", "&"} for w in words
    )


def strip_publisher_suffix(title: str, publisher: str | None = None) -> str:
    """Remove a trailing " - Publisher" / " | Publisher" segment (Google News
    style). At most two segments are removed; the second only when it is a
    known outlet or a domain. Never strips a title down to < 3 words."""
    out = title.strip()
    for attempt in range(2):
        m = _SUFFIX_SEP_RE.search(out)
        if not m:
            break
        head, tail = out[: m.start()].rstrip(" -|"), out[m.end():]
        exact = bool(publisher) and tail.strip().lower() == (publisher or "").strip().lower()
        if not head or (len(head.split()) < 3 and not exact):
            break
        if attempt == 0:
            ok = _looks_like_publisher(tail, publisher)
        else:
            from app.nlp.publishers import is_known_publisher

            ok = bool(_DOMAIN_RE.match(tail.strip())) or is_known_publisher(tail)
        if not ok:
            break
        out = head
    return out


# --------------------------------------------------------------------------- #
# Junk detection
# --------------------------------------------------------------------------- #
# Quote/data pages and auto-generated stubs that aggregators index as "news".
_BOILERPLATE_RE = re.compile(
    r"stock price,? news,? quote|forecast\s*-\s*price target\s*-\s*prediction|"
    r"stock forecast (?:&|and) price target|price prediction 20\d\d|"
    r"stock forecast and price target 20\d\d|insider trading activity 20\d\d|"
    r"^\W*\$?[A-Za-z.]{1,8}\s*\([A-Z.: ]{1,16}\)\W*$|stock quote (?:&|and) (?:chart|summary)|"
    r"live (?:stock )?price (?:chart|today)|real-time (?:stock )?quote|"
    r"\bfor sale in\b|\b[A-HJ-NPR-Z0-9]{17}\b|\bup for auction\b",
    re.IGNORECASE,
)
_CONTENT_WORD_RE = re.compile(r"(?<![$#@\w])[A-Za-z][A-Za-z'&-]*[A-Za-z]")


def content_words(text: str) -> list[str]:
    """Alphabetic words, excluding cashtags/hashtags/mentions."""
    return _CONTENT_WORD_RE.findall(fold(text))


def is_meaningful(text: str | None, min_words: int = 2) -> bool:
    """False for empty items, bare ticker lists ("$GNS $MU $NKE"), emoji-only
    posts and quote/data-page stubs ("Toast, Inc. (TOST) Stock Price, News,
    Quote & History"). `min_words` counts alphabetic words of 2+ letters."""
    if not text:
        return False
    cleaned = clean_text(text)
    if not cleaned or _BOILERPLATE_RE.search(fold(cleaned)):
        return False
    return len(content_words(cleaned)) >= min_words


# --------------------------------------------------------------------------- #
# Dedup normalization
# --------------------------------------------------------------------------- #
_ATTRIBUTION_RE = re.compile(
    r"\s+by investing\.com.*$|\s*\((?:[A-Z]{1,6}[.:]?\s?)?(?:NASDAQ|NYSE|NYSEARCA|NYSEAMERICAN|AMEX|OTC|OTCMKTS|TSX|LSE|ASX)"
    r"(?:\s?:\s?[A-Z.]{1,8})?\)|\s*\([A-Z.]{1,8}:(?:NASDAQ|NYSE|CA|US)\)",
    re.IGNORECASE,
)
_LEAD_LABEL_RE = re.compile(
    r"^(?:update\s*\d*|exclusive|breaking(?: news)?|watch|video|opinion|analysis|live|"
    r"premarket|market wrap|earnings call transcript|weekly recap)\s*[:|-]\s*",
    re.IGNORECASE,
)


def normalize_for_dedup(text: str) -> str:
    """Canonical form for syndication matching: lead labels ("Update:"),
    trailing attributions ("By Investing.com") and exchange tickers in
    parentheses removed, then tokenized (lower-case, money normalized so
    "$150 Billion" == "$150B", punctuation dropped)."""
    out = fold(clean_text(text))
    out = _LEAD_LABEL_RE.sub("", out)
    out = _ATTRIBUTION_RE.sub(" ", out)
    return " ".join(tokenize(out))


# --------------------------------------------------------------------------- #
# Tokenization & stemming
# --------------------------------------------------------------------------- #
def wordset(words: str) -> frozenset[str]:
    """Whitespace-separated word list -> frozenset (keeps long lists readable)."""
    return frozenset(words.split())


STOPWORDS: frozenset[str] = wordset("""
a about above after again against ago all almost along already also although always am among an and another any anyone
anything are aren't around as at away back be became because become been before being below between beyond both but by
came can can't cannot come comes could couldn't did didn't do does doesn't doing don't done down during each either else
even ever every few for from further get gets getting go goes going gone got had hadn't has hasn't have haven't having he
her here here's hers herself him himself his how how's however i i'm if in into is isn't it it's its itself just last
least less let let's like likely made make makes many may me might more most much must my myself near need needs neither
never next no nor not now of off often on once one only onto or other others our ours ourselves out over own per perhaps
put quite rather really s same see seen several shall she should shouldn't since so some something soon still such than
that that's the their theirs them themselves then there there's these they they're this those though through thus to too
toward towards under until up upon us use used very via want was wasn't way we we're well were weren't what what's when
where where's whether which while who who's whom whose why why's will with within without won't would wouldn't yet you
you're your yours yourself yourselves amid amidst despite across inside outside onto unto vs versus etc
""")

# Vocabulary shared by narratives/keywords ----------------------------------- #
# Words that carry no story/keyword identity in financial headlines.
GENERIC_WORDS: frozenset[str] = wordset("""
stock stocks share shares shareholder shareholders investor investors market markets today why here heres what whats
says said say report reports reported update news analyst analysts company companies inc corp corporation co ltd plc
nasdaq nyse wall street year years week weeks month months day days time new big could would should may might will just
now still next first last best better buy buying sell selling hold amid ahead after before over know need thing things
way ways look looks looking watch watching see sees seen get gets got make makes made take takes go goes going come
comes trading trade traders price prices value worth move moves moving lot lots key keys right left long short
huge massive major latest recent ever every much many more most less least one two three four five six seven eight
nine ten nearly almost about around above below likely set sets want wants deal deals plan plans plus via also into
against investing invest invested own owns owning point points case question questions answer answers reason reasons
simple strong message investment investments help helps keep keeps eyes enough number numbers fresh really here's
what's there's it's i'm don't can't won't isn't doesn't didn't let's you're they're we're
""")
# Common headline verbs: weak evidence of what a story is about.
HEADLINE_VERBS: frozenset[str] = wordset("""
unveil unveils unveiled launch launches launched expand expands expanded seek seeks sought face faces faced tap taps
tapped push pushes pushed bring brings brought offer offers offered show shows showed reveal reveals revealed signal
signals warn warns warned add adds added boost boosts boosted lift lifts lifted hit hits pass passes become becomes
remain remains stay stays turn turns turned call calls called name names named announce announces announced plan
plans planned prepare prepares weigh weighs consider considers explore explores join joins reach reaches reached
top tops topped lead leads drive drives driven put puts open opens deliver delivers return returns hold holds
""")
# Price-move words: a move is not a story ("stock rises").
MOVE_WORDS: frozenset[str] = wordset("""
rise rises rising rose risen fall falls falling fell drop drops dropped dropping slide slides sliding slid slip slips
slipped jump jumps jumped jumping climb climbs climbed climbing gain gains gained gaining surge surges surged surging
soar soars soared soaring plunge plunges plunged plunging tumble tumbles tumbled sink sinks sank rally rallies rallied
rallying pop pops popped edge edges edged higher lower up down percent pct rebound rebounds rebounded retreat retreats
retreated sell-off selloff
""")
# Months and weekdays.
CALENDAR_WORDS: frozenset[str] = wordset("january february march april may june july august september october november december jan feb mar "
                  "apr jun jul aug sep sept oct nov dec monday tuesday wednesday thursday friday saturday sunday")

_TOKEN_RE = re.compile(
    r"\$\d[\d,]*(?:\.\d+)?(?:\s?(?:trillion|billion|million|thousand|tn|bn|mn|[tbmk])\b)?"  # money
    r"|\d[\d,]*(?:\.\d+)?%"                                                               # percents
    r"|\$[a-z][a-z.]{0,7}(?<!\.)"                                                         # cashtags
    r"|\d[\d,]*(?:\.\d+)?"                                                                # numbers
    r"|[a-z][a-z0-9&'-]*[a-z0-9]|[a-z]",                                                  # words
)
_MONEY_RE = re.compile(r"^\$(\d[\d,]*(?:\.\d+)?)\s?(trillion|billion|million|thousand|tn|bn|mn|[tbmk])?$")
_MONEY_SCALE = {"trillion": "t", "tn": "t", "t": "t", "billion": "b", "bn": "b", "b": "b",
                "million": "m", "mn": "m", "m": "m", "thousand": "k", "k": "k"}


def normalize_money(token: str) -> str:
    """'$150 billion' / '$150bn' / '$150B' -> '$150b'; '$1,070' -> '$1070'."""
    m = _MONEY_RE.match(token.lower())
    if not m:
        return token
    number = m.group(1).replace(",", "")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return f"${number}{_MONEY_SCALE.get(m.group(2) or '', '')}"


def tokenize(text: str) -> list[str]:
    """Lower-cased tokens: words (possessive 's dropped, inner hyphens kept),
    numbers, percents, cashtags and normalized money amounts ('$150b').
    Spelled-out scales right after a bare number fold into money only when a
    '$' was present ("$1.05 billion")."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(fold(text).lower()):
        if tok.startswith("$") and tok[1:2].isdigit():
            tok = normalize_money(tok)
        elif tok.endswith("'s"):
            tok = tok[:-2]
        tok = tok.strip("'-")
        if tok:
            out.append(tok)
    return out


# Irregular forms map to a base word, which then goes through the rules.
_IRREGULAR = {
    "fell": "fall", "rose": "rise", "risen": "rise", "sank": "sink", "sunk": "sink", "slid": "slide",
    "won": "win", "bought": "buy", "sold": "sell", "led": "lead", "paid": "pay", "said": "say",
    "says": "say", "ran": "run", "grew": "grow", "grown": "grow", "took": "take", "taken": "take",
    "made": "make", "gave": "give", "given": "give", "left": "leave", "lost": "lose", "shook": "shake",
    "sought": "seek", "spent": "spend", "struck": "strike", "fought": "fight", "brought": "bring",
}
_KEEP = frozenset({"news", "series", "analysis", "data", "media", "us", "its", "this", "has", "was",
                   "does", "is", "always", "perhaps", "species", "chaos", "plus", "bonus", "status"})
_VOWELS = frozenset("aeiouy")


def _undouble(w: str) -> str:
    if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
        return w[:-1]
    return w


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """A light, conservative stemmer (plural, tense and final-e rules only) so
    'buybacks'/'buyback', 'unveils'/'unveiled', 'raises'/'raised'/'raising'
    collide. Stems are internal features, never shown to users."""
    w = word.lower()
    if w in _KEEP:
        return w
    w = _IRREGULAR.get(w, w)
    if len(w) <= 3 or not w.replace("-", "").isalpha():
        return w
    # 1) plurals / 3rd person
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "shes", "ches", "xes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    # 2) tense / gerund
    if w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("eed") and len(w) > 4:
        w = w[:-1]
    elif w.endswith("ing") and len(w) > 5 and _VOWELS & set(w[:-3]):
        w = _undouble(w[:-3])
    elif w.endswith("ed") and len(w) > 4 and _VOWELS & set(w[:-2]):
        w = _undouble(w[:-2])
    # 3) final e ("raise"/"rais", "price"/"pricing")
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def word_count(text: str) -> int:
    """Number of whitespace-separated words."""
    return len(text.split())


def is_mostly_upper(text: str) -> bool:
    """True when the text is shouted (>= 70% of letters upper-case)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 6:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.7


def is_title_case(text: str) -> bool:
    """True when most longer words are capitalized (headline style), which
    makes capitalization useless as a proper-noun signal."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text) if len(w) > 3]
    if len(words) < 3:
        return False
    return sum(w[0].isupper() for w in words) / len(words) >= 0.75
