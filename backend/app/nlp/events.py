"""Market-moving event detection in headlines and posts.

`detect_events(text)` returns `DetectedEvent`s in text order. Analyst actions
carry the brokerage (`firm`, canonicalized: "BofA Securities" / "Bank of
America" / "B of A" -> "Bank of America") and the new price target (`value`);
price moves carry the signed percent move. Direction of a price-target change
comes from the numbers when both are present ("raises ... to $69 from $72" is
a cut). Hypothetical/preview phrasing ("Will X Beat Estimates Again?",
"Poised to Beat", "Could Soar") does not produce earnings/price events, but
a reported event inside a question about its consequences does ("Should You
Buy Costco After Its Q4 Earnings Beat?"). Events are headline-level: long
posts are judged on their first sentences (~300 chars).

Patterns were tuned on real Google News / StockTwits phrasing (US, UK and
Nordic broker notes, MarketBeat/TradingView auto-headlines, law-firm blasts);
see tests/nlp/test_events.py for the real cases, false friends included.
Known limit: events are text-level — "Burford stock jumps after jury orders
Apple to pay" yields price_up without saying whose price moved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

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
    wordset,
)
from app.nlp.types import DetectedEvent
from app.sources.base import CompanyRef

EVENT_LABELS: dict[str, str] = {
    "analyst_upgrade": "Analyst upgrade",
    "analyst_downgrade": "Analyst downgrade",
    "analyst_initiate": "Coverage initiated",
    "analyst_top_pick": "Named top pick",
    "pt_raise": "Price target raised",
    "pt_cut": "Price target cut",
    "earnings_beat": "Earnings beat",
    "earnings_miss": "Earnings miss",
    "guidance_raise": "Guidance raised",
    "guidance_cut": "Guidance cut",
    "record_results": "Record results",
    "buyback": "Buyback",
    "dividend_raise": "Dividend raised",
    "dividend_cut": "Dividend cut",
    "layoffs": "Layoffs",
    "lawsuit": "Lawsuit",
    "investigation": "Investigation",
    "settlement": "Settlement",
    "m_and_a": "M&A",
    "partnership": "Partnership",
    "contract_win": "Contract win",
    "product_launch": "Product launch",
    "recall": "Recall",
    "exec_departure": "Executive departure",
    "exec_hire": "Executive hire",
    "offering": "Share offering",
    "bankruptcy": "Bankruptcy risk",
    "delisting": "Delisting risk",
    "short_report": "Short-seller report",
    "insider_buy": "Insider buying",
    "insider_sell": "Insider selling",
    "all_time_high": "All-time high",
    "low_52w": "52-week low",
    "high_52w": "52-week high",
    "stock_split": "Stock split",
    "price_up": "Price jump",
    "price_down": "Price drop",
    "regulatory_approval": "Regulatory approval",
    "regulatory_setback": "Regulatory/trial setback",
    "index_inclusion": "Index inclusion",
    "data_breach": "Breach/outage",
}

EVENT_POLARITY: dict[str, str] = {
    "analyst_upgrade": "bull", "analyst_downgrade": "bear", "analyst_initiate": "neutral",
    "analyst_top_pick": "bull", "pt_raise": "bull", "pt_cut": "bear", "earnings_beat": "bull",
    "earnings_miss": "bear", "guidance_raise": "bull", "guidance_cut": "bear", "record_results": "bull",
    "buyback": "bull", "dividend_raise": "bull", "dividend_cut": "bear", "layoffs": "bear", "lawsuit": "bear",
    "investigation": "bear", "settlement": "neutral", "m_and_a": "neutral", "partnership": "bull",
    "contract_win": "bull", "product_launch": "bull", "recall": "bear", "exec_departure": "bear",
    "exec_hire": "neutral", "offering": "bear", "bankruptcy": "bear", "delisting": "bear", "short_report": "bear",
    "insider_buy": "bull", "insider_sell": "bear", "all_time_high": "bull", "low_52w": "bear", "high_52w": "bull",
    "stock_split": "bull", "price_up": "bull", "price_down": "bear", "regulatory_approval": "bull",
    "regulatory_setback": "bear", "index_inclusion": "bull", "data_breach": "bear",
}

# Theme implied by each event (see app.nlp.themes.THEMES).
EVENT_THEMES: dict[str, str] = {
    "analyst_upgrade": "analyst", "analyst_downgrade": "analyst", "analyst_initiate": "analyst",
    "analyst_top_pick": "analyst", "pt_raise": "analyst", "pt_cut": "analyst", "earnings_beat": "earnings",
    "earnings_miss": "earnings", "record_results": "earnings", "guidance_raise": "guidance",
    "guidance_cut": "guidance", "buyback": "capital_return", "dividend_raise": "capital_return",
    "dividend_cut": "capital_return", "layoffs": "labor", "lawsuit": "legal", "investigation": "legal",
    "settlement": "legal", "bankruptcy": "legal", "m_and_a": "deals", "partnership": "deals",
    "contract_win": "deals", "product_launch": "product", "recall": "product", "exec_departure": "management",
    "exec_hire": "management", "short_report": "trading", "stock_split": "trading", "index_inclusion": "trading",
    "all_time_high": "trading", "low_52w": "trading", "high_52w": "trading",
    "insider_buy": "insider", "insider_sell": "insider", "delisting": "regulatory",
    "regulatory_approval": "regulatory", "regulatory_setback": "regulatory", "data_breach": "legal",
}

# --------------------------------------------------------------------------- #
# Brokerages
# --------------------------------------------------------------------------- #
# canonical -> aliases. Aliases in _CASED_FIRMS are common words and match only
# with their exact capitalization ("Benchmark", "Citizens", "Wood").
_FIRMS: dict[str, tuple[str, ...]] = {
    "Morgan Stanley": ("Morgan Stanley",),
    "Goldman Sachs": ("Goldman Sachs", "Goldman"),
    "JPMorgan": ("JPMorgan Chase & Co.", "JPMorgan Chase", "JPMorgan", "JP Morgan", "J.P. Morgan", "J.P.Morgan"),
    "Bank of America": ("BofA Securities", "BofA Global Research", "Bank of America Securities", "Bank of America",
                        "BofA", "B of A Securities", "B of A", "BoA"),
    "Citi": ("Citigroup", "Citi"),
    "Wells Fargo": ("Wells Fargo & Company", "Wells Fargo"),
    "Barclays": ("Barclays",),
    "UBS": ("UBS",),
    "Deutsche Bank": ("Deutsche Bank",),
    "HSBC": ("HSBC",),
    "Jefferies": ("Jefferies",),
    "Mizuho": ("Mizuho",),
    "Nomura": ("Nomura Instinet", "Nomura"),
    "Macquarie": ("Macquarie",),
    "RBC Capital Markets": ("RBC Capital Markets", "RBC Capital", "RBC"),
    "BMO Capital Markets": ("BMO Capital Markets", "BMO Capital", "BMO"),
    "TD Cowen": ("TD Cowen", "Cowen"),
    "TD Securities": ("TD Securities",),
    "Truist": ("Truist Securities", "Truist Financial", "Truist"),
    "Piper Sandler": ("Piper Sandler",),
    "Raymond James": ("Raymond James Financial", "Raymond James"),
    "Stifel": ("Stifel Nicolaus", "Stifel"),
    "Baird": ("Robert W. Baird", "Robert W Baird", "Baird"),
    "KeyBanc": ("KeyBanc Capital Markets", "KeyBanc"),
    "Wedbush": ("Wedbush Securities", "Wedbush"),
    "Oppenheimer": ("Oppenheimer",),
    "Needham": ("Needham & Company", "Needham"),
    "Bernstein": ("Sanford C. Bernstein", "AllianceBernstein", "Bernstein SocGen", "Bernstein"),
    "Evercore ISI": ("Evercore ISI", "Evercore"),
    "Guggenheim": ("Guggenheim Securities", "Guggenheim"),
    "Loop Capital": ("Loop Capital",),
    "Rosenblatt": ("Rosenblatt Securities", "Rosenblatt"),
    "Cantor Fitzgerald": ("Cantor Fitzgerald", "Cantor"),
    "B. Riley": ("B. Riley Securities", "B. Riley", "B.Riley", "B Riley"),
    "BTIG": ("BTIG Research", "BTIG"),
    "D.A. Davidson": ("D.A. Davidson", "DA Davidson"),
    "Canaccord Genuity": ("Canaccord Genuity", "Canaccord"),
    "Citizens JMP": ("Citizens JMP", "JMP Securities"),
    "Susquehanna": ("Susquehanna",),
    "Melius Research": ("Melius Research", "Melius"),
    "Redburn": ("Rothschild & Co Redburn", "Redburn Atlantic", "Redburn"),
    "Daiwa": ("Daiwa",),
    "CLSA": ("CLSA",),
    "Argus": ("Argus Research", "Argus"),
    "Morningstar": ("Morningstar",),
    "CFRA": ("CFRA",),
    "Wolfe Research": ("Wolfe Research",),
    "Seaport": ("Seaport Research Partners", "Seaport Global", "Seaport"),
    "William Blair": ("William Blair",),
    "Stephens": ("Stephens",),
    "KBW": ("Keefe, Bruyette & Woods", "Keefe Bruyette", "KBW"),
    "Telsey Advisory": ("Telsey Advisory Group", "Telsey Advisory", "Telsey"),
    "BNP Paribas": ("BNP Paribas Exane", "BNP Paribas", "BNP", "Exane"),
    "Societe Generale": ("Societe Generale", "SocGen"),
    "Berenberg": ("Berenberg",),
    "Kepler Cheuvreux": ("Kepler Cheuvreux", "Kepler"),
    "Scotiabank": ("Scotiabank",),
    "CIBC": ("CIBC",),
    "National Bank": ("National Bank Financial",),
    "Desjardins": ("Desjardins",),
    "ATB Capital": ("ATB Capital Markets", "ATB"),
    "H.C. Wainwright": ("HC Wainwright & Co.", "H.C. Wainwright", "HC Wainwright"),
    "Roth": ("Roth Capital", "Roth MKM", "Roth"),
    "Craig-Hallum": ("Craig-Hallum",),
    "Lake Street": ("Lake Street Capital", "Lake Street"),
    "Northland": ("Northland Securities", "Northland"),
    "Ladenburg Thalmann": ("Ladenburg Thalmann", "Ladenburg"),
    "Maxim Group": ("Maxim Group",),
    "Benchmark": ("Benchmark",),
    "Chardan": ("Chardan",),
    "Leerink Partners": ("Leerink Partners", "SVB Leerink", "Leerink"),
    "Janney": ("Janney Montgomery Scott", "Janney"),
    "Compass Point": ("Compass Point",),
    "Freedom Broker": ("Freedom Capital Markets", "Freedom Capital", "Freedom Broker"),
    "Zacks": ("Zacks Investment Research", "Zacks Research", "Zacks"),
    "Wall Street Zen": ("Wall Street Zen",),
    "Weiss Ratings": ("Weiss Ratings",),
    "Phillip Securities": ("Phillip Securities",),
    "DBS": ("DBS Bank", "DBS"),
    "Itau BBA": ("Itau BBA",),
    "BTG Pactual": ("BTG Pactual",),
    "Santander": ("Santander",),
    "Monness Crespi": ("Monness, Crespi, Hardt", "Monness Crespi", "Monness"),
    "New Street Research": ("New Street Research", "New Street"),
    "MoffettNathanson": ("MoffettNathanson", "Moffett Nathanson"),
    "Tigress Financial": ("Tigress Financial", "Tigress"),
    "Erste Group": ("Erste Group", "Erste"),
    "DZ Bank": ("DZ Bank",),
    "Commerzbank": ("Commerzbank",),
    "Investec": ("Investec",),
    "Peel Hunt": ("Peel Hunt",),
    "Panmure Liberum": ("Panmure Liberum", "Panmure Gordon", "Liberum"),
    "Oddo BHF": ("Oddo BHF", "Oddo"),
    "Bryan Garnier": ("Bryan Garnier", "Bryan, Garnier"),
    "Mediobanca": ("Mediobanca",),
    "Pareto Securities": ("Pareto Securities", "Pareto"),
    "DNB Carnegie": ("DNB Carnegie", "DNB Markets", "Carnegie"),
    "SEB": ("SEB",),
    "Nordea": ("Nordea",),
    "Handelsbanken": ("Handelsbanken",),
    "Danske Bank": ("Danske Bank",),
    "SB1 Markets": ("SpareBank 1 Markets", "SB1 Markets", "SB1"),
    "ABG Sundal Collier": ("ABG Sundal Collier", "ABG"),
    "Arctic Securities": ("Arctic Securities",),
    "Inderes": ("Inderes",),
    "Swedbank": ("Swedbank",),
    "Jefferies & Co": (),
    "StoneX": ("StoneX",),
    "Jones Trading": ("JonesTrading", "Jones Trading"),
    "Texas Capital": ("Texas Capital",),
    "Hovde Group": ("Hovde Group", "Hovde"),
    "Rodman & Renshaw": ("Rodman & Renshaw",),
    "D. Boral Capital": ("D. Boral Capital", "D. Boral"),
    "Barrington Research": ("Barrington Research",),
    "Huntington": ("Huntington Securities",),
    "Wood & Company": ("Wood & Company", "Wood & Co", "Wood"),
    "Kenanga": ("Kenanga",),
    "Shenwan Hongyuan": ("Shenwan Hongyuan",),
    "CICC": ("CICC",),
    "Citic Securities": ("CITIC Securities", "Citic Securities"),
    "Haitong": ("Haitong",),
    "MarketsMOJO": ("MarketsMOJO",),
    "GuruFocus": (),
    "Emerging Growth Research": ("Emerging Growth Research",),
    "Redeye": ("Redeye",),
    "Hightower": (),
}
_CASED_FIRMS = frozenset({"Benchmark", "Wood", "Stephens", "Northland", "Roth", "Argus", "Cantor", "Kepler",
                          "Erste", "Oddo", "Santander", "Seaport", "Hovde", "Chardan", "Monness", "Liberum",
                          "Carnegie", "Pareto", "Exane", "Tigress", "Redeye", "Inderes", "Zacks", "Morningstar",
                          "Goldman", "Evercore", "Telsey", "Janney", "Ladenburg", "Leerink", "Melius", "SEB",
                          "Nordea", "BMO", "ATB", "DBS", "ABG", "SB1", "BNP", "BoA", "Baird", "Needham", "Cowen",
                          "Truist", "Stifel", "Macquarie", "Nomura", "Daiwa", "Mizuho"})

_FIRM_CANON: dict[str, str] = {}
for _canon, _aliases in _FIRMS.items():
    for _alias in _aliases:
        _FIRM_CANON[_alias.lower()] = _canon


def _alias_re(alias: str) -> str:
    return re.escape(alias).replace(r"\ ", r"\s+")


_FIRM_ALIASES = sorted((a for aliases in _FIRMS.values() for a in aliases), key=len, reverse=True)
_FIRM_RE = re.compile(
    r"(?<![\w&.])(?:" + "|".join(
        (f"(?-i:{_alias_re(a)})" if a in _CASED_FIRMS else _alias_re(a)) for a in _FIRM_ALIASES
    ) + r")(?:'s)?(?![\w&])",
    re.IGNORECASE,
)
_GENERIC_ACTORS = frozenset({"analyst", "analysts", "wall street", "street", "firm", "brokerage", "bank", "research",
                             "report", "update", "exclusive", "breaking", "watch", "stock", "shares", "the", "this",
                             "why", "how", "what", "it", "investors", "weekly recap"})


def is_known_firm(name: str) -> bool:
    """True when `name` is a brokerage alias ("morgan stanley", "BofA")."""
    return re.sub(r"\s+", " ", re.sub(r"'s$", "", name.strip())).lower() in _FIRM_CANON


def canonical_firm(name: str) -> str:
    """'BofA Securities' -> 'Bank of America'; unknown names are returned trimmed."""
    key = re.sub(r"'s$", "", name.strip()).lower()
    key = re.sub(r"\s+", " ", key)
    return _FIRM_CANON.get(key, re.sub(r"'s$", "", name.strip()))


# --------------------------------------------------------------------------- #
# Shared vocabulary
# --------------------------------------------------------------------------- #
_NUM = r"\d[\d,]*(?:\.\d+)?"
_CUR_PRE = (r"(?:(?:US|C|CA|A|AU|NZ|HK|S)\$|\$|€|£|¥|₹|Rs\.?\s?|"
            r"(?:USD|EUR|GBP|GBp|CAD|AUD|CHF|SEK|NOK|DKK|JPY|INR|HKD|CNY|RMB)\s?)")
_CUR_POST = r"(?:\s?(?:Danish kroner|Norwegian kroner|Swedish kronor|kronor|kroner|euros?|pence|p|USD|EUR|SEK|NOK|DKK))"
_MONEY = rf"(?:{_CUR_PRE}{_NUM}|{_NUM}{_CUR_POST}|{_NUM})"
_RATING = (r"(?:strong[- ]buy|buy|outperform|overweight|accumulate|add|positive|sector outperform|market outperform|"
           r"conviction buy|top pick|equal[- ]weight|equalweight|neutral|hold|market perform|sector perform|"
           r"peer perform|in-line|in line|mixed|sector weight|market weight|underperform|underweight|sell|"
           r"strong[- ]sell|reduce|negative|sector underperform|market underperform|speculative buy)")
_BULL_RATING = re.compile(r"strong[- ]buy|^buy|outperform|overweight|accumulate|^add|positive|top pick|"
                          r"conviction buy|speculative buy", re.IGNORECASE)
_BEAR_RATING = re.compile(r"underperform|underweight|sell|reduce|negative", re.IGNORECASE)

# Price-target nouns; a bare "target" counts only in PT-shaped contexts (not "Target", the retailer).
_PT = (r"(?:(?:12[- ]month\s+|1-year\s+|one-year\s+)?price[- ]targets?|target[- ]prices?|price objectives?|"
       r"\bPTs?\b|(?:(?<=its )|(?<=their )|(?<=the )|(?<=his )|(?<=her )|(?<=a )|(?<=stock )|(?<=share )|"
       r"(?<=street )|(?<=analyst )|(?<=analysts' )|(?<=consensus )|(?<=average ))targets?\b|"
       r"targets?(?=\s+(?:to|from|on|for|of)\b|\s*$|\s*[,.;:-]))")
_PT_UP = (r"raise[sd]?|raising|lift(?:s|ed|ing)?|boost(?:s|ed|ing)?|hike[sd]?|hiking|up(?:s|ped)|upp(?:ed|ing)|"
          r"bump(?:s|ed)?(?: up)?|increase[sd]?|increasing|double[sd]?|doubling|triple[sd]?|tripling|"
          r"nudge[sd]?(?: up)?|ratchet(?:s|ed)? up|lifts?")
_PT_DOWN = (r"cut[s]?|cutting|lower(?:s|ed|ing)?|trim(?:s|med|ming)?|slash(?:es|ed|ing)?|reduce[sd]?|reducing|"
            r"halve[sd]?|halving|chop(?:s|ped)?|pare[sd]?|paring|shave[sd]?|ratchet(?:s|ed)? down")
_PT_NEUTRAL = r"reset[s]?|revamp[s]?|tweak[s]?|adjust(?:s|ed)?|update[sd]?|set[s]?|keep[s]?|kept|maintain(?:s|ed)?|reiterate[sd]?"
_PASSIVE_UP = r"raised|lifted|boosted|hiked|increased|upped|bumped(?: up)?|doubled|tripled|nudged up"
_PASSIVE_DOWN = r"cut|lowered|trimmed|slashed|reduced|halved|chopped|pared|shaved"
# Up to four words between a PT verb and its noun, never crossing another PT verb.
_GAP = (r"(?:(?!(?:rais|lift|boost|hik|cut|lower|trim|slash|reduc|reiterat|maintain|keep|set|upgrad|downgrad)"
        r"\w*\b)[\w.&'$-]+\s+){0,4}?")

_PT_ACTIVE_RE = re.compile(rf"\b(?P<verb>{_PT_UP}|{_PT_DOWN}|{_PT_NEUTRAL})\s+(?:its\s+|their\s+|the\s+)?{_GAP}{_PT}",
                           re.IGNORECASE)
_PT_PASSIVE_RE = re.compile(rf"{_PT}\s+(?:\w+\s+){{0,2}}?(?P<verb>{_PASSIVE_UP}|{_PASSIVE_DOWN})\b", re.IGNORECASE)
_PT_NOUN_RE = re.compile(rf"\b(?P<verb>higher|lower|raised|reduced)\s+(?:[\w.&'-]+\s+){{0,2}}?{_PT}"
                         rf"|(?:price[- ]targets?|(?:(?<=street )|(?<=analyst )|(?<=analysts' )|(?<=stock )|"
                         rf"(?<=share )|(?<=its )|(?<=their ))targets?)\s+(?P<verb2>hikes?|raises?|increases?|boosts?|"
                         rf"cuts?|reductions?)\b(?!\s+(?:prices?|jobs|costs?|staff)\b)", re.IGNORECASE)

_HYPOTHETICAL_RE = re.compile(
    r"(?:\b(?:will|would|could|can|may|might|should|poised to|set to|expected to|likely to|on track to|"
    r"aims? to|hopes? to|if|whether)\s+(?:[\w().&'-]+\s+){0,3}$)",
    re.IGNORECASE,
)
_UPGRADE_NOISE_RE = re.compile(r"upgrade (?:cycle|supercycle|path|program)|(?:guidance|outlook|forecast|credit|"
                               r"earnings|estimate|eps) upgrades?|upgrades? (?:to|for) (?:ios|android|windows|"
                               r"its network|the network|infrastructure|software|firmware|the grid)|network upgrade|"
                               r"software upgrade|free upgrade|upgrade(?:d|s)? (?:its|their|the) (?:network|fleet|"
                               r"facilit|plant|systems?|stores?|app)", re.IGNORECASE)


@dataclass
class _Hit:
    start: int
    key: str
    polarity: str
    firm: str | None = None
    value: float | None = None
    span: str | None = None
    end: int = -1  # end of the matched phrase (for attribution); -1 = start + len(span)

    @property
    def stop(self) -> int:
        return self.end if self.end >= 0 else self.start + len(self.span or "")


def _num(raw: str) -> float | None:
    digits = re.search(_NUM, raw)
    if not digits:
        return None
    try:
        return float(digits.group(0).replace(",", ""))
    except ValueError:
        return None


# Clause boundaries, ignoring initials and abbreviations ("T. Rowe", "Chase & Co.").
_CLAUSE_SPLIT_RE = re.compile(r"(?<!\b[A-Z])(?<!\bInc)(?<!\bCo)(?<!\bCorp)(?<!\bLtd)(?<!\bSt)(?<!\bU\.S)\.\s|"
                              r"[;:!?]\s|\s[-|]\s")


# A reported event inside a question about its consequences is still a fact:
# "Should You Buy Costco Stock After Its Q4 Earnings Beat?", "Will Dividend
# Hike Change City Holding's Narrative?".
_FACT_BEFORE_RE = re.compile(r"\b(?:after|following|on|despite|amid|post)\s+(?:its|the|a|an|their|this|that|"
                             r"[\w.&'-]+'s)?\s*(?:[\w.&'-]+\s+){0,2}$", re.IGNORECASE)
_CONSEQUENCE_AFTER_RE = re.compile(
    r"^[^.?!]{0,40}?\b(?:change|changes|alter|alters|shift|shifts|reshape|lift|lifts|boost|boosts|spark|sparks|drive|"
    r"drives|fuel|fuels|help|helps|hurt|hurts|matter|matters|mean|means|signal|signals|support|supports|keep|keeps|"
    r"justify|justifies|make|makes|put|puts|send|sends|power|powers)\b", re.IGNORECASE)


_NOUN_EVENT_RE = re.compile(r"(?:dividend|payout|earnings|eps|revenue|sales|profit|q[1-4]|guidance|outlook|forecast|"
                            r"buyback|repurchase|delivery|deliveries|stock split|share split)\b", re.IGNORECASE)


def _is_hypothetical(text: str, start: int, end: int | None = None) -> bool:
    """Modal/preview phrasing right before the event, or a yes/no question
    ("Can X spark a rally?") — unless the question is about the consequences
    of an event that already happened (see _FACT_BEFORE_RE)."""
    if _FACT_BEFORE_RE.search(text[max(0, start - 40):start]):
        return False
    if end is not None and _NOUN_EVENT_RE.match(text[start:end]) and _CONSEQUENCE_AFTER_RE.match(text[end:]):
        return False
    clause = _CLAUSE_SPLIT_RE.split(text[max(0, start - 60):start])[-1]
    if _HYPOTHETICAL_RE.search(clause):
        return True
    sentence = _CLAUSE_SPLIT_RE.split(text[:start])[-1]
    rest = text[start:]
    end = re.search(r"[.!?]", rest)
    return (bool(re.match(r"\s*(?:can|could|will|would|should|might|may)\b", sentence, re.IGNORECASE))
            and end is not None and end.group(0) == "?")


_ANALYST_VERB_AFTER_RE = re.compile(
    r"^(?:'s)?\s+(?:analysts?\s+)?(?:maintains?|reiterates?|keeps?|raises?|lifts?|boosts?|hikes?|cuts?|lowers?|trims?|"
    r"slashes?|sets?|upgrades?|downgrades?|initiates?|resumes?|assumes?|reinstates?|starts?|names?|adds?)\b",
    re.IGNORECASE,
)


def _firm_near(text: str, start: int, end: int, allow_actor: bool = False) -> str | None:
    """The brokerage acting in [start, end): inside it, right after it ("at/by/
    from Citi"), or before it in the same clause — preferring a firm followed
    by an analyst verb ("RBC Maintains Goldman Sachs ..., Raises Target" is
    RBC's note), else the nearest. With `allow_actor`, an unknown one-word
    actor opening the clause also counts ("Wood lifts price target") — only
    safe for price-target phrasing, where the subject of "raises price
    target" is always the broker."""
    nearest: tuple[int, str] | None = None
    subject: str | None = None
    for m in _FIRM_RE.finditer(text):
        name = m.group(0)
        if m.start() >= start and m.end() <= end:
            return canonical_firm(name)
        if m.end() <= start:
            gap = text[m.end():start]
            if len(gap) <= 90 and not re.search(r"[.;!?]\s|\s[-|]\s", gap):
                dist = start - m.end()
                if nearest is None or dist < nearest[0]:
                    nearest = (dist, canonical_firm(name))
                if subject is None and _ANALYST_VERB_AFTER_RE.match(text[m.end():m.end() + 40]):
                    subject = canonical_firm(name)
        elif m.start() >= end:
            gap = text[end:m.start()]
            if len(gap) <= 50 and re.search(r"\b(?:by|at|from|with|via)\s+(?:\w+\s+){0,1}$", gap, re.IGNORECASE):
                return canonical_firm(name)
    if subject or nearest:
        return subject or nearest[1]  # type: ignore[index]
    if not allow_actor:
        return None
    m = re.search(r"(?:^|[:;]\s*|\b(?:as|after|amid|following|when|while)\s+)((?:[A-Z][\w.&'-]*\s){1,3})$",
                  text[:start])
    if m:
        actor = m.group(1).strip()
        brokerish = len(actor.split()) == 1 or re.search(
            r"\b(?:Securities|Capital|Research|Partners|Markets|Advisors|Equities|Bank|& Co\.?)$", actor)
        if brokerish and actor.lower() not in _GENERIC_ACTORS:
            return canonical_firm(actor)
    return None


def _pt_values(text: str, end: int) -> tuple[float | None, float | None]:
    """(new, old) price target from the text following a PT phrase."""
    after = text[end:end + 90]
    to_m = re.search(rf"\bto\s+({_MONEY})", after, re.IGNORECASE)
    from_m = re.search(rf"\bfrom\s+({_MONEY})", after, re.IGNORECASE)
    new = _num(to_m.group(1)) if to_m else None
    old = _num(from_m.group(1)) if from_m else None
    if to_m and old is None:  # marketscreener style: "to €67 (64)" / "to SEK 120 (115)"
        paren = re.match(rf"\s*\((?:from\s+)?({_MONEY})\)", after[to_m.end():], re.IGNORECASE)
        if paren:
            old = _num(paren.group(1))
    if new is None:
        before = text[max(0, end - 40):end]
        pre = re.search(rf"({_CUR_PRE}{_NUM})\s+(?:[\w.&'-]+\s+){{0,2}}?(?:price\s+)?targets?\s*$", before,
                        re.IGNORECASE)
        if pre:
            new = _num(pre.group(1))
    return new, old


def _rating_after(text: str, pos: int) -> str | None:
    m = re.search(rf"\b(?:to|at|with(?: an?)?|as)\s+(?:a\s+|an\s+)?['\"]?({_RATING})\b", text[pos:pos + 60],
                  re.IGNORECASE)
    return m.group(1).lower() if m else None


def _analyst_events(text: str) -> list[_Hit]:
    hits: list[_Hit] = []
    # --- price targets ---------------------------------------------------- #
    for regex in (_PT_ACTIVE_RE, _PT_PASSIVE_RE, _PT_NOUN_RE):
        for m in regex.finditer(text):
            if _is_hypothetical(text, m.start()):
                continue
            verb = (m.groupdict().get("verb") or m.groupdict().get("verb2") or "").lower()
            if re.fullmatch(rf"(?:{_PT_UP}|{_PASSIVE_UP}|higher|raised|hikes?|raises?|increases?|boosts?)", verb,
                            re.IGNORECASE):
                direction = 1
            elif re.fullmatch(rf"(?:{_PT_DOWN}|{_PASSIVE_DOWN}|lower|reduced|cuts?|reductions?)", verb,
                              re.IGNORECASE):
                direction = -1
            else:
                direction = 0
            new, old = _pt_values(text, m.end())
            if new is not None and old is not None and new != old:
                direction = 1 if new > old else -1
            if direction == 0:
                continue
            key = "pt_raise" if direction > 0 else "pt_cut"
            span_end = min(len(text), m.end() + 40)
            hits.append(_Hit(m.start(), key, "bull" if direction > 0 else "bear",
                             firm=_firm_near(text, m.start(), m.end(), allow_actor=True), value=new,
                             span=text[m.start():span_end].strip(), end=m.end()))
    # --- rating changes ----------------------------------------------------- #
    rating_patterns = (
        ("analyst_upgrade", re.compile(r"\b(?:double[- ])?upgrad(?:e[sd]?|ing)\b|\bturns? bullish\b|"
                                       r"\b(?:raise[sd]?|lift(?:s|ed)?|boost(?:s|ed)?)\s+(?:its\s+|the\s+)?(?:stock\s+)?"
                                       r"rating\b(?!\s+(?:agency|agencies|outlook))|"
                                       r"\b(?:raise[sd]?|lift(?:s|ed)?|move[sd]?|bump(?:s|ed)?)\s+(?:\w+\s+){0,3}?"
                                       r"to\s+(?:strong[- ]buy|buy|outperform|overweight)\b", re.IGNORECASE)),
        ("analyst_downgrade", re.compile(r"\b(?:double[- ])?downgrad(?:e[sd]?|ing)\b|\bturns? bearish\b|"
                                         r"\b(?:cut[s]?|lower(?:s|ed)?|slash(?:es|ed)?)\s+(?:its\s+|the\s+)?(?:stock\s+)?"
                                         r"rating\b(?!\s+(?:agency|agencies|outlook))|"
                                         r"\b(?:cut[s]?|lower(?:s|ed)?|move[sd]?)\s+(?:\w+\s+){0,3}?"
                                         r"to\s+(?:sell|underperform|underweight|strong[- ]sell|reduce)\b",
                                         re.IGNORECASE)),
        ("analyst_initiate", re.compile(r"\b(?:initiat(?:es|ed|ing|e)|starts?|started|launch(?:es|ed)|begins?|began|"
                                        r"resumes?|resumed|assumes?|assumed|picks? up|reinstates?|reinstated)\s+"
                                        r"(?:\w+\s+){0,2}?coverage\b|\binitiated\s+(?:at|with)\b|"
                                        r"\binitiates?\s+(?:[\w.&'-]+\s+){1,4}?(?:at|with)\s+(?:an?\s+)?"
                                        rf"['\"]?{_RATING}\b", re.IGNORECASE)),
        ("analyst_top_pick", re.compile(r"\b(?:top|best|favou?rite)\s+(?:\w+\s+){0,2}?(?:pick|idea)\b(?!s)|"
                                        r"\b(?:conviction|focus|top picks?) list\b", re.IGNORECASE)),
    )
    for key, regex in rating_patterns:
        for m in regex.finditer(text):
            if _is_hypothetical(text, m.start()):
                continue
            if key in {"analyst_upgrade", "analyst_downgrade"}:
                window = text[max(0, m.start() - 60):m.end() + 60]
                if _UPGRADE_NOISE_RE.search(window):
                    continue
                if not re.search(rf"{_RATING}|analyst|rating|stock|shares|\bpt\b|price target|coverage|"
                                 r"\bto (?:buy|sell|hold)\b", window, re.IGNORECASE) and not _FIRM_RE.search(window):
                    continue
            firm = _firm_near(text, m.start(), m.end())
            if key == "analyst_top_pick" and not (firm or re.search(r"\bnam(?:es|ed)|regains|back as|adds?\b",
                                                                    text[max(0, m.start() - 50):m.start()],
                                                                    re.IGNORECASE)):
                continue
            polarity = EVENT_POLARITY[key]
            if key == "analyst_initiate":
                rating = _rating_after(text, m.start())
                if rating:
                    polarity = "bull" if _BULL_RATING.search(rating) else "bear" if _BEAR_RATING.search(rating) \
                        else "neutral"
            hits.append(_Hit(m.start(), key, polarity, firm=firm,
                             span=text[m.start():min(len(text), m.end() + 50)].strip(), end=m.end()))
    # One note usually comes from one broker: share a firm found by any analyst event.
    known = next((h.firm for h in hits if h.firm), None)
    for h in hits:
        h.firm = h.firm or known
    # Attach a stated target to rating events ("upgrades to Overweight, sets $191 price target").
    stated = next((h.value for h in hits if h.value is not None), None)
    if stated is None:
        m = re.search(rf"({_CUR_PRE}{_NUM})\s+(?:price\s+)?(?:target|PT)\b|(?:price target|target price|PT)\s+"
                      rf"(?:of\s+|to\s+|at\s+)?({_CUR_PRE}{_NUM})|{_RATING}(?:\s+rating)?\s+(?:at|with)\s+(?:an?\s+)?"
                      rf"({_CUR_PRE}{_NUM})", text, re.IGNORECASE)
        if m:
            stated = _num(m.group(1) or m.group(2) or m.group(3))
    for h in hits:
        if h.value is None and h.key in {"analyst_upgrade", "analyst_downgrade", "analyst_initiate", "analyst_top_pick"}:
            h.value = stated
    return hits


# --------------------------------------------------------------------------- #
# Everything else: (key, pattern, hypothetical-guard)
# --------------------------------------------------------------------------- #
_EARN = (r"(?:earnings|eps|profits?|net income|revenues?|sales|results|top[- ]line|bottom[- ]line|quarter|"
         r"quarterly (?:results|report|numbers)|q[1-4]|(?:first|second|third|fourth)[- ]quarter|fiscal q[1-4]|"
         r"deliveries|delivery|comps|same-store sales|comparable sales|bookings|subscribers|margins?|numbers|report)")
_EST = (r"(?:estimates?|expectations|forecasts?|consensus|views?|projections?|wall street|the street|"
        r"(?:earnings|revenue|profit) targets?|analysts'? (?:estimates|expectations|forecasts?))")
_GUIDE = r"(?:guidance|outlook|forecasts?|projections?|guide|view)"
_GUIDE_MID = (r"(?:(?:its|their|the|annual|full[- ]year|fy\s?\d*|fiscal(?: year)?(?: \d{4})?|20\d\d|q[1-4]|quarterly|"
              r"first[- ]half|second[- ]half|h[12]|revenue|sales|profit|earnings|eps|margin|production|delivery|"
              r"growth|capex|spending|2026|2027)\s+){0,3}")
_EXEC = (r"(?:ceo|cfo|coo|cto|cio|c\.e\.o\.|chief\s+\w+\s+officer|chief executive|chief financial officer|chairman|chairwoman|"
         r"chair|president|founder|co-founder|executive|exec|director|svp|evp|vp|general counsel|head of \w+|"
         r"board member|officer|insider)")
_MOVE_UP = (r"jump(?:s|ed)?|soar(?:s|ed)?|surg(?:e|es|ed)|rall(?:y|ies|ied)|spik(?:e|es|ed)|skyrocket(?:s|ed)?|"
            r"rocket(?:s|ed)?|pop(?:s|ped)?|leap(?:s|t|ed)?|climb(?:s|ed)?|gain(?:s|ed)?|ris(?:e|es)|rose|"
            r"advance[sd]?|rebound(?:s|ed)?|zoom(?:s|ed)?|rip(?:s|ped)?|is up|are up|was up|edges? (?:up|higher)|"
            r"inch(?:es)? (?:up|higher)|ticks? (?:up|higher)|surging|soaring|rising|climbing|jumping|rallying|"
            r"gaining|trading higher|trade higher|move higher|moves higher|head higher|heads higher")
_MOVE_DOWN = (r"fall(?:s)?|fell|drop(?:s|ped)?|sink(?:s)?|sank|slid(?:e|es)?|tumbl(?:e|es|ed)|plung(?:e|es|ed)|"
              r"plummet(?:s|ed)?|crash(?:es|ed)?|slump(?:s|ed)?|shed(?:s)?|dip(?:s|ped)?|declin(?:e|es|ed)|"
              r"retreat(?:s|ed)?|slip(?:s|ped)?|tank(?:s|ed)?|lose[s]?|lost|crater(?:s|ed)?|nosedive[sd]?|"
              r"skid(?:s|ded)?|is down|are down|was down|edges? (?:down|lower)|ticks? (?:down|lower)|"
              r"falling|dropping|sinking|sliding|tumbling|plunging|slumping|trading lower|trade lower|"
              r"move lower|moves lower|head lower|heads lower")
_FUNDAMENTAL = (r"(?:revenue|revenues|sales|profit|profits|earnings|eps|margin|margins|income|deliveries|orders|"
                r"traffic|comps|yields?|rates?|inflation|prices|price of|unemployment|guidance|outlook|forecast|"
                r"production|output|demand|volume|bookings|subscribers|users|spending|costs?|debt|cash|losses?|"
                r"wealth|fortune|net worth|index|market|futures|bitcoin|oil|gold|dollar)")

_PCT_AFTER = (rf"(?:\s+(?:by\s+|nearly\s+|almost\s+|over\s+|more than\s+|as much as\s+|about\s+|roughly\s+|"
              rf"another\s+)?(?:{_NUM})\s?(?:%|percent\b|pct\b))")

_RULES: tuple[tuple[str, re.Pattern[str], bool], ...] = tuple((k, re.compile(p, re.IGNORECASE), g) for k, p, g in (
    ("earnings_beat", (
     rf"\b(?:beat|beats|beating|tops?|topped|topping|exceed(?:s|ed|ing)?|surpass(?:es|ed|ing)?|crush(?:es|ed)?|"
     rf"smash(?:es|ed)?|trounce[sd]?|outpace[sd]?|outstrip(?:s|ped)?|blows? past|blew past|sails? past|"
     rf"(?:comes?|came) in above)\s+(?:[\w'$.-]+\s+){{0,4}}?{_EST}\b"
     rf"|\b(?<!guidance )(?<!outlook )(?<!forecast ){_EARN}\s+(?:and\s+\w+\s+|[\w'$.-]+\s+){{0,3}}?"
     r"(?:beat|beats|tops?|topped|exceed(?:s|ed)?|surpass(?:es|ed)?|crush(?:es|ed)?)\b(?!\s+(?:the\s+)?market)"
     rf"|\b(?:{_EARN}|double|top-and-bottom-line)\s+beats?\b|\bbeat[- ]and[- ]raise\b|\bbeats? on (?:\w+\s+){{0,3}}?{_EARN}"
     rf"|\b(?:better|stronger)[- ]than[- ]expected\s+(?:\w+\s+){{0,2}}?{_EARN}"
     ), True),
    ("earnings_miss", (
     rf"\b(?:miss(?:es|ed)?|missing|falls? short of|fell short of|falling short of|lags?|lagged|trails?|trailed|"
     rf"undershoot(?:s)?|undershot|(?:comes?|came) in below|disappoint(?:s|ed)?)\s+(?:[\w'$.-]+\s+){{0,4}}?{_EST}\b"
     rf"|\b{_EARN}\s+(?:[\w'$.-]+\s+){{0,2}}?(?:miss(?:es|ed)?|falls? short|fell short|disappoints?|disappointed)\b"
     rf"(?!\s+of\s+(?!(?:\w+\s+){{0,2}}?{_EST}))"
     rf"|\b(?:wider|bigger|larger|deeper)[- ]than[- ]expected\s+(?:\w+\s+){{0,1}}?loss|\bloss\s+(?:widens|wider than)"
     rf"|\b(?:weaker|worse|softer)[- ]than[- ]expected\s+(?:\w+\s+){{0,2}}?{_EARN}"
     ), True),
    ("guidance_raise", (
     rf"\b(?:raise[sd]?|raising|lift(?:s|ed|ing)?|boost(?:s|ed|ing)?|hike[sd]?|up(?:s|ped)|increase[sd]?|"
     rf"increasing|improve[sd]?|bump(?:s|ed)? up|ratchets? up)\s+{_GUIDE_MID}{_GUIDE}\b"
     rf"|\b(?:upbeat|strong|stronger|robust|bullish|rosy|solid|above-consensus|better-than-expected|raised|"
     rf"upside|blowout|higher|increased|improved|boosted|lifted)\s+(?:[\w-]+\s+){{0,2}}?"
     rf"(?:guidance|outlook|forecast|guide)\b"
     rf"|\b(?:guidance|outlook|forecast|guide)\s+(?:\w+\s+){{0,2}}?(?:tops?|beats?|above|exceeds?|ahead of)\s+"
     rf"(?:\w+\s+){{0,2}}?{_EST}|\bguided? (?:above|ahead of)\b|\bbeat[- ]and[- ]raise\b"
     ), True),
    ("guidance_cut", (
     rf"\b(?:cut[s]?|cutting|lower(?:s|ed|ing)?|slash(?:es|ed|ing)?|trim(?:s|med|ming)?|reduce[sd]?|reducing|"
     rf"withdraw[sn]?|withdrew|pull(?:s|ed)|suspend(?:s|ed)?|scrap(?:s|ped)?|pare[sd]?|walks? back|"
     rf"drops?|dropped|abandons?|abandoned|"
     rf"temper(?:s|ed)?)\s+{_GUIDE_MID}{_GUIDE}\b"
     rf"|\b(?:weak|weaker|soft|softer|disappointing|downbeat|gloomy|cautious|bleak|dismal|lowered|reduced|tepid|"
     rf"lackluster|muted|grim)\s+(?:[\w-]+\s+){{0,2}}?(?:guidance|outlook|forecast|guide)\b"
     rf"|\b(?:guidance|outlook|forecast|guide)\s+(?:\w+\s+){{0,2}}?(?:miss(?:es|ed)?|falls? short|below|"
     rf"disappoints?|trails?|lags?)\b|\bprofit warning\b|\bwarns? (?:on|of) (?:\w+\s+){{0,2}}?(?:profit|revenue|"
     rf"sales|earnings|results|demand)\b|\bforecasts? (?:\w+\s+){{0,2}}?(?:revenue|sales|profit|earnings) "
     rf"(?:drop|decline|fall|slump)"
     ), True),
    ("record_results", (
     r"\brecord (?:(?:quarterly|annual|first[- ]quarter|second[- ]quarter|third[- ]quarter|fourth[- ]quarter|"
     r"q[1-4]|full[- ]year|fiscal|holiday[- ]quarter|monthly)\s+)*(?:revenues?|sales|profits?|earnings|results|"
     r"quarter|deliveries|bookings|net income|eps|margins?|cash flow|orders|backlog)\b"
     r"|\b(?:revenues?|sales|profits?|earnings|deliveries)\s+(?:hits?|reach(?:es)?|at|set)\s+(?:a\s+)?(?:new\s+)?record"
     ), True),
    ("buyback", (
     r"\b(?:buybacks?|buy-backs?|share repurchases?|stock repurchases?|share-repurchase|repurchase (?:program|plan|"
     r"authori[sz]ation)|repurchasing (?:its )?(?:own )?(?:shares|stock)|buy back (?:\$|up to|its|shares|stock|"
     r"\d)|bought back)\b"
     ), False),
    ("dividend_raise", (
     r"\b(?:raise[sd]?|raising|hike[sd]?|hiking|boost(?:s|ed|ing)?|increase[sd]?|increasing|lift(?:s|ed)?|"
     r"up(?:s|ped)|bump(?:s|ed)? up)\s+(?:its\s+|the\s+)?(?:(?:quarterly|annual|monthly|semi-annual|interim|"
     r"final|cash|common)\s+)*(?:dividend|payout|distribution)s?\b(?!\s+(?:tax|taxes|safety|questions|concerns|"
     r"withholding|targets?))|\b(?:dividend|payout)\s+(?:hike|increase|raise|boost|bump)\b|\bspecial dividend\b|\binitiates? (?:a |its first )?(?:quarterly )?dividend\b|"
     r"\bfirst[- ]ever dividend\b"
     ), True),
    ("dividend_cut", (
     r"\b(?:cut[s]?|cutting|slash(?:es|ed|ing)?|suspend(?:s|ed|ing)?|eliminat(?:e|es|ed|ing)|halt(?:s|ed|ing)?|"
     r"reduce[sd]?|reducing|scrap(?:s|ped)?|omit(?:s|ted)?|pause[sd]?|skips?|skipped)\s+(?:its\s+|the\s+)?"
     r"(?:(?:quarterly|annual|monthly|interim|final|cash|common)\s+)*(?:dividend|payout|distribution)s?\b|"
     r"\bdividend (?:cut|suspension|reduction|elimination)s?\b"
     ), True),
    ("layoffs", (
     r"\blayoffs?\b|\blay(?:s|ing)? offs?\b|\blaid off\b|\bjob cuts?\b|\b(?:cut|cuts|cutting|slash(?:es|ing)?|"
     r"eliminat(?:e|es|ing)|shed(?:s|ding)?|trims?|axe[sd]?|axing)\s+(?:about\s+|nearly\s+|some\s+|over\s+|"
     r"more than\s+|another\s+)?(?:[\w,]+\s+){0,2}?(?:jobs|positions|roles|workers|employees|staff|headcount)\b|"
     r"\bworkforce (?:reduction|cuts?)\b|\breduces? (?:its )?workforce\b|\bheadcount reduction\b|"
     r"\bredundanc(?:y|ies)\b"
     ), False),
    ("lawsuit", (
     r"\b(?:lawsuits?|sue[sd]?|suing|class[- ]actions?|litigation|(?:patent|antitrust|securities|copyright|"
     r"trademark|wrongful|privacy|consumer|shareholder)\s+(?:suit|case|trial|claims?|complaint|infringement|"
     r"verdict)|jury (?:verdict|says|orders|finds|awards?)|(?:jury|court|guilty|\$[\d.,]+\s?(?:million|billion|"
     r"m|bn|b)?) verdict|verdict (?:against|over|in (?:the )?(?:case|trial|lawsuit))|lead plaintiff|plaintiffs?|"
     r"files? (?:a )?(?:complaint|suit)|injunction|court (?:rules|ruling|orders|blocks)|ruled against|"
     r"appeals court)\b|(?:\$[\d.,]+\s?(?:million|billion|m|bn|b)?\s+|faces? (?:a )?|filed (?:a )?)suit\b|"
     r"\bsuit (?:over|against|from|alleging|claiming|filed)\b"
     ), False),
    ("investigation", (
     r"\b(?:investigations?|investigat(?:es|ing|ed)|probes?|probing|probed|subpoena(?:s|ed)?|inquiry|inquiries|"
     r"raid(?:ed|s)?|wells notice|under (?:federal )?review)\b"
     ), False),
    ("settlement", (
     r"\b(?:settle[sd]?|settles|settling|settlements?)\b(?=.{0,60}\b(?:lawsuit|suit|case|claims?|charges?|probe|"
     r"sec|ftc|doj|class action|litigation|dispute|allegations|investors|shareholders)\b)|"
     r"\b(?:lawsuit|suit|case|claims?|charges?|probe|class action|litigation|dispute)\b.{0,40}\bsettle[sd]?\b|"
     r"\bagree[sd]? to pay \$"
     ), False),
    ("m_and_a", (
     r"\b(?:acquisitions?|mergers?|takeovers?|buyouts?|tender offer|all-(?:cash|stock) deal|go(?:es|ing)? private|"
     r"take-private|spin-?offs?|spins? off|divest(?:s|ed|iture|itures|ing)?|carve-?out|merge (?:with|into)|"
     r"merging with|(?:will|agrees? to|agreed to|in talks to|nears? (?:a )?deal to|offers? to|bids? to|plans? to|"
     r"seeks? to|deal to|moves? to|looks? to|is set to|aims? to) (?:buy|acquire|purchase|merge with|take over)"
     r"(?! back)|takes? (?:a )?(?:\d+%\s)?stake in|"
     r"bid for|(?:acquires?|acquired|acquiring|buys|bought|snaps up|scoops up)\s+(?:(?-i:[A-Z])[\w&.'-]*|rival|"
     r"startup|majority stake|minority stake|unit|division|maker|developer|operator|provider|business|assets))\b"
     r"|(?<!reason )(?<!reasons )(?<!time )(?<!stock )(?<!stocks )(?<!you )\bto (?:buy|acquire|purchase)\s+"
     r"(?!(?-i:(?:During|Now|Before|After|In|On|At|For|With|From|And|Or|The|This|These|Today|Ahead|Rating|More|"
     r"Shares|Stock|Into|Back|Up|It|Them)\b))(?-i:[A-Z])[\w&.'-]+(?:\s+(?-i:[A-Z])[\w&.'-]+){0,3}"
     r"(?!\s+(?:stock|shares)\b)(?=.*\b(?:deal|billion|million|bn|takeover|acquisition|\$\d))"
     r"|(?:\$|£|€)[\d.,]+\s?(?:billion|bn|b)\s+[^?!]{0,40}?\bdeal\s+(?:closes|closed|completes|completed|clears|"
     r"cleared|wins approval|gets approval|just cleared)\b|"
     r"(?:(?:\$|£|€)[\d.,]+\s?(?:billion|bn|b)|takeover|buyout)\s+deal\s+for\s+(?:rival\s+)?(?-i:[A-Z])[\w&.'-]+|"
     r"\bdeal for rival\b"
     ), False),
    ("partnership", (
     r"\b(?:partner(?:s|ed|ing)? with|partnerships?|teams? up|teamed up|collaborat(?:es|ed|ing|ion)|alliance|"
     r"joint venture|ties up|tie-up|strategic (?:agreement|investment|deal|pact)|signs? (?:a )?(?:deal|pact|"
     r"agreement|mou) with|inks? (?:a )?(?:\w+\s+)?(?:deal|pact|agreement)|strikes? (?:a )?(?:\w+\s+)?(?:deal|pact|"
     r"agreement)|struck (?:a )?deal|(?:supply|licensing|distribution|cloud|chip|ai) (?:deal|pact|agreement) with)\b"
     ), False),
    ("contract_win", (
     r"\b(?:wins?|won|secures?|secured|lands?|landed|awarded|clinch(?:es|ed)?|bags?|bagged|nabs?|gets?|got|"
     r"receives?|received)\s+(?:an?\s+)?(?:[\w$.,'\"-]+\s+){0,4}?(?:contracts?|orders?|award|tender)\b"
     r"(?!\s+(?:extension|talks|negotiations|details|decision))"
     ), False),
    ("product_launch", (
     r"\b(?:launch(?:es|ed|ing)?|unveil(?:s|ed|ing)?|introduc(?:es|ed|ing)|rolls? out|rolled out|rolling out|"
     r"debut(?:s|ed|ing)|releas(?:es|ed) (?:new|its|the|a)|reveal(?:s|ed) (?:new|its)|goes on sale|"
     r"announces? new|showcases?)\b"
     ), False),
    ("recall", (
     r"\brecall(?:s|ed|ing)?\b"
     ), False),
    ("exec_departure", (
     rf"\b{_EXEC}\s+(?:[\w.'-]+\s+){{0,3}}?(?:steps? down|stepping down|stepped down|resign(?:s|ed|ing)?|to resign|"
     rf"quits?|exits?|depart(?:s|ed|ing)?|leaves|leaving|to leave|retir(?:e|es|ed|ing)|to retire|ousted|fired|"
     rf"replaced|to step down|is out|out as)\b|\b(?:resignation|departure|exit|ouster|firing|retirement) of "
     rf"(?:its |the )?(?:ceo|cfo|coo|chief|chairman|president|founder)"
     ), False),
    ("exec_hire", (
     r"\b(?:appoints?|appointed|names?|named|hires?|hired|taps?|tapped|picks?|poach(?:es|ed)|recruits?|"
     r"promotes?|elevat(?:es|ed)|selects?)\s+(?:[\w.'&-]+,?\s+){0,8}?(?:as\s+)?(?:new\s+|interim\s+|its\s+)?"
     r"(?:ceo|cfo|coo|cto|chief\s+\w+\s+officer|chief executive|chairman|president|head of)\b|"
     r"\bnew\s+(?:finance|financial|operating|technology|executive)?\s*chief\s+(?:takes over|steps in|named)\b|"
     r"\btakes? (?:over )?the helm\b|\btakes? over as\s+(?:ceo|cfo|chief|chair|president)\b|"
     r"\b(?:to join|joins?|to lead)\s+(?:[\w.'-]+\s+){0,3}?as\s+(?:its\s+|new\s+)?(?:ceo|cfo|coo|cto|chief|"
     r"president|head)\b|\bnames? new (?:ceo|cfo|chief)"
     ), False),
    ("offering", (
     r"\b(?:(?:public|secondary|stock|share|equity|common stock|registered direct|follow-on|at-the-market|atm|"
     r"preferred stock|common share|convertible(?: senior)? notes?|convertible|debt|bond|notes) offering|"
     r"prices? (?:\$[\d.,]+\s?\w+\s+)?(?:upsized\s+)?(?:offering|placement)|private placement|"
     r"(?:raises?|raising) \$[\d.,]+\s?(?:million|billion|m|b|bn)\s+(?:in|through|via)\s+(?:a\s+)?(?:\w+\s+)?"
     r"(?:stock|share|equity|convertible|notes)|convertible notes|dilution|dilutive|share sale|"
     r"sells? \$[\d.,]+\s?(?:million|billion|m|bn|b) (?:of|in) (?:stock|shares))\b"
     ), False),
    ("bankruptcy", (
     r"\b(?:bankrupt(?:cy|cies)?|chapter (?:11|7|15)|going[- ]concern|insolven(?:t|cy)|receivership|"
     r"files? for (?:bankruptcy|creditor protection)|creditor protection|restructuring support agreement|"
     r"default(?:s|ed)? on (?:its )?(?:debt|bonds?|loans?|notes|payments?)|rescue (?:financing|deal|package|loan)|"
     r"debt restructuring|restructuring talks|forbearance agreement|miss(?:es|ed)? (?:an? )?(?:interest|coupon|debt|bond) "
     r"payments?)\b"
     ), True),
    ("delisting", (
     r"\b(?:delist(?:ing|ed|s)?|(?:nasdaq|nyse) (?:deficiency |non-?compliance |delisting )?notice|"
     r"non-?compliance with (?:nasdaq|nyse|listing)|minimum bid (?:price )?requirement|regain compliance)\b"
     ), False),
    ("short_report", (
     r"\bshort[- ]sell(?:er|ers|ing)?(?:'s)?\s+(?:\w+\s+)?(?:report|attack|alleg\w+|bet|target|campaign|claims|"
     r"dispute|warning|warns?|flags?|accus\w+|raises concerns|spotlights?|questions|scrutiny|targets|pressure|"
     r"bets? against|takes? aim)\b|\bshort report\b|\b(?:hindenburg|muddy waters|citron|blue orca|spruce point|grizzly|"
     r"wolfpack|kerrisdale|fuzzy panda|culper|viceroy|gotham city|hunterbrook|glasshouse|iceberg|bleecker street|"
     r"j capital|scorpion capital|morpheus|night market|jehoshaphat|snowcap|bear cave)(?:\s+(?:research|capital))?"
     r"\s+(?:report|says|alleges|shorts?|takes?|bets?|targets?|is short|discloses|accuses|publishes)\b|"
     r"\b(?:report|attack) (?:from|by) (?:hindenburg|muddy waters|citron|blue orca|spruce point|grizzly|wolfpack|"
     r"kerrisdale|fuzzy panda|culper|viceroy|hunterbrook|glasshouse)"
     ), False),
    ("insider_buy", (
     rf"\b{_EXEC}s?\s+(?:[\w.'-]+\s+){{0,6}}?(?:buys?|bought|purchas(?:es|ed)|acquir(?:es|ed)|adds?|scoops? up|"
     rf"snaps? up|picks? up|loads? up on|subscribes?(?: for| to)?)\s+(?:\$[\d.,]+\w*|[\d.,]+%?\s?(?:million|k|m|"
     rf"thousand)?|more|shares|stock|stake)|\binsider (?:buying|purchases?|buys?)\b|"
     rf"\b{_EXEC}s?\s+(?:[\w.'-]+\s+){{0,6}}?(?:continues?|extends?)\s+(?:\w+\s+)?buying\b|"
     rf"\b{_EXEC}s?\b[^.?!]{{0,60}}\$[\d.,]+\s?(?:million|m|k|thousand)?\s+(?:stock|share)\s+purchase\b"
     ), False),
    ("insider_sell", (
     rf"\b{_EXEC}s?\s+(?:[\w.'-]+\s+){{0,4}}?(?:sells?|sold|unloads?|dumps?|offloads?|disposes? of|trims?|"
     rf"cashes? out|proposes? (?:selling|to sell|a share sale))\b|\binsider (?:selling|sales?|sold)\b|"
     rf"\bproposes? (?:selling|to sell) (?:[\d,]+ )?shares\b|\bform 144\b|\b10b5-1\b"
     ), False),
    ("all_time_high", (
     r"\b(?:hits?|hit|reach(?:es|ed)?|sets?|notch(?:es|ed)?|touch(?:es|ed)?|marks?|soars? to|surges? to|jumps? to|"
     r"climbs? to|rises? to|rall(?:y|ies) to|closes? at|trades? at|breaks? (?:to|out to)|at|to|new|fresh|first|"
     r"heads? for|nears?)\s+(?:a\s+|an\s+|its\s+)?(?:new\s+|fresh\s+|first\s+)?(?:all[- ]time|record)\s+highs?\b|"
     r"\b(?:hits?|reach(?:es|ed)?|sets?|notch(?:es|ed)?|at|first)\s+(?:a\s+)?(?:new\s+|fresh\s+|first\s+)?record\b"
     r"(?!\s+(?:revenue|sales|profit|quarter|earnings|deliveries|low|loss|levels? of|number|amount))|"
     r"\brecord highs?\b|\ball[- ]time highs?\b"
     ), True),
    ("low_52w", (
     r"\b52[- ]week lows?\b|\b(?:multi-?year|decade|\d+-year|\d+-month|record|all[- ]time|lowest level in "
     r"\d+ years?)[- ]lows?\b|\blowest (?:level|close) (?:in|since) (?:\d+|a decade|years)\b"
     ), True),
    ("high_52w", (
     r"\b52[- ]week highs?\b|\b(?:multi-?year|decade|\d+-year|\d+-month)[- ]highs?\b"
     ), True),
    ("stock_split", (
     r"\b(?:stock|share) split\b|\b(?:\d+|two|three|four|five|ten|twenty)[- ](?:for|to)[- ](?:\d+|one) "
     r"(?:stock |share |reverse )?split\b|\breverse split\b|\bsplits? (?:its )?(?:stock|shares)\b|"
     # reverse splits announced without the word: "Every 25 shares will become one", "plans to combine shares"
     r"\b(?:every\s+)?(?:\d+|two|three|four|five|ten|fifteen|twenty|twenty-five|fifty|hundred)\s+(?:[\w.&'-]+\s+){0,2}?"
     r"shares\s+(?:are expected to |will |to |would )?become\s+one\b|\b(?:combine|consolidate)s?\s+(?:its\s+)?"
     r"(?:common\s+)?shares\b|\bshare consolidation\b"
     ), True),
    ("regulatory_approval", (
     r"\bfda (?:approv(?:es|ed|al)|clears?|cleared|clearance|grants?|nod|green[- ]lights?|accepts?)|"
     r"\bapprov(?:al|ed) by (?:the )?(?:fda|ema|regulators)|\b(?:wins?|won|gets?|got|receives?|received|secures?|"
     r"secured)\s+(?:\w+\s+){0,2}?(?:fda|ema|regulatory|antitrust|eu|chmp|ftc|fcc|doj) (?:approval|nod|clearance|"
     r"green light)|\b(?:ema|chmp) (?:recommends|approves|backs)\b"
     ), False),
    ("regulatory_setback", (
     r"\bcomplete response letter\b|\bfda (?:rejects?|rejected|declines?|declined|refuses?|delays?|delayed|"
     r"issues? (?:a )?crl)|\b(?:rejected|declined) by (?:the )?fda|\bclinical hold\b|\bfda (?:panel|advisory "
     r"committee) (?:votes? against|rejects)|\btrial (?:fails?|failed|failure|misses? (?:its )?(?:primary )?"
     r"endpoint)|\b(?:missed|misses|fails? to meet) (?:the |its )?primary endpoint|\bfails? (?:a |its )?"
     r"(?:phase \w+ |late-stage |pivotal )?(?:trial|study)\b"
     ), False),
    ("index_inclusion", (
     r"\b(?:join(?:s|ing)?|added to|to join|set to join|inclusion in|enters?|will be added to|to be added to|"
     r"joins?)\s+(?:the\s+)?(?:s&p 500|s&p 400|s&p 600|s&p midcap 400|s&p smallcap 600|nasdaq[- ]100|"
     r"dow jones industrial average|dow|russell (?:1000|2000|3000)|ftse 100)\b|\b(?:s&p 500|nasdaq[- ]100) "
     r"inclusion\b"
     ), False),
    ("data_breach", (
     r"\b(?:data breach|cyber ?attacks?|cybersecurity incident|hacked|hackers|ransomware|security breach|"
     r"(?:global|massive|major|widespread|nationwide) outage|outage hits|breach of (?:customer|user) data)\b"
     ), False),
    ("price_up", (
     rf"\b(?:stock|shares|share price)\s+(?:\w+\s+){{0,2}}?(?:{_MOVE_UP})\b{_PCT_AFTER}?"
     rf"|\b(?:{_MOVE_UP}){_PCT_AFTER}"
     rf"|\b(?:stock|shares)\s+(?:is\s+|are\s+)?(?:{_MOVE_UP}|up|higher)\b(?!\s+(?:on|in|for|than|from)\b)"
     rf"|\b(?:up|higher){_PCT_AFTER}"
     ), True),
    ("price_down", (
     rf"\b(?:stock|shares|share price)\s+(?:\w+\s+){{0,2}}?(?:{_MOVE_DOWN})\b{_PCT_AFTER}?"
     rf"|\b(?:{_MOVE_DOWN}){_PCT_AFTER}"
     rf"|\b(?:stock|shares)\s+(?:is\s+|are\s+)?(?:{_MOVE_DOWN}|down|lower)\b(?!\s+(?:on|in|for|than|from)\b)"
     rf"|\b(?:down|lower){_PCT_AFTER}"
     ), True),
))

# Matches that must be ignored for a given key (checked on the text around the hit).
_EXCLUDE: dict[str, re.Pattern[str]] = {
    "earnings_miss": re.compile(r"\b(?:give|giving|gave)\s+(?:\w+\s+){0,3}?a miss\b|\bdon'?t miss\b|\bmiss out\b|"
                                r"\bnear miss\b|\bhit or miss\b|\bcan'?t miss\b", re.IGNORECASE),
    "guidance_raise": re.compile(r"\bmoody'?s\b|\bfitch\b|\bs&p global ratings\b|\brating agency\b|"
                                 r"\boutlook to (?:positive|stable|negative)\b|\b(?:imf|world bank|oecd|ecb|"
                                 r"central bank|economists?)\b", re.IGNORECASE),
    "guidance_cut": re.compile(r"\bmoody'?s\b|\bfitch\b|\bs&p global ratings\b|\brating agency\b|"
                               r"\boutlook to (?:positive|stable|negative)\b|\b(?:imf|world bank|oecd|ecb|"
                               r"central bank|economists?)\b", re.IGNORECASE),
    "m_and_a": re.compile(r"\b(?:never|won't|will not|wouldn't|would never|no)\s+(?:\w+\s+)?(?:merge|merger|acquire|"
                          r"acquisition|buy|takeover)\b|"
                          r"\b(?:i|we|i've|we've|i'm|just|finally|first|also)\s+(?:\w+\s+)?(?:bought|buy|buying|"
                          r"purchased|added|picked up)\b|"
                          r"\b(?:stock|shares|stake|position|holdings?|\$[a-z]{1,6})\s+(?:\w+\s+){0,2}?(?:acquired|"
                          r"bought|sold|purchased|cut|raised|lifted|trimmed|boosted|reduced|increased|decreased|"
                          r"grown)\s+by\b|\b(?:customer|user|talent|data|land) acquisition\b|\bacquisition costs?\b|"
                          r"\b(?:acquires?|buys|bought|purchases?|purchased|picks? up) (?:\d[\d,.]*%? (?:more )?)?"
                          r"(?:more )?(?:shares|stock)\b|\b(?:buys|bought|acquires?)\s+(?:UK£|US\$|SEK|[$£€])|"
                          r"\b(?:upgrades?|upgraded|downgrades?|downgraded|raises?|cuts?|moves?)\b.{0,40}\bto (?:buy|acquire)\b|"
                          r"\b(?:buys|bought)\s+(?:put|call|puts|calls|options|bitcoin|ether|gold|the dip)\b|\bacquisition (?:corp|corporation|company|"
                          r"holdings|co)\b|\b(?:tech|ai|hostile) takeover\b", re.IGNORECASE),
    "offering": re.compile(r"\binitial public offering\b|\bipo\b|\b(?:director|officer|ceo|cfo|coo|insider|svp|evp|"
                           r"president|founder|chair(?:man|woman)?|general counsel|executive)\b[^.;:]{0,60}?"
                           r"(?:sells?|sold|proposes?)\b", re.IGNORECASE),
    "product_launch": re.compile(r"\b(?:launch(?:es|ed)?|unveil(?:s|ed)?|introduc(?:es|ed)|rolls? out|announces? new)\s+"
                                 r"(?:an?\s+|its\s+|the\s+)?(?:\$[\d.,]+|(?:[\w$.-]+\s+){0,6}?(?:probe|investigation|inquiry|"
                                 r"lawsuit|coverage|offering|ipo|buyback|repurchase|tender|bid|review|campaign "
                                 r"against|attack|strike|missile|plan to cut|layoffs|restructuring|budget|tariffs?|"
                                 r"dividend|guidance|results|earnings)\b)", re.IGNORECASE),
    "all_time_high": re.compile(r"\b(?:from|below|off|under|shy of|short of|away from|beneath|since|of)\s+"
                                r"(?:its|their|the|a)?\s*(?:\w+\s+)?(?:all[- ]time|record)\s+highs?\b|"
                                r"\bshy of (?:a |its |the )?(?:first |new )?record\b|"
                                r"\b(?:all[- ]time|record) highs? (?:could|may|might|is (?:next|in sight|within reach))\b|"
                                r"\bshort interest\b", re.IGNORECASE),
    "high_52w": re.compile(r"\b(?:from|off|below|under|shy of|short of)\s+(?:its|their|the|a)?\s*(?:\w+\s+)?"
                           r"52[- ]week highs?\b", re.IGNORECASE),
    "low_52w": re.compile(r"\b(?:ratings?|short interest|volatility|vix|unemployment|inventor(?:y|ies)|costs?|"
                          r"rates?|yields?|claims|delinquenc(?:y|ies)|defaults?)\s+(?:\w+\s+){0,2}?(?:hits?|at|to|near|"
                          r"reach(?:es)?|falls? to|drops? to)\s+(?:an?\s+)?(?:new\s+)?(?:\d+-year|multi-?year|decade|record)|"
                          r"\b(?:from|off|above)\s+(?:its|their|the|a)?\s*(?:\w+\s+)?52[- ]week lows?\b|"
                          r"\b(?:invested|investment)\b.{0,40}\b52[- ]week low|"
                          r"low (?:valuation|multiple|p/e|unemployment|rates?)\b", re.IGNORECASE),
    "lawsuit": re.compile(r"\bfollow(?:s|ed)? suit\b", re.IGNORECASE),
    "recall": re.compile(r"\b(?:he|she|i|we|they|ceo|who|fans|founder|still|vividly|fondly|executives?|officials|"
                         r"investors|traders|analysts|people|veterans|employees|workers|residents|experts|"
                         r"economists|survivors|witnesses|colleagues|friends)\s+recall|\brecall(?:s|ed|ing)?\s+"
                         r"(?:how|when|that|the day|his|her|their|being|seeing|meeting|growing|watching|working|"
                         r"a time|moments?|the (?:\d{4}|crisis|crash|era|days|years|time|moment|dot-com))\b|"
                         r"\brecall (?:election|vote|petition|campaign)\b|\btotal recall\b", re.IGNORECASE),
    "investigation": re.compile(r"\b(?:police|homicide|shooting|murder|crash|fire|rape|assault|death)\b",
                                re.IGNORECASE),
}

# Words that make a generic price-move verb refer to something else.
_PRICE_SUBJECT_NOISE = re.compile(rf"\b{_FUNDAMENTAL}\s+(?:\w+\s+)?$", re.IGNORECASE)


def _price_value(match_text: str, key: str) -> float | None:
    pct = re.search(rf"({_NUM})\s?(?:%|percent\b|pct\b)", match_text, re.IGNORECASE)
    if not pct:
        return None
    value = _num(pct.group(1))
    if value is None:
        return None
    return value if key == "price_up" else -value


_ANALYST_TRIGGER = re.compile(r"target|\bpts?\b|grad|initiat|coverage|pick|bullish|bearish|rating|perform|weight|"
                              r"to (?:strong )?(?:buy|sell)|objective")
# Cheap substring pre-filters: a rule's regex only runs when one of its
# trigger fragments occurs in the lower-cased text (keeps 500 texts ~0.1 s).
_TRIGGERS: dict[str, tuple[str, ...]] = {
    "earnings_beat": ("beat", "top", "exceed", "surpass", "crush", "smash", "trounce", "outpac", "outstrip", "past",
                      "above", "better", "stronger"),
    "earnings_miss": ("miss", "short", "lag", "trail", "undershoot", "undershot", "below", "disappoint", "loss",
                      "weaker", "worse", "softer"),
    "guidance_raise": ("guid", "outlook", "forecast", "projection", "view"),
    "guidance_cut": ("guid", "outlook", "forecast", "projection", "view", "warn"),
    "record_results": ("record",),
    "buyback": ("buyback", "buy-back", "repurchas", "buy back", "bought back"),
    "dividend_raise": ("dividend", "payout", "distribution"),
    "dividend_cut": ("dividend", "payout", "distribution"),
    "layoffs": ("layoff", "lay off", "lays off", "laying off", "laid off", "job", "position", "role", "worker",
                "employee", "staff", "headcount", "workforce", "redundanc"),
    "lawsuit": ("suit", "sue", "suing", "class", "litigation", "jury", "verdict", "plaintiff", "complaint",
                "injunction", "court", "ruled", "patent", "antitrust", "securities", "copyright", "trademark",
                "privacy", "wrongful"),
    "investigation": ("investig", "probe", "probing", "subpoena", "inquir", "raid", "wells notice", "under review",
                      "under federal review"),
    "settlement": ("settl", "agree"),
    "m_and_a": ("acqui", "merg", "takeover", "buyout", "tender", "deal", "private", "spin", "divest", "carve",
                "buy", "purchase", "take over", "stake", "bid", "bought", "snaps up", "scoops up"),
    "partnership": ("partner", "team", "collaborat", "alliance", "joint venture", "tie", "strategic", "deal", "pact",
                    "agreement", "struck"),
    "contract_win": ("contract", "order", "award", "tender"),
    "product_launch": ("launch", "unveil", "introduc", "roll", "debut", "releas", "reveal", "on sale", "announce",
                       "showcas"),
    "recall": ("recall",),
    "exec_departure": ("step", "resign", "quit", "exit", "depart", "leav", "retir", "oust", "fired", "replaced",
                       " out"),
    "exec_hire": ("appoint", "name", "hire", "tap", "pick", "poach", "recruit", "promot", "elevat", "select", "join",
                  "to lead", "takes over", "take over", "helm"),
    "offering": ("offering", "placement", "convertible", "dilut", "share sale", "raises $", "raising $", "sells $",
                 "sell $"),
    "bankruptcy": ("bankrupt", "chapter", "going concern", "going-concern", "insolven", "receivership",
                   "creditor protection", "restructuring", "default", "rescue", "forbearance", "payment"),
    "delisting": ("delist", "notice", "compliance", "minimum bid"),
    "short_report": ("short", "hindenburg", "muddy waters", "citron", "blue orca", "spruce point", "grizzly",
                     "wolfpack", "kerrisdale", "fuzzy panda", "culper", "viceroy", "gotham", "hunterbrook",
                     "glasshouse", "iceberg", "bleecker", "j capital", "scorpion", "morpheus", "night market",
                     "jehoshaphat", "snowcap", "bear cave"),
    "insider_buy": ("buy", "bought", "purchas", "acquir", "add", "scoop", "snap", "pick", "load", "subscribe"),
    "insider_sell": ("sell", "sold", "sale", "unload", "dump", "offload", "dispos", "trim", "cash", "form 144", "10b5"),
    "all_time_high": ("high", "record"),
    "low_52w": ("low",),
    "high_52w": ("high",),
    "stock_split": ("split", "become one", "combine", "consolidat"),
    "regulatory_approval": ("fda", "approv", "ema", "chmp", "nod", "clearance"),
    "regulatory_setback": ("complete response", "fda", "clinical hold", "trial", "endpoint", "study"),
    "index_inclusion": ("s&p", "nasdaq-100", "nasdaq 100", "dow", "russell", "ftse"),
    "data_breach": ("breach", "cyber", "hack", "ransomware", "outage"),
}


_LEAD_CHARS = 300
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s+[A-Z$\d\"'(])")


def _lead(text: str) -> str:
    """Events are headline-level: a long post or summary is judged on its
    first sentences (~300 chars), where news is stated; essays further down
    mention "partnerships", "launches" and "mergers" in passing."""
    if len(text) <= _LEAD_CHARS:
        return text
    cut = _LEAD_CHARS
    for m in _SENTENCE_END_RE.finditer(text, 0, _LEAD_CHARS + 1):
        cut = m.end()
    return text[:cut]


def detect_events(text: str, company: CompanyRef | None = None) -> list[DetectedEvent]:
    """Events in `text`, in order of appearance; at most one per (key, firm).

    With `company`, events that belong to *another* entity named in the text
    are dropped: "Tesla rival Nikola files for bankruptcy" carries no
    bankruptcy for Tesla, "EchoStar unit Dish files for bankruptcy after
    delays in AT&T deal" none for AT&T, "Nvidia slips while AMD jumps 5%" no
    price jump for Nvidia (see `_owned`). Texts that never name the company
    keep their events (its owner is unknown; relevance gates them)."""
    if not text:
        return []
    t = _lead(fold(clean_text(text)))
    low = t.lower()
    hits = _analyst_events(t) if _ANALYST_TRIGGER.search(low) else []
    for key, regex, guarded in _RULES:
        triggers = _TRIGGERS.get(key)
        if triggers and not any(trigger in low for trigger in triggers):
            continue
        exclude = _EXCLUDE.get(key)
        for m in regex.finditer(t):
            if guarded and _is_hypothetical(t, m.start(), m.end()):
                continue
            if exclude and _excluded(key, exclude, t, m):
                continue
            if not _context_ok(key, t, m):
                continue
            polarity = _polarity(key, t, m)
            value = _price_value(m.group(0), key) if key in {"price_up", "price_down"} else None
            hits.append(_Hit(m.start(), key, polarity, value=value, span=m.group(0).strip(), end=m.end()))
    if company is not None and hits:
        hits = _attribute(t, hits, company)

    out: list[DetectedEvent] = []
    starts: list[int] = []
    seen: set[tuple[str, str | None]] = set()
    for h in sorted(hits, key=lambda h: (h.start, h.key)):
        ident = (h.key, h.firm)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(DetectedEvent(key=h.key, polarity=h.polarity, firm=h.firm, value=h.value,  # type: ignore[arg-type]
                                 span=(h.span or "")[:120] or None))
        starts.append(h.start)
    return _resolve_conflicts(t, out, starts)


# Exclusions that only void the hit when they overlap it (others void any hit nearby).
_OVERLAP_EXCLUSIONS = frozenset({"m_and_a", "product_launch", "all_time_high", "low_52w", "high_52w"})


def _excluded(key: str, exclude: re.Pattern[str], text: str, m: re.Match[str]) -> bool:
    base = max(0, m.start() - 50)
    local = exclude.search(text[base:m.end() + 50])
    if not local:
        return False
    if key not in _OVERLAP_EXCLUSIONS:
        return True
    return base + local.start() <= m.end() and base + local.end() >= m.start()


def _context_ok(key: str, text: str, m: re.Match[str]) -> bool:
    """Per-event sanity checks on the surrounding text."""
    before = text[max(0, m.start() - 30):m.start()]
    if key in {"price_up", "price_down"}:
        return not _PRICE_SUBJECT_NOISE.search(before)
    if key == "stock_split" and "?" in text:  # "Is a Microsoft Stock Split Coming?" is speculation
        return bool(re.search(r"\b(?:announc|approv|implement|effect|carr(?:y|ies) out|sets?|executes?|enacts?)",
                              text, re.IGNORECASE))
    if key == "all_time_high":
        return not re.search(r"short interest\s+(?:\w+\s+){0,2}$", before, re.IGNORECASE)
    if key in {"earnings_beat", "earnings_miss"}:
        return not re.search(r"\b(?:guidance|outlook|forecasts?|guide)\s+$", text[max(0, m.start() - 20):m.start()],
                             re.IGNORECASE)
    if key in {"guidance_raise", "guidance_cut"}:
        if _MARKET_OUTLOOK_RE.search(m.group(0)):
            return False  # "strong memory pricing outlook": an industry view, not company guidance
        # "JPMorgan Cuts Forecast": a broker's own estimate, not company guidance.
        return not any(f.end() <= m.start() and not re.search(r"[.;:!?]|\s[-|]\s", text[f.end():m.start()])
                       and m.start() - f.end() <= 25 for f in _FIRM_RE.finditer(text))
    return True


_MARKET_OUTLOOK_RE = re.compile(r"\b(?:pricing|price|demand|industry|market|sector|economic|macro|rate|rates|"
                                r"weather|memory|chip|consumer)\s+(?:outlook|forecast|view)s?\b", re.IGNORECASE)


def _polarity(key: str, text: str, m: re.Match[str]) -> str:
    """Default polarity, adjusted where the phrasing flips it."""
    if key == "buyback" and re.search(r"\b(?:suspend|halt|paus|scrap|cancel|end|slash|cut)\w*\s+(?:\w+\s+){0,2}$",
                                      text[max(0, m.start() - 30):m.start()], re.IGNORECASE):
        return "bear"
    if key == "stock_split" and re.search(r"reverse|become one|combine|consolidat", m.group(0), re.IGNORECASE):
        return "bear"
    if key == "bankruptcy" and re.search(r"\b(?:emerg\w+|exit\w*) (?:from )?(?:chapter|bankruptcy)|chapter 11 exit|"
                                         r"\bavoids?\b", text, re.IGNORECASE):
        return "neutral"
    return EVENT_POLARITY[key]


def _named_mover(text: str, start: int) -> bool:
    """True when a price move at `start` has a named subject right before it
    ("Nvidia stock falls", "while AMD jumps") rather than the market at large
    ("Stocks slip 1%")."""
    words = re.findall(r"\$?[A-Za-z][\w&.'-]*", text[max(0, start - 40):start])[-3:]
    title_case = is_title_case(text)
    for word in words:
        bare = word.removesuffix("'s").rstrip(".")
        if bare.startswith("$") or _is_entity_word(bare, title_case):
            return True
        if title_case and bare[:1].isupper() and bare.lower() not in _COMMON_VOCAB:
            return True
    return False


def _resolve_conflicts(text: str, events: list[DetectedEvent], starts: list[int]) -> list[DetectedEvent]:
    """A headline rarely reports one stock both jumping and dropping. When
    both fire, keep the move with a named subject ("Stocks slip 1%, but
    Nvidia stock is rising"); if both have one ("Nvidia stock falls 3% while
    AMD jumps 5%"), the first — the headline's lead subject; if neither, the
    one with a stated magnitude, else the later one."""
    ups = [i for i, e in enumerate(events) if e.key == "price_up"]
    downs = [i for i, e in enumerate(events) if e.key == "price_down"]
    if not (ups and downs):
        return events
    up, down = ups[0], downs[0]
    named_up, named_down = _named_mover(text, starts[up]), _named_mover(text, starts[down])
    if named_up != named_down:
        loser = down if named_up else up
    elif named_up:
        loser = down if starts[up] < starts[down] else up
    elif (events[up].value is None) != (events[down].value is None):
        loser = up if events[up].value is None else down
    else:
        loser = up if starts[up] < starts[down] else down
    return [e for i, e in enumerate(events) if i != loser]


# --------------------------------------------------------------------------- #
# Whose event is it? (company-aware attribution)
# --------------------------------------------------------------------------- #
# Events owned by the subject of their clause ("X files for bankruptcy",
# "X beats estimates", "X stock jumps 5%").
_SUBJECT_EVENTS = frozenset({
    "earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut", "record_results", "buyback",
    "dividend_raise", "dividend_cut", "layoffs", "recall", "exec_departure", "exec_hire", "offering", "bankruptcy",
    "delisting", "insider_buy", "insider_sell", "all_time_high", "low_52w", "high_52w", "stock_split", "price_up",
    "price_down", "regulatory_approval", "regulatory_setback", "index_inclusion", "data_breach",
})
# Events with an actor and a target ("Hindenburg short report on X", "Masimo
# sues X", "SEC probes X"): the target may follow the phrase.
_TARGET_EVENTS = frozenset({"lawsuit", "investigation", "settlement", "short_report"})
# Multi-party events: any party named in the clause owns them.
_PARTY_EVENTS = frozenset({"m_and_a", "partnership", "contract_win", "product_launch"})

_COMMON_VOCAB = STOPWORDS | GENERIC_WORDS | HEADLINE_VERBS | MOVE_WORDS | CALENDAR_WORDS | COMMON_HEADLINE_WORDS | wordset("""
ai ceo cfo coo cto ev evs eps ipo etf gdp cpi us u.s uk fy q1 q2 q3 q4 h1 h2 pc
revenue revenues sales profit profits earnings results quarter quarterly annual fiscal guidance outlook forecast
estimates expectations consensus record high low bankruptcy chapter lawsuit suit probe report short seller sellers
filing files filed recall layoffs jobs dividend buyback offering notice delisting shares stock
even as while whereas but yet after amid following despite since because before ahead though although when once
until why how what where who leader leaders giant maker makers chipmaker automaker retailer firm firms company
companies group business unit""")
_ACRONYM_WORDS = wordset("ai ceo cfo coo cto ev evs eps ipo etf gdp cpi us uk fy q1 q2 q3 q4 h1 h2 pc ii iii yoy qoq")
_WORD_TOKEN_RE = re.compile(r"(?<![\w$&.'-])\$?[A-Za-z][\w&.'-]*")
_SUBORDINATOR_RE = re.compile(
    r"(?:,\s*|\s)(?:even as|even though|as|while|whereas|but|yet|after|amid|following|despite|since|because|before|"
    r"ahead of|though|although|when|once|until)\s",
    re.IGNORECASE,
)
# "as" that is not a conjunction: "such as", "steps down as CEO", "as much as".
_AS_NOT_CLAUSE_BEFORE_RE = re.compile(r"\b(?:such|well|much|many|high|low|long|soon|far|known|named|same|serves?|"
                                      r"served|serving|joins?|joined|rated|seen|down|appointed|ousted|out|resigns?|"
                                      r"resigned|retires?|retired|remains?|stays?|returns?|takes? over|hired|tapped|"
                                      r"exits?|leaves|left|step)\s*$", re.IGNORECASE)
_AS_NOT_CLAUSE_AFTER_RE = re.compile(r"^(?:well|of|part|a result|much|many|expected|usual|planned|needed|soon|"
                                     r"long|if|though|its (?:new )?(?:ceo|cfo|chief|chair|president|head)|"
                                     r"(?:new |interim )?(?:ceo|cfo|coo|cto|chief|chair\w*|president|head|director))\b",
                                     re.IGNORECASE)
# What may sit between an event phrase and the company it happened to:
# "layoffs hit Tesla", "recall of 2 million Tesla vehicles", "lawsuit filed
# against Meta", "short position in Nikola".
_ATTACH_GAP_RE = re.compile(r"^\s*(?:[\w$.,%-]+\s+){0,3}$")
_TARGET_PREPOSITIONS = wordset("on against into in at of over targeting toward towards vs versus with from by")
_GAP_FILLERS = wordset("the a an its their his her new fresh another major big more")
# Gap pieces that keep the company as the subject: its own unit ("EchoStar unit
# Dish DBS"), an executive ("Apple CEO John Ternus is planning layoffs"), a
# ticker in parentheses, coordinated peers ("Ford, GM and Stellantis stock").
_OWN_UNIT_RE = re.compile(r"^(?:'s)?\s+(?:unit|subsidiary|division|arm|affiliate|bank|brand)\b", re.IGNORECASE)
_EXEC_NAME_RE = re.compile(r"(?:'s)?\s*\b(?:ceo|cfo|coo|cto|chief(?:\s+\w+)?(?:\s+officer)?|chair(?:man|woman)?|"
                           r"president|founder|co-founder|director|executive|exec|boss|head)\s+"
                           r"(?:(?-i:[A-Z])[\w.'-]*\s+){1,3}", re.IGNORECASE)
_PAREN_RE = re.compile(r"\([^()]{0,40}\)")
_COORDINATION_RE = re.compile(r"^(?:\s*(?:,|and|&|or)\s+(?:\$?(?-i:[A-Z])[\w&.'-]*\s*){1,3})+")
_LEAD_SUBORDINATOR_RE = re.compile(r"^\s*,?\s*(?:even as|even though|ahead of|\w+)\s+")


def _words(text: str) -> list[str]:
    return [m.group(0).removesuffix("'s").rstrip(".") for m in _WORD_TOKEN_RE.finditer(text)]


def _is_entity_word(word: str, title_case: bool) -> bool:
    """A word that names an organization or person. Title Case headlines
    capitalize everything, so there only entity *shapes* count ("AT&T",
    "AMD", "EchoStar"); sentence case also trusts capitalization."""
    low = word.lower()
    if not word or low in _ACRONYM_WORDS:
        return False
    if word.startswith("$") and len(word) > 1:
        return True
    if "&" in word and len(word) > 2:
        return True
    letters = word.replace(".", "").replace("-", "")
    if letters.isupper() and 2 <= len(letters) <= 6 and letters.isalpha():
        return True
    if any(c.isupper() for c in word[1:]) and any(c.islower() for c in word):
        return True
    if is_known_firm(word):
        return True
    return not title_case and word[:1].isupper() and low not in _COMMON_VOCAB


def _has_entity(text: str, title_case: bool, sentence_start: bool, loose: bool = False) -> bool:
    """True when `text` names someone (see _is_entity_word). A capitalized
    first word of a sentence proves nothing in sentence case. With `loose`,
    a Title Case word outside the common headline vocabulary counts too,
    unless it reads as a verb ("... After Samsung Unveils ...")."""
    for k, word in enumerate(_words(text)):
        if k == 0 and sentence_start and not title_case:
            if _is_entity_word(word, title_case=True):  # only shape counts sentence-initially
                return True
            continue
        if _is_entity_word(word, title_case):
            return True
        if (loose and title_case and word[:1].isupper() and word.lower() not in _COMMON_VOCAB
                and not word.lower().endswith(("ed", "ing"))):
            return True
    return False


def _immediate_subject(segment: str) -> str:
    """The noun phrase right before an event phrase: the last few words after
    the last comma, parenthesized tickers removed."""
    tail = _PAREN_RE.sub(" ", segment).rsplit(",", 1)[-1]
    words = tail.split()
    return " ".join(words[-4:])


def _clauses(text: str) -> list[tuple[int, int]]:
    """Clause spans: hard boundaries (". ", "; ", ": ", " - ", " | ") and
    subordinating conjunctions ("as", "while", "after", "amid" …). Nothing
    inside parentheses splits ("Apple (NASDAQ: AAPL) director …")."""
    masked = _PAREN_RE.sub(lambda m: "x" * len(m.group(0)), text)
    cuts: set[int] = {0, len(text)}
    for m in _CLAUSE_SPLIT_RE.finditer(masked):
        cuts.add(m.end())
    for m in _SUBORDINATOR_RE.finditer(masked):
        word = m.group(0).strip(" ,").lower()
        if word == "as" and (_AS_NOT_CLAUSE_BEFORE_RE.search(masked[max(0, m.start() - 20):m.start()])
                             or _AS_NOT_CLAUSE_AFTER_RE.match(masked[m.end():m.end() + 30])):
            continue
        cuts.add(m.start() + 1)
    points = sorted(cuts)
    return [(a, b) for a, b in pairwise(points) if b > a]


def _sentence_start(text: str, pos: int) -> bool:
    return not text[:pos].strip() or bool(re.search(r"[.!?:\"(]\s*$|\s-\s*$", text[:pos]))


def _new_subject_between(text: str, a: int, b: int, title_case: bool) -> bool:
    """Between our mention [.., a) and an event at b, does another subject
    take over? "AT&T says Dish bankruptcy …" (an entity right before the
    event) or a comma splice "Nvidia stock rises, Intel files for bankruptcy"."""
    gap = _PAREN_RE.sub(" ", text[a:b])
    if _OWN_UNIT_RE.match(gap):  # "EchoStar unit Dish DBS files …": the company's own unit
        return False
    gap = _COORDINATION_RE.sub(" ", _EXEC_NAME_RE.sub(" ", gap, count=1) if _EXEC_NAME_RE.match(gap) else gap)
    if "," in gap:
        head, _sep, tail = gap.rpartition(",")
        if (re.search(r"[A-Za-z]", head) and 0 < len(tail.split()) <= 3
                and _has_entity(tail, title_case, sentence_start=False, loose=True)):
            return True
    if gap.startswith("'s"):  # "Microsoft's X account was hacked": the company's own thing
        return False
    return _has_entity(_immediate_subject(gap), title_case, sentence_start=False)


def _attached_after(text: str, stop: int, mentions: list, kind: str, title_case: bool) -> bool:
    """The company follows the event phrase as its object: "layoffs hit
    Tesla", "lawsuit filed against Meta", "short position in Nikola"."""
    for mention in mentions:
        if mention.start < stop:
            continue
        gap = text[stop:mention.start]
        if not _ATTACH_GAP_RE.match(gap) or re.search(r"[;:!?|]", gap) or _SUBORDINATOR_RE.search(f" {gap} "):
            return False
        if "," in gap and not re.fullmatch(r"\s*,\s*\w+ing\s*", gap):  # ", sending Nvidia to a record"
            return False
        words = [w for w in _words(gap) if w.lower() not in _GAP_FILLERS]
        if _has_entity(gap, title_case, sentence_start=False):
            return False
        if kind == "target":
            return not words or words[-1].lower() in _TARGET_PREPOSITIONS
        return len(words) <= 2
    return False


def _owned(text: str, hit: _Hit, mentions: list, cues: list, clauses: list[tuple[int, int]],
           title_case: bool) -> bool:
    """Does the event at `hit` belong to the company with these mentions?

    Subject events need the company as the subject of their clause, or as
    the subject of the clause they hang off when they have none of their
    own ("Nvidia shares fall after record buyback"); target events also
    accept the company as the object ("Masimo sues Apple"); multi-party
    events accept any non-modifier mention (or brand cue) in the clause."""
    kind = "target" if hit.key in _TARGET_EVENTS else "party" if hit.key in _PARTY_EVENTS else "subject"
    subjects = [m for m in mentions if m.subject_like or (kind == "party" and m.adjunct)]
    if kind == "party":
        subjects = sorted(subjects + cues, key=lambda m: m.start)
    if not subjects:  # only "Tesla rival …" / "… after delays in AT&T deal"
        return False
    start, stop = hit.start, hit.stop
    if any(start <= m.start < stop for m in subjects):
        return True
    ci = next((i for i, (a, b) in enumerate(clauses) if a <= start < b), len(clauses) - 1)
    c_start, c_end = clauses[ci]
    before = [m for m in subjects if c_start <= m.start and m.end <= start]
    if before and (kind != "subject" or not _new_subject_between(text, before[-1].end, start, title_case)):
        return True
    if _attached_after(text, stop, subjects, kind, title_case):
        return True
    if kind == "party" and any(c_start <= m.start < c_end for m in subjects):
        return True
    opener = _immediate_subject(text[c_start:start])
    anaphora = re.match(r"\s*(?:\w+\s+)?(?:its|their)\b", opener, re.IGNORECASE)  # "as its Mastercard launch"
    own_subject = not anaphora and _has_entity(opener, title_case, _sentence_start(text, c_start) and
                                               opener == text[c_start:start].strip(), loose=kind == "party")
    if kind != "target" and own_subject:
        return False  # "… while AMD jumps 5%": that clause has its own subject
    for a, b in reversed(clauses[:ci]):  # inherit the subject of the clause it hangs off
        if any(a <= m.start < b for m in subjects):
            return True
        if _has_entity(text[a:b], title_case, _sentence_start(text, a)):
            return False
    # A subject-less lead clause takes the subject of the next one:
    # "Stock jumps 5% after Apple beats estimates".
    if ci == 0 and len(clauses) > 1 and all(w.lower() in _COMMON_VOCAB for w in _words(text[c_start:start])):
        a, _b = clauses[1]
        lead = _LEAD_SUBORDINATOR_RE.match(text[a:])
        offset = a + (lead.end() if lead else 0)
        return any(offset <= m.start <= offset + 1 for m in subjects)
    return False


def _attribute(text: str, hits: list[_Hit], company: CompanyRef) -> list[_Hit]:
    """Drop company-specific events that belong to another entity."""
    from app.nlp.relevance import brand_cue_mentions, explain_relevance  # local: relevance imports this module

    mentions = explain_relevance(text, company).mentions
    if not mentions:
        return hits  # the company is not named: owner unknown
    cues = brand_cue_mentions(text, company)
    clauses = _clauses(text)
    title_case = is_title_case(text)
    return [h for h in hits if h.key not in _SUBJECT_EVENTS | _TARGET_EVENTS | _PARTY_EVENTS
            or _owned(text, h, mentions, cues, clauses, title_case)]
