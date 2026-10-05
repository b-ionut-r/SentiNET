"""Publisher identity and trust.

Headlines arrive with messy outlet names ("Barron's on MSN", "Yahoo! Finance
Canada", "Investing.com Nigeria", "bloomberg.com"). `canonical_publisher`
folds them to one display name, `publisher_trust` maps that name to a weight
used in aggregation, and `is_press_release` flags self-published copy (wires,
company newsrooms, class-action law-firm blasts) that should not count as
independent coverage.

Trust tiers (multiplicative weights, 1.0 = typical finance outlet):
    1.25  wires & majors (Reuters, Bloomberg, WSJ, FT, AP, CNBC, Barron's …)
    1.05-1.15  strong general/business press (NYT, CNN, Fortune, Nikkei …)
    0.9-1.0    mainstream finance (Yahoo Finance, Motley Fool, Seeking Alpha …)
    0.6-0.8    algorithmic/aggregator outlets (MarketBeat, GuruFocus, Stock Titan …)
    0.55       press-release wires, company newsrooms, low-quality content sites
    0.5        MarketBeat-network auto-content sites
    0.8        unknown

Identity is keyed on a compact form of the name — case, spacing, punctuation
and the domain suffix dropped — so "NETT"/"nett", "AD HOC NEWS"/"ad-hoc-news.de"
and "Foreign Policy Journal"/"foreignpolicyjournal.com" are one outlet with one
trust weight. An outlet outside the table keeps the display name it was first
seen under, so every spelling of it counts once in outlet tallies.
"""
from __future__ import annotations

import re
from functools import lru_cache
from urllib.parse import urlsplit

from app.nlp.text import clean_text, fold

UNKNOWN_TRUST = 0.8
PRESS_RELEASE_TRUST = 0.55

# (canonical name, trust, aliases incl. domains). Aliases are matched after
# normalization (lower-case, no "the", no punctuation, no region suffix).
_PUBLISHERS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    # --- wires & majors -------------------------------------------------- #
    ("Reuters", 1.25, ("reuters", "reuters.com", "thomson reuters")),
    ("Bloomberg", 1.25, ("bloomberg", "bloomberg.com", "bloomberg news", "bloomberg businessweek",
                         "bloomberg law", "bnn bloomberg", "bnnbloomberg.ca")),
    ("WSJ", 1.25, ("wsj", "wall street journal", "wsj.com", "the wall street journal")),
    ("Financial Times", 1.25, ("financial times", "ft", "ft.com")),
    ("AP", 1.25, ("ap", "associated press", "ap news", "apnews.com")),
    ("CNBC", 1.25, ("cnbc", "cnbc.com", "cnbc tv18", "cnbctv18.com")),
    ("Barron's", 1.25, ("barron's", "barrons", "barrons.com")),
    ("MarketWatch", 1.2, ("marketwatch", "marketwatch.com")),
    ("Dow Jones", 1.25, ("dow jones", "dow jones newswires", "djnewswires")),
    ("MT Newswires", 1.0, ("mt newswires", "mtnewswires", "mtnewswires.com")),
    ("The Information", 1.2, ("the information", "theinformation.com")),
    ("Axios", 1.15, ("axios", "axios.com")),
    ("The Economist", 1.2, ("economist", "economist.com")),
    ("Nikkei Asia", 1.15, ("nikkei", "nikkei asia", "asia.nikkei.com")),
    # --- strong general / business press ------------------------------------- #
    ("New York Times", 1.15, ("new york times", "nytimes", "nytimes.com", "nyt")),
    ("Washington Post", 1.1, ("washington post", "washingtonpost.com")),
    ("CNN", 1.1, ("cnn", "cnn business", "cnn.com")),
    ("BBC", 1.1, ("bbc", "bbc news", "bbc.com", "bbc.co.uk")),
    ("The Guardian", 1.05, ("guardian", "theguardian.com")),
    ("Fortune", 1.1, ("fortune", "fortune.com")),
    ("Forbes", 1.0, ("forbes", "forbes.com")),
    ("Business Insider", 1.0, ("business insider", "businessinsider.com", "insider", "markets insider")),
    ("Investor's Business Daily", 1.05, ("investor's business daily", "investors business daily", "ibd",
                                         "investors.com")),
    ("Morningstar", 1.1, ("morningstar", "morningstar.com")),
    ("Fox Business", 1.0, ("fox business", "foxbusiness.com")),
    ("Semafor", 1.05, ("semafor", "semafor.com")),
    ("Politico", 1.05, ("politico", "politico.com")),
    ("TechCrunch", 1.05, ("techcrunch", "techcrunch.com")),
    ("The Verge", 1.0, ("the verge", "verge", "theverge.com")),
    ("Wired", 1.0, ("wired", "wired.com")),
    ("Ars Technica", 1.0, ("ars technica", "arstechnica.com")),
    ("Sherwood News", 1.0, ("sherwood", "sherwood news", "sherwood.news")),
    ("Quartz", 0.95, ("quartz", "qz", "qz.com")),
    ("The Globe and Mail", 1.05, ("globe and mail", "theglobeandmail.com")),
    ("Financial Post", 1.0, ("financial post", "financialpost.com")),
    ("Australian Financial Review", 1.1, ("afr", "australian financial review", "afr.com")),
    ("South China Morning Post", 1.0, ("south china morning post", "scmp", "scmp.com")),
    ("The Times", 1.0, ("the times", "thetimes.co.uk")),
    ("The Telegraph", 1.0, ("telegraph", "telegraph.co.uk")),
    ("CBS News", 1.0, ("cbs news", "cbsnews.com")),
    ("NBC News", 1.0, ("nbc news", "nbcnews.com")),
    ("ABC News", 1.0, ("abc news", "abcnews.go.com")),
    ("NPR", 1.05, ("npr", "npr.org")),
    ("USA Today", 0.95, ("usa today", "usatoday.com")),
    ("Los Angeles Times", 1.0, ("los angeles times", "latimes.com")),
    ("Economic Times", 0.95, ("economic times", "economictimes.indiatimes.com", "the economic times")),
    ("Mint", 0.95, ("mint", "livemint", "livemint.com")),
    ("Business Standard", 0.95, ("business standard", "business-standard.com")),
    ("Moneycontrol", 0.9, ("moneycontrol", "moneycontrol.com")),
    ("Kiplinger", 0.95, ("kiplinger", "kiplinger.com")),
    ("Fast Company", 0.9, ("fast company", "fastcompany.com")),
    ("Inc.", 0.85, ("inc", "inc.com")),
    ("Engadget", 0.9, ("engadget", "engadget.com")),
    ("Tom's Hardware", 0.9, ("tom's hardware", "tomshardware.com")),
    ("CoinDesk", 1.0, ("coindesk", "coindesk.com")),
    ("The Block", 0.95, ("the block", "theblock.co")),
    # --- mainstream finance ---------------------------------------------------- #
    ("Yahoo Finance", 1.0, ("yahoo finance", "yahoo! finance", "finance.yahoo.com", "yahoo", "yahoo.com",
                            "yahoo news")),
    ("Investing.com", 0.95, ("investing.com", "investing com", "investing")),
    ("Investopedia", 0.95, ("investopedia", "investopedia.com")),
    ("Seeking Alpha", 0.95, ("seeking alpha", "seekingalpha.com", "seekingalpha")),
    ("The Motley Fool", 0.9, ("motley fool", "the motley fool", "fool.com", "fool", "fool uk", "fool.co.uk")),
    ("Benzinga", 0.9, ("benzinga", "benzinga.com", "benzinga prediction markets")),
    ("Zacks", 0.85, ("zacks", "zacks investment research", "zacks.com", "zacks research")),
    ("InvestorPlace", 0.85, ("investorplace", "investorplace.com")),
    ("TipRanks", 0.85, ("tipranks", "tipranks.com")),
    ("Barchart", 0.9, ("barchart", "barchart.com")),
    ("TheStreet", 0.9, ("thestreet", "the street", "thestreet.com")),
    ("24/7 Wall St.", 0.85, ("24/7 wall st", "24/7 wall st.", "247wallst.com", "24 7 wall st")),
    ("Nasdaq", 0.9, ("nasdaq", "nasdaq.com")),
    ("TradingView", 0.85, ("tradingview", "tradingview.com")),
    ("Stocktwits", 0.85, ("stocktwits", "stocktwits.com")),
    ("Simply Wall St", 0.8, ("simply wall st", "simply wall street", "simplywall.st")),
    ("RTTNews", 0.8, ("rttnews", "rttnews.com")),
    ("Proactive Investors", 0.75, ("proactive investors", "proactiveinvestors.com", "proactive")),
    ("Bankrate", 0.9, ("bankrate", "bankrate.com")),
    ("9to5Mac", 0.85, ("9to5mac", "9to5mac.com")),
    ("MacRumors", 0.85, ("macrumors", "macrumors.com")),
    ("AppleInsider", 0.85, ("appleinsider", "appleinsider.com")),
    ("Electrek", 0.85, ("electrek", "electrek.co")),
    ("Teslarati", 0.75, ("teslarati", "teslarati.com")),
    ("Cointelegraph", 0.85, ("cointelegraph", "cointelegraph.com")),
    ("Decrypt", 0.85, ("decrypt", "decrypt.co", "decrypt news")),
    ("Trefis", 0.75, ("trefis", "trefis.com")),
    ("TIKR", 0.75, ("tikr", "tikr.com")),
    ("Insider Monkey", 0.7, ("insider monkey", "insidermonkey.com")),
    ("Moomoo", 0.7, ("moomoo", "moomoo.com")),
    ("Finviz", 0.8, ("finviz", "finviz.com")),
    ("MarketScreener", 0.85, ("marketscreener", "marketscreener.com", "zonebourse")),
    ("AOL", 0.9, ("aol", "aol.com", "aol.ca")),
    ("inkl", 0.85, ("inkl", "inkl.com")),
    ("The Business Journals", 1.0, ("the business journals", "bizjournals", "bizjournals.com")),
    ("New York Post", 0.85, ("new york post", "nypost", "nypost.com")),
    ("Times of India", 0.9, ("times of india", "the times of india", "timesofindia.indiatimes.com")),
    ("Deadline", 0.95, ("deadline", "deadline.com")),
    ("Quiver Quantitative", 0.7, ("quiver quantitative", "quiverquant.com")),
    ("Finbold", 0.6, ("finbold", "finbold.com")),
    ("Traders Union", 0.6, ("traders union", "tradersunion.com", "tradersunion")),
    # --- algorithmic / aggregator ------------------------------------------- #
    ("GuruFocus", 0.75, ("gurufocus", "gurufocus.com")),
    ("MarketBeat", 0.65, ("marketbeat", "marketbeat.com")),
    ("Stock Titan", 0.65, ("stock titan", "stocktitan.net", "stocktitan")),
    ("ChartMill", 0.6, ("chartmill", "chart mill", "chartmill.com")),
    ("Kalkine Media", 0.6, ("kalkine", "kalkine media", "kalkinemedia.com")),
    ("StocksToTrade", 0.6, ("stockstotrade", "stockstotrade.com")),
    ("Timothy Sykes", 0.55, ("timothysykes.com", "timothy sykes")),
    ("AD HOC NEWS", 0.55, ("ad hoc news", "ad-hoc-news.de")),
    ("ScanX", 0.55, ("scanx.trade", "scanx")),
    ("Pluang", 0.55, ("pluang",)),
    ("Techi", 0.55, ("techi.com", "techi")),
    ("EquityPandit", 0.55, ("equitypandit.com", "equitypandit")),
    ("BigGo Finance", 0.55, ("finance.biggo.com", "biggo")),
    ("MarketsMojo", 0.55, ("marketsmojo", "markets mojo", "marketsmojo.com")),
    # MarketBeat network auto-content sites.
    ("ETF Daily News", 0.5, ("etf daily news", "etfdailynews.com")),
    ("Defense World", 0.5, ("defense world", "defenseworld.net")),
    ("Ticker Report", 0.5, ("ticker report", "tickerreport.com")),
    ("The Lincolnian Online", 0.5, ("the lincolnian online", "lincolnianonline.com")),
    ("Modern Readers", 0.5, ("modern readers", "modernreaders.com")),
    ("Watch List News", 0.5, ("watch list news", "watchlistnews.com")),
    ("American Banking News", 0.5, ("american banking news", "americanbankingnews.com")),
    ("Dakota Financial News", 0.5, ("dakota financial news", "dakotafinancialnews.com")),
    ("The Enterprise Leader", 0.5, ("the enterprise leader", "theenterpriseleader.com")),
    ("Zolmax", 0.5, ("zolmax", "zolmax.com")),
    ("The Cerbat Gem", 0.5, ("the cerbat gem", "thecerbatgem.com")),
    ("Markets Daily", 0.5, ("markets daily", "marketsdaily.com")),
    # --- press-release wires -------------------------------------------------- #
    ("PR Newswire", PRESS_RELEASE_TRUST, ("pr newswire", "prnewswire", "prnewswire.com", "cision",
                                          "cision pr newswire")),
    ("GlobeNewswire", PRESS_RELEASE_TRUST, ("globenewswire", "globe newswire", "globenewswire.com",
                                            "globe newswire news room")),
    ("Business Wire", PRESS_RELEASE_TRUST, ("business wire", "businesswire", "businesswire.com")),
    ("ACCESS Newswire", PRESS_RELEASE_TRUST, ("accesswire", "access newswire", "accessnewswire.com",
                                              "accesswire.com")),
    ("Newsfile", PRESS_RELEASE_TRUST, ("newsfile", "newsfile corp", "newsfilecorp.com")),
    ("EIN Presswire", PRESS_RELEASE_TRUST, ("ein presswire", "einpresswire", "einpresswire.com", "einnews")),
    ("PRWeb", PRESS_RELEASE_TRUST, ("prweb", "prweb.com")),
    ("Marketwired", PRESS_RELEASE_TRUST, ("marketwired",)),
    ("Newswire.ca", PRESS_RELEASE_TRUST, ("newswire.ca", "cnw", "cnw group")),
    ("NewMediaWire", PRESS_RELEASE_TRUST, ("newmediawire", "newmediawire.com")),
    ("TheNewswire", PRESS_RELEASE_TRUST, ("thenewswire", "thenewswire.com")),
    ("openPR", PRESS_RELEASE_TRUST, ("openpr", "openpr.com")),
    ("Benzinga Press Releases", PRESS_RELEASE_TRUST, ("benzinga press releases",)),
)

# Named explicitly: low-quality content sites share the 0.55 weight but are not wires.
_PR_WIRES = frozenset({
    "PR Newswire", "GlobeNewswire", "Business Wire", "ACCESS Newswire", "Newsfile", "EIN Presswire", "PRWeb",
    "Marketwired", "Newswire.ca", "NewMediaWire", "TheNewswire", "openPR", "Benzinga Press Releases",
})

# Company-owned channels: "NVIDIA Newsroom", "Apple Investor Relations", "NVIDIA Blog".
_COMPANY_CHANNEL_RE = re.compile(r"\b(?:newsroom|press releases?|investor relations|ir|media center)$",
                                 re.IGNORECASE)
# Content of class-action solicitations and paid releases (almost always via PR wires).
_PR_TEXT_RE = re.compile(
    r"shareholder alert|investor alert|deadline alert|lead plaintiff deadline|investors (?:who lost|with losses)|"
    r"class action (?:lawsuit )?(?:filed|notice)|securities (?:class action|fraud) (?:lawsuit|investigation)|"
    r"(?:notifies|reminds|encourages) (?:investors|shareholders)|investigation (?:notice|on behalf of)|"
    r"law firm (?:announces|investigates)|announces (?:the )?(?:pricing|closing|launch) of|"
    r"\b(?:today|hereby) announced\b|^\s*notice (?:of|to)\b|\(?(?:nasdaq|nyse)\s?:\s?[a-z.]{1,6}\)? (?:today )?announce",
    re.IGNORECASE,
)

_REGION_SUFFIX_RE = re.compile(
    r"\s+(?:nigeria|south africa|australia|india|canada|uk|u\.k\.|singapore|philippines|malaysia|new zealand|"
    r"hong kong|ireland|en espanol|espanol|deutschland|france|italia|japan|indonesia|uae|middle east|asia|"
    r"europe|us|u\.s\.|usa|international|global)$",
    re.IGNORECASE,
)
_VIA_RE = re.compile(r"\s+(?:on|via|through)\s+(?:msn|yahoo(?:!)?(?: finance| news)?|aol|flipboard|apple news|"
                     r"google news|newsbreak|smartnews)$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w&'/. ]+")


def _key(name: str) -> str:
    out = fold(name).lower().strip()
    out = out.replace("!", "")
    out = _PUNCT_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip(" .")
    out = out.removeprefix("www.")
    return out


# The public suffix a domain-shaped name ends in: ".com", ".co.uk", ".de".
_DOMAIN_SUFFIX_RE = re.compile(r"(?:\.(?:co|com|net|org|gov|ac|ne|or))?\.[a-z]{2,6}$")
_NOT_ALNUM_RE = re.compile(r"[\W_]+")


def _compact(key: str) -> str:
    """Identity of an outlet name already passed through `_key`: no domain
    suffix, case, spacing or punctuation ("ad hoc news.de" -> "adhocnews",
    "foreign policy journal" -> "foreignpolicyjournal")."""
    return _NOT_ALNUM_RE.sub("", _DOMAIN_SUFFIX_RE.sub("", key))


_ALIAS: dict[str, str] = {}
_COMPACT_ALIAS: dict[str, str] = {}
_TRUST: dict[str, float] = {}
for _name, _trust, _aliases in _PUBLISHERS:
    _TRUST[_name] = _trust
    for _alias in (_name, *_aliases):
        _ALIAS.setdefault(_key(_alias), _name)
        if _key(_alias).startswith("the "):
            _ALIAS.setdefault(_key(_alias)[4:], _name)
        _COMPACT_ALIAS.setdefault(_compact(_key(_alias)), _name)


def _host(value: str) -> str | None:
    if "://" in value:
        host = urlsplit(value).netloc
        return host.lower().split(":")[0] or None
    if re.fullmatch(r"(?:[\w-]+\.)+[a-z]{2,}", value.strip().lower()):
        return value.strip().lower()
    return None


def _brand_display(label: str) -> str:
    """An unknown outlet's domain label as a name, so "ccn.com" and "CCN" (or
    "cryptonews.net" and "Cryptonews") count as one outlet."""
    if len(label) <= 4 or not re.search(r"[aeiouy]", label):
        return label.upper()
    return label[:1].upper() + label[1:]


def _compact_known(compact: str) -> str | None:
    return _COMPACT_ALIAS.get(compact) if len(compact) >= 3 else None


@lru_cache(maxsize=4096)
def _lookup(raw: str) -> tuple[str | None, str, str]:
    """-> (canonical name if known, cleaned display fallback, identity key)."""
    host = _host(raw.strip())
    value = clean_text(raw).strip()
    if not value and not host:
        return None, "", ""
    if host:
        host = host.removeprefix("www.")
        labels = host.split(".")
        for i in range(len(labels) - 1):  # finance.yahoo.com -> yahoo.com
            candidate = ".".join(labels[i:])
            if candidate in _ALIAS:
                return _ALIAS[candidate], host, candidate
        # Brand label before the public suffix: bloomberg.co.jp, reuters.de.
        brand = labels[-3] if len(labels) >= 3 and labels[-2] in {"co", "com", "net", "org"} else labels[-2]
        if brand in _ALIAS and len(brand) >= 3:
            return _ALIAS[brand], host, brand
        compact = _compact(brand)
        return _compact_known(compact), _brand_display(brand), compact  # ad-hoc-news.de -> AD HOC NEWS
    value = _VIA_RE.sub("", value)
    key = _key(value)
    regionless = _REGION_SUFFIX_RE.sub("", key)
    for candidate in (key, regionless, key.removesuffix(".com"), key.removeprefix("the ")):
        if candidate in _ALIAS:
            return _ALIAS[candidate], value, candidate
    compact = _compact(regionless) or _compact(key)
    if value.islower() and " " not in value:
        value = _brand_display(value)  # "nett" is the "NETT" of another feed
    return _compact_known(compact), value, compact


# Display name each unknown outlet was first seen under, by identity key, so
# "Foreign Policy Journal" and "Foreignpolicyjournal" count as one outlet.
# Bounded: past the cap a new outlet keeps its own spelling.
_SEEN_DISPLAY: dict[str, str] = {}
_SEEN_MAX = 20_000



def canonical_publisher(name_or_domain: str | None) -> str | None:
    """One display name per outlet: "Barron's on MSN" -> "Barron's",
    "Yahoo! Finance Canada" -> "Yahoo Finance", "https://www.reuters.com/x" ->
    "Reuters". Unknown outlets keep their (cleaned) name or bare domain."""
    if not name_or_domain:
        return None
    known, fallback, key = _lookup(name_or_domain)
    if known or not fallback:
        return known or None
    if not key:
        return fallback
    seen = _SEEN_DISPLAY.get(key)
    if seen is None and len(_SEEN_DISPLAY) < _SEEN_MAX:
        seen = _SEEN_DISPLAY.setdefault(key, fallback)
    return seen or fallback


def is_known_publisher(name_or_domain: str | None) -> bool:
    """True when the outlet is in the curated table."""
    return bool(name_or_domain) and _lookup(name_or_domain)[0] is not None


def is_press_release(publisher: str | None = None, title: str | None = None) -> bool:
    """True for press-release wires, company-owned channels ("NVIDIA
    Newsroom") and class-action / offering announcements by their wording.
    Wording never overrides an established newsroom: Reuters writing "Nvidia
    today announced record revenue" is still Reuters."""
    if publisher:
        canonical = canonical_publisher(publisher)
        if canonical in _PR_WIRES:
            return True
        if _COMPANY_CHANNEL_RE.search(clean_text(publisher)):
            return True
        if publisher_trust(publisher) >= 0.9:
            return False
    return bool(title) and bool(_PR_TEXT_RE.search(fold(title)))


def publisher_trust(publisher_or_domain: str | None) -> float:
    """Trust weight for an outlet (see module docstring for tiers)."""
    if not publisher_or_domain:
        return UNKNOWN_TRUST
    canonical = canonical_publisher(publisher_or_domain)
    if canonical in _TRUST:
        return _TRUST[canonical]
    if _COMPANY_CHANNEL_RE.search(clean_text(publisher_or_domain)):
        return PRESS_RELEASE_TRUST
    return UNKNOWN_TRUST
