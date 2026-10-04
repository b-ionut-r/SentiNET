"""Turn registry names into the brand names people actually write.

Every text source searches for `CompanyRef.short_name`, so this module decides
search precision for the whole app. Registry names are noisy in different ways:

    "NVIDIA CORP"                      (SEC: all caps, legal suffix)
    "Apple Inc." / "Meta Platforms"    (Yahoo: legal suffix / industry word)
    "Merck &"  "The Goldman Sachs"     (Yahoo displayName artifacts)
    "State Street SPDR S&P 500 ETF T"  (truncated fund names)

`clean_company_name` applies generic, conservative rules; `BRANDS` holds the
curated exceptions where the brand differs from any mechanical cleaning
("Advanced Micro Devices" -> "AMD", SPY -> "S&P 500"). Everything is pure and
deterministic so it is easy to test.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

from app.resolve.words import COMMON_WORDS

# --------------------------------------------------------------------------- #
# Curated brands: ticker -> (short_name, aliases)
# --------------------------------------------------------------------------- #
# Used where mechanical cleaning cannot know how the press refers to a company,
# where the brand is a common English word that needs precise aliases, or where
# a fund is really "about" an index/asset. Aliases are names that identify the
# company on their own (former names, flagship brands) — never product nouns
# that also mean something else.


@dataclass(frozen=True)
class Brand:
    short_name: str
    aliases: tuple[str, ...] = ()


BRANDS: dict[str, Brand] = {
    # Mega caps & household names
    "AAPL": Brand("Apple", ("Apple Inc",)),
    "MSFT": Brand("Microsoft"),
    "GOOGL": Brand("Alphabet", ("Google",)),
    "GOOG": Brand("Alphabet", ("Google",)),
    "AMZN": Brand("Amazon", ("Amazon.com", "AWS")),
    "META": Brand("Meta", ("Meta Platforms", "Facebook")),
    "NVDA": Brand("Nvidia"),
    "TSLA": Brand("Tesla"),
    "BRK-A": Brand("Berkshire Hathaway", ("Berkshire",)),
    "BRK-B": Brand("Berkshire Hathaway", ("Berkshire",)),
    "JPM": Brand("JPMorgan", ("JPMorgan Chase", "JP Morgan", "J.P. Morgan")),
    "BAC": Brand("Bank of America", ("BofA",)),
    "C": Brand("Citigroup", ("Citi",)),
    "WFC": Brand("Wells Fargo"),
    "GS": Brand("Goldman Sachs"),
    "MS": Brand("Morgan Stanley"),
    "SCHW": Brand("Charles Schwab", ("Schwab",)),
    "AXP": Brand("American Express", ("Amex",)),
    "V": Brand("Visa", ("Visa Inc",)),
    "MA": Brand("Mastercard"),
    "PYPL": Brand("PayPal"),
    "XYZ": Brand("Block", ("Block Inc", "Cash App", "Square", "Afterpay")),
    "SQ": Brand("Block", ("Block Inc", "Cash App", "Square", "Afterpay")),
    "COIN": Brand("Coinbase"),
    "HOOD": Brand("Robinhood", ("Robinhood Markets",)),
    "SOFI": Brand("SoFi", ("SoFi Technologies",)),
    "AMD": Brand("AMD", ("Advanced Micro Devices",)),
    "INTC": Brand("Intel"),
    "AVGO": Brand("Broadcom"),
    "QCOM": Brand("Qualcomm"),
    "TSM": Brand("TSMC", ("Taiwan Semiconductor",)),
    "ASML": Brand("ASML"),
    "SMCI": Brand("Supermicro", ("Super Micro Computer", "Super Micro")),
    "ON": Brand("Onsemi", ("ON Semiconductor",)),
    "MU": Brand("Micron", ("Micron Technology",)),
    "ARM": Brand("Arm Holdings", ("Arm",)),
    "IBM": Brand("IBM", ("International Business Machines",)),
    "ORCL": Brand("Oracle", ("Oracle Corp",)),
    "CRM": Brand("Salesforce"),
    "NOW": Brand("ServiceNow"),
    "PLTR": Brand("Palantir", ("Palantir Technologies",)),
    "SNOW": Brand("Snowflake", ("Snowflake Inc",)),
    "ZM": Brand("Zoom", ("Zoom Video", "Zoom Communications")),
    "U": Brand("Unity Software", ("Unity",)),
    "SNAP": Brand("Snap", ("Snap Inc", "Snapchat")),
    "PINS": Brand("Pinterest"),
    "RDDT": Brand("Reddit"),
    "NFLX": Brand("Netflix"),
    "DIS": Brand("Disney", ("Walt Disney",)),
    "TTD": Brand("The Trade Desk", ("Trade Desk",)),
    "SPOT": Brand("Spotify"),
    "UBER": Brand("Uber"),
    "LYFT": Brand("Lyft"),
    "ABNB": Brand("Airbnb"),
    "BKNG": Brand("Booking Holdings", ("Booking.com",)),
    "SHOP": Brand("Shopify"),
    "BABA": Brand("Alibaba"),
    "PDD": Brand("PDD Holdings", ("Temu", "Pinduoduo")),
    "JD": Brand("JD.com"),
    "BIDU": Brand("Baidu"),
    "NIO": Brand("Nio"),
    "RIVN": Brand("Rivian"),
    "LCID": Brand("Lucid", ("Lucid Group", "Lucid Motors")),
    "F": Brand("Ford", ("Ford Motor",)),
    "GM": Brand("General Motors"),
    "GE": Brand("GE Aerospace", ("General Electric",)),
    "BA": Brand("Boeing"),
    "T": Brand("AT&T"),
    "VZ": Brand("Verizon"),
    "TMUS": Brand("T-Mobile"),
    "WMT": Brand("Walmart"),
    "COST": Brand("Costco"),
    "TGT": Brand("Target", ("Target Corp", "Target Corporation")),
    "HD": Brand("Home Depot"),
    "LOW": Brand("Lowe's"),
    "KO": Brand("Coca-Cola", ("Coca Cola",)),
    "PEP": Brand("PepsiCo"),
    "PG": Brand("Procter & Gamble", ("P&G",)),
    "JNJ": Brand("Johnson & Johnson", ("J&J",)),
    "LLY": Brand("Eli Lilly", ("Lilly",)),
    "NVO": Brand("Novo Nordisk"),
    "MRK": Brand("Merck"),
    "PFE": Brand("Pfizer"),
    "UNH": Brand("UnitedHealth"),
    "XOM": Brand("Exxon Mobil", ("ExxonMobil", "Exxon")),
    "CVX": Brand("Chevron"),
    "SHEL": Brand("Shell", ("Shell plc",)),
    "BP": Brand("BP"),
    "OXY": Brand("Occidental", ("Occidental Petroleum",)),
    "DE": Brand("Deere", ("John Deere",)),
    "CAT": Brand("Caterpillar"),
    "MMM": Brand("3M"),
    "UPS": Brand("UPS", ("United Parcel Service",)),
    "FDX": Brand("FedEx"),
    "DAL": Brand("Delta Air Lines"),
    "UAL": Brand("United Airlines"),
    "AAL": Brand("American Airlines"),
    "LUV": Brand("Southwest Airlines"),
    "RCL": Brand("Royal Caribbean"),
    "CCL": Brand("Carnival", ("Carnival Corp",)),
    "MCD": Brand("McDonald's"),
    "SBUX": Brand("Starbucks"),
    "CMG": Brand("Chipotle"),
    "NKE": Brand("Nike"),
    "LULU": Brand("Lululemon"),
    "GPS": Brand("Gap Inc", ("Gap",)),
    "GAP": Brand("Gap Inc", ("Gap",)),
    "EL": Brand("Estée Lauder", ("Estee Lauder",)),
    "GME": Brand("GameStop"),
    "AMC": Brand("AMC Entertainment", ("AMC Theatres",)),
    "BB": Brand("BlackBerry"),
    "MSTR": Brand("Strategy", ("MicroStrategy",)),
    "DELL": Brand("Dell"),
    "HPQ": Brand("HP Inc", ("HP",)),
    "HPE": Brand("Hewlett Packard Enterprise", ("HPE",)),
    "ADBE": Brand("Adobe"),
    "AIG": Brand("AIG", ("American International Group",)),
    "CSCO": Brand("Cisco"),
    "ANET": Brand("Arista", ("Arista Networks",)),
    "PANW": Brand("Palo Alto Networks"),
    "CRWD": Brand("CrowdStrike"),
    "NET": Brand("Cloudflare"),
    "DDOG": Brand("Datadog"),
    "TXN": Brand("Texas Instruments"),
    "AMAT": Brand("Applied Materials"),
    "LRCX": Brand("Lam Research"),
    "ADI": Brand("Analog Devices"),
    "MRVL": Brand("Marvell", ("Marvell Technology",)),
    # Funds: what the fund is *about* is what the news covers.
    "SPY": Brand("S&P 500", ("SPDR S&P 500",)),
    "VOO": Brand("S&P 500", ("Vanguard S&P 500",)),
    "IVV": Brand("S&P 500", ("iShares Core S&P 500",)),
    "QQQ": Brand("Nasdaq 100", ("Nasdaq-100", "Invesco QQQ")),
    "QQQM": Brand("Nasdaq 100", ("Nasdaq-100",)),
    "TQQQ": Brand("Nasdaq 100", ("Nasdaq-100",)),
    "DIA": Brand("Dow Jones", ("Dow Jones Industrial Average",)),
    "IWM": Brand("Russell 2000"),
    "VTI": Brand("US stock market", ("Vanguard Total Stock Market",)),
    "GLD": Brand("Gold", ("gold prices", "SPDR Gold")),
    "IAU": Brand("Gold", ("gold prices",)),
    "SLV": Brand("Silver", ("silver prices",)),
    "USO": Brand("Oil", ("crude oil", "oil prices")),
    "TLT": Brand("Treasury bonds", ("Treasuries", "Treasury yields")),
    "SMH": Brand("Semiconductor stocks", ("chip stocks",)),
    "SOXX": Brand("Semiconductor stocks", ("chip stocks",)),
    "SOXL": Brand("Semiconductor stocks", ("chip stocks",)),
    "XLF": Brand("Financial stocks", ("bank stocks",)),
    "XLK": Brand("Tech stocks", ("technology stocks",)),
    "XLE": Brand("Energy stocks", ("oil stocks",)),
    "ARKK": Brand("ARK Innovation", ("Cathie Wood",)),
    "IBIT": Brand("Bitcoin", ("iShares Bitcoin Trust",)),
    "GBTC": Brand("Bitcoin", ("Grayscale Bitcoin Trust",)),
}

# Bare symbols users type for major cryptocurrencies (see normalize_ticker).
CRYPTO_NAMES: dict[str, str] = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "XRP": "XRP", "SOL": "Solana",
    "DOGE": "Dogecoin", "ADA": "Cardano", "BNB": "BNB", "AVAX": "Avalanche",
    "SHIB": "Shiba Inu", "PEPE": "Pepe", "XLM": "Stellar", "DOT": "Polkadot",
    "LTC": "Litecoin", "LINK": "Chainlink", "TRX": "Tron", "TON": "Toncoin",
    "SUI": "Sui", "HBAR": "Hedera", "BCH": "Bitcoin Cash", "UNI": "Uniswap",
    "ATOM": "Cosmos", "NEAR": "Near Protocol", "APT": "Aptos", "ARB": "Arbitrum",
}

# Bare symbols that are safe to read as the coin. Excluded on purpose because a
# listed security owns the symbol (verified on Yahoo 2026-10-04): LTC (LTC
# Properties), LINK (Interlink), SUI (Sun Communities), BCH (Banco de Chile), TRX,
# ATOM, NEAR, APT, ARB. BTC/ETH/XRP are spot ETFs on the very same coin.
BARE_CRYPTO: frozenset[str] = frozenset({
    "BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "BNB", "AVAX", "SHIB", "PEPE", "XLM", "DOT", "TON", "HBAR", "UNI",
})

CRYPTO_ALIASES: dict[str, tuple[str, ...]] = {
    "ETH": ("Ether",),
    "XRP": ("Ripple",),
    "BNB": ("Binance Coin",),
    "SHIB": ("Shiba Inu coin",),
}

# --------------------------------------------------------------------------- #
# Generic cleaning rules
# --------------------------------------------------------------------------- #
# Pure corporate-structure words: always safe to drop from the end.
_LEGAL = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "limited", "plc", "llc", "lp", "l p", "nv", "n v", "sa", "s a", "spa",
    "s p a", "ag", "se", "ab", "asa", "oyj", "as", "a s", "bv", "kk", "k k",
    "gmbh", "sarl", "pte", "pty", "bhd", "tbk", "holdings", "holding", "group",
    "new", "the", "p l c", "l l c", "com", "aktiengesellschaft", "publ", "a/s",
}
# Industry words dropped only when what remains is one distinctive token
# ("Palantir Technologies" -> "Palantir", but "Palo Alto Networks" stays).
_DESCRIPTORS = {
    "technologies", "technology", "platforms", "communications", "networks",
    "systems", "software", "global", "markets", "worldwide", "international",
    "automotive", "wholesale", "outdoor", "entertainment", "interactive",
    "brands", "enterprises", "solutions", "industries", "therapeutics",
    "pharmaceuticals", "biosciences", "biotherapeutics", "labs", "laboratories",
    "motors", "athletica", "resorts", "hospitality", "financial", "bancorp",
    "sciences", "medicine", "health", "healthcare", "energy", "foods", "restaurants", "aviation",
    "airways", "pharma", "biotech", "biopharma", "biopharmaceuticals", "diagnostics", "genomics",
    "robotics", "oncology", "motor", "computing", "quantum", "ai", "investments", "enterprise",
}
# Descriptors that may also be dropped after a multi-word head
# ("Capital One Financial" -> "Capital One", "Philip Morris International").
_MULTI_OK = {"financial", "international", "worldwide", "global", "investments"}
# All-caps tokens that are real acronyms/stylings, not shouting.
_KEEP_UPPER = {
    "ASML", "AECOM", "AMETEK", "CBRE", "IDEX", "NIO", "IBM", "CSX", "EPAM",
    "SAP", "ICICI", "HDFC", "BHP", "LVMH", "USAA", "AGNC", "EQT", "NRG", "HCA",
    "KLA", "NXP", "PPG", "RPM", "XPO", "DXC", "PDD", "UPS", "CVS", "AMC", "GE",
    "GM", "HP", "BP", "RTX", "AES", "AIG", "ADP", "AMD", "TSMC", "NVR", "PACCAR",
    "CDW", "IQVIA", "MSCI", "STERIS", "ZTO", "HSBC", "UBS", "ING", "RELX",
    "BASF", "ABB", "SPDR", "ETF", "USD", "AT&T", "CME", "ICE",
    "KKR", "TJX", "PNC", "MGM", "NCR", "TPG", "BWX", "SBA", "WEC", "DTE",
    "CMS", "AEP", "PSEG", "NEE", "MDU", "UGI", "AVX", "NOV", "EOG", "APA",
    "IPG", "WPP", "ADT", "AGCO", "MSA", "IQ", "NI", "ITT", "SPX", "UFP", "FMC",
    "IAC", "LKQ", "BJ", "ODP", "JD", "QXO", "AST", "CRH", "BNY", "SM",
}
_SMALL_WORDS = {"of", "and", "the", "for", "de", "du", "la", "le", "von", "van", "&", "y"}
# Three-letter English words that are not acronyms in an all-caps registry title.
_WORDS3 = {"one", "new", "oil", "gas", "air", "sun", "bio", "car", "top", "big", "red", "sky", "sea", "art", "net",
           "web", "box", "pet", "pay", "ice", "inn", "bay", "oak", "key", "map", "fox", "toy", "tea", "joy", "max"}
_ROMAN = re.compile(r"^(?:I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3})$", re.IGNORECASE)
# Single generic heads that are not a brand without their suffix ("News Corp", not "News").
_GENERIC_HEADS = {"news", "public", "general", "first", "national", "american", "united", "western", "eastern",
                  "southern", "northern", "international", "global", "capital", "standard", "republic", "royal",
                  "pacific", "atlantic", "central", "security", "union", "liberty", "federal"}
_STATE_SUFFIX = re.compile(r"(?:\s*/\s*(?:ADR|ADS|[A-Z]{2,3})\b/?)+\s*$|\s*/\s*$", re.IGNORECASE)  # "/DE/", "/ ADR"
_VOWELS = set("AEIOUY")

_SHARE_CLASS = re.compile(
    r"\b(?:class|cl\.?|series)\s+[a-z]\b|\bcommon\s+stock\b|\bordinary\s+shares?\b"
    r"|\bamerican\s+depositary\s+(?:shares?|receipts?)\b|\bdepositary\s+(?:shares?|receipts?)\b"
    r"|\bsponsored\b|\bunsponsored\b|\badrs?\b|\bads\b|\bregistry\s+shares?\b"
    r"|\bcommon\s+st\w*$",  # truncated "Common St…"
    re.IGNORECASE,
)
_PARENS = re.compile(r"\s*\([^)]*\)?\s*$")  # trailing "(The)", "(CAD HEDGED)", truncated "(T"
_DASH_TAIL = re.compile(r"\s+[-–—]\s+.*$")  # "ASML Holding N.V. - New York Re…"
_TOKEN_SPLIT = re.compile(r"[\s,]+")


def _norm_token(tok: str) -> str:
    """Lower-case a token and drop dots for suffix comparison ("N.V." -> "nv")."""
    return tok.lower().replace(".", "").strip()


def _strip_trailing(tokens: list[str], words: set[str]) -> list[str]:
    """Drop trailing tokens (also two-token spellings like "L P") found in `words`."""
    out = list(tokens)
    while len(out) > 1:
        last = _norm_token(out[-1])
        pair = f"{_norm_token(out[-2])} {last}" if len(out) > 2 else ""
        if pair in words:
            out = out[:-2]
        elif last in words or last in {"&", "and"} or last == "":
            out = out[:-1]
        else:
            break
    return out


def _is_shouting(tok: str) -> bool:
    letters = [c for c in tok if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _wordlike(letters: str) -> bool:
    """Pronounceable 4-letter shape ("NIKE", "BANK", "FORD") vs acronym ("ASML", "CBRE")."""
    if sum(c in _VOWELS for c in letters) >= 2:
        return True
    return re.fullmatch(r"[^AEIOUY]{1,2}[AEIOUY][^AEIOUY]{1,2}", letters) is not None


def _fix_case_token(tok: str, *, all_caps_name: bool) -> str:
    """Normalize one token's case: "NVIDIA" -> "Nvidia", keep "IBM", "AT&T", "SoFi"."""
    if not _is_shouting(tok):
        if tok.islower() and tok.isalpha() and tok not in _SMALL_WORDS:
            return tok[:1].upper() + tok[1:]  # "lululemon" -> "Lululemon"
        return tok
    bare = tok.strip(".,")
    if bare.upper() in _KEEP_UPPER or any(c.isdigit() for c in bare) or "&" in bare:
        return tok
    letters = [c for c in bare if c.isalpha()]
    if all_caps_name and bare.lower() in _SMALL_WORDS:
        return tok.lower()
    if all_caps_name and _norm_token(bare) in _LEGAL:
        return tok[:1].upper() + tok[1:].lower()  # "BERKSHIRE HATHAWAY INC" -> "... Inc"
    if _ROMAN.match(bare):
        return tok.upper()  # "Acquisition Corp VIII"
    if all_caps_name and (bare.lower() in _WORDS3 or bare.lower() in COMMON_WORDS):
        return tok[:1].upper() + tok[1:].lower()  # "CAPITAL ONE" -> "Capital One", "OLD DOMINION" -> "Old …"
    if len(letters) <= 3:
        return tok  # acronym-sized: "AMC", "CVS"
    if len(letters) == 4 and not _wordlike(bare.upper()):
        return tok  # "ASML", "INTL", "TSLA": no word shape
    # Word-shaped: title-case each hyphenated part ("COCA-COLA" -> "Coca-Cola", "LOWE'S" -> "Lowe's").
    return "-".join(part[:1].upper() + part[1:].lower() for part in tok.split("-"))


def fix_case(name: str) -> str:
    """Case-normalize a whole name (handles both SEC all-caps and Yahoo mixed case)."""
    tokens = name.split()
    all_caps = all(_is_shouting(t) or not any(c.isalpha() for c in t) for t in tokens)
    fixed = [_fix_case_token(t, all_caps_name=all_caps) for t in tokens]
    if fixed and fixed[0].isalpha() and fixed[0].islower():
        fixed[0] = fixed[0][:1].upper() + fixed[0][1:]
    return " ".join(fixed)


def ascii_fold(text: str) -> str:
    """"Estée" -> "Estee" (search engines and GDELT are inconsistent with accents)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _base_clean(raw: str) -> list[str]:
    """Shared first pass: entities, parentheses, share classes, legal suffixes."""
    return _base_clean_with_tail(raw)[0]


def _base_clean_with_tail(raw: str) -> tuple[list[str], list[str]]:
    """`_base_clean` plus the legal tokens it removed from the end."""
    name = html.unescape(raw or "").replace("\u00a0", " ").strip()
    name = _STATE_SUFFIX.sub("", name)
    name = _DASH_TAIL.sub("", name)
    name = _PARENS.sub("", name)
    name = _SHARE_CLASS.sub(" ", name)
    # "Amazon.com" -> "Amazon", but keep "JD.com" (the bare head would be too short).
    name = re.sub(r"\b([A-Za-z][\w&'-]{2,})\.com\b", r"\1", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" ,.-")
    tokens = [t for t in _TOKEN_SPLIT.split(name) if t]
    if tokens and tokens[0].lower() == "the" and len(tokens) > 1:
        tokens = tokens[1:]
    kept = _strip_trailing(tokens, _LEGAL)
    return [t.rstrip(",") for t in kept], tokens[len(kept):]


def _distinctive(token: str) -> bool:
    bare = token.strip(".,&'!")
    return len(bare) >= 3 and bare.lower() not in COMMON_WORDS


def _strip_descriptors(tokens: list[str]) -> list[str]:
    """Drop trailing industry words when the remaining head still identifies the firm."""
    head = _strip_trailing(tokens, _DESCRIPTORS | _LEGAL)
    if head and len(head) < len(tokens):
        if len(head) == 1 and _distinctive(head[0]):
            return head
        removed = {_norm_token(t) for t in tokens[len(head):]}
        if len(head) > 1 and removed <= (_MULTI_OK | _LEGAL):
            return head
    # Fall back to dropping only the generic corporate descriptors
    # ("Charles River Laboratories International" -> "Charles River Laboratories").
    head = _strip_trailing(tokens, _MULTI_OK | _LEGAL)
    if len(head) > 1 or (len(head) == 1 and _distinctive(head[0])):
        return head
    return tokens


def _needs_suffix(head: str) -> bool:
    """A lone head that is no brand on its own: a generic word ("News") or a tiny word ("On", "Nu")."""
    bare = head.lower().strip(".,")
    letters = [c for c in head if c.isalpha()]
    return bare in _GENERIC_HEADS or (len(letters) <= 2 and not _is_shouting(head) and not any(c.isdigit() for c in head))


def _legal_alias(short: str, candidates: list[str]) -> str | None:
    """"Sea" + "Sea Limited" -> "Sea Limited": the precise form of an everyday-word brand."""
    for cand in candidates:
        tokens, tail = _base_clean_with_tail(cand)
        if tail and " ".join(tokens).lower().strip(" ,.") == short.lower():
            suffix = tail[0].rstrip(".,")
            if _norm_token(suffix) not in {"the", "new", "com", "&", "and"}:
                return f"{short} {fix_case(suffix) if _is_shouting(suffix) else suffix}"
    return None


def clean_company_name(raw: str) -> str:
    """Registry name -> brand name used in headlines.

    >>> clean_company_name("NVIDIA CORP")
    'Nvidia'
    >>> clean_company_name("Alphabet Inc. Class A")
    'Alphabet'
    >>> clean_company_name("Palantir Technologies Inc.")
    'Palantir'
    >>> clean_company_name("Palo Alto Networks, Inc.")
    'Palo Alto Networks'
    """
    tokens, legal_tail = _base_clean_with_tail(raw)
    if not tokens:
        return (raw or "").strip()
    tokens = _strip_descriptors(tokens)
    if len(tokens) == 1 and legal_tail and _needs_suffix(tokens[0]):
        tokens = [tokens[0], legal_tail[0].rstrip(".,")]  # "NEWS CORP" -> "News Corp", "On Holding"
    name = " ".join(tokens).strip(" ,.&")
    name = name.removesuffix(" and")
    return fix_case(name)


# Fund issuers / wrappers that say nothing about what the fund holds.
_FUND_NOISE = re.compile(
    r"\b(?:state street|spdr|ishares|vanguard|invesco|proshares|direxion(?: daily)?|schwab|"
    r"global x|vaneck|first trust|wisdomtree|fidelity|jpmorgan|j\.p\. morgan|ark|"
    r"select sector|index fund|index|etf|etn|fund|trust|shares|series \d+|portfolio|core|"
    r"ultrapro|ultrashort|ultra|bull|bear|[1-3]x|daily|leveraged|inverse|msci)\b",
    re.IGNORECASE,
)


def clean_fund_name(raw: str) -> str:
    """"iShares Russell 2000 ETF" -> "Russell 2000" (best effort for uncurated ETFs)."""
    name = html.unescape(raw or "")
    name = _PARENS.sub("", name)
    core = re.sub(r"\s+", " ", _FUND_NOISE.sub(" ", name)).strip(" ,.-&")
    return core if len(core) >= 3 else (name.strip() or raw)


def clean_crypto_name(raw: str) -> str:
    """"Bitcoin USD" -> "Bitcoin"."""
    name = re.sub(r"\s+(?:USD|USDT|USDC|EUR|GBP|BTC)$", "", (raw or "").strip(), flags=re.IGNORECASE)
    return name or raw


@dataclass
class NameResult:
    short_name: str
    aliases: list[str] = field(default_factory=list)


def _sane_display_name(name: str | None) -> bool:
    if not name or len(name.strip()) < 2:
        return False
    return not re.search(r"(?:&|\band)\s*$", name.strip(), re.IGNORECASE)  # "Merck &" is truncated


def derive_names(
    ticker: str,
    quote_type: str,
    *,
    display_name: str | None = None,
    long_name: str | None = None,
    short_name: str | None = None,
    registry_name: str | None = None,
    crypto_name: str | None = None,
) -> NameResult:
    """Pick the best brand name + aliases from every name we know.

    Preference: curated brand > Yahoo displayName (already a brand) > long name
    > Yahoo short name > SEC registry title > the ticker itself.
    """
    ticker = ticker.upper()
    base = ticker.split("-")[0] if quote_type == "CRYPTOCURRENCY" else ticker
    candidates = [n for n in (long_name, short_name, registry_name) if n]

    if ticker in BRANDS:
        brand = BRANDS[ticker]
        short = brand.short_name
        aliases = list(brand.aliases)
    elif quote_type == "CRYPTOCURRENCY":
        short = CRYPTO_NAMES.get(base) or clean_crypto_name(crypto_name or long_name or short_name or base)
        aliases = list(CRYPTO_ALIASES.get(base, ()))
    elif quote_type in {"ETF", "MUTUALFUND"}:
        source = long_name or short_name or registry_name or ticker
        short = clean_fund_name(source)
        aliases = [fix_case(source.strip())] if source and source != short else []
    else:
        if _sane_display_name(display_name):
            short = clean_company_name(display_name or "")
        elif candidates:
            short = clean_company_name(candidates[0])
        else:
            short = ticker
        aliases = []
        # The fuller legal-cleaned form is a strong identifier too ("Palantir Technologies").
        for cand in candidates[:1]:
            full = fix_case(" ".join(_base_clean(cand)).strip(" ,.&"))
            if full and full.lower() != short.lower():
                aliases.append(full)
        # An everyday-word brand is unambiguous with its legal suffix ("Sea Limited", "Pool Corporation").
        if is_common_word_name(short) and (legal := _legal_alias(short, candidates)):
            aliases.append(legal)

    folded = ascii_fold(short)
    if folded and folded != short:
        aliases.append(folded)
    seen = {short.lower()}
    unique: list[str] = []
    for alias in aliases:
        key = alias.lower()
        if alias and key not in seen and key != ticker.lower() and not _needs_suffix(alias):
            seen.add(key)
            unique.append(alias)
    return NameResult(short_name=short or ticker, aliases=unique)


def is_common_word_name(short_name: str) -> bool:
    """True when the brand is also an ordinary word (search needs disambiguation)."""
    head = short_name.split()[0].lower().strip(".,") if short_name else ""
    return len(short_name.split()) == 1 and (head in COMMON_WORDS or len(head) <= 3)
