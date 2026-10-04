"""Search-term construction shared by the keyword-search sources.

Precision *and* recall: a search source should return items whose headline
actually names the asset (that is what the downstream relevance filter keeps),
without drowning in homonyms ("Target" the retailer vs. "price target",
"$SPY" vs. spies, "Shell" vs. bash shell).

`search_terms(company)` derives, per asset class:

* **names** — distinctive names, most specific first: the cleaned brand name
  plus aliases (equities); the coin name (crypto, "Bitcoin USD" -> "Bitcoin");
  for ETFs the *underlying theme*, because news is written about "the S&P 500",
  not about "SPDR S&P 500 ETF Trust" (known funds via a small table, others by
  stripping issuer/boilerplate words from the fund name).
* **symbol_searchable** — whether the bare symbol is safe to search alone
  (not a word like NOW/ALL/SPY, not 1-2 letters, not identical to the name).
* **ambiguous** — the primary name is an everyday word, so it must always be
  paired with finance context.
* **context** — finance words that anchor a full-text search to markets.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.sources.base import CompanyRef

Asset = Literal["equity", "etf", "crypto", "other"]

# Brand names whose everyday meaning dominates general text ("price target", "bash shell",
# "H-1B visa", "snap packages"): never search them without finance context. Names whose
# capitalized use mostly means the company (Apple, Amazon, Meta, Ford) are not listed.
COMMON_WORD_NAMES = frozenset(
    ["target", "block", "visa", "shell", "snap", "gap", "coach", "match", "zoom", "unity", "sea", "ball", "box", "ring", "square", "delta", "southern", "progressive", "general", "national", "united", "american", "first", "global", "pioneer", "digital", "energy", "root", "rocket", "lemonade", "dollar", "crown", "sun", "edge", "monster", "marathon", "riot", "compass", "fortune", "genius", "carnival", "progress", "premier", "frontier", "liberty", "pinnacle", "summit"]
)

# Symbols that read as words/abbreviations in ordinary text.
WORD_TICKERS = frozenset(
    ["A", "AI", "ALL", "AM", "AN", "ANY", "APP", "ARE", "ARM", "AT", "BE", "BEST", "BIG", "BOX", "BUY", "CAN", "CAR", "CASH", "CAT", "CEO", "COST", "DD", "DO", "DOG", "DOT", "EAT", "EDIT", "EOD", "EV", "EVER", "EYE", "FAST", "FLY", "FOR", "FUN", "GAIN", "GAS", "GO", "GOLD", "GOOD", "GROW", "HAS", "HE", "HOME", "HOPE", "HUGE", "IT", "JOB", "KEY", "KIDS", "LAND", "LIFE", "LINK", "LIVE", "LOVE", "LOW", "MAIN", "MAN", "MARK", "MIND", "MOVE", "NEAR", "NEW", "NEXT", "NICE", "NOW", "OK", "ON", "ONE", "OPEN", "OUT", "PAY", "PEAK", "PLAY", "PLUG", "PUMP", "REAL", "RIDE", "RUN", "SAFE", "SAVE", "SEE", "SHOP", "SKY", "SNOW", "SO", "SOL", "SPY", "STAR", "TECH", "TELL", "TEN", "TOP", "TRUE", "TWO", "UK", "UP", "US", "USA", "WELL", "WORK", "YOU"]
)

FINANCE_CONTEXT: tuple[str, ...] = ("stock", "shares", "investors", "analyst", "earnings")
ETF_CONTEXT: tuple[str, ...] = ("stocks", "index", "market", "ETF")
CRYPTO_CONTEXT: tuple[str, ...] = ("crypto", "price", "token", "traders")

# Funds whose news flow is about an index/asset rather than the wrapper itself.
ETF_THEMES: dict[str, tuple[str, ...]] = {
    "SPY": ("S&P 500",), "VOO": ("S&P 500",), "IVV": ("S&P 500",), "SPLG": ("S&P 500",),
    "QQQ": ("Nasdaq 100", "Nasdaq"), "QQQM": ("Nasdaq 100", "Nasdaq"), "TQQQ": ("Nasdaq 100", "Nasdaq"),
    "DIA": ("Dow Jones",), "IWM": ("Russell 2000",), "VTI": ("stock market",),
    "GLD": ("gold price", "gold"), "IAU": ("gold price", "gold"), "SLV": ("silver price", "silver"),
    "USO": ("oil prices", "crude oil"), "TLT": ("Treasury yields", "Treasuries"),
    "SMH": ("chip stocks", "semiconductor stocks"), "SOXX": ("chip stocks", "semiconductor stocks"),
    "ARKK": ("ARK Innovation", "Cathie Wood"), "IBIT": ("Bitcoin ETF", "Bitcoin"),
    "XLK": ("tech stocks",), "XLF": ("bank stocks", "financial stocks"), "XLE": ("energy stocks", "oil stocks"),
}

_ETF_NOISE = re.compile(
    r"\b(spdr|ishares|vanguard|invesco|schwab|proshares|direxion|global x|ark|first trust|vaneck|wisdomtree|"
    r"jpmorgan|fidelity|franklin|xtrackers|amplify|roundhill|grayscale|defiance|yieldmax|simplify|pacer|"
    r"state street|dimensional|avantis|graniteshares|etfs?|trust|fund|index|shares|portfolio|series \d+|"
    r"daily|ultrapro|ultra|ultrashort|bull|bear|[23]x|leveraged|select sector|core|total|tr|the)\b",
    re.IGNORECASE,
)
_LEGAL_SUFFIX = re.compile(
    r"(,|\s)+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|n\.?v|s\.?a|ag|se|holdings?|group|"
    r"class [a-c]|common stock|ordinary shares|adr|american depositary shares|\(the\))\.?$",
    re.IGNORECASE,
)
_CRYPTO_QUOTE = re.compile(r"[\s-]+(usd|usdt|eur|btc)$", re.IGNORECASE)


@dataclass(frozen=True)
class SearchTerms:
    names: tuple[str, ...]  # never empty; most specific first
    symbol: str  # base symbol, e.g. "NVDA", "BTC", "BRK-B"
    symbol_searchable: bool
    ambiguous: bool
    context: tuple[str, ...]
    asset: Asset

    @property
    def primary(self) -> str:
        return self.names[0]

    @property
    def needs_context(self) -> bool:
        """Name-only searches are unsafe for everyday-word names and ETF themes."""
        return self.ambiguous or self.asset == "etf"


def clean_name(name: str) -> str:
    """'Apple Inc.' -> 'Apple', 'The Boeing Company' -> 'Boeing', 'Alphabet Inc. Class A' -> 'Alphabet'."""
    text = re.sub(r"[®™©]", "", name or "").strip().strip('"')
    text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE)
    previous = None
    while previous != text:
        previous = text
        text = _LEGAL_SUFFIX.sub("", text).strip(" ,.")
    return text


def is_word_ticker(symbol: str) -> bool:
    core = symbol.split("-")[0].split(".")[0].upper()
    return len(core) <= 2 or core in WORD_TICKERS


def etf_theme(company: CompanyRef) -> tuple[str, ...]:
    """What an ETF's news is actually about ("S&P 500", "semiconductor stocks"…); () if unknown."""
    known = ETF_THEMES.get(company.ticker.upper())
    if known:
        return known
    core = re.sub(r"\s+", " ", _ETF_NOISE.sub(" ", company.name or "")).strip(" ,.-&")
    if not core or core.upper() == company.base_symbol.upper():
        return ()
    if len(core.split()) == 1 and not any(ch.isdigit() for ch in core):
        core = f"{core} stocks"  # "Technology Select Sector SPDR" -> "Technology stocks"
    return (core,)


def search_terms(company: CompanyRef) -> SearchTerms:
    symbol = company.base_symbol.upper()
    theme: tuple[str, ...] = ()
    if company.is_crypto:
        asset: Asset = "crypto"
        primary = _CRYPTO_QUOTE.sub("", clean_name(company.short_name or company.name))
        context = CRYPTO_CONTEXT
    elif company.quote_type in {"ETF", "INDEX"}:  # both trade on an index/asset theme
        asset = "etf"
        theme = etf_theme(company)
        primary = theme[0] if theme else clean_name(company.short_name or company.name)
        context = ETF_CONTEXT
    else:
        asset = "equity" if company.quote_type == "EQUITY" else "other"
        primary = clean_name(company.short_name or company.name)
        context = FINANCE_CONTEXT
    primary = primary or symbol

    names: list[str] = [primary]
    for alias in (*theme[1:], *company.aliases):
        alias = clean_name(alias)
        if len(alias) >= 3 and alias.lower() not in {n.lower() for n in names} and alias.upper() != symbol:
            names.append(alias)

    ambiguous = primary.lower() in COMMON_WORD_NAMES or len(primary) <= 2
    searchable = (
        re.fullmatch(r"[A-Z]{3,5}", symbol) is not None  # "BRK-B"/"^GSPC" are not written that way in prose
        and not is_word_ticker(symbol)
        and symbol.lower() not in {n.lower() for n in names}
    )
    return SearchTerms(tuple(names[:3]), symbol, searchable, ambiguous, context, asset)


def us_symbol(company: CompanyRef) -> str | None:
    """Exchange-style US symbol for symbol-keyed feeds ("BRK-B" -> "BRK.B").

    None for crypto/indices and for foreign listings ("SHOP.TO", "AIR.PA"):
    their base symbol can belong to a *different* US company (AIR = AAR Corp).
    """
    ticker = company.ticker.upper()
    if company.quote_type not in {"EQUITY", "ETF"} or "." in ticker or ticker.startswith("^"):
        return None
    return ticker.replace("-", ".")


def or_group(terms: Sequence[str]) -> str:
    """('Nvidia', 'S&P 500') -> '(Nvidia OR "S&P 500")'; quotes anything that is not one plain word."""
    parts = [t if re.fullmatch(r"[A-Za-z0-9]+", t) else f'"{t}"' for t in terms]
    return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"
