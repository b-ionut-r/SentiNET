"""Search terms and mention matching shared by the keyword-search sources.

Precision *and* recall: a search source should return items that are really
about the asset, without drowning in homonyms ("Target" vs. "price target",
"ICE" the agency, "$SPY" vs. spies, "SoFi Stadium", "Trump" the politician).

* `search_terms(company)` derives, per asset class, the distinctive **names**
  to search (brand + aliases for equities; the coin name for crypto; for funds,
  indices, futures and FX the *underlying theme* — news is written about "the
  S&P 500", not "SPDR S&P 500 ETF Trust"), whether the bare **symbol** is safe
  to search, whether the name is an everyday word (**ambiguous**), and the
  finance **context** words that anchor a full-text search.
* `Mentions(terms)` decides whether a fetched text talks about the asset, with
  the same rules for every source (case-sensitive for word-like names, venue
  phrases removed, bare symbols only with market vocabulary).
* `issuer_symbols` / `crowd_symbol_ambiguous` handle share classes and
  ticker-words on crowd leaderboards.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.sources.base import CompanyRef
from app.sources.vocab import (
    CASE_SENSITIVE_NAMES,
    COMMON_WORD_NAMES,
    CRYPTO_NAMES,
    REDDIT_ACRONYMS,
    SELF_EVIDENT_THEMES,
    SHARE_CLASS_GROUPS,
    SHOUTED_WORDS,
    THEMES,
    WORD_TICKERS,
)

# "etf" = anything traded on a theme rather than a company: funds, indices, futures, FX.
Asset = Literal["equity", "etf", "crypto", "other"]
THEME_TYPES = frozenset({"ETF", "INDEX", "MUTUALFUND", "FUTURE", "CURRENCY"})

FINANCE_CONTEXT: tuple[str, ...] = ("stock", "shares", "investors", "analyst", "earnings")
ETF_CONTEXT: tuple[str, ...] = ("stocks", "index", "market", "ETF")
FUTURES_CONTEXT: tuple[str, ...] = ("prices", "futures", "traders", "market")
FX_CONTEXT: tuple[str, ...] = ("currency", "forex", "traders", "dollar")
CRYPTO_CONTEXT: tuple[str, ...] = ("crypto", "token", "coin", "price")

_ETF_NOISE = re.compile(
    r"\b(spdr|ishares|vanguard|invesco|schwab|proshares|direxion|global x|ark|first trust|vaneck|wisdomtree|"
    r"jpmorgan|fidelity|franklin|xtrackers|amplify|roundhill|grayscale|defiance|yieldmax|simplify|pacer|"
    r"state street|dimensional|avantis|graniteshares|etfs?|trust|fund|index|shares|portfolio|series \d+|"
    r"daily|ultrapro|ultra|ultrashort|bull|bear|[23]x|leveraged|select sector|core|total|tr|the|admiral|"
    r"investor|institutional|class [a-z]|futures?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|"
    r"july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|'?\d{2}(?=\s*$)|-\d{4})\b",
    re.IGNORECASE,
)
_THEME_CANON = {"500": "S&P 500", "s&p 500": "S&P 500", "nasdaq-100": "Nasdaq 100"}
_LEGAL_SUFFIX = re.compile(
    r"(,|\s)+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|n\.?v|s\.?a|ag|se|holdings?|group|"
    r"class [a-c]|common stock|ordinary shares|adr|american depositary shares|\(the\))\.?$",
    re.IGNORECASE,
)
_CRYPTO_QUOTE = re.compile(r"[\s-]+(usd|usdt|eur|btc)$", re.IGNORECASE)
_CRYPTO_SITE = re.compile(r"\.(dev|io|org|com|xyz|fi|finance|network)$", re.IGNORECASE)  # "APEcoin.dev"
_FINANCE_NOUN = re.compile(
    r"(?i)\b(stocks?|shares|prices?|yields?|index|indices|etfs?|futures|bonds?|treasur\w*|reits?|markets?|"
    r"banks|miners|homebuilders|equities|caps)\b"
)
_CRYPTO_NOUN = re.compile(r"(?i)(coin|token|protocol|swap|chain|network)\b")
_VENUE = r"(?:Stadium|Arena|Center|Centre|Field|Park|Bowl|Theat(?:er|re)|Tukker|Hall|Amphitheat(?:er|re)|Pavilion|Plaza|Dome|Garden)"

# Market vocabulary. `_MARKET_WORDS` is enough to keep a post that *names* the asset;
# a bare symbol or hashtag needs `_STRONG_WORDS` ("ICE shares about detentions" must fail).
_STRONG = (
    r"stocks?|earnings|revenue|guidance|analysts?|price targets?|upgraded?|downgraded?|upgrades?|downgrades?|"
    r"investors?|valuation|market cap|bullish|bearish|short(?:s|ing|ed| sellers?)|rall(?:y|ies|ied)|"
    r"sell-?off|ipo|dividends?|buybacks?|nasdaq|nyse|eps|all-time high|record high|52-week|calls|puts|"
    r"options|ticker|quarter(?:ly)?|q[1-4]|premarket|pre-market|after-hours|wall street|hedge funds?"
)
_STRONG_WORDS = re.compile(rf"(?i)\b(?:{_STRONG})\b")
_MARKET_WORDS = re.compile(
    rf"(?i)\b(?:{_STRONG}|shares|shareholders?|investing|investment|trad(?:e|es|ing|ers?)|portfolio|fiscal|"
    r"profits?|margins?|deliveries|outlook|forecast|market|futures|prices?|yields?|bonds?|etfs?|s&p 500)\b"
)
_CRYPTO_WORDS = re.compile(
    r"(?i)\b(?:crypto\w*|tokens?|coins?|memecoins?|meme ?coins?|altcoins?|blockchain|defi|nfts?|on-?chain|"
    r"wallets?|binance|coinbase|kraken|hodl|airdrops?|staking|mainnet|whales?|halving|btc|eth|sol|usdt|usdc|"
    r"stablecoins?|dex|satoshis?|sats)\b"
)
_EXCHANGES = r"(?:NYSE|NASDAQ|Nasdaq|NasdaqGS|NasdaqGM|NasdaqCM|AMEX|NYSEARCA|NYSE American|OTC|TSX|LSE|CBOE)"


@dataclass(frozen=True)
class SearchTerms:
    names: tuple[str, ...]  # never empty; most specific first
    symbol: str  # base symbol, e.g. "NVDA", "BTC", "BRK-B"
    symbol_searchable: bool
    ambiguous: bool  # primary name is an everyday word: always pair it with context
    context: tuple[str, ...]
    asset: Asset
    self_evident: tuple[str, ...] = ()  # names that are market talk on their own ("S&P 500", "Bitcoin")

    @property
    def primary(self) -> str:
        return self.names[0]

    @property
    def needs_context(self) -> bool:
        """Name-only searches are unsafe for everyday-word names and themes."""
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
    """Symbols a case-insensitive search engine would confuse with words (NOW, SPY, F, TRUMP)."""
    core = symbol.split("-")[0].split(".")[0].upper()
    return len(core) <= 2 or core in WORD_TICKERS


def crowd_symbol_ambiguous(symbol: str) -> bool:
    """True when a Reddit leaderboard's count for this symbol measures a word, not the stock.

    Boards count upper-case tokens, so length is not the issue (MU, GM, KO are real
    chatter); typed-in-caps words and trading lingo are (YOU, ALL, ES, DTE, HYSA).
    """
    core = symbol.upper().replace(".", "-")
    return core in SHOUTED_WORDS or core in REDDIT_ACRONYMS


def issuer_symbols(company: CompanyRef) -> frozenset[str]:
    """The symbol plus its share-class siblings, in '-', '.' and '/' spellings (GOOGL -> GOOG, GOOGL)."""
    symbol = company.ticker.upper()
    group = next((g for g in SHARE_CLASS_GROUPS if symbol in g), frozenset({symbol}))
    return frozenset(s.replace("-", sep) for s in group for sep in ("-", ".", "/"))


def etf_theme(company: CompanyRef) -> tuple[str, ...]:
    """What a fund/index/future's news is about ("S&P 500", "semiconductor stocks", "gold price"); () if unknown."""
    known = THEMES.get(company.ticker.upper())
    if known:
        return known
    core = re.sub(r"\s+", " ", _ETF_NOISE.sub(" ", company.name or "")).strip(" ,.-&")
    core = _THEME_CANON.get(core.lower(), core)
    if not core or core.upper() == company.base_symbol.upper() or not re.search(r"[A-Za-z]", core):
        return ()
    if not _FINANCE_NOUN.search(core) and not re.search(r"\d", core):
        noun = {"FUTURE": "prices", "CURRENCY": ""}.get((company.quote_type or "").upper(), "stocks")
        core = f"{core} {noun}".strip()  # "Health Care" -> "Health Care stocks", "Lumber" -> "Lumber prices"
    return (core,)


def _crypto_names(company: CompanyRef, symbol: str) -> tuple[list[str], bool]:
    known = CRYPTO_NAMES.get(symbol)
    if known:
        return list(known[0]), known[1]
    raw = clean_name(company.short_name or company.name)
    name = _CRYPTO_SITE.sub("", _CRYPTO_QUOTE.sub("", raw)).strip()
    return [name or symbol], True  # unlisted coins: precision first, always anchored


def search_terms(company: CompanyRef) -> SearchTerms:
    symbol = company.base_symbol.upper()
    quote_type = (company.quote_type or "EQUITY").upper()
    if company.is_crypto:
        asset: Asset = "crypto"
        names, ambiguous = _crypto_names(company, symbol)
        context = CRYPTO_CONTEXT
    elif quote_type in THEME_TYPES:
        asset = "etf"
        names = list(etf_theme(company)) or [clean_name(company.short_name or company.name)]
        ambiguous = False
        context = {"FUTURE": FUTURES_CONTEXT, "CURRENCY": FX_CONTEXT}.get(quote_type, ETF_CONTEXT)
    else:
        asset = "equity" if quote_type == "EQUITY" else "other"
        names = [clean_name(company.short_name or company.name)]
        ambiguous = names[0].lower() in COMMON_WORD_NAMES or len(names[0]) <= 2
        context = FINANCE_CONTEXT
    names = [n for n in names if n] or [symbol]
    for alias in company.aliases:
        alias = clean_name(alias)
        if len(alias) >= 3 and alias.lower() not in {n.lower() for n in names} and alias.upper() != symbol:
            names.append(alias)
    names = names[:3]

    searchable = (
        re.fullmatch(r"[A-Z]{3,5}", symbol) is not None  # "BRK-B"/"^GSPC" are not written that way in prose
        and not is_word_ticker(symbol)
        and symbol.lower() not in {n.lower() for n in names}
    )
    if asset == "crypto":
        evident = [n for n in names if not ambiguous or _CRYPTO_NOUN.search(n)]
    elif asset == "etf":
        evident = [n for n in names if _FINANCE_NOUN.search(n) or re.search(r"\d", n) or n.lower() in SELF_EVIDENT_THEMES]
    else:
        evident = []
    return SearchTerms(tuple(names), symbol, searchable, ambiguous, context, asset, tuple(evident))


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


def _case_sensitive(name: str, terms: SearchTerms) -> bool:
    """Acronyms ("UPS", "AMD"), word-like brands ("Apple", "Micron") and ambiguous primaries keep their casing."""
    return name.isupper() or name.lower() in CASE_SENSITIVE_NAMES or (terms.ambiguous and name == terms.primary)


def _name_pattern(name: str, case_sensitive: bool) -> str:
    body = re.escape(name)
    if case_sensitive:  # as written or in capitals ("Apple"/"APPLE", never "apple")
        body = f"(?:{body}|{re.escape(name.upper())})"
    else:
        body = f"(?i:{body})"
    return rf"(?<![\w$@]){body}(?![\w&])"


class Mentions:
    """Does a text talk about the asset? One rule set for every source.

    * `$SYM` (any case) always counts.
    * A name counts when it appears with the right casing outside venue phrases
      ("SoFi Stadium", "Big Apple"); in `social` mode it also needs market
      vocabulary unless the name is self-evident ("S&P 500", "Bitcoin").
    * Exchange notation ("NYSE: TGT") counts; a bare upper-case symbol or `#SYM`
      needs strong market vocabulary ("ICE shares about detentions" fails).
    """

    def __init__(self, terms: SearchTerms) -> None:
        self.terms = terms
        symbol = re.escape(terms.symbol).replace(r"\-", "[-./]")  # $BRK.B / $BRK-B / $BRK/B
        self._cashtag = re.compile(rf"(?i)(?<![\w$])\${symbol}(?![\w])")
        self._hashtag = re.compile(rf"(?i)(?<![\w#])#{symbol}(?![\w])")
        self._listed = re.compile(rf"\b{_EXCHANGES}\s*:\s*{symbol}\b")
        self._bare = re.compile(rf"(?<![\w$#.-]){symbol}(?![\w])") if terms.symbol_searchable else None
        case = {n: _case_sensitive(n, terms) for n in terms.names}
        self._names = [(n, re.compile(_name_pattern(n, case[n]))) for n in terms.names]
        alts = "|".join(re.escape(n) for n in terms.names)
        self._venue = re.compile(rf"(?i:(?:{alts})\s+{_VENUE})|\bBig Apple\b")
        self._crypto = terms.asset == "crypto"

    def cashtag(self, text: str) -> bool:
        return bool(self._cashtag.search(text))

    def named(self, text: str) -> str | None:
        """The first name found (venue phrases removed), else None."""
        cleaned = self._venue.sub(" ", text)
        return next((n for n, pattern in self._names if pattern.search(cleaned)), None)

    def _intent(self, text: str, strong: bool) -> bool:
        words = _STRONG_WORDS if strong else _MARKET_WORDS
        return bool(words.search(text) or (self._crypto and _CRYPTO_WORDS.search(text)))

    def about(self, text: str, *, social: bool = False) -> bool:
        if not text:
            return False
        if self.cashtag(text) or self._listed.search(text):
            return True
        name = self.named(text)
        if name is not None and (not social or name in self.terms.self_evident or self._intent(text, strong=False)):
            return True
        symbol_hit = self._hashtag.search(text) or (self._bare is not None and self._bare.search(text))
        return bool(symbol_hit and self._intent(text, strong=True))
