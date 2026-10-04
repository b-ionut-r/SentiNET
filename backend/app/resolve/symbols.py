"""Ticker normalization, company resolution and symbol search.

Resolution fuses two free registries:

* SEC `company_tickers.json` (US-listed issuers -> CIK + legal title), and
* Yahoo Finance (via yfinance) for display names, asset type, exchange, sector.

The brand name every source searches for (`CompanyRef.short_name`) comes from
`app.resolve.names`. Resolution never raises for a well-formed ticker: when the
registries are unreachable it degrades to whatever is known (worst case the
ticker itself), and only the complete result is cached for long.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from app.core.cache import cached
from app.core.sync import run_yahoo
from app.resolve.names import BARE_CRYPTO, CRYPTO_NAMES, derive_names, fix_case
from app.schemas import SymbolMatch
from app.sources.base import CompanyRef

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Ticker normalization
# --------------------------------------------------------------------------- #
_VALID = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")
_CLASS_SHARE = re.compile(r"^([A-Z]{1,5})[./]([ABC])$")  # BRK.B, BF/B -> BRK-B, BF-B
_CRYPTO_PAIR = re.compile(r"^([A-Z0-9]{2,10})[-/]?(USD|USDT|USDC)$")
_CRYPTO_STOCKTWITS = re.compile(r"^([A-Z0-9]{2,10})\.X$")  # StockTwits style: BTC.X
_EXCHANGE_PREFIX = re.compile(
    r"^(?:NASDAQ|NYSE|NYSEARCA|NYSEAMERICAN|AMEX|ARCA|BATS|OTC|TSX|LSE)\s*:\s*", re.IGNORECASE
)

# Yahoo exchange codes -> display names.
EXCHANGES: dict[str, str] = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ", "NAS": "NASDAQ",
    "NYQ": "NYSE", "NYS": "NYSE", "ASE": "NYSE American", "PCX": "NYSE Arca",
    "BTS": "Cboe BZX", "CXI": "Cboe", "PNK": "OTC", "OQB": "OTC", "OQX": "OTC",
    "CCC": "Crypto", "CME": "CME", "CBT": "CBOT", "NYM": "NYMEX", "CMX": "COMEX",
    "TOR": "TSX", "VAN": "TSXV", "LSE": "LSE", "GER": "XETRA", "FRA": "Frankfurt",
    "PAR": "Euronext Paris", "AMS": "Euronext Amsterdam", "HKG": "HKEX",
    "JPX": "Tokyo", "ASX": "ASX", "SNP": "S&P", "DJI": "Dow Jones", "NIM": "NASDAQ",
}

_SEARCH_TYPES = {"EQUITY", "ETF", "CRYPTOCURRENCY", "INDEX", "MUTUALFUND"}


def normalize_ticker(raw: str) -> str | None:
    """Canonical Yahoo-style symbol, or None when `raw` cannot be a ticker.

    Rules (in order):
      * trim, drop a leading ``$``/``#`` cashtag and an ``EXCHANGE:`` prefix, upper-case;
      * share classes ``BRK.B`` / ``BRK/B`` -> ``BRK-B`` (only A/B/C; ``.L``/``.TO`` etc.
        are exchange suffixes and are kept);
      * crypto: ``BTC.X`` (StockTwits), ``BTCUSD``, ``BTC/USD`` -> ``BTC-USD``; a bare
        major-coin symbol (``BTC``, ``ETH``, ``SOL``… see `BARE_CRYPTO`) -> ``-USD``,
        because in a sentiment terminal "BTC" means bitcoin, not the Grayscale mini trust.
        Coins whose bare symbol belongs to a listed stock (``LTC``, ``LINK``, ``SUI``…)
        stay stocks; ask for ``LTC-USD`` explicitly;
      * indices keep ``^`` (``^VIX``), futures keep ``=F`` (``GC=F``);
      * anything else must match ``[A-Z0-9][A-Z0-9.-=]{0,14}`` and contain a letter.
    """
    if not raw or not isinstance(raw, str):
        return None
    sym = raw.strip().lstrip("$#").strip()
    sym = _EXCHANGE_PREFIX.sub("", sym).strip().upper()
    if not sym or len(sym) > 16:
        return None
    if m := _CRYPTO_STOCKTWITS.match(sym):
        sym = f"{m.group(1)}-USD"
    elif m := _CLASS_SHARE.match(sym):
        sym = f"{m.group(1)}-{m.group(2)}"
    elif sym in BARE_CRYPTO:
        sym = f"{sym}-USD"
    elif (m := _CRYPTO_PAIR.match(sym)) and m.group(1) in CRYPTO_NAMES:
        sym = f"{m.group(1)}-{m.group(2)}"
    if not _VALID.match(sym) or not re.search(r"[A-Z]", sym):
        return None
    return sym


def is_crypto_symbol(symbol: str) -> bool:
    return bool(re.match(r"^[A-Z0-9]{2,10}-(USD|USDT|USDC|EUR|GBP|BTC)$", symbol))


def exchange_label(code: str | None, full_name: str | None = None) -> str | None:
    """Yahoo exchange code -> display label ("NMS" -> "NASDAQ")."""
    if code and code.upper() in EXCHANGES:
        return EXCHANGES[code.upper()]
    return full_name or code or None


# --------------------------------------------------------------------------- #
# Logos
# --------------------------------------------------------------------------- #
def logo_url_for(symbol: str, website: str | None = None) -> str | None:
    """Best-guess logo URL (no network). The frontend should fall back on error.

    StockTwits' CDN serves logos for US-listed symbols (class shares as ``BRK.B``)
    and crypto (``BTC.X``); foreign listings, indices and futures fall back to the
    company website's favicon when we know the site.
    """
    sym = (symbol or "").upper()
    if not sym:
        return None
    if is_crypto_symbol(sym):
        return f"https://logos.stocktwits-cdn.com/{sym.split('-')[0]}.X.png"
    us_listed = re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", sym) is not None
    if us_listed:
        return f"https://logos.stocktwits-cdn.com/{sym.replace('-', '.')}.png"
    domain = _domain(website)
    if domain:
        return f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    return None


def _domain(website: str | None) -> str | None:
    if not website:
        return None
    url = website if "://" in website else f"https://{website}"
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else (host or None)


# --------------------------------------------------------------------------- #
# Company resolution
# --------------------------------------------------------------------------- #
def _infer_quote_type(symbol: str) -> str:
    if is_crypto_symbol(symbol):
        return "CRYPTOCURRENCY"
    if symbol.startswith("^"):
        return "INDEX"
    if symbol.endswith("=F"):
        return "FUTURE"
    return "EQUITY"


def build_company_ref(
    ticker: str,
    info: dict[str, Any] | None,
    sec_entry: tuple[str, str] | None,
) -> CompanyRef:
    """Pure merge of Yahoo `info` and the SEC map entry into a CompanyRef."""
    info = info or {}
    cik, sec_title = sec_entry if sec_entry else (None, None)
    quote_type = str(info.get("quoteType") or _infer_quote_type(ticker)).upper()
    long_name = info.get("longName") or None
    short_yahoo = info.get("shortName") or None
    names = derive_names(
        ticker,
        quote_type,
        display_name=info.get("displayName"),
        long_name=long_name,
        short_name=short_yahoo,
        registry_name=sec_title,
        crypto_name=info.get("name"),
    )
    if quote_type == "CRYPTOCURRENCY":
        official = names.short_name
    else:
        official = long_name or short_yahoo or (fix_case(sec_title) if sec_title else None) or ticker
    return CompanyRef(
        ticker=ticker,
        name=str(official),
        short_name=names.short_name,
        aliases=names.aliases,
        quote_type=quote_type,
        cik=cik if quote_type in {"EQUITY", "ETF"} else None,
        exchange=exchange_label(info.get("exchange"), info.get("fullExchangeName")),
        sector=info.get("sector") or None,
        industry=info.get("industry") or None,
        website=info.get("website") or None,
    )


@cached(ttl=86400, none_ttl=60)
async def _resolve_complete(ticker: str) -> CompanyRef | None:
    """Full resolution (cached 24h); None when Yahoo had nothing (not cached long)."""
    from app.intel.market_data import get_info
    from app.intel.sec import get_cik_map

    info_res, cik_res = await asyncio.gather(get_info(ticker), get_cik_map(), return_exceptions=True)
    info = info_res if isinstance(info_res, dict) else None
    cik_map = cik_res if isinstance(cik_res, dict) else {}
    if not info or not (info.get("longName") or info.get("shortName")):
        return None
    return build_company_ref(ticker, info, cik_map.get(ticker))


async def resolve_company(ticker: str) -> CompanyRef:
    """Resolve a ticker into a `CompanyRef`; never raises for a well-formed ticker."""
    sym = normalize_ticker(ticker) or ticker.strip().upper()
    try:
        ref = await _resolve_complete(sym)
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the analysis here
        logger.warning("resolve %s failed: %s", sym, exc)
        ref = None
    if ref is not None:
        return replace(ref, aliases=list(ref.aliases))  # callers may mutate; keep the cached copy pristine
    sec_entry: tuple[str, str] | None = None
    try:
        from app.intel.sec import get_cik_map

        sec_entry = (await get_cik_map()).get(sym)
    except Exception as exc:  # noqa: BLE001
        logger.info("SEC map unavailable for %s: %s", sym, exc)
    return build_company_ref(sym, None, sec_entry)


# --------------------------------------------------------------------------- #
# Symbol search
# --------------------------------------------------------------------------- #
def _yahoo_search(q: str, limit: int) -> list[dict[str, Any]]:
    import yfinance as yf

    return list(yf.Search(q, max_results=max(limit * 2, 10), news_count=0, lists_count=0,
                          include_nav_links=False, raise_errors=False).quotes or [])


def _rank_key(match: SymbolMatch, wanted: str | None) -> tuple[int, int]:
    """Exact symbol first, then US listings before foreign lines (stable otherwise)."""
    foreign = "." in match.symbol or (match.type == "CRYPTOCURRENCY" and not match.symbol.endswith("-USD"))
    return (0 if match.symbol == wanted else 1, 1 if foreign else 0)


def matches_from_yahoo(quotes: list[dict[str, Any]], q: str, limit: int) -> list[SymbolMatch]:
    """Pure: Yahoo search quotes -> ranked SymbolMatch list."""
    wanted = normalize_ticker(q)
    out: list[SymbolMatch] = []
    seen: set[str] = set()
    for item in quotes:
        sym = str(item.get("symbol") or "").upper()
        qtype = str(item.get("quoteType") or "").upper()
        if not sym or sym in seen or qtype not in _SEARCH_TYPES:
            continue
        seen.add(sym)
        name = str(item.get("longname") or item.get("shortname") or sym)
        if any(m.name == name and sym.startswith(m.symbol) for m in out):
            continue  # same issuer's units/warrants/foreign line (BIXIU, NVDA.TO): noise in autocomplete
        out.append(SymbolMatch(
            symbol=sym,
            name=name,
            exchange=item.get("exchDisp") or exchange_label(item.get("exchange")),
            type=qtype,
            logo_url=logo_url_for(sym),
        ))
    out.sort(key=lambda m: _rank_key(m, wanted))
    return out[:limit]


def matches_from_sec(cik_map: dict[str, tuple[str, str]], q: str, limit: int) -> list[SymbolMatch]:
    """Pure fallback: exact ticker, ticker prefix, then name word-prefix matches."""
    query = q.strip().upper()
    if not query:
        return []
    wanted = normalize_ticker(query) or query
    words = re.findall(r"[A-Z0-9&]+", query)
    scored: list[tuple[int, int, str]] = []
    for sym, (_cik, title) in cik_map.items():
        title_u = title.upper()
        if sym == wanted:
            rank = 0
        elif sym.startswith(query):
            rank = 1
        elif words and all(re.search(rf"\b{re.escape(w)}", title_u) for w in words):
            rank = 2 if title_u.startswith(words[0]) else 3
        else:
            continue
        scored.append((rank, len(sym), sym))
    scored.sort()
    return [
        SymbolMatch(symbol=sym, name=fix_case(cik_map[sym][1]), exchange=None, type="EQUITY",
                    logo_url=logo_url_for(sym))
        for _, _, sym in scored[:limit]
    ]


def _search_text(q: str) -> str:
    """What to send to Yahoo: a cashtag/class-share query is searched by its symbol."""
    raw = q.strip()
    if raw.startswith("$") or re.search(r"[./-]", raw):
        return normalize_ticker(raw) or raw
    return raw


@cached(ttl=3600, none_ttl=60)
async def _search_cached(q: str, limit: int) -> list[SymbolMatch]:
    wanted = normalize_ticker(q)
    results: list[SymbolMatch] = []
    try:
        quotes = await run_yahoo(_yahoo_search, _search_text(q), limit)
        results = matches_from_yahoo(quotes, q, limit)
    except Exception as exc:  # noqa: BLE001 - fall back to the SEC map
        logger.info("Yahoo search failed for %r: %s", q, exc)
    if wanted and is_crypto_symbol(wanted) and all(m.symbol != wanted for m in results):
        base = wanted.split("-")[0]
        results.insert(0, SymbolMatch(symbol=wanted, name=CRYPTO_NAMES.get(base, base), exchange="Crypto",
                                      type="CRYPTOCURRENCY", logo_url=logo_url_for(wanted)))
    have_exact = any(m.symbol == wanted for m in results)
    if len(results) < limit or (wanted and not have_exact):
        try:
            from app.intel.sec import get_cik_map

            have = {m.symbol for m in results}
            extra = [m for m in matches_from_sec(await get_cik_map(), q, limit) if m.symbol not in have]
            results.extend(m for m in extra if m.symbol == wanted)
            results.extend(m for m in extra if m.symbol != wanted)
        except Exception as exc:  # noqa: BLE001
            logger.info("SEC search fallback failed for %r: %s", q, exc)
    results.sort(key=lambda m: _rank_key(m, wanted))
    return results[:limit]


async def search_symbols(q: str, limit: int = 8) -> list[SymbolMatch]:
    """Autocomplete: Yahoo search (equities, ETFs, crypto, indices), SEC map fallback."""
    query = (q or "").strip()
    if not query:
        return []
    return await _search_cached(query.lower(), max(1, min(limit, 20)))
