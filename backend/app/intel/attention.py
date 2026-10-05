"""Public attention: daily English-Wikipedia pageviews of the company's article.

Pageviews are a clean, free proxy for retail curiosity (they spike on earnings
shocks, scandals and meme rallies before the crowd metrics catch up).

Finding the article: one MediaWiki API request looks up the likely titles
directly ("NVIDIA Corporation" redirects to "Nvidia") and returns their
descriptions and last 60 days of views (`prop=pageviews`); only when no title
is clearly the company do we fall back to CirrusSearch (restricted to pages with
a company infobox for equities). Windows beyond 60 days come from the Wikimedia
REST pageviews API.

Transport: Wikimedia refuses HTTP/1.1 clients from cloud IPs (403 "robot
policy" / 429, verified 2026-10-04) while answering HTTP/2 normally, so these
hosts are called through curl_cffi (HTTP/2; already installed as yfinance's
transport) with a contact-bearing User-Agent. A refusal pauses that host (fail
fast with a clear message) and results from the last 24 h are served meanwhile.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote

from app.config import settings
from app.core.cache import cached
from app.core.http import UpstreamError
from app.core.ratelimit import HostLimiter
from app.resolve.names import ascii_fold
from app.sources.base import CompanyRef

logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
REST_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
            "all-access/user/{title}/daily/{start}/{end}")
ACTION_API_MAX_DAYS = 60
TIMEOUT = 10.0

# en.wikipedia.org is not in the shared host table; be polite on our own.
_limiter = HostLimiter({"en.wikipedia.org": 1.0, "wikimedia.org": 1.0})
# After a refusal (403/429) a host is left alone for this long (seconds): the
# anonymous per-IP budget refills within minutes, hammering it never helps.
API_PAUSE = 120.0
REST_PAUSE = 600.0
_paused_until: dict[str, float] = {}

_COMPANY_WORDS = re.compile(
    r"\b(company|corporation|conglomerate|multinational|manufacturer|retailer|chain|bank|"
    r"airline|brand|holding|firm|maker|provider|developer|operator|producer|platform|"
    r"insurer|insurance|semiconductor|automaker|pharmaceutical|biotechnology|software|"
    r"technology|media|restaurant|cruise|railroad|utility|trust|reit|exchange|distributor|"
    r"supplier|wholesaler|publisher|carrier|lender|e-commerce|streaming|business|enterprise|contractor|"
    r"subsidiary|marketplace|app)\b",
    re.IGNORECASE,
)
_FUND_WORDS = re.compile(r"\b(index|fund|etf|exchange-traded|stock market|commodity|bond|metal|futures?|"
                         r"benchmark|precious)\b", re.IGNORECASE)
_CRYPTO_WORDS = re.compile(r"\b(cryptocurrency|blockchain|token|coin|digital currency)\b", re.IGNORECASE)
_LEGAL_TAIL = re.compile(r"\b(inc|incorporated|corp|corporation|company|co|ltd|limited|plc|group|holdings?|"
                         r"n\.?v|s\.?a|ag|se)\b\.?", re.IGNORECASE)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\(.*?\)", " ", text)
    text = _LEGAL_TAIL.sub(" ", text)
    return re.sub(r"[^a-z0-9&]+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Article choice (pure)
# --------------------------------------------------------------------------- #
def candidate_titles(company: CompanyRef) -> list[str]:
    """Titles worth a direct lookup, most specific first (redirects resolve the rest)."""
    short = company.short_name.strip()
    if company.is_crypto:
        titles = [short, f"{short} (cryptocurrency)", *company.aliases]
    elif company.quote_type == "EQUITY":
        titles = [company.name, short, f"{short} (company)", *company.aliases]
    else:
        titles = [short, *company.aliases, company.name]
    out: list[str] = []
    for title in titles:
        title = re.sub(r"\s+", " ", (title or "").replace("|", " ")).strip()
        if len(title) >= 2 and title.lower() not in {t.lower() for t in out}:
            out.append(title)
    return out[:12]


def search_query(company: CompanyRef) -> str:
    """CirrusSearch query that lands on the right article for this asset type."""
    if company.is_crypto:
        return f"{company.short_name} cryptocurrency"
    if company.quote_type == "EQUITY":
        base = " ".join(re.sub(r"[,.]", " ", company.name or company.short_name).split())
        return f'{base} hastemplate:"Infobox company"'
    return company.short_name


def _kind_pattern(company: CompanyRef) -> re.Pattern[str]:
    if company.is_crypto:
        return _CRYPTO_WORDS
    return _COMPANY_WORDS if company.quote_type == "EQUITY" else _FUND_WORDS


def pick_article(pages: list[dict[str, Any]], company: CompanyRef, *, require_kind: bool = False) -> dict[str, Any] | None:
    """Pure: choose the article that is really about this company (or None).

    `require_kind` (direct title lookups): the description must say what the
    asset is ("…technology company", "Exchange-traded fund"), so the "Apple"
    fruit article never stands in for Apple Inc.
    """
    wanted = {_norm(n) for n in [company.name, company.short_name, *company.aliases] if n}
    wanted.discard("")
    kind_re = _kind_pattern(company)
    best: tuple[float, dict[str, Any]] | None = None
    for page in pages:
        title = str(page.get("title") or "")
        desc = str(page.get("description") or "")
        overview = re.match(r"(list|history|timeline) of ", title, re.IGNORECASE)
        disambiguation = "disambiguation" in (page.get("pageprops") or {})
        if not title or page.get("missing") or overview or disambiguation or "disambiguation" in (title + desc).lower():
            continue
        norm_title = _norm(title)
        score = 0.0
        if norm_title in wanted:
            score += 3.0
        elif any(norm_title.startswith(w) or w.startswith(norm_title) for w in wanted if w):
            score += 1.0
        kind = bool(kind_re.search(desc))
        if require_kind and not kind:
            continue
        score += 1.0 if kind else 0.0
        index = int(page.get("index") or 10)
        score += max(0.0, (5 - index) * 0.2)
        if score >= 2.0 and (best is None or score > best[0]):
            best = (score, page)
    return best[1] if best else None


def views_from_page(page: dict[str, Any]) -> list[tuple[date, float]]:
    """Pure: MediaWiki `pageviews` map -> [(day, views)] oldest first, gaps dropped."""
    out: list[tuple[date, float]] = []
    for day, views in sorted((page.get("pageviews") or {}).items()):
        if views is None:
            continue  # not yet computed (today / yesterday)
        try:
            out.append((date.fromisoformat(day), float(views)))
        except (TypeError, ValueError):
            continue
    return out


def views_from_rest(payload: Any) -> list[tuple[date, float]]:
    """Pure: Wikimedia REST per-article payload -> [(day, views)]."""
    out: list[tuple[date, float]] = []
    for item in (payload or {}).get("items") or []:
        ts = str(item.get("timestamp") or "")
        try:
            out.append((date(int(ts[:4]), int(ts[4:6]), int(ts[6:8])), float(item["views"])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(out)


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
def _headers() -> dict[str, str]:
    # Wikimedia requires a contact-bearing UA (no URL needed; SEC's WAF even rejects one).
    return {"User-Agent": f"SentiNET/2.0 ({settings.contact_email})", "Accept": "application/json"}


async def _http_get(url: str, params: dict[str, Any] | None) -> tuple[int, str]:
    """HTTP/2 GET via curl_cffi -> (status, body). Raises UpstreamError on transport failure."""
    from curl_cffi.requests import AsyncSession

    try:
        async with AsyncSession() as session:
            resp = await session.get(url, params=params, headers=_headers(), timeout=TIMEOUT)
            return resp.status_code, resp.text
    except Exception as exc:  # noqa: BLE001 - curl errors carry no stable type
        raise UpstreamError(f"{url.split('/')[2]} unreachable ({type(exc).__name__})") from exc


def _pause_left(host: str) -> float:
    return max(0.0, _paused_until.get(host, 0.0) - time.monotonic())


async def _wiki_json(url: str, params: dict[str, Any] | None = None, *, pause: float = API_PAUSE) -> Any:
    """GET JSON from a Wikimedia host; a refusal pauses the host for `pause` seconds."""
    host = url.split("/")[2]
    if (left := _pause_left(host)) > 0:
        raise UpstreamError(f"{host} is rate-limiting this server's IP; next try in {left:.0f} s")
    await _limiter(host)
    status, body = await _http_get(url, params)
    if status in (403, 429):
        _paused_until[host] = time.monotonic() + pause
        raise UpstreamError(f"{host} refused the request (HTTP {status}, rate limit); pausing {pause / 60:.0f} min")
    if status == 404:
        return None
    if status >= 400:
        raise UpstreamError(f"{host}: HTTP {status}")
    try:
        return json.loads(body)
    except ValueError as exc:
        raise UpstreamError(f"{host} returned non-JSON: {body[:80]!r}") from exc


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _query_params(days: int) -> dict[str, str]:
    return {"action": "query", "format": "json", "formatversion": "2", "redirects": "1",
            "prop": "pageviews|description|pageprops", "ppprop": "disambiguation",
            "pvipdays": str(min(days, ACTION_API_MAX_DAYS))}


async def find_article(company: CompanyRef, days: int = ACTION_API_MAX_DAYS) -> dict[str, Any] | None:
    """The company's article (with up to 60 days of views), by title lookup, else search."""
    data = await _wiki_json(WIKI_API, {**_query_params(days), "titles": "|".join(candidate_titles(company))})
    pages = ((data or {}).get("query") or {}).get("pages") or []
    page = pick_article(pages, company, require_kind=True)
    if page is not None:
        return page
    data = await _wiki_json(WIKI_API, {**_query_params(days), "generator": "search",
                                       "gsrsearch": search_query(company), "gsrlimit": "5", "gsrnamespace": "0"})
    pages = ((data or {}).get("query") or {}).get("pages") or []
    return pick_article(pages, company)


@cached(ttl=settings.history_cache_ttl, none_ttl=900)
async def _pageviews(
    ticker: str, name: str, short_name: str, aliases: tuple[str, ...], quote_type: str, days: int
) -> list[tuple[date, float]] | None:
    company = CompanyRef(ticker=ticker, name=name, short_name=short_name, aliases=list(aliases), quote_type=quote_type)
    page = await find_article(company, days)
    if page is None:
        return None
    views = views_from_page(page)
    if days > ACTION_API_MAX_DAYS:
        end = datetime.now(UTC).date() - timedelta(days=1)
        start = end - timedelta(days=days - 1)
        title = quote(str(page["title"]).replace(" ", "_"), safe="")
        try:
            url = REST_URL.format(title=title, start=f"{start:%Y%m%d}00", end=f"{end:%Y%m%d}00")
            rest = views_from_rest(await _wiki_json(url, pause=REST_PAUSE))
            if len(rest) > len(views):
                views = rest
        except UpstreamError as exc:
            logger.info("Wikimedia REST pageviews unavailable (%s); using %d days", exc, len(views))
    return views or None


_LAST_GOOD: dict[str, tuple[float, list[tuple[date, float]]]] = {}
STALE_MAX_SECONDS = 24 * 3600


async def get_wiki_pageviews(company: CompanyRef, days: int = 90) -> list[tuple[date, float]] | None:
    """Daily Wikipedia pageviews (oldest first) for the company's article, or None.

    Raises `UpstreamError` when Wikipedia refuses unless a result from the last
    24 h can be served instead.
    """
    aliases = tuple(dict.fromkeys(a for a in (*company.aliases, ascii_fold(company.short_name)) if a))
    span = max(7, min(int(days), 365))
    key = f"{company.ticker}:{span}"
    try:
        views = await _pageviews(company.ticker, company.name, company.short_name, aliases, company.quote_type, span)
    except UpstreamError:
        stale = _LAST_GOOD.get(key)
        if stale and time.monotonic() - stale[0] < STALE_MAX_SECONDS:
            return stale[1]
        raise
    if views:
        if len(_LAST_GOOD) > 512:
            _LAST_GOOD.pop(next(iter(_LAST_GOOD)))
        _LAST_GOOD[key] = (time.monotonic(), views)
    return views


def reset_state() -> None:
    """Test helper: forget stale results and host pauses."""
    _LAST_GOOD.clear()
    _paused_until.clear()
