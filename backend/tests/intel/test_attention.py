"""Wikipedia pageviews: article choice, parsing, REST extension and failure modes."""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

from app.core.http import UpstreamError
from app.core.ratelimit import HostLimiter
from app.intel import attention
from app.intel.attention import (
    candidate_titles,
    lookup_order,
    pick_article,
    search_query,
    views_from_page,
    views_from_rest,
)
from app.sources.base import CompanyRef
from tests.intel.helpers import load_json

TARGET = CompanyRef(ticker="TGT", name="Target Corporation", short_name="Target",
                    aliases=["Target Corp", "Target Corporation"])
APPLE = CompanyRef(ticker="AAPL", name="Apple Inc.", short_name="Apple", aliases=["Apple Inc"])

Handler = Callable[[str, dict[str, Any] | None], tuple[Any, ...]]  # (status, body[, retry_after])


@pytest.fixture
def wiki(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], list[tuple[str, dict[str, Any]]]]:
    """Route Wikimedia requests to a handler: (url, params) -> (status, json-or-text)."""
    import json

    monkeypatch.setattr(attention, "_limiter", HostLimiter({}))
    calls: list[tuple[str, dict[str, Any]]] = []

    def install(handler: Handler) -> list[tuple[str, dict[str, Any]]]:
        async def fake(url: str, params: dict[str, Any] | None) -> tuple[int, str, float | None]:
            calls.append((url, dict(params or {})))
            status, body, *hint = handler(url, params)
            return status, body if isinstance(body, str) else json.dumps(body), (hint[0] if hint else None)

        monkeypatch.setattr(attention, "_http_get", fake)
        return calls

    return install


def _titles_payload(*pages: dict[str, Any]) -> dict[str, Any]:
    return {"batchcomplete": True, "query": {"pages": list(pages)}}


def _is_search(params: dict[str, Any] | None) -> bool:
    return bool(params and params.get("generator") == "search")


# --------------------------------------------------------------------------- #
# Article choice
# --------------------------------------------------------------------------- #
def test_candidate_titles_and_search_query_by_asset_type() -> None:
    assert candidate_titles(APPLE)[:3] == ["Apple Inc.", "Apple", "Apple (company)"]
    btc = CompanyRef(ticker="BTC-USD", name="Bitcoin", short_name="Bitcoin", quote_type="CRYPTOCURRENCY")
    assert candidate_titles(btc) == ["Bitcoin", "Bitcoin (cryptocurrency)"]
    assert search_query(TARGET) == 'Target Corporation hastemplate:"Infobox company"'
    assert search_query(btc) == "Bitcoin cryptocurrency"
    spy = CompanyRef(ticker="SPY", name="SPDR S&P 500 ETF Trust", short_name="S&P 500", quote_type="ETF")
    assert search_query(spy) == "S&P 500" and candidate_titles(spy)[0] == "S&P 500"


def test_title_lookup_never_takes_the_namesake() -> None:
    """Real answer for titles=Apple Inc.|Apple|Apple (company)|Apple Inc: the fruit must lose."""
    pages = load_json("wiki/apple_titles.json")["query"]["pages"]
    assert {p["title"] for p in pages} == {"Apple Inc.", "Apple"}
    best = pick_article(pages, APPLE, require_kind=True)
    assert best is not None and best["title"] == "Apple Inc." and len(views_from_page(best)) >= 55
    fruit = [p for p in pages if p["title"] == "Apple"]
    assert pick_article(fruit, APPLE, require_kind=True) is None
    disamb = [{"title": "Target", "description": "Topics referred to by the same term",
               "pageprops": {"disambiguation": ""}}]
    assert pick_article(disamb, TARGET) is None


def test_official_name_beats_a_product_alias() -> None:
    """Real answer for SNAP: "Snapchat" (an app, also matches) must lose to "Snap Inc."."""
    data = load_json("wiki/snap_titles.json")
    query = data["query"]
    snap = CompanyRef(ticker="SNAP", name="Snap Inc.", short_name="Snap", aliases=["Snap Inc", "Snapchat"])
    titles = candidate_titles(snap)
    order = lookup_order(query, titles)
    assert order["Snap Inc."] == 0 and order["Snapchat"] == titles.index("Snapchat")
    for pages in (query["pages"], list(reversed(query["pages"]))):  # API order must not matter
        best = pick_article(pages, snap, require_kind=True, order=order)
        assert best is not None and best["title"] == "Snap Inc."


def test_lookup_order_follows_normalization_and_redirects() -> None:
    query = {"normalized": [{"from": "nvidia corporation", "to": "Nvidia corporation"}],
             "redirects": [{"from": "Nvidia corporation", "to": "Nvidia"}, {"from": "NVDA", "to": "Nvidia"}]}
    assert lookup_order(query, ["nvidia corporation", "NVDA", "Other"]) == {"Nvidia": 0, "Other": 2}


def test_pick_article_prefers_the_company_not_its_namesakes() -> None:
    pages = load_json("wiki/target_search.json")["query"]["pages"]
    assert {p["title"] for p in pages} >= {"Target Australia", "Target Corporation"}
    best = pick_article(pages, TARGET)
    assert best is not None and best["title"] == "Target Corporation"


def test_pick_article_rejects_unrelated_hits() -> None:
    pages = [
        {"title": "Apple (disambiguation)", "index": 1, "description": "Topics referred to by the same term"},
        {"title": "List of apple cultivars", "index": 2, "description": ""},
        {"title": "Malus", "index": 3, "description": "Genus of plants"},
    ]
    assert pick_article(pages, APPLE) is None


def test_views_parsing() -> None:
    page = load_json("wiki/target_search.json")["query"]["pages"][1]
    views = views_from_page(page)
    assert len(views) >= 55 and all(v > 0 for _, v in views)
    assert [d for d, _ in views] == sorted(d for d, _ in views)  # nulls (not yet computed) dropped
    rest = views_from_rest({"items": [{"timestamp": "2026090100", "views": 10}, {"timestamp": "bad"},
                                      {"timestamp": "2026083100", "views": 7}]})
    assert rest == [(date(2026, 8, 31), 7.0), (date(2026, 9, 1), 10.0)]


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def _target_page() -> dict[str, Any]:
    return next(p for p in load_json("wiki/target_search.json")["query"]["pages"] if p["title"] == "Target Corporation")


async def test_title_lookup_needs_one_request(wiki) -> None:
    calls = wiki(lambda url, params: (200, _titles_payload(_target_page())))
    views = await attention.get_wiki_pageviews(TARGET, days=60)
    assert views and len(views) >= 55
    assert len(calls) == 1 and not _is_search(calls[0][1]) and "Target Corporation" in calls[0][1]["titles"]


async def test_search_is_the_fallback(wiki) -> None:
    search = load_json("wiki/target_search.json")

    def handler(url: str, params: dict[str, Any] | None) -> tuple[int, Any]:
        if _is_search(params):
            return 200, search
        return 200, _titles_payload({"title": "Target", "description": "Topics referred to by the same term",
                                     "pageprops": {"disambiguation": ""}})

    calls = wiki(handler)
    views = await attention.get_wiki_pageviews(TARGET, days=60)
    assert views and len(views) >= 55 and len(calls) == 2 and _is_search(calls[1][1])


async def test_long_windows_extend_with_rest(wiki) -> None:
    rest_items = {"items": [{"timestamp": f"2026{m:02d}{d:02d}00", "views": 100 + d}
                            for m in (7, 8, 9) for d in range(1, 29)]}

    def handler(url: str, params: dict[str, Any] | None) -> tuple[int, Any]:
        if url.startswith("https://wikimedia.org/"):
            assert "/Target_Corporation/daily/" in url
            return 200, rest_items
        return 200, _titles_payload(_target_page())

    wiki(handler)
    views = await attention.get_wiki_pageviews(TARGET, days=90)
    assert views and len(views) == 84


async def test_rest_refusal_keeps_60_days_and_pauses_rest(wiki) -> None:
    def handler(url: str, params: dict[str, Any] | None) -> tuple[int, Any]:
        if url.startswith("https://wikimedia.org/"):
            return 429, "You are making too many requests to the API."
        return 200, _titles_payload(_target_page())

    calls = wiki(handler)
    views = await attention.get_wiki_pageviews(TARGET, days=90)
    assert views and 55 <= len(views) <= 60
    other = CompanyRef(ticker="TGT2", name="Target Corporation", short_name="Target", aliases=["Target Corp"])
    assert await attention.get_wiki_pageviews(other, days=120)
    assert sum(url.startswith("https://wikimedia.org/") for url, _ in calls) == 1  # REST paused after refusal


async def test_refusal_pauses_the_host_and_fails_fast(wiki) -> None:
    calls = wiki(lambda url, params: (403, "Please respect our robot policy"))
    with pytest.raises(UpstreamError, match=r"HTTP 403"):
        await attention.get_wiki_pageviews(TARGET, days=60)
    with pytest.raises(UpstreamError, match=r"next try in \d+ s"):
        await attention.get_wiki_pageviews(APPLE, days=60)
    assert len(calls) == 1  # no retry, nothing sent while paused


async def test_short_retry_after_is_honored_once(wiki, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention, "_sleep", _no_sleep)
    answers = iter([(429, "Too many requests", 4.0), (200, _titles_payload(_target_page()))])
    calls = wiki(lambda url, params: next(answers))
    assert await attention.get_wiki_pageviews(TARGET, days=60)
    assert len(calls) == 2


async def test_long_retry_after_sets_the_pause(wiki, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention, "_sleep", _no_sleep)
    calls = wiki(lambda url, params: (429, "Too many requests", 46.0))
    with pytest.raises(UpstreamError, match=r"pausing 46 s"):
        await attention.get_wiki_pageviews(TARGET, days=60)
    assert len(calls) == 1 and 40 < attention._pause_left("en.wikipedia.org") <= 46


async def _no_sleep(_seconds: float) -> None:
    """Skip Retry-After waits in tests."""


async def test_recent_series_is_reused_across_restarts(wiki) -> None:
    from app.core import cache

    calls = wiki(lambda url, params: (200, _titles_payload(_target_page())))
    fresh = await attention.get_wiki_pageviews(TARGET, days=60)
    cache.clear_all()
    attention.reset_state()  # memory gone, disk copy stays
    assert await attention.get_wiki_pageviews(TARGET, days=60) == fresh
    assert len(calls) == 1  # no second request


async def test_no_article_is_none(wiki) -> None:
    wiki(lambda url, params: (200, {"batchcomplete": True}))
    other = CompanyRef(ticker="QQQQ", name="Nothing Corp", short_name="Nothing")
    assert await attention.get_wiki_pageviews(other, days=60) is None


async def test_serves_recent_views_when_wikipedia_refuses(wiki) -> None:
    from app.core import cache

    wiki(lambda url, params: (200, _titles_payload(_target_page())))
    fresh = await attention.get_wiki_pageviews(TARGET, days=60)
    cache.clear_all()
    wiki(lambda url, params: (429, "Too many requests"))
    assert await attention.get_wiki_pageviews(TARGET, days=60) == fresh


async def test_bare_ref_is_not_looked_up_by_its_ticker(wiki) -> None:
    """Delisted X (U.S. Steel): the lookup "X (company)" picked Musk's "X Corp." (review, 2026-10-05)."""
    calls = wiki(lambda url, params: (200, _titles_payload({"title": "X Corp.", "description": "American technology company",
                                                            "pageviews": {"2026-10-01": 771}})))
    bare = CompanyRef(ticker="X", name="X", short_name="X")
    assert await attention.get_wiki_pageviews(bare, days=60) is None
    assert calls == []
    # A bare ref of a curated brand still knows its names offline.
    nvda = CompanyRef(ticker="NVDA", name="NVDA", short_name="NVDA")
    wiki(lambda url, params: (200, _titles_payload({"title": "Nvidia", "description": "American technology company",
                                                    "pageviews": {"2026-10-01": 30000}})))
    assert await attention.get_wiki_pageviews(nvda, days=60) == [(date(2026, 10, 1), 30000.0)]
