"""Wikipedia pageviews: article choice, parsing, REST fallback and failure modes."""
from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from app.core.http import UpstreamError
from app.core.ratelimit import HostLimiter
from app.intel import attention
from app.intel.attention import (
    pick_article,
    search_query,
    views_from_page,
    views_from_rest,
)
from app.sources.base import CompanyRef
from tests.intel.helpers import load_json

TARGET = CompanyRef(ticker="TGT", name="Target Corporation", short_name="Target",
                    aliases=["Target Corp", "Target Corporation"])


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core import http

    monkeypatch.setattr(attention, "_limiter", HostLimiter({}))
    attention._LAST_GOOD.clear()
    monkeypatch.setattr(http, "_retry_delay", lambda resp, attempt: 0.0)


def test_search_query_by_asset_type() -> None:
    assert search_query(TARGET) == 'Target Corporation hastemplate:"Infobox company"'
    assert search_query(CompanyRef(ticker="BTC-USD", name="Bitcoin", short_name="Bitcoin",
                                   quote_type="CRYPTOCURRENCY")) == "Bitcoin cryptocurrency"
    assert search_query(CompanyRef(ticker="SPY", name="SPDR", short_name="S&P 500", quote_type="ETF")) == "S&P 500"


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
    company = CompanyRef(ticker="AAPL", name="Apple Inc.", short_name="Apple")
    assert pick_article(pages, company) is None


def test_views_parsing() -> None:
    page = load_json("wiki/target_search.json")["query"]["pages"][1]
    views = views_from_page(page)
    assert len(views) >= 55 and all(v > 0 for _, v in views)
    assert [d for d, _ in views] == sorted(d for d, _ in views)  # nulls (not yet computed) dropped
    rest = views_from_rest({"items": [{"timestamp": "2026090100", "views": 10}, {"timestamp": "bad"},
                                      {"timestamp": "2026083100", "views": 7}]})
    assert rest == [(date(2026, 8, 31), 7.0), (date(2026, 9, 1), 10.0)]


async def test_get_wiki_pageviews_60_days_without_rest() -> None:
    payload = load_json("wiki/target_search.json")
    with respx.mock as mock:
        mock.get(attention.WIKI_API).mock(return_value=httpx.Response(200, json=payload))
        views = await attention.get_wiki_pageviews(TARGET, days=60)
    assert views and len(views) >= 55


async def test_get_wiki_pageviews_rest_refused_keeps_60_days() -> None:
    payload = load_json("wiki/target_search.json")
    with respx.mock as mock:
        mock.get(attention.WIKI_API).mock(return_value=httpx.Response(200, json=payload))
        rest = mock.get(url__startswith="https://wikimedia.org/api/rest_v1/").mock(
            return_value=httpx.Response(429, text="You are making too many requests"))
        views = await attention.get_wiki_pageviews(TARGET, days=90)
        assert rest.called
    assert views and 55 <= len(views) <= 60


async def test_get_wiki_pageviews_errors_and_no_article() -> None:
    with respx.mock as mock:
        mock.get(attention.WIKI_API).mock(return_value=httpx.Response(403, text="Please respect our robot policy"))
        with pytest.raises(UpstreamError):
            await attention.get_wiki_pageviews(TARGET, days=60)
    other = CompanyRef(ticker="QQQQ", name="Nothing Corp", short_name="Nothing")
    with respx.mock as mock:
        mock.get(attention.WIKI_API).mock(return_value=httpx.Response(200, json={"batchcomplete": True}))
        assert await attention.get_wiki_pageviews(other, days=60) is None


async def test_serves_recent_views_when_wikipedia_refuses() -> None:
    from app.core import cache

    payload = load_json("wiki/target_search.json")
    with respx.mock as mock:
        mock.get(attention.WIKI_API).mock(return_value=httpx.Response(200, json=payload))
        fresh = await attention.get_wiki_pageviews(TARGET, days=60)
    cache.clear_all()
    with respx.mock as mock:
        mock.get(attention.WIKI_API).mock(return_value=httpx.Response(429, text="Too many requests"))
        assert await attention.get_wiki_pageviews(TARGET, days=60) == fresh
