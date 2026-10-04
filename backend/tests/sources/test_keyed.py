"""Free-key sources (Finnhub, Marketaux, Alpha Vantage, Reddit): skipped without keys, parse correctly
with them, and never leak the key into error messages."""
from __future__ import annotations

import httpx
import pytest
import respx

from app.config import settings
from app.core.http import UpstreamError
from app.sources import alphavantage, finnhub, marketaux, reddit
from app.sources.util import DailyBudget, is_listing_page
from tests.conftest import load_json_fixture
from tests.sources.conftest import company

SECRET = "sk-test-SECRET-123"


def fx(name: str):
    return load_json_fixture(f"sources/{name}")


@pytest.mark.parametrize(
    "source",
    [finnhub.FinnhubSource(), marketaux.MarketauxSource(), alphavantage.AlphaVantageSource(), reddit.RedditSource()],
)
async def test_unconfigured_without_keys(no_keys, source):
    assert source.requires_key and not source.configured()
    assert not await source.fetch(company("NVDA"))  # defensive: no network, empty batch


# --------------------------------------------------------------------------- Finnhub
@pytest.fixture(autouse=True)
def _fresh_budgets(monkeypatch):
    monkeypatch.setattr(alphavantage, "BUDGET", DailyBudget(25))
    monkeypatch.setattr(marketaux, "BUDGET", DailyBudget(100))


@respx.mock
async def test_finnhub_parses_and_marks_only_named_items_ticker_specific(monkeypatch, frozen_now):
    monkeypatch.setattr(settings, "finnhub_api_key", SECRET)
    route = respx.get(finnhub.URL).mock(return_value=httpx.Response(200, json=fx("finnhub_company_news_handmade.json")))
    batch = await finnhub.FinnhubSource().fetch(company("AAPL"))
    request = route.calls[0].request
    params = request.url.params
    assert params["symbol"] == "AAPL" and params["from"] == "2026-09-27" and params["to"] == "2026-10-04"
    # The key travels in a header, never in the URL (URLs end up in access/proxy logs).
    assert request.headers["X-Finnhub-Token"] == SECRET and SECRET not in str(request.url)
    specific = {s.title: s.ticker_specific for s in batch.signals}
    assert specific == {  # option-chain page + empty headline dropped
        "Example: Apple faces example lawsuit & probe": True,
        "Example: Stocks settle higher as example rate worries ease": False,  # market wrap: relevance decides
        "Example: Apple shares rise after example upgrade": True,
    }
    assert batch.signals[0].body == "Synthetic HTML summary."
    assert all(s.timestamp.tzinfo for s in batch.signals)


@respx.mock
async def test_finnhub_error_does_not_leak_key(monkeypatch):
    monkeypatch.setattr(settings, "finnhub_api_key", SECRET)
    respx.get(finnhub.URL).mock(return_value=httpx.Response(401, json={"error": "Invalid API key."}))
    with pytest.raises(UpstreamError) as info:
        await finnhub.FinnhubSource().fetch(company("AAPL"))
    assert SECRET not in str(info.value) and "401" in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_finnhub_supports_us_equities_only():
    src = finnhub.FinnhubSource()
    assert src.supports(company("AAPL")) and not src.supports(company("SPY")) and not src.supports(company("BTC-USD"))


# --------------------------------------------------------------------------- Marketaux
def test_marketaux_title_highlight_means_ticker_specific():
    signals = marketaux.parse_items(fx("marketaux_news_handmade.json"), "TSLA")
    assert [s.ticker_specific for s in signals] == [True, False]
    assert signals[0].extra["provider_score"] == pytest.approx(0.6124)
    assert signals[0].publisher == "example.com"


@respx.mock
async def test_marketaux_reads_pages_tolerates_one_failure_and_caches(monkeypatch):
    monkeypatch.setattr(settings, "marketaux_api_key", SECRET)
    route = respx.get(marketaux.URL).mock(
        side_effect=[httpx.Response(200, json=fx("marketaux_news_handmade.json")), httpx.Response(402)]
    )
    src = marketaux.MarketauxSource()
    batch = await src.fetch(company("TSLA"))
    assert len(batch.signals) == 2
    batch.signals.clear()  # callers get a private copy of the cached batch
    assert len((await src.fetch(company("TSLA"))).signals) == 2 and route.call_count == 2  # served from cache
    assert marketaux.BUDGET.remaining == 98


@respx.mock
async def test_marketaux_spent_budget_is_reported_plainly(monkeypatch):
    monkeypatch.setattr(settings, "marketaux_api_key", SECRET)
    monkeypatch.setattr(marketaux, "BUDGET", DailyBudget(0))
    route = respx.get(marketaux.URL).mock(return_value=httpx.Response(200, json=fx("marketaux_news_handmade.json")))
    with pytest.raises(UpstreamError, match="daily free quota"):
        await marketaux.MarketauxSource().fetch(company("TSLA"))
    assert route.call_count == 0


@respx.mock
async def test_marketaux_all_pages_failing_raises_sanitized(monkeypatch):
    monkeypatch.setattr(settings, "marketaux_api_key", SECRET)
    respx.get(marketaux.URL).mock(return_value=httpx.Response(401))
    with pytest.raises(UpstreamError) as info:
        await marketaux.MarketauxSource().fetch(company("TSLA"))
    assert SECRET not in str(info.value)


# --------------------------------------------------------------------------- Alpha Vantage
def test_alphavantage_real_demo_payload():
    payload = fx("alphavantage_news_aapl_demo.json")
    batch = alphavantage.parse_payload(payload, "AAPL")
    articles = [item for item in payload["feed"] if not is_listing_page(item["title"])]
    assert len(articles) == 10 and len(batch.signals) == 10  # 2 quote/ratings pages dropped
    # Provider view: relevance-weighted mean of AAPL scores over relevant (>= 0.5) articles.
    pairs = [
        (float(t["relevance_score"]), float(t["ticker_sentiment_score"]))
        for item in articles
        for t in item["ticker_sentiment"]
        if t["ticker"] == "AAPL" and float(t["relevance_score"]) >= 0.5
    ]
    expected = sum(r * s for r, s in pairs) / sum(r for r, _ in pairs)
    assert batch.metrics["av_sentiment"] == pytest.approx(expected, abs=1e-4)
    assert batch.metrics["av_articles"] == len(pairs)
    specific = [s for s in batch.signals if s.ticker_specific]
    assert specific and all(s.extra["provider_relevance"] >= 0.9 for s in specific)
    assert all(s.timestamp and s.timestamp.tzinfo for s in batch.signals)


def test_alphavantage_throttle_or_demo_message_is_an_error():
    with pytest.raises(UpstreamError, match="demo"):
        alphavantage.parse_payload(fx("alphavantage_demo_key_refusal.json"), "NVDA")


def test_alphavantage_invalid_symbol_is_empty():
    assert not alphavantage.parse_payload({"Information": "Invalid inputs. Please refer to the API documentation"}, "X")


def test_alphavantage_time_and_symbols():
    assert alphavantage.parse_time("20261004T151215").isoformat() == "2026-10-04T15:12:15+00:00"
    assert alphavantage.av_symbol(company("BTC-USD")) == "CRYPTO:BTC"
    assert alphavantage.av_symbol(company("BRK-B")) == "BRK.B"


@respx.mock
async def test_alphavantage_fetch_sends_window_caches_and_sanitizes(monkeypatch, frozen_now):
    monkeypatch.setattr(settings, "alphavantage_api_key", SECRET)
    route = respx.get(alphavantage.URL).mock(return_value=httpx.Response(200, json=fx("alphavantage_news_aapl_demo.json")))
    src = alphavantage.AlphaVantageSource()
    first = await src.fetch(company("AAPL"))
    params = route.calls[0].request.url.params
    assert params["tickers"] == "AAPL" and params["time_from"] == "20260927T2230" and params["sort"] == "LATEST"
    assert (await src.fetch(company("AAPL"))).metrics == first.metrics and route.call_count == 1  # 3 h cache
    assert alphavantage.BUDGET.remaining == 24
    route.mock(return_value=httpx.Response(500))
    monkeypatch.setattr("app.core.http.asyncio.sleep", _no_sleep)
    with pytest.raises(UpstreamError) as info:
        await src.fetch(company("NVDA"))  # cache miss -> real (failing) call
    assert SECRET not in str(info.value)


@respx.mock
async def test_alphavantage_spent_budget_is_reported_plainly(monkeypatch):
    monkeypatch.setattr(settings, "alphavantage_api_key", SECRET)
    monkeypatch.setattr(alphavantage, "BUDGET", DailyBudget(0))
    route = respx.get(alphavantage.URL).mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(UpstreamError, match="daily free quota"):
        await alphavantage.AlphaVantageSource().fetch(company("AAPL"))
    assert route.call_count == 0


async def _no_sleep(*_args, **_kwargs):
    return None


# --------------------------------------------------------------------------- Reddit
def test_reddit_listing_skips_stickied_nsfw_and_posts_not_naming_the_asset():
    from app.sources.query import Mentions, search_terms

    assert len(reddit.parse_listing(fx("reddit_search_handmade.json"))) == 2  # without a matcher
    signals = reddit.parse_listing(fx("reddit_search_handmade.json"), Mentions(search_terms(company("NVDA"))))
    assert [s.title for s in signals] == ["Example: NVDA earnings play discussion"]
    s = signals[0]
    assert s.engagement == 165 and s.publisher == "r/wallstreetbets" and s.body == "Synthetic & body text"
    assert s.url == "https://www.reddit.com/r/wallstreetbets/comments/abc123/example/"


def test_reddit_query_never_searches_word_tickers_bare():
    from app.sources.query import search_terms

    assert reddit.build_query(search_terms(company("NVDA"))) == '"$NVDA" OR NVDA OR "Nvidia"'
    assert reddit.build_query(search_terms(company("SPY"))) == '"$SPY" OR "S&P 500"'
    assert reddit.build_query(search_terms(company("BRK-B"))) == '"$BRK.B" OR "Berkshire Hathaway"'


@respx.mock
async def test_reddit_oauth_flow(monkeypatch):
    monkeypatch.setattr(settings, "reddit_client_id", "cid")
    monkeypatch.setattr(settings, "reddit_client_secret", SECRET)
    token = respx.post(reddit.TOKEN_URL).mock(return_value=httpx.Response(200, json={"access_token": "tok"}))
    search = respx.get(reddit.SEARCH_URL.format(subs=reddit.STOCK_SUBS)).mock(
        return_value=httpx.Response(200, json=fx("reddit_search_handmade.json"))
    )
    batch = await reddit.RedditSource().fetch(company("NVDA"))
    assert token.calls[0].request.headers["Authorization"].startswith("Basic ")
    assert search.calls[0].request.headers["Authorization"] == "Bearer tok"
    assert search.calls[0].request.url.params["restrict_sr"] == "1"
    assert len(batch.signals) == 1


@respx.mock
async def test_reddit_bad_credentials_are_sanitized(monkeypatch):
    monkeypatch.setattr(settings, "reddit_client_id", "cid")
    monkeypatch.setattr(settings, "reddit_client_secret", SECRET)
    respx.post(reddit.TOKEN_URL).mock(return_value=httpx.Response(401))
    with pytest.raises(UpstreamError) as info:
        await reddit.RedditSource().fetch(company("NVDA"))
    assert SECRET not in str(info.value) and "401" in str(info.value)
