"""Keyless social/crowd sources: StockTwits, Hacker News, Bluesky, ApeWisdom, Tradestie (offline)."""
from __future__ import annotations

import json

import httpx
import respx

from app.config import settings
from app.sources import apewisdom, bluesky, hackernews, stocktwits, tradestie
from app.sources.base import CompanyRef
from app.sources.query import search_terms
from tests.conftest import load_fixture, load_json_fixture
from tests.sources.conftest import company


def fx(name: str):
    return load_json_fixture(f"sources/{name}")


# --------------------------------------------------------------------------- StockTwits
def test_stocktwits_attribution_rule():
    sym = "NVDA"
    assert stocktwits.attributable({"symbols": [{"symbol": "NVDA"}], "body": "anything"}, sym)
    assert stocktwits.attributable({"symbols": [{"symbol": "NVDA"}, {"symbol": "AMD"}], "body": "$NVDA > $AMD"}, sym)
    # Tag belongs to the first cashtag ($MU), not to every symbol mentioned.
    assert not stocktwits.attributable({"symbols": [{"symbol": "MU"}, {"symbol": "NVDA"}], "body": "$MU and $NVDA"}, sym)
    many = [{"symbol": s} for s in ("NVDA", "AMD", "MU", "INTC")]
    assert not stocktwits.attributable({"symbols": many, "body": "$NVDA $AMD $MU $INTC"}, sym)


def test_stocktwits_batch_metrics_from_real_pages():
    p1, p2 = fx("stocktwits_nvda_p1.json"), fx("stocktwits_nvda_p2.json")
    batch = stocktwits.build_batch([p1, p2], "NVDA")
    m = batch.metrics
    assert m["stocktwits_messages"] == len(p1["messages"]) + len(p2["messages"]) == len(batch.signals)
    assert m["stocktwits_watchers"] == p1["symbol"]["watchlist_count"]
    labeled = [s for s in batch.signals if s.user_label]
    assert m["stocktwits_bullish"] + m["stocktwits_bearish"] == len(labeled)
    assert all(s.ticker_specific for s in labeled)  # stance only counted when attributable
    raw_tags = sum(1 for p in (p1, p2) for msg in p["messages"] if (msg["entities"].get("sentiment") or {}))
    assert len(labeled) < raw_tags  # multi-symbol spam tags were not attributed to NVDA
    assert m["stocktwits_span_hours"] > 0
    assert all(s.url and s.url.startswith("https://stocktwits.com/") for s in batch.signals)
    assert all("&amp;" not in s.title for s in batch.signals)


def test_stocktwits_crypto_stream():
    assert stocktwits.stream_symbol(company("BTC-USD")) == "BTC.X"
    batch = stocktwits.build_batch([fx("stocktwits_btc.json")], "BTC.X")
    assert batch.signals and any(s.user_label for s in batch.signals)
    assert sum(1 for s in batch.signals if s.ticker_specific) >= len(batch.signals) // 2


@respx.mock
async def test_stocktwits_fetch_paginates_with_max_cursor():
    p1, p2 = fx("stocktwits_nvda_p1.json"), fx("stocktwits_nvda_p2.json")
    url = stocktwits.URL.format(symbol="NVDA")
    route = respx.get(url).mock(side_effect=[httpx.Response(200, json=p1), httpx.Response(200, json=p2)])
    batch = await stocktwits.StockTwitsSource().fetch(company("NVDA"))
    assert route.call_count == 2
    assert route.calls[1].request.url.params["max"] == str(p1["cursor"]["max"])
    assert batch.metrics["stocktwits_messages"] == 20


@respx.mock
async def test_stocktwits_keeps_page_one_if_page_two_fails():
    url = stocktwits.URL.format(symbol="NVDA")
    respx.get(url).mock(side_effect=[httpx.Response(200, json=fx("stocktwits_nvda_p1.json")), httpx.Response(403)])
    batch = await stocktwits.StockTwitsSource().fetch(company("NVDA"))
    assert batch.metrics["stocktwits_messages"] == 12


@respx.mock
async def test_stocktwits_unknown_symbol_is_empty():
    respx.get(stocktwits.URL.format(symbol="NVDA")).mock(
        return_value=httpx.Response(404, text=load_fixture("sources/stocktwits_not_found.json"))
    )
    assert not await stocktwits.StockTwitsSource().fetch(company("NVDA"))


# --------------------------------------------------------------------------- Hacker News
def test_hn_requests_disable_typos_and_skip_ambiguous_story_search():
    nvda = hackernews.build_requests(search_terms(company("NVDA")))
    assert [r["tags"] for r in nvda] == ["story", "comment", "(story,comment)"]
    assert nvda[0]["restrictSearchableAttributes"] == "title"
    tgt = hackernews.build_requests(search_terms(company("TGT")))
    assert all(r["tags"] != "story" for r in tgt)  # "Target" titles are mostly not the retailer
    assert hackernews.STRICT["typoTolerance"] == "false" and hackernews.STRICT["queryType"] == "prefixNone"


def test_hn_parse_stories_and_comments():
    stories = [hackernews.parse_hit(h) for h in fx("hackernews_stories_nvidia.json")["hits"]]
    comments = [hackernews.parse_hit(h) for h in fx("hackernews_comments_nvidia.json")["hits"]]
    assert all(s and s.extra["type"] == "story" and "Nvidia".lower() in s.title.lower() for s in stories)
    assert all(s.url.startswith("https://news.ycombinator.com/item?id=") for s in stories)
    assert any(s.engagement > 0 for s in stories)
    assert all(c and c.extra["type"] == "comment" and "<p>" not in c.title for c in comments)


@respx.mock
async def test_hn_fetch_dedupes_across_queries():
    stories, comments = fx("hackernews_stories_nvidia.json"), fx("hackernews_comments_nvidia.json")

    def route(request: httpx.Request) -> httpx.Response:
        assert request.url.params["numericFilters"].startswith("created_at_i>")
        tags = request.url.params["tags"]
        payload = stories if tags == "story" else comments if tags == "comment" else stories
        return httpx.Response(200, json=payload)

    respx.get(hackernews.URL).mock(side_effect=route)
    batch = await hackernews.HackerNewsSource().fetch(company("NVDA"))
    assert len(batch.signals) == len(stories["hits"]) + len(comments["hits"])


def test_hn_does_not_support_etfs():
    assert not hackernews.HackerNewsSource().supports(company("SPY"))


# --------------------------------------------------------------------------- Bluesky
def test_bluesky_filter_rejects_homonyms_but_keeps_market_talk():
    keep = bluesky.PostFilter(search_terms(company("SOFI")))
    assert not keep("SoFi stadium is Star Trek and Allegiant is Star Wars")
    assert not keep("Purple Hat- SOFI TUKKER")
    assert not keep("Use my link to get up to $450 in cash bonuses with SoFi")
    assert keep("SoFi Stock: Why the Opportunity May Be Too Good to Ignore")
    assert keep("loading up on $SOFI")
    spy = bluesky.PostFilter(search_terms(company("SPY")))
    assert keep("#SOFI earnings") and spy("The S&P 500 closed at a record") and not spy("a spy thriller")


def test_bluesky_parse_real_name_search_keeps_only_finance():
    keep = bluesky.PostFilter(search_terms(company("SOFI")))
    posts = fx("bluesky_sofi_name.json")["posts"]
    kept = [s for s in (bluesky.parse_post(p, keep) for p in posts) if s]
    assert 1 <= len(kept) <= 2 < len(posts)
    assert any("SoFi Stock" in s.title for s in kept)


def test_bluesky_parse_cashtag_posts():
    keep = bluesky.PostFilter(search_terms(company("NVDA")))
    posts = fx("bluesky_nvda_cashtag.json")["posts"]
    kept = [s for s in (bluesky.parse_post(p, keep) for p in posts) if s]
    assert kept and all(s.url.startswith("https://bsky.app/profile/") for s in kept)
    assert all(s.timestamp and s.timestamp.tzinfo for s in kept)


def test_bluesky_future_created_at_falls_back_to_indexed_at():
    post = {
        "uri": "at://did:plc:x/app.bsky.feed.post/abc",
        "author": {"handle": "a.bsky.social"},
        "record": {"text": "$NVDA up", "createdAt": "2030-01-01T00:00:00Z"},
        "indexedAt": "2026-10-04T12:00:00Z",
        "likeCount": 2,
        "repostCount": 1,
        "replyCount": 1,
    }
    signal = bluesky.parse_post(post, bluesky.PostFilter(search_terms(company("NVDA"))))
    assert signal.timestamp.year == 2026 and signal.engagement == 4


@respx.mock
async def test_bluesky_fetch_public_and_metric(no_keys):
    payload = fx("bluesky_nvda_cashtag.json")
    route = respx.get(bluesky.PUBLIC_URL).mock(return_value=httpx.Response(200, json=payload))
    batch = await bluesky.BlueskySource().fetch(company("NVDA"))
    assert route.call_count == 2
    assert {c.request.url.params["q"] for c in route.calls} == {"$NVDA", "Nvidia stock"}
    assert batch.metrics["bluesky_posts"] == len(batch.signals) > 0


@respx.mock
async def test_bluesky_uses_session_token_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "bluesky_handle", "example.bsky.social")
    monkeypatch.setattr(settings, "bluesky_app_password", "app-pass")
    session = respx.post(f"{bluesky.AUTH_HOST}/xrpc/com.atproto.server.createSession").mock(
        return_value=httpx.Response(200, json=fx("bluesky_session_handmade.json"))
    )
    search = respx.get(f"{bluesky.AUTH_HOST}/xrpc/app.bsky.feed.searchPosts").mock(
        return_value=httpx.Response(200, json=fx("bluesky_nvda_cashtag.json"))
    )
    batch = await bluesky.BlueskySource().fetch(company("NVDA"))
    assert session.call_count == 1 and search.call_count == 2
    assert search.calls[0].request.headers["Authorization"] == "Bearer test-access-jwt"
    assert json.loads(session.calls[0].request.content)["password"] == "app-pass"
    assert batch.signals


@respx.mock
async def test_bluesky_bad_credentials_fall_back_to_public(monkeypatch):
    monkeypatch.setattr(settings, "bluesky_handle", "example.bsky.social")
    monkeypatch.setattr(settings, "bluesky_app_password", "wrong")
    respx.post(f"{bluesky.AUTH_HOST}/xrpc/com.atproto.server.createSession").mock(return_value=httpx.Response(401))
    public = respx.get(bluesky.PUBLIC_URL).mock(return_value=httpx.Response(200, json=fx("bluesky_nvda_cashtag.json")))
    assert await bluesky.BlueskySource().fetch(company("NVDA"))
    assert public.call_count == 2


# --------------------------------------------------------------------------- ApeWisdom
def _ape_routes():
    first, last = fx("apewisdom_stocks_p1.json"), fx("apewisdom_stocks_last.json")
    route = respx.get(url__regex=r"https://apewisdom\.io/api/v1\.0/filter/all-stocks/page/\d+").mock(
        side_effect=lambda request: httpx.Response(200, json=first if request.url.path.endswith("/1") else last)
    )
    return first, route


@respx.mock
async def test_apewisdom_reads_whole_board_once_and_caches():
    first, route = _ape_routes()
    src = apewisdom.ApeWisdomSource()
    nvda = await src.fetch(company("NVDA"))
    row = next(r for r in first["results"] if r["ticker"] == "NVDA")
    assert nvda.metrics["reddit_mentions"] == row["mentions"]
    assert nvda.metrics["reddit_rank"] == row["rank"]
    assert nvda.metrics["reddit_mentions_prev"] == row["mentions_24h_ago"]
    assert nvda.metrics["reddit_rank_prev"] == row["rank_24h_ago"]
    assert nvda.metrics["reddit_tracked"] > 0 and not nvda.signals  # metrics only, never synthetic text
    calls = route.call_count
    assert calls == first["pages"]
    await src.fetch(company("SPY"))
    assert route.call_count == calls  # served from the 10-minute board cache


@respx.mock
async def test_apewisdom_absent_ticker_is_empty():
    _ape_routes()
    assert not await apewisdom.ApeWisdomSource().fetch(CompanyRef(ticker="ZZZZ", name="Z Corp", short_name="Z"))


@respx.mock
async def test_apewisdom_crypto_board():
    crypto = fx("apewisdom_crypto_p1.json")
    crypto["pages"] = 1  # trimmed fixture holds a single page
    respx.get("https://apewisdom.io/api/v1.0/filter/all-crypto/page/1").mock(return_value=httpx.Response(200, json=crypto))
    batch = await apewisdom.ApeWisdomSource().fetch(company("BTC-USD"))
    assert batch.metrics["reddit_rank"] == 1


# --------------------------------------------------------------------------- Tradestie
def test_tradestie_frozen_scores_are_detected_on_real_data():
    today = tradestie._index(fx("tradestie_2026-10-02.json"))
    past = tradestie._index(fx("tradestie_2026-09-04.json"))
    assert not tradestie.scores_are_live(today, past)


def test_tradestie_live_scores_pass_the_gate():
    today = tradestie._index(fx("tradestie_2026-10-02.json"))
    past = {k: (rank, {**row, "sentiment_score": (row["sentiment_score"] or 0) + 0.1}) for k, (rank, row) in today.items()}
    assert tradestie.scores_are_live(today, past)


@respx.mock
async def test_tradestie_never_emits_frozen_sentiment(frozen_now):
    def route(request: httpx.Request) -> httpx.Response:
        name = "tradestie_2026-09-04.json" if "date" in request.url.params else "tradestie_2026-10-02.json"
        return httpx.Response(200, json=fx(name))

    respx.get(tradestie.URL).mock(side_effect=route)
    batch = await tradestie.TradestieSource().fetch(company("NVDA"))
    assert batch.metrics == {"wsb_rank": 6, "wsb_comments": 24}
    assert "wsb_sentiment" not in batch.metrics and "wsb_label" not in batch.metrics


def test_tradestie_metrics_when_gate_passes():
    m = tradestie.metrics_for(2, {"no_of_comments": 44, "sentiment_score": 0.525, "sentiment": "Bullish"}, True)
    assert m == {"wsb_rank": 2, "wsb_comments": 44, "wsb_sentiment": 0.525, "wsb_label": "bullish"}


def test_tradestie_skips_word_tickers_but_not_spy():
    src = tradestie.TradestieSource()
    ai = CompanyRef(ticker="AI", name="C3.ai, Inc.", short_name="C3.ai")
    assert not src.supports(ai) and src.supports(company("SPY")) and not src.supports(company("BTC-USD"))


def test_apewisdom_metrics_skip_missing_fields():
    assert apewisdom.metrics_for({"mentions": 3, "rank": 9, "mentions_24h_ago": None}, 100) == {
        "reddit_mentions": 3,
        "reddit_rank": 9,
        "reddit_tracked": 100,
    }
