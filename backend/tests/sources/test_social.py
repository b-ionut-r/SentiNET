"""Keyless social/crowd sources: StockTwits, Hacker News, Bluesky, ApeWisdom, Tradestie (offline)."""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime

import httpx
import pytest
import respx

from app.config import settings
from app.core.http import UpstreamError
from app.sources import apewisdom, bluesky, hackernews, stocktwits, tradestie
from app.sources.base import CompanyRef, RawSignal
from app.sources.query import Mentions, search_terms
from tests.conftest import load_fixture, load_json_fixture
from tests.sources.conftest import CAPTURE_NOW, company


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
    batch = stocktwits.build_batch([p1, p2], "NVDA", now=CAPTURE_NOW)
    m = batch.metrics
    assert m["stocktwits_messages"] == len(p1["messages"]) + len(p2["messages"]) == len(batch.signals)
    assert m["stocktwits_watchers"] == p1["symbol"]["watchlist_count"]
    labeled = [s for s in batch.signals if s.user_label]
    assert m["stocktwits_bullish"] + m["stocktwits_bearish"] == len(labeled)
    assert all(s.ticker_specific for s in labeled)  # stance only counted when attributable
    raw_tags = sum(1 for p in (p1, p2) for msg in p["messages"] if (msg["entities"].get("sentiment") or {}))
    assert len(labeled) < raw_tags  # multi-symbol spam tags were not attributed to NVDA
    assert m["stocktwits_span_hours"] > 0
    assert m["stocktwits_bull_authors"] + m["stocktwits_bear_authors"] <= len(labeled)
    assert all(s.url and s.url.startswith("https://stocktwits.com/") for s in batch.signals)
    assert all("&amp;" not in s.title for s in batch.signals)


def test_stocktwits_stance_only_counts_the_last_72_hours():
    # Real quiet-name pages (TGT, captured 2026-10-04 23:33 UTC): 60 messages span 16 days and
    # carry 10 attributable tags, 4 of them from one author. Only the 3 recent ones are "the crowd now".
    pages = [fx("stocktwits_tgt_p1.json"), fx("stocktwits_tgt_p2.json")]
    now = datetime(2026, 10, 4, 23, 33, tzinfo=UTC)
    batch = stocktwits.build_batch(pages, "TGT", now=now)
    # Labels stay on the text items; as text only the last 14 days are returned (9 of 10 tags, 58 of 60 posts).
    assert sum(1 for s in batch.signals if s.user_label) == 9 and len(batch.signals) == 58
    m = batch.metrics
    assert (m["stocktwits_bullish"], m["stocktwits_bearish"]) == (2, 1)
    assert (m["stocktwits_bull_authors"], m["stocktwits_bear_authors"]) == (2, 1)
    assert m["stocktwits_window_hours"] == 72.0 and m["stocktwits_span_hours"] > 300


def test_stocktwits_author_votes_once_with_latest_tag():
    template = fx("stocktwits_nvda_p1.json")["messages"][0]

    def msg(i, user, stance, hour):
        return {
            **template,
            "id": i,
            "body": "$NVDA view",
            "symbols": [{"symbol": "NVDA"}],
            "user": {"username": user, "followers": 1},
            "entities": {"sentiment": {"basic": stance}},
            "created_at": f"2026-10-04T{hour:02d}:00:00Z",
        }

    spammer = [msg(i, "prolific", "Bearish", 10 + i) for i in range(5)]
    page = {"symbol": {"watchlist_count": 1}, "messages": [*spammer, msg(9, "prolific", "Bullish", 20), msg(10, "other", "Bullish", 9)]}
    m = stocktwits.build_batch([page], "NVDA", now=CAPTURE_NOW).metrics
    assert (m["stocktwits_bullish"], m["stocktwits_bearish"]) == (2, 5)  # per message: one author dominates
    assert (m["stocktwits_bull_authors"], m["stocktwits_bear_authors"]) == (2, 0)  # per author: latest tag wins


def test_stocktwits_crypto_stream():
    assert stocktwits.stream_symbol(company("BTC-USD")) == "BTC.X"
    batch = stocktwits.build_batch([fx("stocktwits_btc.json")], "BTC.X")
    assert batch.signals and any(s.user_label for s in batch.signals)
    assert sum(1 for s in batch.signals if s.ticker_specific) >= len(batch.signals) // 2


@respx.mock
async def test_stocktwits_hot_names_page_deeper_with_max_cursor():
    # Two real NVDA pages cover only ~3.4 h, so the source keeps paging (up to 4 pages).
    p1, p2 = fx("stocktwits_nvda_p1.json"), fx("stocktwits_nvda_p2.json")
    url = stocktwits.URL.format(symbol="NVDA")
    route = respx.get(url).mock(side_effect=[httpx.Response(200, json=p) for p in (p1, p2, p2, p2)])
    batch = await stocktwits.StockTwitsSource().fetch(company("NVDA"))
    assert route.call_count == stocktwits.MAX_PAGES == 4
    assert route.calls[1].request.url.params["max"] == str(p1["cursor"]["max"])
    assert route.calls[2].request.url.params["max"] == str(p2["cursor"]["max"])
    assert batch.metrics["stocktwits_messages"] == 20  # repeated pages dedupe by message id


@respx.mock
async def test_stocktwits_quiet_names_stop_after_two_pages():
    pages = [fx("stocktwits_tgt_p1.json"), fx("stocktwits_tgt_p2.json")]
    route = respx.get(stocktwits.URL.format(symbol="TGT")).mock(side_effect=[httpx.Response(200, json=p) for p in pages])
    batch = await stocktwits.StockTwitsSource().fetch(company("TGT"))
    assert route.call_count == 2 and batch.metrics["stocktwits_messages"] == 60


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
def test_hn_requests_are_name_based_and_strict():
    nvda = hackernews.build_requests(search_terms(company("NVDA")))
    assert [(r["tags"], r["query"]) for r in nvda] == [("story", "Nvidia"), ("comment", "Nvidia stock")]
    assert nvda[0]["restrictSearchableAttributes"] == "title"
    strict = hackernews.STRICT
    assert strict["typoTolerance"] == "false" and strict["queryType"] == "prefixNone" and strict["ignorePlurals"] == "false"
    # Brand + a distinct alias get story searches (Alphabet news on HN says "Google").
    assert [r["query"] for r in hackernews.build_requests(search_terms(company("GOOGL")))] == ["Alphabet", "Google", "Alphabet stock"]
    # "&" names: Algolia drops the "&", so "AT&T stock" would match any comment saying "stock".
    assert [r["tags"] for r in hackernews.build_requests(search_terms(company("T")))] == ["story"]


def test_hn_skips_everyday_word_names_entirely():
    assert hackernews.build_requests(search_terms(company("TGT"))) == []
    assert not hackernews.HackerNewsSource().supports(company("TGT"))
    assert hackernews.HackerNewsSource().supports(company("NVDA"))


@respx.mock
async def test_hn_drops_hits_that_do_not_name_the_company():
    # Real Algolia hits for the bare symbol "MAR" (Mars, Champ de Mars, "Mar 31st", job posts).
    respx.get(hackernews.URL).mock(return_value=httpx.Response(200, json=fx("hackernews_mar_symbol_noise.json")))
    assert not await hackernews.HackerNewsSource().fetch(company("MAR"))


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


def test_hn_long_comment_excerpt_keeps_the_mention():
    # Real comment whose only "Nvidia" sits past the first 500 chars: the stored text is
    # shifted to that sentence, so downstream relevance still sees the name.
    hit = fx("hackernews_comments_nvidia.json")["hits"][0]
    signal = hackernews.parse_hit(hit, Mentions(search_terms(company("NVDA"))))
    assert "Nvidia" in signal.title and signal.title.startswith("…If that means") and len(signal.title) <= 501


def test_hn_does_not_support_etfs():
    assert not hackernews.HackerNewsSource().supports(company("SPY"))


# --------------------------------------------------------------------------- Bluesky
def _kept(ticker: str, fixture: str):
    mentions = Mentions(search_terms(company(ticker)))
    return [s for s in (bluesky.parse_post(p, mentions) for p in fx(fixture)["posts"]) if s]


def test_bluesky_name_search_keeps_only_finance_posts():
    kept = _kept("SOFI", "bluesky_sofi_name.json")  # real: stadium, "Sofi Tukker", referral bonuses…
    assert len(kept) == 1 and "SoFi Stock" in kept[0].title


def test_bluesky_rejects_real_homonym_floods():
    # Real search results (2026-10-04): "Health Care" policy chatter for XLV, political "Trump"
    # posts for the TRUMP memecoin. Neither is about the instrument.
    assert _kept("XLV", "bluesky_health_care_name.json") == []
    assert _kept("TRUMP-USD", "bluesky_trump_cashtag.json") == []


def test_bluesky_parse_cashtag_posts():
    kept = _kept("NVDA", "bluesky_nvda_cashtag.json")
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
    signal = bluesky.parse_post(post, Mentions(search_terms(company("NVDA"))))
    assert signal.timestamp.year == 2026 and signal.engagement == 4


@respx.mock
async def test_bluesky_fetch_public_and_metrics(no_keys, frozen_now):
    payload = fx("bluesky_nvda_cashtag.json")
    route = respx.get(bluesky.PUBLIC_URL).mock(return_value=httpx.Response(200, json=payload))
    batch = await bluesky.BlueskySource().fetch(company("NVDA"))
    assert route.call_count == 2
    assert {c.request.url.params["q"] for c in route.calls} == {"$NVDA", "Nvidia stock"}
    m = batch.metrics
    assert m["bluesky_posts"] == len(batch.signals) > 0
    assert 0 < m["bluesky_authors"] <= m["bluesky_posts"]
    authors = Counter(s.author for s in batch.signals)
    assert max(authors.values()) <= bluesky.PER_AUTHOR  # bot control
    assert m["bluesky_saturated"] is False and m["bluesky_span_hours"] == 168.0  # fixture < 100 posts per query
    assert m["bluesky_posts_per_day"] == pytest.approx(m["bluesky_posts"] / 7, abs=0.1)


def test_bluesky_rate_uses_each_querys_own_coverage():
    now = datetime(2026, 10, 4, 12, tzinfo=UTC)

    def post(uri, created):
        return {"uri": uri, "record": {"createdAt": created}}

    # Query 1 hit the cap within 11.5 h; 20 of its posts are on-topic. Query 2 covers the week: 7 kept.
    capped = [post(f"q1/{i}", f"2026-10-04T{i % 12:02d}:30:00Z") for i in range(bluesky.LIMIT)]
    week = [post(f"q2/{i}", f"2026-09-{28 + i % 3}T10:00:00Z") for i in range(7)]
    kept_uris = {f"q1/{i}" for i in range(20)} | {f"q2/{i}" for i in range(7)}
    per_day, hours, saturated = bluesky.query_rate(capped, kept_uris, now)
    assert saturated and hours == 11.5 and per_day == pytest.approx(20 * 24 / 11.5)
    assert bluesky.query_rate(week, kept_uris, now) == (pytest.approx(1.0), 168.0, False)
    kept = [RawSignal(title="x", author=f"a{i}") for i in range(27)]
    m = bluesky.crowd_metrics(kept, [capped, week], kept_uris, now)
    assert m["bluesky_saturated"] and m["bluesky_span_hours"] == 11.5
    assert m["bluesky_posts_per_day"] == pytest.approx(41.7, abs=0.1)
    # A junk-saturated query (no kept posts) must not zero the rate the other query measured.
    junk = bluesky.crowd_metrics(kept[:7], [[post(f"j/{i}", "2026-10-04T11:00:00Z") for i in range(100)], week], kept_uris, now)
    assert junk["bluesky_posts_per_day"] == pytest.approx(1.0) and junk["bluesky_span_hours"] == 168.0


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
async def test_apewisdom_partial_board_serves_found_rows_but_never_claims_absence():
    first = fx("apewisdom_stocks_p1.json")
    respx.get(url__regex=r"https://apewisdom\.io/api/v1\.0/filter/all-stocks/page/\d+").mock(
        side_effect=lambda request: httpx.Response(200, json=first) if request.url.path.endswith("/1") else httpx.Response(503)
    )
    src = apewisdom.ApeWisdomSource()
    assert (await src.fetch(company("NVDA"))).metrics["reddit_rank"] > 0  # on page 1
    with pytest.raises(UpstreamError, match="partially unavailable"):
        await src.fetch(CompanyRef(ticker="ZZZZ", name="Z Corp", short_name="Z"))


def test_apewisdom_word_tickers_are_unsupported_on_the_stock_board():
    # Real board (2026-10-04): "YOU" (CLEAR Secure) ranked 9th, "ES", "CD", "DTE" in the top 15 —
    # counts of the words/lingo, not the stocks. MU (rank 2) is real chatter.
    board = {r["ticker"] for r in fx("apewisdom_stocks_p1_live.json")["results"]}
    assert {"YOU", "ES", "CD", "DTE", "MU"} <= board
    src = apewisdom.ApeWisdomSource()
    for word in ("YOU", "ES", "CD", "DTE"):
        assert not src.supports(CompanyRef(ticker=word, name=f"{word} Corp", short_name=word))
    assert src.supports(company("MU")) and src.supports(company("BTC-USD"))


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
    assert batch.metrics == {"wsb_rank": 6, "wsb_comments": 24, "wsb_board_total": 350}
    assert "wsb_sentiment" not in batch.metrics and "wsb_label" not in batch.metrics


def test_tradestie_metrics_when_gate_passes():
    row = {"no_of_comments": 44, "sentiment_score": 0.525, "sentiment": "Bullish"}
    m = tradestie.metrics_for(2, row, True, 746)
    assert m == {"wsb_rank": 2, "wsb_comments": 44, "wsb_board_total": 746, "wsb_sentiment": 0.525, "wsb_label": "bullish"}


def test_tradestie_rank_is_withheld_on_a_thin_board():
    # Real Sunday list: the whole top-50 summed to 100 comments; rank 26 there means nothing.
    board = tradestie._index(fx("tradestie_2026-10-04_sunday.json"))
    total = tradestie.board_total(board)
    rank, row = board["MSFT"]
    assert total == 100 and tradestie.metrics_for(rank, row, False, total) == {"wsb_comments": 2, "wsb_board_total": 100}
    assert "wsb_rank" not in tradestie.metrics_for(1, {"no_of_comments": 3}, False, 800)  # too few for the ticker


def test_tradestie_gate_is_about_words_not_length():
    src = tradestie.TradestieSource()
    for word in ("AI", "UK", "CD", "HYSA", "CAPE"):  # all on the real Sunday board
        assert not src.supports(CompanyRef(ticker=word, name=f"{word} Inc.", short_name=word))
    assert src.supports(company("MU")) and src.supports(company("SPY")) and not src.supports(company("BTC-USD"))


def test_apewisdom_metrics_skip_missing_fields():
    # Live TRUMP.X row (2026-10-04): no mentions_24h_ago and rank_24h_ago 0 = not ranked yesterday.
    assert apewisdom.metrics_for({"mentions": 3, "rank": 9, "mentions_24h_ago": None, "rank_24h_ago": 0}, 100) == {
        "reddit_mentions": 3,
        "reddit_rank": 9,
        "reddit_tracked": 100,
    }
