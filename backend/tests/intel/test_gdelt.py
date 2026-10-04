"""GDELT tone trend: query design, parsing, statistics and rate-limit handling."""
from __future__ import annotations

import asyncio
import re
from datetime import date, timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from app.core.http import UpstreamError
from app.core.ratelimit import limiter
from app.intel import gdelt
from app.intel.gdelt import (
    build_query,
    build_trend,
    merge_series,
    parse_timeline,
    tone_stats,
)
from app.schemas import TonePoint
from app.sources.base import CompanyRef
from tests.intel.helpers import load_json

RATE_TEXT = ("Please limit requests to one every 5 seconds or contact kalev.leetaru5@gmail.com for larger "
             "queries.")


@pytest.fixture(autouse=True)
def _fast_gdelt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdelt, "RATE_LIMIT_WAIT", 0.0)
    monkeypatch.setattr(gdelt, "SPACING", 0.0)
    monkeypatch.setitem(limiter._spacing, "api.gdeltproject.org", 0.0)
    gdelt.reset_state()


def ref(ticker: str, short: str, aliases: list[str] | None = None, qtype: str = "EQUITY") -> CompanyRef:
    return CompanyRef(ticker=ticker, name=short, short_name=short, aliases=aliases or [], quote_type=qtype)


# --------------------------------------------------------------------------- #
# Query design
# --------------------------------------------------------------------------- #
def test_query_distinctive_name() -> None:
    assert build_query(ref("NVDA", "Nvidia")) == '"Nvidia" sourcelang:english'
    q = build_query(ref("PLTR", "Palantir", ["Palantir Technologies"]))
    assert q == '"Palantir" sourcelang:english'  # alias already contains the name: redundant


def test_query_curated_common_words() -> None:
    apple = build_query(ref("AAPL", "Apple"))
    assert apple.startswith('"Apple" (iPhone OR') and apple.endswith("sourcelang:english")
    meta = build_query(ref("META", "Meta"))
    assert "Zuckerberg" in meta and "Instagram" not in meta  # page chrome ("Follow us on Instagram")
    assert '"S&P 500"' in build_query(ref("SPY", "S&P 500", qtype="ETF"))


def test_query_uncurated_common_word_is_anchored() -> None:
    q = build_query(ref("CHWY", "Chewy", ["Chewy Inc"]))
    assert q == '("Chewy Inc" OR "Chewy Corp" OR "Chewy CEO") sourcelang:english'
    # Stopwords vanish from GDELT phrases: "Target shares" would match "price target on shares".
    assert "shares" not in build_query(ref("POOL", "Pool", ["Pool Corporation"]))


def test_query_short_names_only_inside_phrases() -> None:
    q = build_query(ref("IBM", "IBM", ["International Business Machines"]))
    assert q == ('("IBM Inc" OR "IBM Corp" OR "IBM CEO" OR "International Business Machines" OR "IBM shares" '
                 'OR "IBM stock") sourcelang:english')
    nike = build_query(ref("NKE", "Nike"))
    assert '"Nike shares"' in nike and '"Nike" ' not in nike


def test_no_query_term_is_too_short_for_gdelt() -> None:
    """GDELT rejects quoted words under 5 characters and silently ignores bare ones."""
    assert gdelt.fallback_query(ref("META", "Meta", ["Meta Platforms"])) == '"Meta Platforms" sourcelang:english'
    assert build_query(ref("ZZ", "Kora")) == '("Kora shares" OR "Kora stock" OR "Kora CEO") sourcelang:english'
    assert build_query(ref("XX", "X")) == '("X Inc" OR "X Corp" OR "X CEO" OR "X shares" OR "X stock") sourcelang:english'
    queries = [*gdelt.CURATED.values(), build_query(ref("IBM", "IBM")), build_query(ref("U", "Unity")),
               build_query(ref("NKE", "Nike")), build_query(ref("CHWY", "Chewy"))]
    for query in queries:
        stripped = re.sub(r'"[^"]*"|\bOR\b|sourcelang:\w+|[()]', " ", query)
        for phrase in re.findall(r'"([^"]*)"', query):
            assert len(phrase) >= gdelt.MIN_PHRASE or phrase == "AT&T", (phrase, query)  # AT&T: verified earlier
        for word in stripped.split():
            assert len(word) >= gdelt.MIN_PHRASE, (word, query)  # bare short keywords match nothing


def test_query_aliases_are_ored() -> None:
    q = build_query(ref("ZZZ", "Acmecorp", ["Acme Rockets", "AR"]))  # 2-letter alias dropped
    assert q == '("Acmecorp" OR "Acme Rockets") sourcelang:english'
    assert build_query(ref("BTC-USD", "Bitcoin", qtype="CRYPTOCURRENCY")) == '"Bitcoin" sourcelang:english'


def _assert_gdelt_syntax(query: str) -> None:
    assert query.count('"') % 2 == 0, query
    groups = re.findall(r"\(([^()]*)\)", query)
    assert query.count("(") == query.count(")") == len(groups), query  # no nesting
    for group in groups:
        assert " OR " in group, query  # GDELT: parentheses only around OR'd statements


def test_queries_have_valid_gdelt_syntax() -> None:
    for query in gdelt.CURATED.values():
        _assert_gdelt_syntax(query)
    for company in (ref("NVDA", "Nvidia"), ref("CHWY", "Chewy"), ref("XYZQ", "QRS"),
                    ref("ZZZ", "Acmecorp", ["Acme Rockets"]), ref("T", "AT&T"), ref("ETH-USD", "Ethereum", qtype="CRYPTOCURRENCY")):
        _assert_gdelt_syntax(build_query(company))


# --------------------------------------------------------------------------- #
# Parsing & statistics
# --------------------------------------------------------------------------- #
def test_parse_real_tone_payload() -> None:
    tone = parse_timeline(load_json("gdelt/nvidia_timelinetone.json"))
    assert len(tone) >= 60
    days = sorted(tone)
    assert days[-1] - days[0] <= timedelta(days=92)
    assert all(-10 < v < 10 and norm is None for v, norm in tone.values())
    assert parse_timeline({}) == {} and parse_timeline("Please limit requests") == {}


def _volume_payload(days: dict[date, int]) -> dict:
    return {"timeline": [{"series": "Article Count", "data": [
        {"date": f"{d:%Y%m%d}T000000Z", "value": v, "norm": 100_000} for d, v in days.items()]}]}


def test_merge_drops_partial_today_and_unions_days() -> None:
    today = date(2026, 10, 4)
    tone = {date(2026, 10, 2): (0.5, None), date(2026, 10, 4): (1.0, None)}
    vol = parse_timeline(_volume_payload({date(2026, 10, 1): 40, date(2026, 10, 2): 50, today: 3}))
    series = merge_series(tone, vol, today=today)
    assert [p.date for p in series] == [date(2026, 10, 1), date(2026, 10, 2)]
    assert series[0].tone is None and series[0].volume == 40 and series[1].tone == 0.5


def test_tone_stats_volume_weighted_and_percentile() -> None:
    start = date(2026, 7, 1)
    points = [TonePoint(date=start + timedelta(days=i), tone=0.0, volume=100) for i in range(80)]
    # last week: two very negative days with heavy coverage, five mildly positive light days
    for i, (t, v) in enumerate([(-2.0, 400), (-2.0, 400), (0.5, 50), (0.5, 50), (0.5, 50), (0.5, 50), (0.5, 50)]):
        points[-7 + i] = TonePoint(date=points[-7 + i].date, tone=t, volume=v)
    stats = tone_stats(points)
    expected_7d = (-2.0 * 800 + 0.5 * 250) / 1050
    assert stats["tone_7d"] == pytest.approx(expected_7d, abs=1e-3)
    assert stats["tone_7d"] < stats["tone_30d"] < 0.01
    assert stats["change_7d_vs_30d"] == pytest.approx(stats["tone_7d"] - stats["tone_30d"], abs=2e-3)
    assert stats["percentile_7d"] is not None and stats["percentile_7d"] < 0.1  # 90-day low


def test_tone_stats_without_volume_uses_simple_mean() -> None:
    start = date(2026, 9, 1)
    points = [TonePoint(date=start + timedelta(days=i), tone=float(i % 3), volume=None) for i in range(21)]
    stats = tone_stats(points)
    last7 = [p.tone for p in points[-7:]]
    assert stats["tone_7d"] == pytest.approx(sum(last7) / 7, abs=1e-3)
    assert tone_stats([])["tone_7d"] is None


def test_build_trend_from_real_tone() -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")
    last_day = max(parse_timeline(payload))
    trend = build_trend('"Nvidia" sourcelang:english', payload, {}, today=last_day + timedelta(days=1))
    assert trend is not None and trend.query.startswith('"Nvidia"')
    assert trend.series[-1].date == last_day and trend.tone_7d is not None and trend.tone_90d is not None
    assert 0.0 <= (trend.percentile_7d or 0) <= 1.0
    assert build_trend("q", {}, {}, today=last_day) is None


# --------------------------------------------------------------------------- #
# Fetching: rate limits, cooldown, rejected queries, caching, serialization
# --------------------------------------------------------------------------- #
def _mode(request: httpx.Request) -> str:
    return parse_qs(urlsplit(str(request.url)).query)["mode"][0]


def _query(request: httpx.Request) -> str:
    return parse_qs(urlsplit(str(request.url)).query)["query"][0]


async def test_rate_limit_text_is_retried_once() -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")
    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(side_effect=[
            httpx.Response(200, text=RATE_TEXT),  # GDELT sometimes refuses with HTTP 200 + text
            httpx.Response(200, json=payload),
        ])
        assert await gdelt._gdelt('"Nvidia"', "timelinetone", 90) == payload
        assert route.call_count == 2
    assert gdelt.cooling_down() == 0


async def test_persistent_rate_limit_trips_cooldown_and_fails_fast() -> None:
    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(return_value=httpx.Response(429, text=RATE_TEXT))
        with pytest.raises(gdelt.GdeltRateLimited):
            await gdelt._gdelt('"Nvidia"', "timelinetone", 90)
        assert route.call_count == 2
        assert gdelt.cooling_down() > 20
        # While cooling down nothing is sent and the caller learns why at once.
        with pytest.raises(gdelt.GdeltRateLimited, match=r"next try in \d+ s"):
            await gdelt.get_tone_trend(ref("AMZN", "Amazon"))
        assert route.call_count == 2


async def test_cooldown_escalates_and_resets_on_success() -> None:
    gdelt._breaker.trip()
    first = gdelt.cooling_down()
    gdelt._breaker.trip()
    assert gdelt.cooling_down() > first * 1.5
    gdelt._breaker.until = 0.0  # pause elapsed
    payload = load_json("gdelt/nvidia_timelinetone.json")
    with respx.mock as mock:
        mock.get(gdelt.API_URL).mock(return_value=httpx.Response(200, json=payload))
        await gdelt._gdelt('"Nvidia"', "timelinetone", 90)
    assert gdelt._breaker.strikes == 0


async def test_query_errors_are_reported_verbatim() -> None:
    with respx.mock as mock:
        mock.get(gdelt.API_URL).mock(return_value=httpx.Response(200, text="The specified phrase is too short.\n"))
        with pytest.raises(gdelt.GdeltQueryRejected, match="phrase is too short"):
            await gdelt._gdelt('"Meta"', "timelinetone", 90)
        mock.get(gdelt.API_URL).mock(return_value=httpx.Response(502, text="Bad gateway"))
        with pytest.raises(UpstreamError, match="GDELT HTTP 502"):
            await gdelt._gdelt('"Meta Platforms"', "timelinetone", 90)
    assert gdelt.cooling_down() == 0  # not a rate limit: no cooldown


async def test_cold_call_waits_for_tone_only_and_volume_refusal_keeps_tone() -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")

    def answer(request: httpx.Request) -> httpx.Response:
        if _mode(request) == "timelinetone":
            return httpx.Response(200, json=payload)
        return httpx.Response(429, text=RATE_TEXT)

    with respx.mock as mock:
        mock.get(gdelt.API_URL).mock(side_effect=answer)
        trend = await gdelt.get_tone_trend(ref("NVDA", "Nvidia"))
        await gdelt.drain()
    assert trend is not None and trend.tone_7d is not None
    assert all(p.volume is None for p in trend.series)
    assert gdelt.cooling_down() > 0  # the volume refusal paused further requests


async def test_tone_and_volume_are_merged_once_both_arrive() -> None:
    tone = load_json("gdelt/nvidia_timelinetone.json")
    volume = load_json("gdelt/nvidia_timelinevolraw.json")
    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(
            side_effect=lambda r: httpx.Response(200, json=tone if _mode(r) == "timelinetone" else volume))
        company = ref("NVDA", "Nvidia")
        await gdelt.get_tone_trend(company)
        await gdelt.drain()
        trend = await gdelt.get_tone_trend(company)
        assert route.call_count == 2  # tone + volume once; the second call is served from cache
    assert trend is not None and any(p.volume for p in trend.series)


async def test_rejected_query_falls_back_to_plain_name() -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")
    seen: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        query = _query(request)
        seen.append(query)
        if "OR" in query:
            return httpx.Response(200, text="The specified phrase is too short.")
        return httpx.Response(200, json=payload if _mode(request) == "timelinetone" else {})

    with respx.mock as mock:
        mock.get(gdelt.API_URL).mock(side_effect=answer)
        trend = await gdelt.get_tone_trend(ref("ZZZ", "Acmecorp", ["Acme Rockets"]))
        await gdelt.drain()
    assert trend is not None and trend.query == '"Acmecorp" sourcelang:english'
    assert seen[0].startswith('("Acmecorp" OR') and seen[-1] == '"Acmecorp" sourcelang:english'
    assert len(seen) == 3  # rejected, fallback tone, fallback volume


async def test_no_coverage_is_none_not_an_error() -> None:
    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(return_value=httpx.Response(200, text=""))
        assert await gdelt.get_tone_trend(ref("ZZZ", "Obscurecorp")) is None
        await gdelt.drain()
        assert await gdelt.get_tone_trend(ref("ZZZ", "Obscurecorp")) is None
        assert route.call_count == 2  # the empty answer is cached like any other


async def test_concurrent_callers_share_one_request_and_requests_never_overlap() -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")
    in_flight = peak = 0

    async def answer(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return httpx.Response(200, json=payload if _mode(request) == "timelinetone" else {})

    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(side_effect=answer)
        nvda, amd = ref("NVDA", "Nvidia"), ref("AVGO", "Broadcom")
        results = await asyncio.gather(*(gdelt.get_tone_trend(c) for c in (nvda, nvda, nvda, amd)))
        await gdelt.drain()
    assert all(r is not None for r in results)
    tone_calls = [c for c in route.calls if _mode(c.request) == "timelinetone"]
    assert len(tone_calls) == 2  # one per distinct query, however many callers
    assert peak == 1  # GDELT counts overlapping requests against the per-IP quota


async def test_stale_trend_is_served_while_refreshing(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")
    company = ref("NVDA", "Nvidia")
    with respx.mock as mock:
        mock.get(gdelt.API_URL).mock(return_value=httpx.Response(200, json=payload))
        fresh = await gdelt.get_tone_trend(company)
        await gdelt.drain()
    monkeypatch.setattr(gdelt.settings, "history_cache_ttl", 0)  # everything is stale now
    monkeypatch.setattr(gdelt, "REFRESH_GAP", 0.0)
    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(return_value=httpx.Response(429, text=RATE_TEXT))
        stale = await gdelt.get_tone_trend(company)  # answered from cache, refresh in background
        await gdelt.drain()
        assert stale == fresh and route.call_count == 2  # the refresh tried (and was refused)
        again = await gdelt.get_tone_trend(company)  # cooling down: no new request
        assert again == fresh and route.call_count == 2
    gdelt.reset_state()
    with pytest.raises(UpstreamError):
        with respx.mock as mock:
            mock.get(gdelt.API_URL).mock(return_value=httpx.Response(429, text=RATE_TEXT))
            await gdelt.get_tone_trend(company)


def test_real_volume_drops_day_still_being_ingested() -> None:
    """GDELT's newest day had norm 46,896 vs ~150k typical: incomplete, so excluded."""
    tone = load_json("gdelt/nvidia_timelinetone.json")
    volume = load_json("gdelt/nvidia_timelinevolraw.json")
    raw_days = sorted(parse_timeline(volume))
    trend = build_trend('"Nvidia" sourcelang:english', tone, volume, today=date(2026, 10, 4))
    assert trend is not None
    assert trend.series[-1].date == raw_days[-2]  # the partial last day is gone
    assert all(p.volume is not None and p.volume > 50 for p in trend.series)
    assert trend.tone_7d is not None and trend.tone_30d is not None and trend.percentile_7d is not None




async def test_refreshes_are_spaced_per_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trend whose volume keeps failing is not re-requested on every analysis."""
    payload = load_json("gdelt/nvidia_timelinetone.json")

    def answer(request: httpx.Request) -> httpx.Response:
        if _mode(request) == "timelinetone":
            return httpx.Response(200, json=payload)
        return httpx.Response(502, text="Bad gateway")  # not a rate limit: no cooldown

    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(side_effect=answer)
        company = ref("NVDA", "Nvidia")
        for _ in range(3):
            assert await gdelt.get_tone_trend(company) is not None
            await gdelt.drain()
        assert route.call_count == 2  # tone + one volume attempt within REFRESH_GAP


def test_funds_indices_and_futures_search_their_theme() -> None:
    kre = CompanyRef(ticker="KRE", name="SPDR S&P Regional Banking ETF", short_name="S&P Regional Banking",
                     quote_type="ETF")
    assert build_query(kre) == '("regional banks" OR "regional bank stocks") sourcelang:english'
    gold = CompanyRef(ticker="GC=F", name="Gold Dec 26", short_name="Gold", aliases=["gold prices", "gold futures"],
                      quote_type="FUTURE")
    assert build_query(gold) == '("gold prices" OR "gold futures") sourcelang:english'  # never bare "Gold"
    coffee = CompanyRef(ticker="KC=F", name="Coffee Dec 26", short_name="Coffee", quote_type="FUTURE")
    assert build_query(coffee) == '("Coffee prices" OR "Coffee futures") sourcelang:english'
    vix = CompanyRef(ticker="^VIX", name="CBOE Volatility Index", short_name="VIX",
                     aliases=["Cboe Volatility Index", "volatility index"], quote_type="INDEX")
    assert build_query(vix) == '("Cboe Volatility Index" OR "volatility index") sourcelang:english'


async def test_unsearchable_company_returns_none_without_a_request() -> None:
    """No name of 5+ characters (and no aliases): GDELT cannot match anything, so do not ask."""
    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(return_value=httpx.Response(200, json={}))
        assert await gdelt.get_tone_trend(ref("BNB-USD", "BNB", qtype="CRYPTOCURRENCY")) is None
        assert route.call_count == 0
