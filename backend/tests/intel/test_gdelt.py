"""GDELT tone trend: query design, parsing, statistics and rate-limit handling."""
from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from app.core.ratelimit import limiter
from app.intel import gdelt
from app.intel.gdelt import build_query, build_trend, merge_series, parse_timeline, tone_stats
from app.schemas import TonePoint
from app.sources.base import CompanyRef
from tests.intel.helpers import load_json

RATE_TEXT = ("Please limit requests to one every 5 seconds or contact kalev.leetaru5@gmail.com for larger "
             "queries.")


@pytest.fixture(autouse=True)
def _fast_gdelt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gdelt, "RATE_LIMIT_WAIT", 0.0)
    monkeypatch.setitem(limiter._spacing, "api.gdeltproject.org", 0.0)


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
    assert "Zuckerberg" in build_query(ref("META", "Meta"))
    assert '"S&P 500"' in build_query(ref("SPY", "S&P 500", qtype="ETF"))


def test_query_uncurated_common_word_is_anchored() -> None:
    q = build_query(ref("CHWY", "Chewy"))
    assert q.startswith('("Chewy Inc" OR "Chewy Corp" OR "Chewy shares"') and '"Chewy" ' not in q


def test_query_acronym_needs_corporate_context() -> None:
    q = build_query(ref("XYZQ", "QRS"))
    assert q.startswith('"QRS" (shares OR stock') and q.endswith("sourcelang:english")


def test_query_aliases_are_ored() -> None:
    q = build_query(ref("ZZZ", "Acmecorp", ["Acme Rockets", "AR"]))  # 2-letter alias dropped
    assert q == '("Acmecorp" OR "Acme Rockets") sourcelang:english'
    assert build_query(ref("BTC-USD", "Bitcoin", qtype="CRYPTOCURRENCY")) == '"Bitcoin" sourcelang:english'


def test_queries_have_balanced_syntax() -> None:
    for ticker, query in gdelt.CURATED.items():
        assert query.count("(") == query.count(")") <= 1, ticker  # GDELT allows one OR group
        assert query.count('"') % 2 == 0, ticker


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
# Fetching: rate limits, rejected queries, partial failure
# --------------------------------------------------------------------------- #
def _mode(request: httpx.Request) -> str:
    return parse_qs(urlsplit(str(request.url)).query)["mode"][0]


async def test_rate_limit_text_is_retried_once() -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")
    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(side_effect=[
            httpx.Response(200, text=RATE_TEXT),  # GDELT sometimes refuses with HTTP 200 + text
            httpx.Response(200, json=payload),
        ])
        assert await gdelt._gdelt('"Nvidia"', "timelinetone", 90) == payload
        assert route.call_count == 2


async def test_persistent_rate_limit_raises() -> None:
    with respx.mock as mock:
        mock.get(gdelt.API_URL).mock(return_value=httpx.Response(429, text=RATE_TEXT))
        with pytest.raises(gdelt.GdeltRateLimited):
            await gdelt._gdelt('"Nvidia"', "timelinetone", 90)


async def test_trend_with_volume_refused_keeps_tone() -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")

    def answer(request: httpx.Request) -> httpx.Response:
        if _mode(request) == "timelinetone":
            return httpx.Response(200, json=payload)
        return httpx.Response(429, text=RATE_TEXT)

    with respx.mock as mock:
        mock.get(gdelt.API_URL).mock(side_effect=answer)
        trend = await gdelt.get_tone_trend(ref("NVDA", "Nvidia"))
    assert trend is not None and trend.tone_7d is not None
    assert all(p.volume is None for p in trend.series)


async def test_rejected_query_falls_back_to_plain_name() -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")
    seen: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        query = parse_qs(urlsplit(str(request.url)).query)["query"][0]
        seen.append(query)
        if "OR" in query:
            return httpx.Response(200, text="The specified phrase is too short.")
        return httpx.Response(200, json=payload if _mode(request) == "timelinetone" else {})

    with respx.mock as mock:
        mock.get(gdelt.API_URL).mock(side_effect=answer)
        trend = await gdelt.get_tone_trend(ref("ZZZ", "Acmecorp", ["Acme Rockets"]))
    assert trend is not None and trend.query == '"Acmecorp" sourcelang:english'
    assert seen[0].startswith('("Acmecorp" OR') and seen[-1] == '"Acmecorp" sourcelang:english'


async def test_tone_trend_is_cached() -> None:
    payload = load_json("gdelt/nvidia_timelinetone.json")
    with respx.mock as mock:
        route = mock.get(gdelt.API_URL).mock(return_value=httpx.Response(200, json=payload))
        company = ref("NVDA", "Nvidia")
        await gdelt.get_tone_trend(company)
        await gdelt.get_tone_trend(company)
        assert route.call_count == 2  # tone + volume once; second call served from cache


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
