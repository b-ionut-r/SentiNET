"""History alignment, lead/lag correlation and significance-aware interpretation."""
from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from app.analytics.stats import aligned_series, betainc, build_history, corr_pvalue, pearson, t_pvalue
from app.schemas import TonePoint, ToneTrend
from tests.analytics.factories import NOW, snapshot

END = date(2026, 10, 2)  # a Friday


@pytest.mark.parametrize(("t", "df", "p"), [
    (2.0, 10, 0.073388), (2.228139, 10, 0.05), (1.96, 1000, 0.050275), (3.5, 30, 0.001478), (0.0, 12, 1.0),
])
def test_t_pvalues_match_reference_values(t: float, df: int, p: float) -> None:
    assert t_pvalue(t, df) == pytest.approx(p, abs=2e-5)


def test_betainc_and_pearson() -> None:
    assert betainc(2, 3, 0.4) == pytest.approx(0.5248, abs=1e-4)
    assert betainc(1, 1, 0.3) == pytest.approx(0.3)
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    assert corr_pvalue(0.3, 62) == pytest.approx(0.0178, abs=1e-4)
    assert corr_pvalue(0.99, 2) == 1.0


def calendar(days: int) -> list[date]:
    return [END - timedelta(days=days - 1 - i) for i in range(days)]


def series(rng: random.Random, days: int, lead: int | None, strength: float = 1.2):
    """Daily tone + closes on weekdays; returns follow tone `lead` trading days later (None: independent)."""
    dates = calendar(days)
    trading = [d for d in dates if d.weekday() < 5]
    tone = {d: rng.gauss(0, 1) for d in trading}
    closes, price = [], 100.0
    for i, d in enumerate(trading):
        signal = strength * tone[trading[i - lead]] if lead is not None and i - lead >= 0 else 0.0
        price *= 1 + (signal + rng.gauss(0, 1)) / 100
        closes.append((d, round(price, 4)))
    points = [TonePoint(date=d, tone=round(tone[d], 4), volume=100.0) for d in trading]
    return ToneTrend(query="q", series=points), closes


def test_tone_leading_returns_is_detected_and_explained() -> None:
    tone, closes = series(random.Random(7), 130, lead=1)
    h = build_history("ACME", 120, tone, closes, [], None, {"tone": "ok", "price": "ok"})
    assert h.best_lag is not None and h.best_lag.lag_days == 1 and h.best_lag.r > 0.4
    assert h.best_lag.p_value < 0.001 and h.best_lag.n >= 70
    assert h.interpretation.startswith("News tone tends to lead this stock")
    assert "1 trading day later" in h.interpretation and "p < 0.001" in h.interpretation
    assert [s.lag_days for s in h.lags] == [-3, -2, -1, 0, 1, 2, 3]


def test_independent_series_report_no_reliable_relationship() -> None:
    tone, closes = series(random.Random(11), 130, lead=None)
    h = build_history("ACME", 120, tone, closes, [], None, {})
    assert h.interpretation.startswith(("No reliable relationship", "Only a weak hint"))
    assert h.best_lag is not None and abs(h.best_lag.r) < 0.3


def test_price_leading_the_news() -> None:
    rng = random.Random(3)
    dates = [d for d in calendar(150) if d.weekday() < 5]
    rets = [rng.gauss(0, 1.5) for _ in dates]
    closes, price = [], 50.0
    for d, r in zip(dates, rets, strict=True):
        price *= 1 + r / 100
        closes.append((d, price))
    # Coverage reacts two sessions after the move.
    tone = ToneTrend(query="q", series=[
        TonePoint(date=d, tone=(0.8 * rets[i - 2] if i >= 2 else 0) + rng.gauss(0, 0.5), volume=50.0)
        for i, d in enumerate(dates)])
    h = build_history("ACME", 140, tone, closes, [], None, {})
    assert h.best_lag.lag_days == -2 and h.interpretation.startswith("Price leads the news")


def test_weekend_tone_rolls_into_monday() -> None:
    fri, sat, sun, mon = date(2026, 9, 25), date(2026, 9, 26), date(2026, 9, 27), date(2026, 9, 28)
    tone = {d: TonePoint(date=d, tone=t, volume=v) for d, t, v in
            ((sat, 5.0, 100.0), (sun, 5.0, 100.0), (mon, -1.0, 200.0))}
    tones, rets = aligned_series(tone, {fri: 100.0, mon: 102.0})
    assert tones == [pytest.approx(2.0)] and rets == [pytest.approx(2.0)]  # (5·100 + 5·100 − 1·200) / 400


def test_points_merge_all_series() -> None:
    tone = ToneTrend(query="q", series=[TonePoint(date=END - timedelta(days=i), tone=0.5, volume=10.0)
                                        for i in range(10)])
    closes = [(END - timedelta(days=i), 100.0 + i) for i in range(10) if (END - timedelta(days=i)).weekday() < 5]
    snaps = [snapshot(30, 55, 0.1), snapshot(26, 61, 0.2)]  # same day: the later one wins
    wiki = [(END - timedelta(days=i), 1000.0 + i) for i in range(10)]
    h = build_history("ACME", 7, tone, closes, snaps, wiki, {"tone": "ok"})
    assert len(h.points) == 7 and h.points[-1].date == END
    by_day = {p.date: p for p in h.points}
    assert by_day[END].close == 100.0 and by_day[END].ret_pct == pytest.approx((100 / 101 - 1) * 100, abs=1e-3)
    snap_day = (NOW - timedelta(hours=26)).date()
    assert by_day[snap_day].snapshot_score == 61
    assert by_day[END].wiki_views == 1000.0 and by_day[END].tone == 0.5
    saturday = END - timedelta(days=6)
    assert by_day[saturday].close is None and by_day[saturday].ret_pct is None
    assert h.status == {"tone": "ok"}
    assert "Too few" in h.interpretation  # 5 trading days are not enough to say anything


def test_missing_series_are_explained() -> None:
    closes = [(END - timedelta(days=i), 10.0) for i in range(30)]
    assert build_history("X", 30, None, closes, [], None, {}).interpretation.startswith("No GDELT tone history")
    empty = build_history("X", 30, None, [], [], None, {"tone": "error: x"})
    assert empty.points == [] and empty.best_lag is None and empty.status == {"tone": "error: x"}


def test_temporarily_missing_tone_is_not_called_absent() -> None:
    # Live: GDELT rate-limited / still loading, yet the text said the data did not exist.
    closes = [(END - timedelta(days=i), 10.0 + i % 3) for i in range(40)]
    for status in ("error: still loading after 23s; ready on next refresh",
                   "error: GDELT is rate-limiting this server's IP"):
        h = build_history("NVDA", 30, None, closes, [], None, {"tone": status, "price": "ok"})
        assert h.interpretation.startswith("Global news tone (GDELT) is temporarily unavailable")
        assert "reload" in h.interpretation
    h = build_history("NVDA", 30, None, closes, [], None, {"tone": "empty", "price": "ok"})
    assert h.interpretation.startswith("No GDELT tone history")


def test_zero_article_days_are_not_a_neutral_tone() -> None:
    # Live KOSS: 3 articles in 90 days (GDELT zero-fills tone and volume on uncovered days) produced
    # 'Price leads the news … r = +0.44, p < 0.001, n = 58'.
    rng = random.Random(7)
    days = calendar(90)
    covered = {days[20], days[45], days[70]}
    tone = ToneTrend(query="q", series=[TonePoint(date=d, tone=(rng.uniform(-3, 3) if d in covered else 0.0),
                                                  volume=(1.0 if d in covered else 0.0)) for d in days])
    closes = [(d, 10.0 + rng.uniform(-1, 1)) for d in days if d.weekday() < 5]
    h = build_history("KOSS", 90, tone, closes, [], None, {"tone": "ok"})
    assert h.lags == [] and h.best_lag is None
    assert h.interpretation.startswith("Coverage too sparse (3 articles in 90 days")
    # Zero-article days never enter the aligned series as tone.
    from app.analytics.stats import aligned_series
    tones, _ = aligned_series({p.date: p for p in tone.series}, dict(closes))
    assert sum(t is not None for t in tones) <= 3
