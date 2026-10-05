"""Tone ↔ price history and the lead/lag question: "does news tone lead this stock?".

Alignment is on trading days: each trading day's return (close vs previous
close) is paired with the mean GDELT tone of the calendar days since the
previous close (weekend news counts toward Monday). For lag k:

    r(k) = corr(tone[t], return[t + k])     k > 0: tone leads returns by k trading days

p-values are two-sided, from Student's t with n − 2 degrees of freedom
(regularized incomplete beta, no SciPy). Seven lags are tested, so a link is
only called reliable when it survives a Bonferroni correction (p × 7 < 0.05)
with |r| >= 0.2.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from app.analytics.util import signed
from app.schemas import HistoryPoint, HistoryResponse, LagStat, Snapshot, ToneTrend

LAGS = range(-3, 4)
MIN_PAIRS = 10  # below this a lag is not reported
MIN_RELIABLE_PAIRS = 20
ALPHA = 0.05
MIN_R = 0.2


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_pvalue(t: float, df: float) -> float:
    """Two-sided p-value of Student's t."""
    if df <= 0 or math.isnan(t):
        return 1.0
    return min(1.0, max(0.0, betainc(df / 2.0, 0.5, df / (df + t * t))))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / math.sqrt(sxx * syy)


def corr_pvalue(r: float, n: int) -> float:
    """Two-sided p-value for H0: no correlation."""
    if n < 3:
        return 1.0
    if abs(r) >= 1.0:
        return 0.0
    return t_pvalue(r * math.sqrt((n - 2) / (1.0 - r * r)), n - 2)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def build_history(ticker: str, days: int, tone: ToneTrend | None, closes: list[tuple[date, float]],
                  snapshots: list[Snapshot], wiki: list[tuple[date, float]] | None,
                  status: dict[str, str]) -> HistoryResponse:
    """Aligned daily series over the last `days` days plus tone→return lead/lag stats."""
    days = max(1, int(days))
    tone_by_day = {p.date: p for p in (tone.series if tone else [])}
    close_by_day = {d: float(c) for d, c in closes if c is not None and c > 0}
    wiki_by_day = {d: float(v) for d, v in (wiki or []) if v is not None}
    snap_by_day: dict[date, int] = {}
    for snap in snapshots:  # oldest first: the day's last snapshot wins
        snap_by_day[snap.at.date()] = snap.sentinel_score

    all_days = set(tone_by_day) | set(close_by_day) | set(wiki_by_day) | set(snap_by_day)
    if not all_days:
        return HistoryResponse(ticker=ticker, days=days, interpretation="No history available for this ticker yet.",
                               status=dict(status))
    end = max(all_days)
    start = end - timedelta(days=days - 1)
    returns = daily_returns(close_by_day)

    points = []
    for d in sorted(x for x in all_days if start <= x <= end):
        tp = tone_by_day.get(d)
        points.append(HistoryPoint(
            date=d, tone=tp.tone if tp else None, volume=tp.volume if tp else None,
            close=close_by_day.get(d), ret_pct=returns.get(d), snapshot_score=snap_by_day.get(d),
            wiki_views=wiki_by_day.get(d),
        ))

    window_closes = {d: c for d, c in close_by_day.items() if start - timedelta(days=7) <= d <= end}
    tone_series = {d: p for d, p in tone_by_day.items() if start <= d <= end}
    lags = lag_stats(tone_series, window_closes)
    best = best_lag(lags)
    text = interpret(lags, best, has_tone=bool(tone_series), has_price=len(window_closes) >= 2)
    return HistoryResponse(ticker=ticker, days=days, points=points, lags=lags, best_lag=best,
                           interpretation=text, status=dict(status))


def daily_returns(closes: dict[date, float]) -> dict[date, float]:
    """Percent return of each trading day vs the previous available close."""
    out: dict[date, float] = {}
    ordered = sorted(closes)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        out[cur] = round((closes[cur] / closes[prev] - 1.0) * 100.0, 4)
    return out


def aligned_series(tone: dict, closes: dict[date, float]) -> tuple[list[float | None], list[float]]:
    """(tone per trading day, return per trading day) on the trading-day calendar.

    A trading day's tone is the (volume-weighted) mean tone of the calendar days
    after the previous close up to and including that day."""
    days = sorted(closes)
    tones: list[float | None] = []
    rets: list[float] = []
    for prev, cur in zip(days, days[1:], strict=False):
        rets.append((closes[cur] / closes[prev] - 1.0) * 100.0)
        acc = wsum = 0.0
        d = prev + timedelta(days=1)
        while d <= cur:
            p = tone.get(d)
            if p is not None and p.tone is not None:
                w = p.volume if p.volume and p.volume > 0 else 1.0
                acc += p.tone * w
                wsum += w
            d += timedelta(days=1)
        tones.append(acc / wsum if wsum else None)
    return tones, rets


def lag_stats(tone: dict, closes: dict[date, float]) -> list[LagStat]:
    tones, rets = aligned_series(tone, closes)
    out: list[LagStat] = []
    for k in LAGS:
        xs, ys = [], []
        for i, t in enumerate(tones):
            j = i + k
            if t is not None and 0 <= j < len(rets):
                xs.append(t)
                ys.append(rets[j])
        if len(xs) < MIN_PAIRS:
            continue
        r = pearson(xs, ys)
        if r is None:
            continue
        out.append(LagStat(lag_days=k, r=round(r, 4), n=len(xs), p_value=round(corr_pvalue(r, len(xs)), 4)))
    return out


def best_lag(lags: list[LagStat]) -> LagStat | None:
    """Most significant lag (enough pairs preferred), ties broken by |r|."""
    if not lags:
        return None
    pool = [s for s in lags if s.n >= MIN_RELIABLE_PAIRS] or lags
    return min(pool, key=lambda s: (s.p_value, -abs(s.r), abs(s.lag_days)))


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def _lag_meaning(k: int) -> str:
    if k == 0:
        return "same day"
    return f"tone leading price by {_days(k)}" if k > 0 else f"price leading tone by {_days(k)}"


def _days(k: int) -> str:
    k = abs(k)
    return f"{k} trading day{'s' if k != 1 else ''}"


def interpret(lags: list[LagStat], best: LagStat | None, has_tone: bool, has_price: bool) -> str:
    """Plain-English reading that respects significance."""
    if not has_tone:
        return "No GDELT tone history for this period, so the tone ↔ price link can't be measured."
    if not has_price:
        return "No price history for this period, so the tone ↔ price link can't be measured."
    if best is None:
        return (f"Too few overlapping trading days (need {MIN_PAIRS}+) to measure a tone ↔ price relationship.")
    stats = f"r = {signed(best.r)}, {_p(best.p_value)}, n = {best.n}"
    reliable = (best.n >= MIN_RELIABLE_PAIRS and abs(best.r) >= MIN_R
                and best.p_value * len(LAGS) < ALPHA)
    if not reliable:
        where = f"lag {best.lag_days:+d}, {_lag_meaning(best.lag_days)}"
        if best.p_value < ALPHA and abs(best.r) >= 0.15:
            return (f"Only a weak hint: the strongest link ({where}) has {stats}, but it does not survive testing "
                    f"{len(LAGS)} lags — treat it as noise.")
        return (f"No reliable relationship: the strongest link is r = {signed(best.r)} ({where}; "
                f"{_p(best.p_value)}, n = {best.n}).")
    k, positive = best.lag_days, best.r > 0
    if k > 0:
        if positive:
            return (f"News tone tends to lead this stock: more positive global coverage has been followed by "
                    f"higher returns {_days(k)} later ({stats}).")
        return (f"News tone leads inversely: more positive coverage has been followed by weaker returns "
                f"{_days(k)} later ({stats}) — a contrarian pattern.")
    if k < 0:
        if positive:
            return (f"Price leads the news: gains have been followed by more positive coverage {_days(k)} later "
                    f"({stats}) — the coverage chases the stock.")
        return (f"Price leads the news inversely: gains have been followed by more negative coverage {_days(k)} "
                f"later ({stats}).")
    if positive:
        return (f"Tone and price move together on the same day ({stats}): coverage mirrors the day's move "
                f"rather than predicting it.")
    return f"Tone and same-day returns move in opposite directions ({stats})."
