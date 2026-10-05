"""Pure Yahoo transforms, tested on real recorded yfinance payloads."""
from __future__ import annotations

import math
from datetime import date, datetime, UTC
from itertools import pairwise

import pandas as pd
import pytest

from app.intel import transforms as tx
from tests.intel import helpers as fx

NOW = datetime(2026, 10, 4, 22, 0, tzinfo=UTC)  # fixtures were captured on Sunday 2026-10-04
TODAY = NOW.date()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_coercions() -> None:
    assert tx.num("1.5") == 1.5 and tx.num(float("nan")) is None and tx.num(None) is None and tx.num(True) is None
    assert tx.pos(0) is None and tx.pos(-1) is None and tx.pos(3) == 3.0
    assert tx.to_date(1790812800) == date(2026, 10, 1)
    assert tx.to_date("2026-11-17") == date(2026, 11, 17)
    assert tx.to_date(pd.Timestamp("2026-11-17 15:00", tz="America/New_York")) == date(2026, 11, 17)
    assert tx.to_date(pd.NaT) is None and tx.to_date("") is None and tx.to_date(None) is None
    assert tx.to_utc("2026-10-01 10:34:36").tzinfo is not None
    assert tx.date_noon_utc(date(2026, 10, 6)).hour == 12
    assert tx.pct(110, 100) == 10.0 and tx.pct(1, 0) is None and tx.pct(None, 1) is None


# --------------------------------------------------------------------------- #
# quote / profile / candles
# --------------------------------------------------------------------------- #
def test_quote_from_info_nvda() -> None:
    q = tx.quote_from_info(fx.info("NVDA"))
    assert q is not None
    assert q.price == 233.95 and q.previous_close == 230.86
    assert q.change_pct == pytest.approx(1.338, abs=1e-3)
    assert q.currency == "USD" and q.market_cap and q.market_cap > 1e12
    assert q.as_of is not None and q.as_of.tzinfo is not None
    assert q.year_high and q.year_low and q.year_low < q.price <= q.year_high * 1.01


def test_quote_from_info_crypto_and_missing() -> None:
    q = tx.quote_from_info(fx.info("BTC-USD"))
    assert q is not None and q.price and q.price > 1000
    assert tx.quote_from_info({}) is None
    assert tx.quote_from_info({"regularMarketPrice": 0}) is None


def test_quote_from_history_fallback() -> None:
    q = tx.quote_from_history(fx.bars("NVDA"), "USD")
    df = fx.bars("NVDA")
    assert q is not None and q.price == pytest.approx(df["Close"].iloc[-1])
    assert q.change_pct == pytest.approx((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100, abs=0.01)


def test_profile_from_info_trims_summary() -> None:
    p = tx.profile_from_info(fx.info("AAPL"), symbol="AAPL", name="Apple Inc.", short_name="Apple",
                             quote_type="EQUITY", exchange="NASDAQ", cik="0000320193", logo_url=None)
    assert p.sector == "Technology" and p.employees and p.employees > 100_000
    assert p.summary and len(p.summary) <= 901 and p.summary.endswith(".")
    assert p.financial_currency == "USD"


@pytest.mark.parametrize(("sym", "quote_ccy", "reporting"), [
    ("SHOP.TO", "CAD", "USD"),  # TSX line of a company that reports in dollars
    ("VOD.L", "GBp", "EUR"),  # London pence quote, euro reporting
    ("7203.T", "JPY", "JPY"),
    ("BTC-USD", "USD", None),  # coins (and funds) have no reporting currency
])
def test_profile_reporting_currency(sym: str, quote_ccy: str, reporting: str | None) -> None:
    info = fx.info(sym)
    p = tx.profile_from_info(info, symbol=sym, name=sym, short_name=None, quote_type=None, exchange=None,
                             cik=None, logo_url=None)
    assert info["currency"] == quote_ccy and p.financial_currency == reporting


def test_iso_currency() -> None:
    assert tx.iso_currency("usd") == "USD" and tx.iso_currency(" EUR ") == "EUR" and tx.iso_currency("GBp") == "GBP"
    assert tx.iso_currency(None) is None and tx.iso_currency("") is None and tx.iso_currency("US$") is None
    assert tx.iso_currency(840) is None and tx.iso_currency("EURO") is None


def test_candles_intraday_and_crypto() -> None:
    candles = tx.candles_from_history(fx.bars("NVDA", "5m"))
    assert len(candles) >= 70
    assert all(c.t.tzinfo == UTC for c in candles)
    assert all(c.l <= min(c.o, c.c) + 1e-6 and c.h >= max(c.o, c.c) - 1e-6 for c in candles)
    assert [c.t for c in candles] == sorted(c.t for c in candles)
    assert tx.candles_from_history(None) == [] and tx.candles_from_history(pd.DataFrame()) == []


def test_closes_use_exchange_session_dates() -> None:
    closes = tx.closes_from_history(fx.bars("NVDA"))
    assert closes[-1][0] == date(2026, 10, 2)  # NY session date, not the UTC timestamp's date
    assert all(a[0] < b[0] for a, b in pairwise(closes))


# --------------------------------------------------------------------------- #
# technicals
# --------------------------------------------------------------------------- #
def test_wilder_rsi_reference_series() -> None:
    # StockCharts' published RSI example: first 14-period RSI ≈ 70.5
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    assert tx.wilder_rsi(closes) == pytest.approx(70.5, abs=0.3)
    assert tx.wilder_rsi([1.0] * 5) is None
    assert tx.wilder_rsi([float(i) for i in range(30)]) == 100.0


def test_technicals_match_independent_pandas_math() -> None:
    df = fx.bars("NVDA")
    t = tx.technicals_from_history(df, now=NOW)
    assert t is not None
    close = df["Close"]
    last = close.iloc[-1]
    assert t.return_1d == pytest.approx((last / close.iloc[-2] - 1) * 100, abs=0.01)
    assert t.return_5d == pytest.approx((last / close.iloc[-6] - 1) * 100, abs=0.01)
    assert t.vs_50dma_pct == pytest.approx((last / close.iloc[-50:].mean() - 1) * 100, abs=0.01)
    assert t.vs_200dma_pct == pytest.approx((last / close.iloc[-200:].mean() - 1) * 100, abs=0.01)
    rets = (close.iloc[-31:] / close.iloc[-31:].shift(1)).dropna().map(math.log)
    assert t.volatility_30d == pytest.approx(rets.std() * math.sqrt(252) * 100, abs=0.1)
    high = df["High"][df.index > df.index[-1] - pd.DateOffset(years=1)].max()
    assert t.pct_from_52w_high == pytest.approx((last / high - 1) * 100, abs=0.01)
    prior = close[[ts.year < 2026 for ts in close.index]].iloc[-1]
    assert t.return_ytd == pytest.approx((last / prior - 1) * 100, abs=0.01)
    vol = df["Volume"]
    assert t.volume_ratio == pytest.approx(vol.iloc[-1] / vol.iloc[-64:-1].mean(), abs=0.01)
    assert 0 <= (t.rsi_14 or -1) <= 100
    assert t.trend in {"uptrend", "downtrend", "sideways"}


def test_technicals_skip_partial_session_volume() -> None:
    df = fx.bars("NVDA")
    during = datetime(2026, 10, 2, 18, 0, tzinfo=UTC)  # Fri 14:00 New York, market open
    t = tx.technicals_from_history(df, now=during)
    vol = df["Volume"]
    assert t is not None and t.volume_ratio == pytest.approx(vol.iloc[-2] / vol.iloc[-65:-2].mean(), abs=0.01)


def test_technicals_crypto_annualizes_with_365_days() -> None:
    df = fx.bars("BTC-USD")
    t_crypto = tx.technicals_from_history(df, now=NOW, is_crypto=True)
    t_equity = tx.technicals_from_history(df, now=NOW, is_crypto=False)
    assert t_crypto and t_equity and t_crypto.volatility_30d and t_equity.volatility_30d
    assert t_crypto.volatility_30d / t_equity.volatility_30d == pytest.approx(math.sqrt(365 / 252), rel=0.01)


def _closes_frame(closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range(end="2026-10-02", periods=len(closes), tz="America/New_York")
    return pd.DataFrame({"Close": closes, "High": closes, "Volume": [1e6] * len(closes)}, index=idx)


def test_breakout_above_both_averages_is_an_uptrend() -> None:
    """GME 2026-10 (+30% 1M, +20% vs 50-DMA, +11% vs 200-DMA, 50 < 200) was labelled "sideways"."""
    closes = [30 - 0.05 * i for i in range(230)] + [18.5 + 0.35 * i for i in range(22)]  # long slide, sharp rally
    t = tx.technicals_from_history(_closes_frame(closes), now=NOW)
    assert t is not None and t.vs_50dma_pct and t.vs_50dma_pct > 10 and t.vs_200dma_pct and t.vs_200dma_pct > 0
    sma50, sma200 = sum(closes[-50:]) / 50, sum(closes[-200:]) / 200
    assert sma50 < sma200 and t.trend == "uptrend"
    # Mirror image: a stock that broke below both averages while the 50 still sits over the 200.
    t_down = tx.technicals_from_history(_closes_frame([10 + 0.05 * i for i in range(230)]
                                                     + [21.5 - 0.35 * i for i in range(22)]), now=NOW)
    assert t_down is not None and t_down.trend == "downtrend"


@pytest.mark.parametrize(("last", "sma20", "sma50", "sma200", "ret_1m", "vol", "expected"), [
    (120, 112, 100, 90, 3.0, 30.0, "uptrend"),  # fully aligned
    (80, 88, 100, 110, -3.0, 30.0, "downtrend"),
    (120.4, 104, 100, 108, 30.2, 39.8, "uptrend"),  # GME: above both, 50 still < 200
    (116.6, 104, 100, 100.5, 22.9, 46.3, "uptrend"),  # META
    (101.5, 100, 100, 99, 1.0, 30.0, "uptrend"),  # aligned: a quiet stretch inside an uptrend
    (101.5, 100, 100, 101, 1.0, 30.0, "sideways"),  # above both but hugging the 50-DMA, quiet month
    (85, 95, 100, 90, -12.0, 30.0, "downtrend"),  # below both, 50 still > 200: rolling over
    (106, 101, 100, 110, 10.3, 51.7, "sideways"),  # KOSS: between the averages, sub-1σ month
    (106, 101, 100, 110, 16.0, 51.7, "uptrend"),  # ... unless the month moved it decisively
    (106, 101, 100, 110, -16.0, 51.7, "sideways"),  # ... the same way as the price sits
    (94, 99, 100, 90, -6.0, 20.0, "downtrend"),  # a dip above the 200-DMA on a ≥1σ month
    (105, 103, 100, None, None, None, "uptrend"),  # < 200 bars: 20 vs 50 stands in
    (105, 98, 100, None, None, None, "sideways"),
])
def test_trend_label(last: float, sma20: float, sma50: float, sma200: float | None, ret_1m: float | None,
                     vol: float | None, expected: str) -> None:
    assert tx.trend_label(last, sma20=sma20, sma50=sma50, sma200=sma200, return_1m=ret_1m,
                          volatility=vol) == expected


def test_technicals_short_history() -> None:
    df = fx.bars("NVDA").tail(30)
    t = tx.technicals_from_history(df, now=NOW)
    assert t is not None and t.vs_200dma_pct is None and t.vs_50dma_pct is None and t.trend is None
    assert t.return_1d is not None and t.return_1m is not None
    assert tx.technicals_from_history(df.tail(1), now=NOW) is None


# --------------------------------------------------------------------------- #
# analysts
# --------------------------------------------------------------------------- #
def test_analysts_nvda() -> None:
    view = tx.analysts_from_frames(fx.info("NVDA"), fx.recommendations("NVDA"), fx.upgrades("NVDA"),
                                   price=233.95, now=NOW)
    assert view is not None
    # From the counts the view shows (10/48/2/1/0), not Yahoo's recommendationMean 1.30 (another panel).
    assert view.consensus == "buy" and view.mean_rating == pytest.approx(1.90, abs=0.01)
    assert view.counts and view.counts.period == "0m"
    assert view.total == view.counts.strong_buy + view.counts.buy + view.counts.hold + view.counts.sell + view.counts.strong_sell
    assert view.target_mean == 327.7 and view.upside_pct == pytest.approx(40.07, abs=0.01)
    assert view.target_low <= view.target_median <= view.target_high
    assert 0 < len(view.actions) <= 25
    assert [a.date for a in view.actions] == sorted((a.date for a in view.actions), reverse=True)
    assert [rc.period for rc in view.trend] == ["0m", "-1m", "-2m", "-3m"]
    init = next(a for a in view.actions if a.action == "init")
    assert init.prior_target is None and init.from_grade is None


def test_analyst_revision_counting_windows() -> None:
    frame = pd.DataFrame(
        {
            "Firm": ["A", "B", "C", "D", "E", "F", "F"],
            "ToGrade": ["Buy", "Sell", "Buy", "Buy", "Hold", "Buy", "Buy"],
            "FromGrade": ["Hold", "Hold", "Buy", "Buy", "Buy", "", ""],
            "Action": ["up", "down", "main", "main", "down", "init", "init"],
            "priceTargetAction": ["Raises", "Lowers", "Raises", "", "Lowers", "Announces", "Announces"],
            "currentPriceTarget": [120.0, 80.0, 130.0, 95.0, 70.0, 100.0, 100.0],
            "priorPriceTarget": [100.0, 90.0, 110.0, 100.0, 90.0, 0.0, 0.0],
        },
        index=pd.DatetimeIndex(pd.to_datetime([
            "2026-10-01 12:00", "2026-09-20 12:00", "2026-09-15 12:00", "2026-09-10 12:00",
            "2026-07-10 12:00", "2026-09-30 12:00", "2026-09-30 12:00",  # last row: exact duplicate
        ]), name="GradeDate"),
    )
    view = tx.analysts_from_frames({"targetMeanPrice": 110.0}, None, frame, price=100.0, now=NOW)
    assert view is not None
    assert view.upgrades_90d == 1 and view.downgrades_90d == 2  # Jul 10 is within 90 days of Oct 4
    assert view.pt_raises_30d == 2  # A (label), C (label)
    assert view.pt_cuts_30d == 2  # B (label), D (numbers: 100 -> 95, no label)
    assert len(view.actions) == 6  # duplicate init collapsed
    assert view.upside_pct == 10.0 and view.consensus is None


def test_consensus_agrees_with_the_counts_shown() -> None:
    """TGT: Yahoo's mean 2.47 says "buy" while 21 of 38 counted analysts say hold (4 strong sell)."""
    view = tx.analysts_from_frames(fx.info("TGT"), fx.recommendations("TGT"), fx.upgrades("TGT"), price=156.0, now=NOW)
    assert view is not None and view.counts is not None
    assert (view.counts.hold, view.total) == (21, 38)
    assert view.consensus == "hold" and view.mean_rating == pytest.approx(2.82, abs=0.01)
    # AAPL's recorded feed has rows without a priceTargetAction label (NaN): counted from the numbers.
    aapl = tx.analysts_from_frames(fx.info("AAPL"), fx.recommendations("AAPL"), fx.upgrades("AAPL"), price=None, now=NOW)
    assert aapl is not None and aapl.pt_raises_30d >= 1
    # Too few counted analysts: the provider's mean is the better estimate.
    thin = pd.DataFrame([{"period": "0m", "strongBuy": 1, "buy": 1, "hold": 0, "sell": 0, "strongSell": 0}])
    view = tx.analysts_from_frames({"recommendationMean": 2.1, "targetMeanPrice": 10.0}, thin, None, price=9.0, now=NOW)
    assert view is not None and view.mean_rating == 2.1 and view.consensus == "buy" and view.total == 2


def test_frozen_upgrade_feed_reads_as_unknown_not_zero() -> None:
    """META's real feed stopped at 2024-09-30: two-year-old reiterations are not "recent actions"."""
    raw = fx.upgrades("META")
    assert raw.index.max() < pd.Timestamp("2024-10-01")
    view = tx.analysts_from_frames(fx.info("META"), fx.recommendations("META"), raw, price=728.08, now=NOW)
    assert view is not None and view.actions == []
    assert view.upgrades_90d == view.pt_raises_30d == 0 and view.consensus == "buy" and view.total == 63
    # A live feed keeps actions up to a year old, nothing older.
    nvda = tx.analysts_from_frames(fx.info("NVDA"), fx.recommendations("NVDA"), fx.upgrades("NVDA"), price=233.95,
                                   now=NOW)
    assert nvda is not None and nvda.actions and all(NOW - a.date <= tx.ACTION_MAX_AGE for a in nvda.actions)


def test_analysts_none_without_coverage() -> None:
    assert tx.analysts_from_frames({}, None, None, price=10.0, now=NOW) is None
    assert tx.consensus_for(1.2) == "strong_buy" and tx.consensus_for(2.6) == "hold" and tx.consensus_for(4.8) == "strong_sell"


# --------------------------------------------------------------------------- #
# earnings & dividends
# --------------------------------------------------------------------------- #
def test_earnings_nvda() -> None:
    view = tx.earnings_from_frames(fx.calendar("NVDA"), fx.earnings_dates("NVDA"), fx.info("NVDA"), today=TODAY)
    assert view is not None
    assert view.next_date == date(2026, 11, 17) and view.days_until == 44
    assert view.eps_low <= view.eps_estimate <= view.eps_high
    assert view.revenue_estimate and view.revenue_estimate > 1e10
    assert len(view.history) == 8 and view.history[0].date > view.history[-1].date
    assert all(e.eps_actual is not None for e in view.history)
    beats = sum(e.eps_actual > e.eps_estimate for e in view.history)
    assert view.beat_rate == pytest.approx(beats / 8, abs=1e-3)


def test_earnings_sofi_beat_rate_and_no_stale_next_date() -> None:
    view = tx.earnings_from_frames(fx.calendar("SOFI"), fx.earnings_dates("SOFI"), fx.info("SOFI"), today=TODAY)
    assert view is not None and view.next_date and view.next_date >= TODAY
    later = tx.earnings_from_frames(fx.calendar("SOFI"), fx.earnings_dates("SOFI"), {}, today=date(2027, 6, 1))
    assert later is not None and later.next_date is None and later.eps_estimate is None  # past date never shown as "next"


def test_earnings_history_fallback_table() -> None:
    hist = pd.DataFrame(
        {"epsActual": [1.30, 1.62], "epsEstimate": [1.2565, 1.6381], "surprisePercent": [0.0346, -0.011]},
        index=pd.DatetimeIndex(["2025-10-31", "2026-01-31"], name="quarter"),
    )
    view = tx.earnings_from_frames({}, None, {}, today=TODAY, earnings_history=hist)
    assert view is not None and view.beat_rate == 0.5 and view.history[0].surprise_pct == pytest.approx(-1.1)


def test_dividend_catalysts() -> None:
    jpm = tx.dividend_catalysts(fx.calendar("JPM"), fx.info("JPM"), today=TODAY)
    assert jpm and jpm[0].kind == "dividend" and jpm[0].upcoming
    assert jpm[0].date.date() >= TODAY and "Ex-dividend" in jpm[0].title
    assert "yield" in (jpm[0].detail or "")
    nvda = tx.dividend_catalysts(fx.calendar("NVDA"), fx.info("NVDA"), today=TODAY)
    assert all(c.date.date() >= TODAY for c in nvda)  # past ex-date (Sep 10) never reported as upcoming
    assert tx.dividend_catalysts({}, {}, today=TODAY) == []


def test_dividend_detail_states_facts_in_the_quote_currency() -> None:
    # JPM: Yahoo's per-share value ($1.50) is July's dividend, not October's (forward $6.60/yr implies $1.65).
    jpm = tx.dividend_catalysts(fx.calendar("JPM"), fx.info("JPM"), today=TODAY)
    assert [c.title for c in jpm] == ["Ex-dividend date in 2 days", "Dividend payment"]
    assert all(c.detail == "last $1.50/share · $6.60/yr · yield 1.99%" for c in jpm)
    # Toyota: yen, never "$"; no trading advice appended.
    toyota = tx.dividend_catalysts(fx.calendar("7203.T"), fx.info("7203.T"), today=TODAY)
    assert len(toyota) == 1 and toyota[0].detail == "¥50/share · ¥100/yr · yield 3.50%"
    # Vodafone quotes in pence; Yahoo's amounts are pounds (4p/yr is what squares with the 3.14% yield).
    vod = tx.dividend_catalysts(fx.calendar("VOD.L"), fx.info("VOD.L"), today=date(2026, 5, 20))
    assert len(vod) == 1 and vod[0].detail == "2.36p/share · 4p/yr · yield 3.14%"
    # NVDA after its Sep 10 ex-date: the Oct 1 payment settles that same dividend, so no "last".
    nvda = tx.dividend_catalysts(fx.calendar("NVDA"), fx.info("NVDA"), today=date(2026, 9, 20))
    assert [(c.title, c.detail) for c in nvda] == [("Dividend payment", "$0.25/share · $1.00/yr · yield 0.43%")]
    for c in [*jpm, *toyota, *vod, *nvda]:
        assert "buy before" not in (c.detail or "").lower()
    # Unknown currency: amounts without a symbol; no amounts at all: no detail rather than filler.
    bare = tx.dividend_catalysts({"Ex-Dividend Date": date(2026, 10, 9)},
                                 {"lastDividendValue": 0.5, "lastDividendDate": "2026-10-09"}, today=TODAY)
    assert bare[0].detail == "0.50/share"
    assert tx.dividend_catalysts({"Ex-Dividend Date": date(2026, 10, 9)}, {}, today=TODAY)[0].detail is None


@pytest.mark.parametrize(("value", "currency", "text"), [
    (1.5, "USD", "$1.50"), (0.272, "USD", "$0.272"), (0.0832, "USD", "$0.0832"), (1234.5, "usd", "$1,234.50"),
    (1.76, "CAD", "C$1.76"), (1.88, "EUR", "€1.88"), (3.1, "CHF", "3.10 CHF"), (22.5, "JPY", "¥22.5"),
    (0.023625, "GBp", "2.36p"), (0.04, "GBX", "4p"), (9.02, "ZAc", "902c"), (0.25, None, "0.25"),
])
def test_cash_per_share(value: float, currency: str | None, text: str) -> None:
    assert tx.cash_per_share(value, currency) == text


# --------------------------------------------------------------------------- #
# insiders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Purchase at price 18.06 per share.", "buy"),
        ("Sale at price 222.19 - 223.75 per share.", "sell"),
        ("Stock Award(Grant) at price 0.00 per share.", "award"),
        ("Conversion of Exercise of derivative security at price 3.50 per share.", "exercise"),
        ("Stock Gift at price 0.00 per share.", "gift"),
        ("", "other"),
        # UK PDMR notices (VOD.L, BARC.L)
        ("Bought at price 1.58 per share.", "buy"),
        ("Sold at price 1.70 per share.", "sell"),
        ("Buy Back at price 1.27 per share.", "other"),  # the company's own buyback
        ("Decrease at price 6.10 per share.", "other"),
        # Canada SEDI (SHOP.TO, RY.TO)
        ("Acquisition in the public market at price 20.10 per share.", "buy"),
        ("Disposition in the public market at price 138.31 per share.", "sell"),
        ("Disposition under a purchase/ownership plan at price 149.72 per share.", "sell"),  # ASDP ~ 10b5-1
        ("Acquisition under a purchase/ownership plan at price 120.00 per share.", "other"),  # ESPP
        ("Redemption, retraction, cancelation, repurchase at price 199.70 per share.", "other"),
        ("Disposition carried out privately at price 50.00 per share.", "other"),
        ("Exercise of options at price 3.12 per share.", "exercise"),
        ("Grant of options", "award"),
    ],
)
def test_classify_insider(text: str, kind: str) -> None:
    assert tx.classify_insider(text) == kind


@pytest.mark.parametrize(
    ("raw", "pretty"),
    [
        ("TETER TIMOTHY S", "Timothy S. Teter"),
        ("NOTO ANTHONY J.", "Anthony J. Noto"),
        ("O'BRIEN DEIRDRE", "Deirdre O'Brien"),
        ("KEOUGH KELLI ALLEN", "Kelli Allen Keough"),
        ("SMITH JOHN JR", "John Smith Jr."),
        ("FORD HENRY III", "Henry Ford III"),
        ("HELMAN WILLIAM W IV", "William W. Helman IV"),
        ("NORA JOHNSON SUZANNE M", "Nora Johnson Suzanne M."),  # ambiguous: order kept
        ("BERKSHIRE HATHAWAY INC", "Berkshire Hathaway Inc"),
        ("Jensen Huang", "Jensen Huang"),
        ("Lutke (Tobias Albin)", "Tobias Albin Lutke"),  # SEDI / UK "Last (First)"
        ("van Boxmeer (Jean-Francois M.L.)", "Jean-Francois M.L. van Boxmeer"),
        ("Soci\ufffdt\ufffd G\ufffdn\ufffdrale S.A", "Société Générale S.A"),  # Yahoo lost the accents
        ("M\ufffdller (J\ufffdrg)", "Jörg Müller"),
        ("Xq\ufffdz Holdings Plc", "Xqz Holdings Plc"),  # unknown word: the marks are dropped, not guessed
        ("Vodafone Group Plc", "Vodafone Group Plc"),
    ],
)
def test_pretty_insider_name(raw: str, pretty: str) -> None:
    assert tx.pretty_insider_name(raw) == pretty


def test_insiders_sofi_ceo_buying() -> None:
    df = fx.insiders("SOFI")
    view = tx.insiders_from_frame(df, now=NOW)
    assert view is not None
    since = pd.Timestamp(NOW.date()) - pd.Timedelta(days=180)
    recent = df[df["Start Date"] >= since]
    buys = recent[recent["Text"].str.startswith("Purchase")]
    sells = recent[recent["Text"].str.startswith("Sale")]
    assert view.buys == len(buys) and view.buy_value == pytest.approx(buys["Value"].sum())
    assert view.sells == len(sells) and view.sell_value == pytest.approx(sells["Value"].sum())
    assert view.ratio == pytest.approx((view.buy_value - view.sell_value) / (view.buy_value + view.sell_value), abs=1e-3)
    assert view.buys >= 1 and any(t.kind == "buy" and "Noto" in t.insider for t in view.transactions)
    assert len(view.transactions) <= 25
    assert [t.date for t in view.transactions] == sorted((t.date for t in view.transactions), reverse=True)
    trades = [t for t in view.transactions if t.kind in {"buy", "sell"} and t.date >= since.date()]
    assert len(trades) == view.buys + view.sells or len(view.transactions) == 25  # trades never crowded out


def _window(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Start Date"] >= pd.Timestamp(NOW.date()) - pd.Timedelta(days=180)]


def test_insiders_uk_pdmr_wording() -> None:
    """VOD.L showed "no open-market insider trades in 180d" while executives traded (Yahoo, 2026-10-05)."""
    df = fx.insiders("VOD.L")
    view = tx.insiders_from_frame(df, now=NOW)
    assert view is not None and view.buys >= 3 and view.sells >= 5
    reiter = next(t for t in view.transactions if t.kind == "sell" and t.insider == "Joakim Reiter")
    assert reiter.date == date(2026, 9, 18) and reiter.value == 850_500 and reiter.shares == 500_000
    recent = _window(df)
    sold = recent[recent["Text"].str.startswith("Sold")]
    assert view.sells == len(sold) and view.sell_value == pytest.approx(sold["Value"].sum())
    # "Buy Back" is Vodafone buying its own shares, never an insider purchase.
    assert all(t.kind == "other" for t in view.transactions if t.insider == "Vodafone Group Plc")
    assert all("\ufffd" not in t.insider for t in view.transactions)
    assert any(t.insider == "Société Générale S.A" for t in view.transactions)


def test_insiders_canada_sedi_wording() -> None:
    shop = tx.insiders_from_frame(fx.insiders("SHOP.TO"), now=NOW)
    assert shop is not None and shop.sells >= 20 and shop.buys == 0
    lutke = [t for t in shop.transactions if t.insider == "Tobias Albin Lutke" and t.date == date(2026, 9, 30)]
    assert lutke and all(t.kind == "sell" and "plan" in (t.text or "") for t in lutke)  # pre-arranged plan sale
    ry = tx.insiders_from_frame(fx.insiders("RY.TO"), now=NOW)
    assert ry is not None and ry.sells >= 20
    # The bank's own redemptions/repurchases (200,000 shares a day) are not insider selling.
    assert all(t.kind == "other" for t in ry.transactions if t.insider == "Royal Bank of Canada")
    recent = _window(fx.insiders("RY.TO"))
    public = recent[recent["Text"].str.startswith("Disposition in the public market")]
    assert ry.sells == len(public) and ry.sell_value == pytest.approx(public["Value"].sum())


def test_same_day_same_price_purchases_are_a_plan_not_a_cluster_buy() -> None:
    """BARC.L: six insiders "bought" ~100 shares each at 6.31 on 2026-09-22 (share plan / DRIP)."""
    view = tx.insiders_from_frame(fx.insiders("BARC.L"), now=NOW)
    assert view is not None
    plan = [t for t in view.transactions if t.date == date(2026, 9, 22)]
    assert len(plan) == 7 and all(t.kind == "other" and "plan purchase: 6 insiders" in (t.text or "") for t in plan)
    lone = [t for t in view.transactions if t.kind == "buy"]
    assert [(t.date, t.insider) for t in lone][:1] == [(date(2026, 9, 11), "Craig Bright")]
    assert view.buys == len(lone) == 3 and view.sells >= 10

    def buy(who: str, price: float, day: date = date(2026, 9, 1)) -> tx.InsiderTxn:
        return tx.InsiderTxn(date=day, insider=who, kind="buy", shares=100, value=100 * price,
                             text=f"Purchase at price {price:.2f} per share.")

    # Two insiders at one price, or three at different prices, are still individual decisions.
    rows = [buy("A", 10.0), buy("B", 10.0), buy("C", 10.07), buy("D", 9.95, date(2026, 9, 2))]
    assert tx.insider_view(rows, since=date(2026, 4, 1), window_days=180).buys == 4  # type: ignore[union-attr]
    rows.append(buy("E", 10.0))
    view = tx.insider_view(rows, since=date(2026, 4, 1), window_days=180)
    assert view is not None and view.buys == 2 and {t.insider for t in view.transactions if t.kind == "buy"} == {"C", "D"}


def test_insiders_nvda_values_and_indirect_flag() -> None:
    view = tx.insiders_from_frame(fx.insiders("NVDA"), now=NOW)
    assert view is not None and view.sells > 0 and view.buys == 0 and view.ratio == -1.0
    assert any("(indirect)" in (t.text or "") for t in view.transactions)
    assert all(t.value is None for t in view.transactions if t.kind in {"award", "gift"})
    assert tx.insiders_from_frame(pd.DataFrame(), now=NOW) is None


# --------------------------------------------------------------------------- #
# indices
# --------------------------------------------------------------------------- #
def test_indices_from_download() -> None:
    names = {"SPY": "S&P 500", "^VIX": "VIX", "BTC-USD": "Bitcoin", "NOPE": "Missing"}
    quotes = {q.symbol: q for q in tx.indices_from_download(fx.indices(), names)}
    spy = quotes["SPY"]
    assert spy.price and spy.change_pct is not None and 15 <= len(spy.spark) <= 22
    assert spy.spark[-1] == spy.price
    assert quotes["BTC-USD"].price and quotes["^VIX"].price
    assert quotes["NOPE"].price is None and quotes["NOPE"].spark == []


def test_crypto_profile_drops_stale_price_sentences() -> None:
    p = tx.profile_from_info(fx.info("BTC-USD"), symbol="BTC-USD", name="Bitcoin", short_name="Bitcoin",
                             quote_type="CRYPTOCURRENCY", exchange="Crypto", cik=None, logo_url=None)
    assert p.summary and "Bitcoin" in p.summary
    assert "last known price" not in p.summary and "24 hours" not in p.summary


def test_earnings_reported_today_is_not_next() -> None:
    dates = fx.earnings_dates("NVDA")
    reported_day = date(2026, 8, 26)
    view = tx.earnings_from_frames({"Earnings Date": [reported_day]}, dates, {}, today=reported_day)
    assert view is not None and view.next_date == date(2026, 11, 17)  # the Aug 26 report already happened


# --------------------------------------------------------------------------- #
# crypto calendar
# --------------------------------------------------------------------------- #
def test_crypto_quote_previous_close_matches_its_24h_change() -> None:
    """Yahoo's crypto change is rolling 24 h; its previousClose is the 00:00 UTC open."""
    info = fx.info("BTC-USD")
    q = tx.quote_from_info(info)
    assert q is not None and q.change is not None and q.previous_close is not None
    assert q.price - q.previous_close == pytest.approx(q.change, abs=1e-3)
    assert q.previous_close != info["regularMarketPreviousClose"]


def test_crypto_returns_never_span_a_missing_daily_bar() -> None:
    """Live 2026-10-05: Yahoo had no BTC bar for 10-04, and "1d" silently became a 2-day move."""
    df = fx.bars("BTC-USD")
    full = tx.technicals_from_history(df, now=NOW, is_crypto=True)
    gap = tx.technicals_from_history(df.drop(df.index[-2]), now=NOW, is_crypto=True)
    assert full is not None and full.return_1d is not None
    assert gap is not None and gap.return_1d is None and gap.return_5d == full.return_5d
    idx = fx.indices()
    btc_dates = idx[("Close", "BTC-USD")].dropna().index
    holed = idx.drop(btc_dates[-2])
    quotes = {q.symbol: q for q in tx.indices_from_download(holed, {"SPY": "S&P 500", "BTC-USD": "Bitcoin"})}
    assert quotes["BTC-USD"].change_pct is None and quotes["BTC-USD"].price is not None
    assert quotes["SPY"].change_pct is not None
