"""Graceful degradation: malformed provider values, failing parts and failing NLP never cost the analysis."""
from __future__ import annotations

import math
from datetime import datetime

import pytest

from app.analytics import build as build_module
from app.analytics import textkit
from app.analytics.build import build_analysis
from app.analytics.composite import DEGRADED_MAX_DISTANCE
from app.schemas import Analysis
from tests.analytics import scenarios
from tests.analytics.factories import GOOGLE, NOW, STOCKTWITS, company, inputs, post, raw, run


def comp(a: Analysis, key: str):
    return next(c for c in a.verdict.components if c.key == key)


def insight(a: Analysis, prefix: str):
    return next((i for i in a.insights if i.title.startswith(prefix)), None)


# --------------------------------------------------------------------------- #
# Non-finite numbers and naive datetimes from providers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["technicals.return_1m", "technicals.volatility_30d", "analysts.mean_rating",
                                   "analysts.upside_pct", "insiders.sell_value", "quote.market_cap"])
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_intel_values_never_crash_the_analysis(field: str, bad: float) -> None:
    inp = scenarios.bullish_large_cap()
    model_name, attr = field.split(".")
    model = getattr(inp, model_name)
    setattr(inp, model_name, model.model_copy(update={attr: bad}))
    a = build_analysis(inp)
    assert 0 <= a.verdict.score <= 100 and a.verdict.headline
    # The bad number is treated as missing (or as the field's numeric default), never as a value.
    cleaned = getattr(a, model_name) if model_name in ("technicals", "analysts", "insiders", "quote") else None
    if cleaned is not None:
        value = getattr(cleaned, attr)
        assert value is None or math.isfinite(value)
    a.model_dump_json()  # JSON-safe (NaN would not serialize as valid JSON numbers)


def test_naive_analyst_dates_are_read_as_utc() -> None:
    inp = scenarios.bullish_large_cap()
    naive = [x.model_copy(update={"date": x.date.replace(tzinfo=None)}) for x in inp.analysts.actions]
    inp.analysts = inp.analysts.model_copy(update={"actions": naive})
    a = build_analysis(inp)
    analyst_catalysts = [c for c in a.catalysts if c.kind == "analyst"]
    assert analyst_catalysts and all(c.date.tzinfo is not None for c in analyst_catalysts)
    assert comp(a, "analysts").available


def test_a_failing_component_is_left_out_and_reported(monkeypatch) -> None:
    def broken(*_args, **_kw):
        raise ValueError("cannot convert float NaN to integer")

    monkeypatch.setattr(build_module, "technicals_part", broken)
    a = build_analysis(scenarios.bullish_large_cap())
    tech = comp(a, "technicals")
    assert not tech.available and tech.detail == "could not be computed"
    assert a.verdict.score > 55  # the rest of the read stands
    note = insight(a, "Part of the analysis could not be computed")
    assert note is not None and note.kind == "quality" and "Technicals" in note.detail


def test_failing_catalysts_do_not_sink_the_analysis(monkeypatch) -> None:
    monkeypatch.setattr(build_module, "build_catalysts", lambda f: 1 / 0)
    a = build_analysis(scenarios.bullish_large_cap())
    assert a.catalysts == [] and insight(a, "Part of the analysis could not be computed") is not None


# --------------------------------------------------------------------------- #
# Sentiment engine failure: structured data only, capped, and said plainly
# --------------------------------------------------------------------------- #
def test_engine_failure_caps_the_read_and_says_why(monkeypatch) -> None:
    healthy = build_analysis(scenarios.crypto())

    def fail(texts, kinds=None, company=None):
        raise RuntimeError("engine mid-edit")

    monkeypatch.setattr(textkit, "analyze", fail)
    degraded = build_analysis(scenarios.crypto())
    v = degraded.verdict
    # Live finding: BTC went from 67 Bullish to 77 'Strongly Bullish' when its texts could not be scored.
    assert abs(v.score - 50) <= DEGRADED_MAX_DISTANCE
    assert abs(v.score - 50) <= abs(healthy.verdict.score - 50) + 1
    assert v.label in ("Leaning Bullish", "Neutral", "Leaning Bearish")
    assert "from structured data only" in v.headline and "could not be scored" in v.headline
    assert "0 relevant items" not in v.headline
    assert not comp(degraded, "news").available
    assert insight(degraded, "Sentiment engine failed").severity == "alert"


# --------------------------------------------------------------------------- #
# Data-quality notes always say when there is no text at all
# --------------------------------------------------------------------------- #
def test_zero_source_runs_are_reported() -> None:
    inp = scenarios.bullish_large_cap()
    inp.source_runs = []
    a = build_analysis(inp)
    note = insight(a, "No relevant news or social items")
    assert note is not None and "no text source ran" in note.detail


def test_all_empty_sources_are_reported() -> None:
    inp = inputs(company(), [run(GOOGLE, status="empty"), run(STOCKTWITS, status="empty")])
    a = build_analysis(inp)
    note = insight(a, "No relevant news or social items")
    assert note is not None and "2 sources answered with nothing" in note.detail
    assert insight(a, "Thin coverage") is None  # one note, not two


def test_off_topic_items_are_counted_in_the_note() -> None:
    inp = inputs(company(), [run(GOOGLE, [raw("Globex wins a big contract", 3), raw("Initech misses", 4)])])
    note = insight(build_analysis(inp), "No relevant news or social items")
    assert note is not None and "2 items fetched, 2 off-topic" in note.detail


def test_slow_gdelt_note_matches_what_momentum_actually_uses() -> None:
    inp = inputs(company(), [run(GOOGLE, [raw("Acme beats estimates", 3)]),
                             run(STOCKTWITS, [post("$ACME calls", 1, "a", "bullish")])],
                 intel_status={"tone": "error: still loading after 12s; ready on next refresh"})
    note = insight(build_analysis(inp), "Global news tone still loading")
    assert note is not None and note.severity == "info"
    assert "momentum component is n/a" in note.detail  # 1 headline: no 48h-vs-prior comparison exists
    flow = ([raw(f"Acme wins order number {i} as demand surges", 2 + 4 * i, o) for i, o in enumerate(
                ["Reuters", "Bloomberg", "CNBC", "Barron's", "MarketWatch", "Zacks"])]
            + [raw(f"Acme supplier update number {i}", 60 + 10 * i, o) for i, o in enumerate(
                ["Reuters", "Bloomberg", "CNBC", "Barron's", "MarketWatch", "Zacks"])])
    rich = inputs(company(), [run(GOOGLE, flow)],
                  intel_status={"tone": "error: still loading after 12s; ready on next refresh"})
    note = insight(build_analysis(rich), "Global news tone still loading")
    assert note is not None and "last 48 h of headlines (6) vs the prior days (6)" in note.detail


# --------------------------------------------------------------------------- #
# Small honesty fixes
# --------------------------------------------------------------------------- #
def test_not_applicable_components_say_so_for_crypto_and_etfs() -> None:
    a = build_analysis(scenarios.crypto())
    assert comp(a, "analysts").detail == "n/a for crypto" and comp(a, "insiders").detail == "n/a for crypto"
    etf = scenarios.bullish_large_cap()
    etf.company = company("SPY", "SPDR S&P 500 ETF Trust", "SPDR S&P 500", quote_type="ETF", aliases=["S&P 500"])
    etf.analysts = etf.insiders = None
    b = build_analysis(etf)
    assert comp(b, "analysts").detail == "n/a for ETFs" and comp(b, "insiders").detail == "n/a for ETFs"


def test_fresh_items_count_as_fresh_in_confidence() -> None:
    # age 0.0 (an item stamped at `now`) once read as stale: confidence fell from high to medium.
    def at(stamp: datetime):
        signals = [raw(f"Acme story number {i} beats estimates", 0, o) for i, o in
                   enumerate(["Reuters", "Bloomberg", "CNBC", "Barron's", "MarketWatch", "Zacks"])]
        for s in signals:
            s.timestamp = stamp
        inp = scenarios.bullish_large_cap()
        inp.source_runs = [run(GOOGLE, signals)]
        return build_analysis(inp).verdict.confidence_value

    from datetime import timedelta
    assert at(NOW) == pytest.approx(at(NOW - timedelta(seconds=60)), abs=0.01)
