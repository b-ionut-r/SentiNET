"""Builders for realistic, fully synthetic analytics inputs (no network)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from app.analytics.inputs import AnalysisInputs, SourceRun
from app.analytics.util import stable_id
from app.schemas import (
    AnalystAction,
    AnalystView,
    Catalyst,
    EarningsEvent,
    EarningsView,
    Filing,
    InsiderTxn,
    InsiderView,
    Quote,
    RatingCounts,
    Snapshot,
    Technicals,
    TonePoint,
    ToneTrend,
)
from app.sources.base import CompanyRef, RawSignal, SourceBatch

NOW = datetime(2026, 10, 2, 18, 0, tzinfo=UTC)  # a Friday afternoon


@dataclass
class FakeSource:
    key: str
    label: str
    kind: str = "news"
    weight: float = 1.0
    requires_key: bool = False
    description: str = "test source"
    docs_url: str | None = None

    def configured(self) -> bool:
        return True

    def supports(self, company: CompanyRef) -> bool:
        return True

    async def fetch(self, company: CompanyRef) -> SourceBatch:  # pragma: no cover
        return SourceBatch()


GOOGLE = FakeSource("google_news", "Google News", "news", 1.2)
BING = FakeSource("bing_news", "Bing News", "news", 1.1)
YAHOO = FakeSource("yahoo_news", "Yahoo Finance", "news", 1.1)
STOCKTWITS = FakeSource("stocktwits", "StockTwits", "social", 0.8)
BLUESKY = FakeSource("bluesky", "Bluesky", "social", 0.6)
APEWISDOM = FakeSource("apewisdom", "Reddit buzz (ApeWisdom)", "social", 0.5)
TRADESTIE = FakeSource("tradestie", "WSB buzz (Tradestie)", "social", 0.4)
FINNHUB = FakeSource("finnhub", "Finnhub", "news", 1.15, requires_key=True)

OUTLETS = ["Reuters", "Bloomberg", "CNBC", "MarketWatch", "Barron's", "The Motley Fool", "Benzinga",
           "Investopedia", "Yahoo Finance", "TipRanks", "Zacks", "Seeking Alpha"]


def company(ticker: str = "ACME", name: str = "Acme Corporation", short: str = "Acme", quote_type: str = "EQUITY",
            aliases: list[str] | None = None) -> CompanyRef:
    return CompanyRef(ticker=ticker, name=name, short_name=short, quote_type=quote_type, aliases=aliases or [])


def ago(hours: float) -> datetime:
    return NOW - timedelta(hours=hours)


def raw(title: str, hours: float = 6.0, publisher: str | None = "Reuters", **kw: Any) -> RawSignal:
    return RawSignal(title=title, timestamp=ago(hours), publisher=publisher,
                     url=kw.pop("url", f"https://example.com/{stable_id(title, hours, publisher)}"), **kw)


def post(text: str, hours: float = 2.0, author: str = "trader", label: str | None = None, likes: int = 0,
         **kw: Any) -> RawSignal:
    return RawSignal(title=text, timestamp=ago(hours), publisher="StockTwits", author=author, engagement=likes,
                     user_label=label, ticker_specific=kw.pop("ticker_specific", True), **kw)


def news_flow(headlines: list[str], hours_step: float = 3.0, start: float = 1.0) -> list[RawSignal]:
    """Headlines spread over time and across outlets."""
    return [raw(h, start + i * hours_step, OUTLETS[i % len(OUTLETS)]) for i, h in enumerate(headlines)]


def run(source: FakeSource, signals: list[RawSignal] | None = None, metrics: dict[str, Any] | None = None,
        status: str = "ok", error: str | None = None, latency: int = 420) -> SourceRun:
    batch = None if status in ("error", "disabled", "unconfigured") else SourceBatch(signals or [], metrics or {})
    return SourceRun(source=source, status=status, batch=batch, latency_ms=latency if status != "unconfigured" else None,
                     error=error)


def inputs(comp: CompanyRef, runs: list[SourceRun], **kw: Any) -> AnalysisInputs:
    status = kw.pop("intel_status", None)
    built = AnalysisInputs(company=comp, now=kw.pop("now", NOW), engine_name="sentinel", source_runs=runs, **kw)
    if status is None:
        status = {}
        for key, attr in (("quote", "quote"), ("technicals", "technicals"), ("analysts", "analysts"),
                          ("insiders", "insiders"), ("earnings", "earnings"), ("tone", "tone"), ("wiki", "wiki_views")):
            status[key] = "ok" if getattr(built, attr) else "empty"
    built.intel_status = status
    return built


# --------------------------------------------------------------------------- #
# Structured intel
# --------------------------------------------------------------------------- #
def quote(price: float = 100.0, market_cap: float | None = 50e9) -> Quote:
    return Quote(price=price, change_pct=0.8, market_cap=market_cap, currency="USD")


def technicals(r1m: float = 4.0, r3m: float = 12.0, vs50: float = 5.0, vs200: float = 12.0, rsi: float = 60.0,
               vol: float = 30.0, r5d: float = 1.0, trend: str = "uptrend") -> Technicals:
    return Technicals(return_1d=0.5, return_5d=r5d, return_1m=r1m, return_3m=r3m, vs_50dma_pct=vs50,
                      vs_200dma_pct=vs200, rsi_14=rsi, volatility_30d=vol, trend=trend)  # type: ignore[arg-type]


def action(days: float, firm: str, kind: str, to: str | None = "Buy", pt: float | None = None,
           prior: float | None = None, frm: str | None = None) -> AnalystAction:
    return AnalystAction(date=NOW - timedelta(days=days), firm=firm, action=kind, to_grade=to,  # type: ignore[arg-type]
                         price_target=pt, prior_target=prior, from_grade=frm)


def analysts(mean: float = 1.8, total: int = 30, upside: float | None = 20.0, actions: list[AnalystAction] | None = None,
             up90: int = 0, down90: int = 0, price: float = 100.0) -> AnalystView:
    consensus = ("strong_buy" if mean < 1.5 else "buy" if mean < 2.5 else "hold" if mean < 3.5 else "sell"
                 if mean < 4.5 else "strong_sell")
    return AnalystView(
        consensus=consensus, mean_rating=mean, total=total,
        counts=RatingCounts(period="0m", strong_buy=total // 3, buy=total // 3, hold=total - 2 * (total // 3)),
        target_mean=round(price * (1 + (upside or 0) / 100), 2) if upside is not None else None,
        upside_pct=upside, actions=actions or [], upgrades_90d=up90, downgrades_90d=down90,
    )


def insider(days: float, who: str, kind: str, value: float, position: str = "Director") -> InsiderTxn:
    return InsiderTxn(date=(NOW - timedelta(days=days)).date(), insider=who, position=position,
                      kind=kind, shares=round(value / 50), value=value)  # type: ignore[arg-type]


def insiders(txns: list[InsiderTxn]) -> InsiderView:
    buys = [t for t in txns if t.kind == "buy"]
    sells = [t for t in txns if t.kind == "sell"]
    bv, sv = sum(t.value or 0 for t in buys), sum(t.value or 0 for t in sells)
    return InsiderView(buys=len(buys), sells=len(sells), buy_value=bv, sell_value=sv, net_value=bv - sv,
                       ratio=(bv - sv) / (bv + sv) if bv + sv else None, transactions=txns)


def earnings(days_until: int = 20, beat_rate: float = 0.75, quarters: int = 8) -> EarningsView:
    history = [EarningsEvent(date=(NOW - timedelta(days=91 * (i + 1))).date(), eps_estimate=1.0, eps_actual=1.1,
                             surprise_pct=10.0) for i in range(quarters)]
    nxt = (NOW + timedelta(days=days_until)).date()
    return EarningsView(next_date=nxt, days_until=days_until, eps_estimate=1.25, eps_low=1.1, eps_high=1.4,
                        revenue_estimate=5.2e9, history=history, beat_rate=beat_rate)


def tone_trend(base: float = 0.5, recent: float | None = None, days: int = 90, volume: float = 200.0,
               spike: float | None = None, percentile: float | None = None) -> ToneTrend:
    """Daily GDELT-like series; `recent` overrides the last 7 days, `spike` the last 2 days' volume."""
    series = []
    end = NOW.date() - timedelta(days=1)
    for i in range(days):
        d = end - timedelta(days=days - 1 - i)
        wobble = 0.15 * ((i * 7919) % 11 - 5) / 5
        tone = (recent if recent is not None and i >= days - 7 else base) + wobble
        vol = volume * (1 + 0.1 * ((i * 104729) % 7 - 3) / 3)
        if spike is not None and i >= days - 2:
            vol = spike
        series.append(TonePoint(date=d, tone=round(tone, 3), volume=round(vol, 1)))
    t7 = recent if recent is not None else base
    return ToneTrend(query="test", tone_7d=t7, tone_30d=base, tone_90d=base, change_7d_vs_30d=round(t7 - base, 3),
                     percentile_7d=percentile if percentile is not None else (0.5 if recent is None else
                                                                            (0.95 if t7 > base else 0.05)),
                     series=series)


def filing(days: float, form: str, title: str, items: list[str], importance: str = "low",
           polarity: str = "neutral") -> Filing:
    return Filing(form=form, date=(NOW - timedelta(days=days)).date(), title=title, items=items,
                  importance=importance, polarity=polarity, url="https://www.sec.gov/x")  # type: ignore[arg-type]


def ex_dividend(days: int, detail: str = "$0.25/share · yield 1.10%") -> Catalyst:
    return Catalyst(date=datetime.combine((NOW + timedelta(days=days)).date(), datetime.min.time(), tzinfo=UTC)
                    + timedelta(hours=12), kind="dividend", title=f"Ex-dividend date in {days} days",
                    detail=detail, upcoming=True)


def snapshot(hours: float, sentinel: int, score: float, label: str = "neutral", price: float | None = 100.0,
             narratives: list[str] | None = None, ticker: str = "ACME") -> Snapshot:
    return Snapshot(ticker=ticker, at=ago(hours), sentinel_score=sentinel, score=score, label=label,  # type: ignore[arg-type]
                    n_signals=50, price=price, narratives=narratives or [])


def daily(values: list[float], end: date | None = None) -> list[tuple[date, float]]:
    end = end or NOW.date()
    return [(end - timedelta(days=len(values) - 1 - i), v) for i, v in enumerate(values)]


@dataclass
class Scenario:
    """A named input bundle (for readable parametrized tests)."""

    name: str
    inputs: AnalysisInputs
    notes: list[str] = field(default_factory=list)
