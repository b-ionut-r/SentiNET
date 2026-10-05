"""A fake world for platform tests: stand-ins for every provider, NLP and analytics
module the orchestrator imports lazily.

`FakeWorld.install(monkeypatch)` registers fake modules in `sys.modules`
(restored after each test), so these tests never touch the network and do not
depend on the real implementations, which other build agents own.
"""
from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from app.analytics.inputs import AnalysisInputs
from app.nlp.types import DetectedEvent, TextAnalysis
from app.schemas import (
    Analysis,
    AnalystAction,
    AnalystView,
    AttentionView,
    Brief,
    Candle,
    Component,
    FearGreed,
    HistoryResponse,
    IndexQuote,
    MarketOverview,
    Narrative,
    PriceResponse,
    Profile,
    Quote,
    Reason,
    SentimentStat,
    Signal,
    SourceReport,
    SymbolMatch,
    Technicals,
    TonePoint,
    ToneTrend,
    TrendingTicker,
    Verdict,
)
from app.sources.base import CompanyRef, RawSignal, SourceBatch

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


class Sentinel:
    """Marks an intel result that should raise / hang instead of returning."""

    def __init__(self, exc: BaseException | None = None, delay: float = 0.0, value: Any = None) -> None:
        self.exc, self.delay, self.value = exc, delay, value


class FakeSource:
    weight = 1.0
    docs_url = "https://example.com/docs"

    def __init__(self, key: str, kind: str = "news", *, signals: int = 3, metrics: dict[str, Any] | None = None,
                 delay: float = 0.0, error: BaseException | None = None, requires_key: bool = False,
                 configured: bool = True, ticker_specific: bool = True) -> None:
        self.key = key
        self.label = key.replace("_", " ").title()
        self.kind = kind
        self.requires_key = requires_key
        self.description = f"Fake {key} source"
        self._configured = configured
        self.n_signals = signals
        self.metrics = metrics or {}
        self.delay = delay
        self.error = error
        self.ticker_specific = ticker_specific  # False: a keyword search (matches any string)
        self.calls = 0

    def configured(self) -> bool:
        return self._configured

    def supports(self, company: CompanyRef) -> bool:
        return True

    async def fetch(self, company: CompanyRef) -> SourceBatch:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        signals = [
            RawSignal(title=f"{company.short_name} headline {i} from {self.key}", url=f"https://news.example/{self.key}/{i}",
                      publisher="Reuters", timestamp=NOW - timedelta(hours=i), ticker_specific=self.ticker_specific)
            for i in range(self.n_signals)
        ]
        return SourceBatch(signals=signals, metrics=dict(self.metrics))


def make_verdict(score: int = 64, headline: str = "Bullish: analysts and news align.",
                 evidence: bool = True) -> Verdict:
    """A verdict; `evidence=False` is the real "No read" shape (score 50, no available component)."""
    if not evidence:
        return Verdict(
            score=50, label="No read", stance="neutral", confidence="low", confidence_value=0.0,
            headline="No read: every source and data feed came back empty or failed",
            components=[Component(key=k, label=k.title(), score=None, weight=0.2, available=False, detail="n/a")
                        for k in ("news", "social", "analysts", "insiders", "momentum", "technicals")],
        )
    stance = "bullish" if score >= 55 else "bearish" if score <= 45 else "neutral"
    return Verdict(
        score=score, label="Bullish" if score >= 62 else "Neutral", stance=stance, confidence="medium",
        confidence_value=0.6, headline=headline,
        reasons=[Reason(text="3 price-target raises in 30 days", polarity="bull", weight=0.8, ref="analysts")],
        components=[Component(key="news", label="News", score=61.0, weight=0.3, available=True,
                              detail="+0.21 across 12 articles", confidence=0.7),
                    Component(key="insiders", label="Insiders", score=None, weight=0.1, available=False,
                              detail="no open-market trades")],
    )


@dataclass
class FakeWorld:
    """Configurable fake providers. Intel values may be `Sentinel`s to raise or hang."""

    company: CompanyRef = field(default_factory=lambda: CompanyRef(
        ticker="NVDA", name="NVIDIA Corporation", short_name="Nvidia", cik="0001045810", exchange="NMS"))
    sources: list[tuple[FakeSource, str]] = field(default_factory=lambda: [
        (FakeSource("google_news"), "enabled"),
        (FakeSource("stocktwits", "social", signals=2,
                    metrics={"stocktwits_bullish": 30, "stocktwits_bearish": 10}), "enabled"),
        (FakeSource("finnhub", requires_key=True, configured=False), "unconfigured"),
    ])
    resolve: Any = None  # None → `company`; Sentinel → raise/hang
    intel: dict[str, Any] = field(default_factory=dict)
    score: int = 64
    narratives: list[Narrative] = field(default_factory=list)
    attention: AttentionView | None = None
    analyst_actions: list[AnalystAction] = field(default_factory=list)
    build_error: BaseException | None = None
    build_delay: float = 0.0
    evidence: bool | None = True  # verdict has an available component; None: decided like the real composite
    witness: Any = field(default_factory=lambda: Quote(price=600.0))  # SPY quote: Yahoo answering (None: outage)

    calls: dict[str, list[Any]] = field(default_factory=dict)
    inputs: list[AnalysisInputs] = field(default_factory=list)

    def __post_init__(self) -> None:
        defaults: dict[str, Any] = {
            "profile": Profile(symbol="NVDA", name="NVIDIA Corporation", short_name="Nvidia", sector="Technology"),
            "quote": Quote(price=180.0, change_pct=1.5, currency="USD"),
            "technicals": Technicals(return_1m=4.2, rsi_14=58.0, trend="uptrend"),
            "analysts": "auto",
            "earnings": None,
            "calendar": [],
            "insiders": None,
            "filings": [],
            "tone": ToneTrend(query='"Nvidia"', tone_7d=1.2, tone_30d=0.8,
                              series=[TonePoint(date=date(2026, 10, 1), tone=1.0, volume=120.0)]),
            "wiki": [(date(2026, 10, 1), 15000.0), (date(2026, 10, 2), 16000.0)],
        }
        for k, v in defaults.items():
            self.intel.setdefault(k, v)

    # ---- helpers -------------------------------------------------------------- #
    def _record(self, name: str, *args: Any) -> None:
        self.calls.setdefault(name, []).append(args)

    async def _intel(self, key: str, *args: Any) -> Any:
        if key == "quote" and args and args[0] == "SPY":  # the analyzer's market-data witness
            self._record("witness", *args)
            if isinstance(self.witness, Sentinel):
                if self.witness.exc is not None:
                    raise self.witness.exc
                return self.witness.value
            return self.witness
        self._record(key, *args)
        value = self.intel.get(key)
        if isinstance(value, Sentinel):
            if value.delay:
                await asyncio.sleep(value.delay)
            if value.exc is not None:
                raise value.exc
            return value.value
        if key == "analysts" and value == "auto":
            price = args[1] if len(args) > 1 else None
            return AnalystView(consensus="buy", total=40, target_mean=200.0,
                               upside_pct=((200.0 / price - 1) * 100) if price else None,
                               actions=list(self.analyst_actions))
        return value

    def build(self, inputs: AnalysisInputs) -> Analysis:
        self.inputs.append(inputs)
        if self.build_delay:
            import time

            time.sleep(self.build_delay)
        if self.build_error is not None:
            raise self.build_error
        signals: list[Signal] = []
        reports: list[SourceReport] = []
        for run in inputs.source_runs:
            batch = run.batch or SourceBatch()
            for i, raw in enumerate(batch.signals):
                signals.append(Signal(
                    id=f"{run.source.key}-{i}", source=run.source.key, source_label=run.source.label,
                    kind=run.source.kind, title=raw.title, url=raw.url, publisher=raw.publisher,
                    timestamp=raw.timestamp, score=0.3, label="bullish", confidence=0.7, weight=1.0,
                    themes=["earnings"], events=["pt_raise"],
                ))
            reports.append(SourceReport(key=run.source.key, label=run.source.label, kind=run.source.kind,
                                        status=run.status, fetched=len(batch.signals), kept=len(batch.signals),
                                        latency_ms=run.latency_ms, error=run.error,
                                        requires_key=run.source.requires_key))
        n = len(signals)
        evidence = self.evidence
        if evidence is None:  # like the real composite: nothing scored and no structured input → "No read"
            evidence = bool(n) or any(x is not None for x in (inputs.analysts, inputs.technicals, inputs.insiders))
        return Analysis(
            ticker=inputs.company.ticker, generated_at=inputs.now, engine=inputs.engine_name,
            profile=inputs.profile, quote=inputs.quote, technicals=inputs.technicals,
            verdict=make_verdict(self.score, evidence=evidence), brief=Brief(summary="Fake brief."),
            sentiment=SentimentStat(score=0.3, label="bullish", n=n, bullish=n),
            news=SentimentStat(score=0.25, label="bullish", n=max(0, n - 2)),
            social=SentimentStat(score=0.4, label="bullish", n=min(2, n)),
            narratives=list(self.narratives), analysts=inputs.analysts, attention=self.attention,
            tone=inputs.tone, sources=reports, signals=signals,
        )

    # ---- install --------------------------------------------------------------- #
    def install(self, monkeypatch: Any) -> FakeWorld:
        world = self

        def module(name: str, **attrs: Any) -> types.ModuleType:
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            monkeypatch.setitem(sys.modules, name, mod)
            return mod

        def normalize_ticker(raw: str) -> str | None:
            sym = raw.strip().lstrip("$").upper().replace(".", "-") if raw else ""
            if not sym or not all(c.isalnum() or c in "-^=" for c in sym) or len(sym) > 12:
                return None
            return sym

        async def resolve_company(ticker: str) -> CompanyRef:
            world._record("resolve", ticker)
            if isinstance(world.resolve, Sentinel):
                if world.resolve.delay:
                    await asyncio.sleep(world.resolve.delay)
                if world.resolve.exc is not None:
                    raise world.resolve.exc
                return world.resolve.value
            if world.resolve is not None:
                return world.resolve
            return CompanyRef(**{**world.company.__dict__, "ticker": ticker})

        async def search_symbols(q: str, limit: int = 8) -> list[SymbolMatch]:
            world._record("search", q, limit)
            if q == "boom":
                raise RuntimeError("search provider down")
            return [SymbolMatch(symbol="NVDA", name="NVIDIA Corporation", exchange="NMS", type="EQUITY")][:limit]

        module("app.resolve.symbols", normalize_ticker=normalize_ticker, resolve_company=resolve_company,
               search_symbols=search_symbols)
        module("app.sources.registry",
               enabled_sources=lambda company: list(world.sources),
               all_sources=lambda: [s for s, _ in world.sources],
               get_source=lambda key: next((s for s, _ in world.sources if s.key == key), None))

        def intel_fn(key: str) -> Any:
            async def fn(*args: Any) -> Any:
                return await world._intel(key, *args)
            return fn

        async def get_price_history(ticker: str, rng: str) -> PriceResponse:
            world._record("price", ticker, rng)
            if ticker == "FAIL":
                raise RuntimeError("yahoo exploded token=SUPERSECRET123")
            return PriceResponse(ticker=ticker, range=rng, interval="1d",
                                 candles=[Candle(t=NOW, o=1, h=2, l=0.5, c=1.5, v=100)])

        async def get_daily_closes(ticker: str, days: int = 400) -> list[tuple[date, float]]:
            world._record("closes", ticker, days)
            return [(date(2026, 10, 1), 100.0), (date(2026, 10, 2), 101.0)]

        async def get_indices() -> list[IndexQuote]:
            return [IndexQuote(symbol="SPY", name="S&P 500", price=600.0, change_pct=0.4, spark=[1, 2, 3])]

        module("app.intel.market_data",
               get_profile=intel_fn("profile"), get_quote=intel_fn("quote"), get_technicals=intel_fn("technicals"),
               get_analysts=intel_fn("analysts"), get_earnings=intel_fn("earnings"),
               get_calendar_catalysts=intel_fn("calendar"), get_insiders=intel_fn("insiders"),
               get_price_history=get_price_history, get_daily_closes=get_daily_closes, get_indices=get_indices)
        module("app.intel.sec", get_filings=intel_fn("filings"))
        module("app.intel.gdelt", get_tone_trend=intel_fn("tone"))
        module("app.intel.attention", get_wiki_pageviews=intel_fn("wiki"))

        async def get_cnn_fear_greed() -> FearGreed:
            return FearGreed(score=38.0, rating="fear")

        async def get_crypto_fear_greed() -> FearGreed:
            raise RuntimeError("alternative.me down")

        async def get_trending() -> list[TrendingTicker]:
            return [TrendingTicker(symbol="NVDA", source="reddit", rank=1, mentions=900)]

        async def get_market_headlines() -> list[RawSignal]:
            return [RawSignal(title="Stocks rally as yields fall", publisher="CNBC", timestamp=NOW)]

        module("app.intel.market", get_cnn_fear_greed=get_cnn_fear_greed, get_crypto_fear_greed=get_crypto_fear_greed,
               get_trending=get_trending, get_market_headlines=get_market_headlines)

        module("app.analytics.build", build_analysis=world.build)

        def build_history(ticker, days, tone, closes, snapshots, wiki, status) -> HistoryResponse:
            world._record("build_history", ticker, days, tone, closes, snapshots, wiki, status)
            return HistoryResponse(ticker=ticker, days=days, interpretation="No reliable relationship.",
                                   status=status)

        module("app.analytics.stats", build_history=build_history)

        def build_market_overview(fear_greed, crypto_fg, indices, trending, headline_raws, status, now):
            world._record("build_market", fear_greed, crypto_fg, indices, trending, headline_raws, status, now)
            return MarketOverview(generated_at=now, regime="Risk-off: Fear", regime_detail="F&G 38", status=status,
                                  fear_greed=fear_greed, crypto_fear_greed=crypto_fg, indices=indices,
                                  trending=trending)

        module("app.analytics.market", build_market_overview=build_market_overview)

        class Engine:
            name = "sentinel"

            def score(self, texts: list[str], kinds: list[str] | None = None,
                      targets: object = None) -> list[TextAnalysis]:
                return [self._one(t) for t in texts]

            @staticmethod
            def _one(text: str) -> TextAnalysis:
                low = text.lower()
                if "beat" in low or "upgrade" in low:
                    return TextAnalysis(0.6, "bullish", 0.8, [("beat", 0.5)], ["earnings"],
                                        [DetectedEvent("earnings_beat", "bull")])
                if "miss" in low or "lawsuit" in low:
                    return TextAnalysis(-0.5, "bearish", 0.7, [("miss", -0.4)], ["legal"],
                                        [DetectedEvent("lawsuit", "bear")])
                return TextAnalysis(0.0, "neutral", 0.3, [], [], [])

        engine = Engine()

        def label_for(score: float) -> str:
            return "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"

        module("app.nlp.engine", get_engine=lambda: engine, label_for=label_for, NEUTRAL_BAND=0.05)
        module("app.nlp.pipeline", analyze_texts=lambda texts, kinds=None, company=None: engine.score(texts, kinds))
        module("app.nlp.relevance", relevance=lambda text, company: 1.0 if company.short_name in text else 0.2)
        module("app.nlp.themes", THEMES={"earnings": "Earnings", "legal": "Legal"})
        return self
