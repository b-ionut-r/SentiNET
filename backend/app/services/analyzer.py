"""Analysis orchestrator: fan out to every provider, then synthesize one `Analysis`.

    normalize → resolve company → [text/crowd sources ‖ structured intel] → previous
    snapshot → analytics (`build_analysis`, CPU pool) → snapshot + alert rules → cache

Guarantees
* A slow or failing provider never fails the analysis: each task is time-boxed
  and reported (status, latency, error) while the rest is synthesized.
* One run per ticker at a time (single-flight). Concurrent callers share it and
  late joiners get the progress events so far replayed, in order.
* A caller that goes away (closed SSE stream) does not abort the shared run;
  the result is still stored and cached.
* Fresh results are cached for `settings.analyze_cache_ttl` and served with
  `cached=True`.

Provider modules are imported inside functions so this module imports cleanly
even while those packages are being edited; tests replace them with fakes.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import settings
from app.core.cache import TTLStore
from app.core.sync import run_cpu
from app.schemas import (
    Analysis,
    AnalystView,
    Catalyst,
    EarningsView,
    Filing,
    InsiderView,
    Profile,
    ProgressEvent,
    Quote,
    Snapshot,
    Technicals,
    ToneTrend,
)
from app.services.errors import (
    AnalysisFailed,
    InvalidTicker,
    ServiceError,
    UnknownSymbol,
)
from app.services.tasks import Outcome, deferred, describe_error, has_data, run_bounded
from app.sources.base import CompanyRef, Source, SourceBatch

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]

PREVIOUS_MIN_AGE = timedelta(minutes=15)  # "what changed" compares against a snapshot at least this old
ANALYTICS_TIMEOUT = 45.0
STORAGE_TIMEOUT = 5.0
LISTENER_TIMEOUT = 2.0
TONE_DAYS = 90  # keep in sync with the history endpoint so both share GDELT's cache
LATEST_KEEP = 64
# Time budget of a cold run, counted from the start (symbol resolution
# included) to synthesis, which then takes well under a second: < 12 s total.
RUN_BUDGET = 11.0
RESOLVE_TIMEOUT = 8.0
MIN_TIME_BOX = 1.0
# Tail rule: once every source and every core intel task has answered, the slow
# name-search tasks (GDELT tone, Wikipedia) get at most this much longer instead
# of idling to the end of the budget. Stragglers are not cancelled: they finish
# in the background, warm their provider cache (GDELT allows 1 request / 5 s)
# and, when they bring data, invalidate the cached analysis so the next load
# recomputes with it.
TAIL_GRACE = 3.0
TAIL_KEYS = frozenset({"tone", "wiki"})
# A `refresh` within this many seconds of the last run returns that run (flagged
# cached): news does not change that fast, and free APIs deserve politeness.
MIN_REFRESH_SECONDS = 45.0
# Fresh runs allowed at once. Time boxes include rate-limiter waits, so a burst
# of parallel runs would degrade *every* result; extra runs queue instead.
MAX_CONCURRENT_RUNS = 4


# --------------------------------------------------------------------------- #
# Module state
# --------------------------------------------------------------------------- #
_cache = TTLStore(ttl=settings.analyze_cache_ttl, maxsize=128)
_latest: OrderedDict[str, Analysis] = OrderedDict()  # last result per ticker, no TTL (exports)
_runs: dict[str, _Run] = {}
_slots: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None
# symbol -> generated_at of a cached analysis superseded by late-arriving data
_stale: dict[str, datetime] = {}
# Negative cache: symbols just found not to exist (a typo re-submitted should not
# fan out to ~30 provider calls again). `refresh=True` bypasses it.
UNKNOWN_TTL = 600
_unknown = TTLStore(ttl=UNKNOWN_TTL, maxsize=256)


def _run_slots() -> asyncio.Semaphore:
    """The run-concurrency semaphore for the current event loop (tests use several loops)."""
    global _slots
    loop = asyncio.get_running_loop()
    if _slots is None or _slots[0] is not loop:
        _slots = (loop, asyncio.Semaphore(MAX_CONCURRENT_RUNS))
    return _slots[1]


_FALLBACK_SYMBOL = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


def _fallback_normalize(text: str) -> str | None:
    """Conservative normalization used only if the resolver module cannot be imported."""
    sym = text.lstrip("$").upper()
    return sym if _FALLBACK_SYMBOL.match(sym) else None


def normalize(raw: str) -> str:
    """Canonical symbol for user input, or `InvalidTicker` with a helpful message."""
    try:
        from app.resolve.symbols import normalize_ticker
    except ImportError:  # resolver mid-edit: degrade rather than 500
        logger.exception("symbol resolver unavailable")
        normalize_ticker = _fallback_normalize

    text = (raw or "").strip()
    symbol = normalize_ticker(text) if 0 < len(text) <= 24 else None
    if not symbol:
        shown = text[:24] or "(empty)"
        raise InvalidTicker(
            f"'{shown}' is not a valid ticker. Use a symbol like AAPL, BRK-B, SHOP.TO or BTC-USD."
        )
    return symbol


def _cache_get(symbol: str) -> Analysis | None:
    """Cached analysis unless late data (a straggler that has since arrived) superseded it."""
    hit = _cache.get(symbol)
    if hit is None:
        return None
    stale = _stale.get(symbol)
    return None if stale is not None and hit.generated_at <= stale else hit


def _mark_stale(symbol: str, generated_at: datetime) -> None:
    if symbol not in _stale or _stale[symbol] < generated_at:
        _stale[symbol] = generated_at


def cached_analysis(symbol: str) -> Analysis | None:
    """The fresh cached analysis for a canonical symbol, if any."""
    hit = _cache_get(symbol)
    return hit.model_copy(update={"cached": True}) if hit is not None else None


def latest_analysis(symbol: str) -> Analysis | None:
    """The most recent analysis computed in this process (even if its cache TTL lapsed)."""
    return _latest.get(symbol)


def reset() -> None:
    """Drop cached results and run bookkeeping (tests)."""
    global _slots
    _cache.clear()
    _latest.clear()
    _runs.clear()
    _stale.clear()
    _unknown.clear()
    _slots = None


async def shutdown() -> None:
    """Cancel in-flight runs and kept-alive provider calls (application shutdown)."""
    from app.services.tasks import cancel_background

    tasks = [r.task for r in _runs.values() if r.task is not None and not r.task.done()]
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _runs.clear()
    await cancel_background()


# --------------------------------------------------------------------------- #
# Single-flight run with replayable progress
# --------------------------------------------------------------------------- #
async def _deliver(cb: ProgressCallback, event: ProgressEvent) -> bool:
    """Send one event to a listener; False if the listener is broken or stalled."""
    try:
        await asyncio.wait_for(cb(event), LISTENER_TIMEOUT)
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a bad listener must not affect the run
        logger.debug("dropping progress listener: %s", exc)
        return False


class _Run:
    """One in-flight analysis of a ticker, shared by every concurrent caller."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.events: list[ProgressEvent] = []
        self.task: asyncio.Task[Analysis] | None = None
        self._listeners: list[ProgressCallback] = []
        self._lock = asyncio.Lock()

    async def emit(self, event: ProgressEvent) -> None:
        async with self._lock:
            self.events.append(event)
            for cb in list(self._listeners):
                if not await _deliver(cb, event):
                    self._listeners.remove(cb)

    async def attach(self, cb: ProgressCallback) -> None:
        """Replay past events, then subscribe (atomically w.r.t. `emit`, so order is kept)."""
        async with self._lock:
            for event in self.events:
                if not await _deliver(cb, event):
                    return
            self._listeners.append(cb)

    def detach(self, cb: ProgressCallback) -> None:
        if cb in self._listeners:
            self._listeners.remove(cb)


def _finish_run(symbol: str, run: _Run, task: asyncio.Task[Analysis]) -> None:
    if _runs.get(symbol) is run:
        del _runs[symbol]
    if not task.cancelled():
        task.exception()  # mark retrieved: every caller may have gone away


async def analyze(ticker: str, refresh: bool = False, progress: ProgressCallback | None = None) -> Analysis:
    """Full analysis of `ticker` (cached unless `refresh`); streams `ProgressEvent`s to `progress`.

    Raises `ServiceError` subclasses only (invalid/unknown ticker, synthesis
    failure); provider failures are absorbed into the result.
    """
    symbol = normalize(ticker)
    hit = _cache_get(symbol)
    if hit is not None:
        age = (datetime.now(UTC) - hit.generated_at).total_seconds()
        if not refresh or age < MIN_REFRESH_SECONDS:
            if progress is not None:
                await _deliver(progress, ProgressEvent(
                    stage="done", key="done", label="Served from cache", status="ok",
                    count=len(hit.signals), ms=0, detail=f"computed {_ago(int(age))} ago",
                ))
            return hit.model_copy(update={"cached": True})
    known_unknown = _unknown.get(symbol)
    if known_unknown is not None and not refresh:
        if progress is not None:
            await _deliver(progress, ProgressEvent(stage="done", key="done", label="Analysis stopped",
                                                   status="error", detail=known_unknown))
        raise UnknownSymbol(known_unknown)

    run = _runs.get(symbol)
    if run is None:
        run = _Run(symbol)
        run.task = asyncio.create_task(_execute(run), name=f"analyze:{symbol}")
        run.task.add_done_callback(functools.partial(_finish_run, symbol, run))
        _runs[symbol] = run
    if progress is not None:
        await run.attach(progress)
    try:
        assert run.task is not None
        return await asyncio.shield(run.task)
    finally:
        if progress is not None:
            run.detach(progress)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
@dataclass
class _IntelSpec:
    key: str
    label: str
    call: Callable[[], Awaitable[Any]]
    expect: type | tuple[type, ...]
    skip: str | None = None  # reason, when not applicable to this asset
    keep_alive: bool = True


_INTEL_LABELS = {
    "profile": "Company profile",
    "quote": "Live quote",
    "technicals": "Technicals",
    "analysts": "Analyst ratings",
    "earnings": "Earnings",
    "calendar": "Dividend calendar",
    "insiders": "Insider trades",
    "filings": "SEC filings",
    "tone": "GDELT global tone",
    "wiki": "Wikipedia attention",
}

# Structured intel that does not exist for an asset class (saves Yahoo/SEC calls).
_NO_ISSUER = {"analysts", "insiders", "earnings", "calendar", "filings"}  # nothing issues these
_NOT_APPLICABLE: dict[str, set[str]] = {
    "CRYPTOCURRENCY": _NO_ISSUER,
    "ETF": {"analysts", "insiders", "earnings", "filings"},
    "MUTUALFUND": {"analysts", "insiders", "earnings", "filings"},
    "INDEX": _NO_ISSUER,
    "FUTURE": _NO_ISSUER,
    "CURRENCY": _NO_ISSUER,
}

_ASSET_NAMES = {"CRYPTOCURRENCY": "crypto", "ETF": "ETFs", "MUTUALFUND": "funds", "INDEX": "indices",
                "FUTURE": "futures", "CURRENCY": "currencies"}

# Evidence that an instrument exists must be keyed by the *symbol*: keyword
# searches (news, Bluesky, Hacker News) return something for almost any string.
_SYMBOL_KEYED_METRICS = ("stocktwits_", "reddit_", "wsb_")


async def _execute(run: _Run) -> Analysis:
    """Run wrapper: anything unexpected becomes a clean `AnalysisFailed` (never a raw 500)."""
    try:
        slots = _run_slots()
        if slots.locked():
            ahead = sum(1 for r in _runs.values() if r is not run)
            await run.emit(ProgressEvent(stage="resolve", key="queue", label="Queued", status="running",
                                         detail=f"waiting for a free slot ({ahead} analyses in flight)"))
            async with slots:
                await run.emit(ProgressEvent(stage="resolve", key="queue", label="Queued", status="ok"))
                return await _execute_inner(run)
        async with slots:
            return await _execute_inner(run)
    except asyncio.CancelledError:
        raise
    except ServiceError as exc:
        await run.emit(ProgressEvent(stage="done", key="done", label="Analysis stopped", status="error",
                                     detail=str(exc)))
        raise
    except Exception as exc:
        logger.exception("analysis of %s failed", run.symbol)
        detail = describe_error(exc)
        await run.emit(ProgressEvent(stage="done", key="done", label="Analysis failed", status="error",
                                     detail=detail))
        raise AnalysisFailed(f"Analysis of {run.symbol} failed ({detail}). Please retry.") from exc


async def _execute_inner(run: _Run) -> Analysis:
    from app.analytics.inputs import AnalysisInputs

    started = time.perf_counter()
    deadline = time.monotonic() + RUN_BUDGET  # resolution counts against the budget too
    now = datetime.now(UTC)
    symbol = run.symbol

    company = await _resolve(run, deadline)
    planned, skipped_runs = _plan_sources(company)
    specs = _intel_specs(company)
    for source in planned:
        await run.emit(ProgressEvent(stage="source", key=source.key, label=source.label, status="running"))
    for spec in specs:
        await run.emit(ProgressEvent(
            stage="intel", key=spec.key, label=spec.label,
            status="skipped" if spec.skip else "running", detail=spec.skip,
        ))
    for skipped in skipped_runs:
        hint = "needs a free API key" if skipped.status == "unconfigured" else "disabled in settings"
        await run.emit(ProgressEvent(stage="source", key=skipped.source.key, label=skipped.source.label,
                                     status="skipped", detail=hint))

    core_done = asyncio.Event()  # every source and non-tail intel task has answered
    try:
        async with asyncio.TaskGroup() as tg:
            engine_task = tg.create_task(run_cpu(_engine_name))
            previous_task = tg.create_task(_previous_snapshot(symbol, now))
            source_tasks = [tg.create_task(_run_source(run, s, company, deadline)) for s in planned]
            intel_tasks = {
                s.key: tg.create_task(_run_intel(run, s, deadline, core_done if s.key in TAIL_KEYS else None))
                for s in specs if s.key != "analysts"
            }
            analysts_spec = next((s for s in specs if s.key == "analysts"), None)
            if analysts_spec is not None:
                intel_tasks["analysts"] = tg.create_task(
                    _run_analysts(run, analysts_spec, symbol, intel_tasks.get("quote"), deadline)
                )
            core = [*source_tasks, *(t for k, t in intel_tasks.items() if k not in TAIL_KEYS)]
            tg.create_task(_set_when_done(core, core_done))
            # Fail fast on typos: decide "unknown symbol" as soon as market data and
            # sources have answered (raising here cancels the slower tasks).
            tg.create_task(_early_unknown_check(symbol, company, source_tasks, skipped_runs, intel_tasks))
    except* UnknownSymbol as group:
        _unknown.set(symbol, str(group.exceptions[0]))
        raise group.exceptions[0] from None
    source_runs = [t.result() for t in source_tasks] + skipped_runs
    intel = {k: t.result() for k, t in intel_tasks.items()}
    for out in intel.values():
        if out.pending is not None:
            out.pending.add_done_callback(functools.partial(_straggler_done, symbol, now))
    skipped_keys = {s.key for s in specs if s.skip}
    status = {
        k: (("ok" if has_data(o.value) else "empty") if o.ok else f"error: {o.error}")
        for k, o in intel.items() if k not in skipped_keys
    }

    def val(key: str) -> Any:
        out = intel.get(key)
        return out.value if out is not None and out.ok else None

    inputs = AnalysisInputs(
        company=company,
        now=now,
        engine_name=engine_task.result(),
        profile=val("profile"),
        quote=val("quote"),
        technicals=val("technicals"),
        analysts=val("analysts"),
        insiders=val("insiders"),
        earnings=val("earnings"),
        filings=list(val("filings") or []),
        calendar_catalysts=list(val("calendar") or []),
        tone=val("tone"),
        wiki_views=val("wiki") or None,
        source_runs=source_runs,
        previous=previous_task.result(),
        intel_status=status,
    )
    analysis = await _synthesize(run, inputs)
    analysis = analysis.model_copy(update={
        "ticker": symbol, "generated_at": now, "cached": False,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    })
    await _persist_and_alert(analysis)

    _cache.set(symbol, analysis)
    _latest[symbol] = analysis
    _latest.move_to_end(symbol)
    while len(_latest) > LATEST_KEEP:
        _latest.popitem(last=False)

    v = analysis.verdict
    await run.emit(ProgressEvent(
        stage="done", key="done", label="Analysis complete", status="ok",
        count=len(analysis.signals), ms=analysis.elapsed_ms, detail=f"SentiNET {v.score} · {v.label}",
    ))
    return analysis


async def _set_when_done(tasks: list[asyncio.Task[Any]], event: asyncio.Event) -> None:
    if tasks:
        await asyncio.wait(tasks)
    event.set()


def _straggler_done(symbol: str, generated_at: datetime, task: asyncio.Task[Any]) -> None:
    """A kept-alive provider call finished after its run: if it brought data, the cached
    analysis of that run is superseded (the next load recomputes, now with this data)."""
    if task.cancelled() or task.exception() is not None or not has_data(task.result()):
        return
    _mark_stale(symbol, generated_at)
    logger.info("late data for %s (%s); cached analysis superseded", symbol, task.get_name())


def _time_box(limit: float, deadline: float | None) -> float:
    """`limit`, cut to what is left of the run budget (never below `MIN_TIME_BOX`)."""
    if deadline is None:
        return limit
    return min(limit, max(MIN_TIME_BOX, deadline - time.monotonic()))


async def _resolve(run: _Run, deadline: float | None = None) -> CompanyRef:
    """Resolve the company; on failure fall back to a bare ref so the rest still runs."""
    await run.emit(ProgressEvent(stage="resolve", key="resolve", label="Resolve symbol", status="running"))
    # Kept alive: a slow first resolve (SEC ticker map download) still lands in its cache.
    out = await run_bounded(deferred("app.resolve.symbols", "resolve_company", run.symbol),
                            _time_box(min(RESOLVE_TIMEOUT, settings.intel_timeout), deadline),
                            keep_alive=True, name="resolve")
    company = out.value if out.ok and isinstance(out.value, CompanyRef) else None
    if company is None:
        error = f"slow resolver ({out.ms / 1000:.0f}s)" if out.timed_out else (out.error or "unexpected payload")
        await run.emit(ProgressEvent(stage="resolve", key="resolve", label="Resolve symbol", status="error",
                                     ms=out.ms, detail=f"{error}; continuing with the bare symbol"))
        return bare_company(run.symbol)
    parts = [company.name, company.quote_type] + ([company.exchange] if company.exchange else [])
    await run.emit(ProgressEvent(stage="resolve", key="resolve", label="Resolve symbol", status="ok",
                                 ms=out.ms, detail=" · ".join(parts)))
    return company


async def resolve_or_bare(symbol: str, timeout: float | None = None) -> CompanyRef:
    """Resolved company for a canonical symbol, or a bare ref if resolution fails/times out."""
    out = await run_bounded(deferred("app.resolve.symbols", "resolve_company", symbol),
                            timeout or settings.intel_timeout, name="resolve")
    return out.value if out.ok and isinstance(out.value, CompanyRef) else bare_company(symbol)


def bare_company(symbol: str) -> CompanyRef:
    """Minimal ref when resolution fails, so sources can still search by symbol."""
    quote_type = "EQUITY"
    if symbol.startswith("^"):
        quote_type = "INDEX"
    elif symbol.rsplit("-", 1)[-1] in {"USD", "USDT", "USDC", "EUR", "BTC"} and "-" in symbol:
        quote_type = "CRYPTOCURRENCY"
    return CompanyRef(ticker=symbol, name=symbol, short_name=symbol, quote_type=quote_type)


def _engine_name() -> str:
    try:
        from app.nlp.engine import get_engine

        return str(get_engine().name)
    except Exception as exc:  # noqa: BLE001 - name is cosmetic; analytics loads the engine itself
        logger.warning("sentiment engine unavailable: %s", exc)
        return settings.sentiment_engine


async def _previous_snapshot(symbol: str, now: datetime) -> Snapshot | None:
    from app.storage import db

    out = await run_bounded(lambda: db.latest_snapshot(symbol, before=now - PREVIOUS_MIN_AGE),
                            STORAGE_TIMEOUT, name="previous-snapshot")
    return out.value if out.ok else None


# ---- sources ---------------------------------------------------------------- #
def _plan_sources(company: CompanyRef) -> tuple[list[Source], list[Any]]:
    """(sources to run, SourceRuns for disabled/unconfigured ones). Unsupported ones are omitted."""
    from app.analytics.inputs import SourceRun

    try:
        from app.sources.registry import enabled_sources

        pairs = enabled_sources(company)
    except Exception:
        logger.exception("source registry failed")
        return [], []
    planned: list[Source] = []
    skipped: list[SourceRun] = []
    for source, state in pairs:
        if state == "enabled":
            planned.append(source)
        elif state in ("disabled", "unconfigured"):
            skipped.append(SourceRun(source=source, status=state))
    return planned, skipped


async def _run_source(run: _Run, source: Source, company: CompanyRef, deadline: float | None = None) -> Any:
    from app.analytics.inputs import SourceRun

    out = await run_bounded(lambda: source.fetch(company), _time_box(settings.source_timeout, deadline),
                            keep_alive=False, name=f"source:{source.key}")
    if not out.ok:
        await run.emit(ProgressEvent(stage="source", key=source.key, label=source.label,
                                     status="error", ms=out.ms, detail=out.error))
        return SourceRun(source=source, status="error", latency_ms=out.ms, error=out.error)
    batch = out.value if isinstance(out.value, SourceBatch) else SourceBatch()
    state = "ok" if batch else "empty"
    await run.emit(ProgressEvent(stage="source", key=source.key, label=source.label, status=state,
                                 count=len(batch.signals), ms=out.ms, detail=_source_detail(batch)))
    return SourceRun(source=source, status=state, batch=batch, latency_ms=out.ms)


def _source_detail(batch: SourceBatch) -> str | None:
    """Terse, real-number summary for the progress panel."""
    parts = [f"{len(batch.signals)} items"] if batch.signals else []
    m = batch.metrics
    try:
        bull, bear = m.get("stocktwits_bullish"), m.get("stocktwits_bearish")
        if isinstance(bull, int) and isinstance(bear, int) and bull + bear:
            parts.append(f"{bull / (bull + bear):.0%} bulls of {bull + bear} tagged")
        if m.get("reddit_rank") is not None:
            parts.append(f"Reddit #{m['reddit_rank']} · {m.get('reddit_mentions', '?')} mentions")
        if m.get("wsb_label"):
            parts.append(f"WSB {m['wsb_label']}")
    except (TypeError, ValueError):
        pass
    if not parts and m:
        parts.append(f"{len(m)} metrics")
    return " · ".join(parts) or None


# ---- intel -------------------------------------------------------------------- #
def _intel_specs(company: CompanyRef) -> list[_IntelSpec]:
    sym = company.ticker
    na = _NOT_APPLICABLE.get(company.quote_type, set())
    asset = _ASSET_NAMES.get(company.quote_type, company.quote_type.lower())
    calls: list[tuple[str, Callable[[], Awaitable[Any]], Any]] = [
        ("profile", deferred("app.intel.market_data", "get_profile", company), Profile),
        ("quote", deferred("app.intel.market_data", "get_quote", sym), Quote),
        ("technicals", deferred("app.intel.market_data", "get_technicals", sym), Technicals),
        ("analysts", deferred("app.intel.market_data", "get_analysts", sym, None), AnalystView),
        ("earnings", deferred("app.intel.market_data", "get_earnings", sym), EarningsView),
        ("calendar", deferred("app.intel.market_data", "get_calendar_catalysts", sym), list),
        ("insiders", deferred("app.intel.market_data", "get_insiders", sym), InsiderView),
        ("filings", deferred("app.intel.sec", "get_filings", company), list),
        ("tone", deferred("app.intel.gdelt", "get_tone_trend", company, TONE_DAYS), ToneTrend),
        ("wiki", deferred("app.intel.attention", "get_wiki_pageviews", company, TONE_DAYS), list),
    ]
    specs = []
    for key, call, expect in calls:
        skip = f"n/a for {asset}" if key in na else None
        if key == "filings" and not skip and not company.cik:
            skip = "no SEC registrant found for this symbol"
        specs.append(_IntelSpec(key, _INTEL_LABELS[key], call, expect, skip))
    return specs


async def _run_intel(run: _Run, spec: _IntelSpec, deadline: float | None = None,
                     tail: asyncio.Event | None = None) -> Outcome[Any]:
    """One intel task, time-boxed by the run budget (and the tail rule when `tail` is given)."""
    if spec.skip:
        return Outcome()
    out = await run_bounded(spec.call, _time_box(settings.intel_timeout, deadline), keep_alive=spec.keep_alive,
                            name=f"intel:{spec.key}", until=tail, grace=TAIL_GRACE)
    if out.ok and out.value is not None and not isinstance(out.value, spec.expect):
        logger.warning("intel %s returned %s, expected %s", spec.key, type(out.value).__name__, spec.expect)
        out = Outcome(error=f"unexpected payload ({type(out.value).__name__})", ms=out.ms)
    if out.ok and isinstance(out.value, list) and spec.key in ("calendar", "filings"):
        want = Catalyst if spec.key == "calendar" else Filing
        out.value = [x for x in out.value if isinstance(x, want)]
    if out.ok:
        state = "ok" if has_data(out.value) else "empty"
        await run.emit(ProgressEvent(stage="intel", key=spec.key, label=spec.label, status=state,
                                     count=_count(out.value), ms=out.ms, detail=_intel_detail(spec.key, out.value)))
    else:
        await run.emit(ProgressEvent(stage="intel", key=spec.key, label=spec.label, status="error",
                                     ms=out.ms, detail=out.error))
    return out


async def _run_analysts(run: _Run, spec: _IntelSpec, symbol: str,
                        quote_task: asyncio.Task[Outcome[Any]] | None, deadline: float | None) -> Outcome[Any]:
    """Analysts need the live price for upside; wait for the quote first (its own time box)."""
    if spec.skip:
        return Outcome()
    price = None
    if quote_task is not None:
        quote_out = await quote_task
        if quote_out.ok and isinstance(quote_out.value, Quote):
            price = quote_out.value.price
    call = deferred("app.intel.market_data", "get_analysts", symbol, price)
    return await _run_intel(run, _IntelSpec(spec.key, spec.label, call, spec.expect), deadline)


def _count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, ToneTrend):
        return len(value.series)
    if isinstance(value, AnalystView):
        return value.total
    if isinstance(value, InsiderView):
        return len(value.transactions)
    if isinstance(value, EarningsView):
        return len(value.history)
    return None


def _intel_detail(key: str, value: Any) -> str | None:
    """One-line, number-backed summary of an intel result for the progress panel."""
    try:
        if isinstance(value, Quote) and value.price is not None:
            chg = f" {value.change_pct:+.2f}%" if value.change_pct is not None else ""
            return f"{value.price:,.2f} {value.currency or ''}{chg}".replace("  ", " ").strip()
        if isinstance(value, Profile):
            return " · ".join(x for x in (value.sector, value.industry) if x) or value.name
        if isinstance(value, Technicals):
            bits = []
            if value.trend:
                bits.append(value.trend)
            if value.rsi_14 is not None:
                bits.append(f"RSI {value.rsi_14:.0f}")
            if value.return_1m is not None:
                bits.append(f"1M {value.return_1m:+.1f}%")
            return " · ".join(bits) or None
        if isinstance(value, AnalystView):
            bits = [f"{value.total} analysts"] if value.total else []
            if value.consensus:
                bits.append(value.consensus.replace("_", " "))
            if value.upside_pct is not None:
                bits.append(f"target {value.upside_pct:+.0f}%")
            return " · ".join(bits) or None
        if isinstance(value, EarningsView):
            if value.next_date is not None:
                return f"next {value.next_date.isoformat()}" + (
                    f" (in {value.days_until}d)" if value.days_until is not None else "")
            return f"{len(value.history)} past reports" if value.history else None
        if isinstance(value, InsiderView):
            return f"{value.buys} buys / {value.sells} sells ({value.window_days}d)"
        if isinstance(value, ToneTrend) and value.tone_7d is not None:
            return f"7d tone {value.tone_7d:+.2f}" + (
                f" vs 30d {value.tone_30d:+.2f}" if value.tone_30d is not None else "")
        if key == "wiki" and isinstance(value, list) and value:
            recent = [float(v) for _, v in value[-7:]]
            return f"{sum(recent) / len(recent):,.0f} views/day (7d)"
        if key == "filings" and isinstance(value, list):
            high = sum(1 for f in value if getattr(f, "importance", "") == "high")
            return f"{len(value)} filings" + (f" · {high} high-importance" if high else "")
        if key == "calendar" and isinstance(value, list) and value:
            return "; ".join(c.title for c in value[:2])
    except (TypeError, ValueError, AttributeError):
        return None
    return None


# ---- synthesis, persistence ------------------------------------------------------ #
async def _early_unknown_check(symbol: str, company: CompanyRef, source_tasks: list[asyncio.Task[Any]],
                               skipped_runs: list[Any], intel_tasks: dict[str, asyncio.Task[Outcome[Any]]]) -> None:
    decisive = {k: intel_tasks[k] for k in ("quote", "technicals", "profile") if k in intel_tasks}
    pending = [*decisive.values(), *source_tasks]
    if pending:
        await asyncio.wait(pending)
    _check_known(symbol, company, [t.result() for t in source_tasks] + skipped_runs,
                 {k: t.result() for k, t in decisive.items()})


def _symbol_coverage(run: Any) -> bool:
    """True when a source answered *for this symbol*: a ticker-specific item (symbol
    stream / issuer-tagged feed) or crowd metrics looked up by the symbol."""
    batch = getattr(run, "batch", None)
    if getattr(run, "status", None) != "ok" or batch is None:
        return False
    return (any(s.ticker_specific for s in batch.signals)
            or any(k.startswith(_SYMBOL_KEYED_METRICS) for k in batch.metrics))


def _check_known(symbol: str, company: CompanyRef, runs: list[Any], intel: dict[str, Outcome[Any]]) -> None:
    """404 when market data positively says "no such instrument" and nothing keyed by the symbol exists.

    Yahoo's quote is the authority on whether a symbol trades: only an *answer*
    of "nothing" counts (a failed quote could be an outage, so the degraded
    analysis proceeds and reports it). Keyword searches (news feeds, Bluesky,
    Hacker News, GDELT, Wikipedia) say nothing about existence — "APPL" or a
    delisted "SIVB" still matches articles — so only symbol-keyed coverage
    (StockTwits stream, issuer-tagged items, Reddit/WSB boards) counts.
    """
    if company.cik or company.name != symbol:
        return
    quote = intel.get("quote")
    if quote is None or not quote.ok or has_data(quote.value):
        return
    technicals = intel.get("technicals")
    if technicals is not None and technicals.ok and has_data(technicals.value):
        return
    profile = intel.get("profile")
    p = profile.value if profile is not None and profile.ok else None
    if isinstance(p, Profile) and (p.name != symbol or p.sector or p.exchange):
        return
    if any(_symbol_coverage(r) for r in runs):
        return
    raise UnknownSymbol(_unknown_message(symbol))


def _unknown_message(symbol: str) -> str:
    return (f"No market data or symbol coverage found for '{symbol}' — it may be mistyped or delisted. "
            "Check the symbol (exchange suffix like SHOP.TO, crypto like BTC-USD).")


async def ensure_known(symbol: str) -> CompanyRef:
    """The company behind a symbol about to be watched/alerted; `UnknownSymbol` (404) if nothing knows it.

    Cheap: a stored snapshot or recent analysis proves it; otherwise the
    (cached) resolver, then one quote lookup. Only a positive "no quote" for an
    unresolvable symbol rejects — provider outages accept (the monitor backs
    off from tickers that keep failing).
    """
    from app.storage import db

    recent = latest_analysis(symbol)
    if recent is not None:
        name = recent.profile.name if recent.profile is not None else symbol
        return CompanyRef(ticker=symbol, name=name or symbol, short_name=name or symbol)
    company = await resolve_or_bare(symbol, timeout=RESOLVE_TIMEOUT)
    if company.cik or company.name != symbol:
        return company
    stored = await run_bounded(lambda: db.latest_snapshot(symbol), STORAGE_TIMEOUT, name="known-snapshot")
    if stored.ok and stored.value is not None:
        return company
    quote = await run_bounded(deferred("app.intel.market_data", "get_quote", symbol), RESOLVE_TIMEOUT,
                              name="known-quote")
    if not quote.ok or has_data(quote.value):
        return company
    raise UnknownSymbol(_unknown_message(symbol))


async def _synthesize(run: _Run, inputs: Any) -> Analysis:
    n_texts = sum(len(r.batch.signals) for r in inputs.source_runs if r.batch is not None)
    await run.emit(ProgressEvent(stage="analytics", key="synthesis", label="Scoring & synthesis",
                                 status="running", count=n_texts,
                                 detail=f"{n_texts} texts · {inputs.engine_name} engine"))
    t0 = time.perf_counter()
    try:
        from app.analytics.build import build_analysis

        analysis = await asyncio.wait_for(run_cpu(build_analysis, inputs), ANALYTICS_TIMEOUT)
        if not isinstance(analysis, Analysis):
            raise TypeError(f"build_analysis returned {type(analysis).__name__}")
    except Exception as exc:
        logger.exception("analytics failed for %s", run.symbol)
        detail = describe_error(exc)
        await run.emit(ProgressEvent(stage="analytics", key="synthesis", label="Scoring & synthesis",
                                     status="error", ms=int((time.perf_counter() - t0) * 1000), detail=detail))
        raise AnalysisFailed(f"Synthesis failed for {run.symbol} ({detail}). Please retry.") from exc
    ms = int((time.perf_counter() - t0) * 1000)
    await run.emit(ProgressEvent(
        stage="analytics", key="synthesis", label="Scoring & synthesis", status="ok",
        count=len(analysis.signals), ms=ms,
        detail=f"{analysis.sentiment.n} signals kept · {len(analysis.narratives)} narratives · "
               f"{len(analysis.insights)} insights",
    ))
    return analysis


async def _persist_and_alert(analysis: Analysis) -> None:
    """Store the snapshot and evaluate alert rules; failures are logged, never raised."""
    from app.storage import db

    out = await run_bounded(lambda: db.save_snapshot(analysis), STORAGE_TIMEOUT, name="save-snapshot")
    if not out.ok:
        logger.warning("snapshot not saved for %s: %s", analysis.ticker, out.error)
        return
    from app.services.alerts import process_analysis

    res = await run_bounded(lambda: process_analysis(analysis, out.value), STORAGE_TIMEOUT, name="alerts")
    if not res.ok:
        logger.warning("alert evaluation failed for %s: %s", analysis.ticker, res.error)


def _ago(seconds: int) -> str:
    if seconds < 90:
        return f"{max(seconds, 0)}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"
