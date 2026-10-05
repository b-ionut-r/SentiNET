"""Sentiment lab: score arbitrary texts with the production engine, explainably.

Each text gets the same treatment as a live signal (score, label, confidence,
driver terms, themes, events); with a ticker, also its relevance to that
company. The summary is a confidence- (and relevance-) weighted mean.

Lab work is user-submitted CPU work, so it is bounded and kept away from
analyses: at most `MAX_TOTAL_CHARS` per request (~1-2 s of scoring), one
request scored at a time on the lab's own worker thread (never the shared CPU
pool that analyses synthesize on), and at most `MAX_WAITING` queued behind it;
beyond that the lab answers 503 "busy" with `Retry-After` instead of queueing
work that would stall every analysis on the server.
"""
from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.schemas import (
    Driver,
    ScoredText,
    ScoreRequest,
    ScoreResponse,
    SentimentStat,
    ThemeStat,
)
from app.services.analyzer import normalize, resolve_or_bare
from app.services.errors import Busy, InvalidInput, Unavailable
from app.services.tasks import describe_error
from app.sources.base import CompanyRef

MAX_TEXT_CHARS = 10_000
# Total characters scored per request: the engine scores ~100-200k chars/s, so
# one full request is ~1-2 s of CPU. Mirrored in the web lab.
MAX_TOTAL_CHARS = 250_000
MAX_WAITING = 2  # requests queued behind the one being scored; more → 503 busy
NEUTRAL_BAND = 0.05  # fallback when the engine module doesn't export `label_for`

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lab")
_gate: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None
_waiting = 0


def validate_texts(texts: list[str]) -> list[str]:
    """Strip, drop blanks, and bound per-text and total length (`InvalidInput` with a precise message)."""
    cleaned = [t.strip() for t in texts]
    for i, text in enumerate(cleaned, start=1):
        if len(text) > MAX_TEXT_CHARS:
            raise InvalidInput(f"Text #{i} has {len(text):,} characters; the limit is {MAX_TEXT_CHARS:,}.")
    kept = [t for t in cleaned if t]
    if not kept:
        raise InvalidInput("Provide at least one non-empty text to score.")
    total = sum(len(t) for t in kept)
    if total > MAX_TOTAL_CHARS:
        raise InvalidInput(
            f"{len(kept)} texts with {total:,} characters in all; one run scores at most "
            f"{MAX_TOTAL_CHARS:,} characters. Split the batch into smaller runs."
        )
    return kept


def _slot() -> asyncio.Semaphore:
    """The one-at-a-time lab semaphore of the running loop (tests use several loops)."""
    global _gate, _waiting
    loop = asyncio.get_running_loop()
    if _gate is None or _gate[0] is not loop:
        _gate, _waiting = (loop, asyncio.Semaphore(1)), 0
    return _gate[1]


async def _run_admitted(fn: Any, *args: Any) -> Any:
    """Run `fn(*args)` on the lab thread, one at a time, refusing when the queue is full.

    The slot is held until the thread finishes — even if the caller disconnects —
    so an abandoned request still counts against the queue while it burns CPU.
    """
    global _waiting
    slot = _slot()
    if slot.locked() and _waiting >= MAX_WAITING:
        raise Busy("The sentiment lab is busy scoring other requests; retry in a few seconds.")
    _waiting += 1
    try:
        await slot.acquire()
    finally:
        _waiting -= 1
    try:
        fut = asyncio.get_running_loop().run_in_executor(_executor, functools.partial(fn, *args))
    except BaseException:
        slot.release()
        raise

    def finished(f: asyncio.Future[Any]) -> None:
        slot.release()
        if not f.cancelled():
            f.exception()  # mark retrieved: the caller may have gone away

    fut.add_done_callback(finished)
    return await asyncio.shield(fut)


def _label_fn() -> Any:
    try:
        from app.nlp.engine import label_for

        return label_for
    except (ImportError, AttributeError):
        return lambda s: "bullish" if s > NEUTRAL_BAND else "bearish" if s < -NEUTRAL_BAND else "neutral"


def _score_sync(texts: list[str], company: CompanyRef | None) -> ScoreResponse:
    from app.nlp.engine import get_engine
    from app.nlp.pipeline import analyze_texts
    from app.nlp.themes import THEMES

    analyses = analyze_texts(texts, None, company)
    relevance_fn = None
    if company is not None:
        from app.nlp.relevance import relevance as relevance_fn
    label_for = _label_fn()

    results: list[ScoredText] = []
    for text, a in zip(texts, analyses, strict=True):
        results.append(ScoredText(
            text=text,
            score=round(a.score, 4),
            label=a.label,
            confidence=round(a.confidence, 4),
            drivers=[Driver(term=t, impact=round(v, 4)) for t, v in a.drivers],
            themes=list(a.themes),
            events=[e.key for e in a.events],
            relevance=round(relevance_fn(text, company), 4) if relevance_fn else None,
        ))

    weights = [max(r.confidence, 0.05) * (r.relevance if r.relevance is not None else 1.0) for r in results]
    total = sum(weights)
    mean = sum(w * r.score for w, r in zip(weights, results, strict=True)) / total if total > 0 else 0.0
    summary = SentimentStat(
        score=round(mean, 4),
        label=label_for(mean),
        n=len(results),
        bullish=sum(r.label == "bullish" for r in results),
        bearish=sum(r.label == "bearish" for r in results),
        neutral=sum(r.label == "neutral" for r in results),
        confidence=round(sum(r.confidence for r in results) / len(results), 4),
    )

    by_theme: dict[str, list[float]] = {}
    for r in results:
        for theme in r.themes:
            by_theme.setdefault(theme, []).append(r.score)
    themes = sorted(
        (ThemeStat(theme=k, label=THEMES.get(k, k.replace("_", " ").title()), count=len(v),
                   score=round(sum(v) / len(v), 4), share=round(len(v) / len(results), 4))
         for k, v in by_theme.items()),
        key=lambda t: (-t.count, t.theme),
    )
    return ScoreResponse(engine=str(get_engine().name), results=results, summary=summary, themes=themes)


async def score_texts(req: ScoreRequest) -> ScoreResponse:
    """Score the request's texts (raises `InvalidInput` for bad input, `Unavailable` if NLP fails)."""
    texts = validate_texts(req.texts)
    company = None
    if req.ticker and req.ticker.strip():
        company = await resolve_or_bare(normalize(req.ticker), timeout=8.0)
    try:
        return await _run_admitted(_score_sync, texts, company)
    except Busy:
        raise
    except Exception as exc:
        raise Unavailable(f"Scoring failed ({describe_error(exc)}).") from exc
