"""Sentiment lab: score arbitrary texts with the production engine, explainably.

Each text gets the same treatment as a live signal (score, label, confidence,
driver terms, themes, events); with a ticker, also its relevance to that
company. The summary is a confidence- (and relevance-) weighted mean.
"""
from __future__ import annotations

from typing import Any

from app.core.sync import run_cpu
from app.schemas import (
    Driver,
    ScoredText,
    ScoreRequest,
    ScoreResponse,
    SentimentStat,
    ThemeStat,
)
from app.services.analyzer import normalize, resolve_or_bare
from app.services.errors import InvalidInput, Unavailable
from app.services.tasks import describe_error
from app.sources.base import CompanyRef

MAX_TEXT_CHARS = 10_000
NEUTRAL_BAND = 0.05  # fallback when the engine module doesn't export `label_for`


def validate_texts(texts: list[str]) -> list[str]:
    """Strip, drop blanks, and bound length (`InvalidInput` with a precise message)."""
    cleaned = [t.strip() for t in texts]
    for i, text in enumerate(cleaned, start=1):
        if len(text) > MAX_TEXT_CHARS:
            raise InvalidInput(f"Text #{i} has {len(text):,} characters; the limit is {MAX_TEXT_CHARS:,}.")
    kept = [t for t in cleaned if t]
    if not kept:
        raise InvalidInput("Provide at least one non-empty text to score.")
    return kept


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
        return await run_cpu(_score_sync, texts, company)
    except Exception as exc:
        raise Unavailable(f"Scoring failed ({describe_error(exc)}).") from exc
