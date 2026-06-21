"""Turn RawSignals into scored, deduplicated Signals."""
from __future__ import annotations

from app.schemas import Signal
from app.sentiment.base import label_for
from app.sentiment.factory import get_engine
from app.sources.registry import SourceResult
from app.utils.text import clean_text, is_meaningful, normalize_for_dedup


def _blend(engine_score: float, prelabeled: float | None) -> float:
    """Combine the engine score with a source-provided label hint.

    When a source ships its own bull/bear tag (StockTwits, Tradestie), we trust
    it partially but let the text-based engine score dominate.
    """
    if prelabeled is None:
        return engine_score
    return max(-1.0, min(1.0, 0.6 * engine_score + 0.4 * prelabeled))


def normalize_and_score(results: list[SourceResult]) -> list[Signal]:
    # Collect meaningful raw signals with their source weight/name.
    pending: list[tuple[str, float, object]] = []  # (source_name, weight, RawSignal)
    seen: set[str] = set()
    for result in results:
        for raw in result.signals:
            text = clean_text(raw.text)
            if not is_meaningful(text):
                continue
            key = normalize_for_dedup(text)
            if key in seen:
                continue
            seen.add(key)
            pending.append((result.source.name, result.source.weight, raw))

    if not pending:
        return []

    engine = get_engine()
    texts = [clean_text(raw.text) for _, _, raw in pending]
    scores = engine.score_batch(texts)

    signals: list[Signal] = []
    for (source_name, _weight, raw), text, escore in zip(pending, texts, scores):
        final = _blend(escore, raw.prelabeled_score)
        signals.append(
            Signal(
                source=source_name,
                text=text,
                url=raw.url,
                author=raw.author,
                timestamp=raw.timestamp,
                engagement=raw.engagement,
                score=round(final, 4),
                label=label_for(final),
            )
        )
    return signals
