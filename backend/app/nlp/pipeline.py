"""One call from text to explained sentiment: score + themes + events.

`analyze_texts(texts, kinds, company)` runs the configured sentiment engine
on the batch and enriches each `TextAnalysis` with detected events (analyst
actions with firm/target, earnings beats, lawsuits…) and themes. Themes
implied by high-precision events come first ("pt_raise" => "analyst"), then
the phrase-based themes. With `company`, events that happened to another
entity named in the text are dropped ("Tesla rival Nikola files for
bankruptcy" is no bankruptcy for Tesla) — always pass it when the batch is
about one company.
"""
from __future__ import annotations

from collections.abc import Sequence

from app.nlp.events import EVENT_THEMES, detect_events
from app.nlp.themes import classify_themes
from app.nlp.types import TextAnalysis
from app.sources.base import CompanyRef

MAX_THEMES = 4


def _merge_themes(*groups: Sequence[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for key in group:
            if key and key not in out:
                out.append(key)
    return out[:MAX_THEMES]


def enrich(text: str, analysis: TextAnalysis, company: CompanyRef | None = None) -> TextAnalysis:
    """Attach events (attributed to `company` when given) and themes for
    `text` to an engine result (in place)."""
    events = detect_events(text, company)
    implied = [EVENT_THEMES[e.key] for e in events if e.key in EVENT_THEMES]
    analysis.events = list(analysis.events or []) + [e for e in events if e.key not in {x.key for x in analysis.events}]
    analysis.themes = _merge_themes(analysis.themes or [], implied, classify_themes(text))
    return analysis


def analyze_texts(texts: Sequence[str], kinds: Sequence[str] | None = None,
                  company: CompanyRef | None = None) -> list[TextAnalysis]:
    """Score a batch with the configured engine and add events/themes.

    `kinds[i]` ("news" | "social" | …) lets the engine adapt to register;
    `company` attributes events to the company the batch is about (see
    module docstring). Returns exactly one TextAnalysis per input text, in
    order."""
    if not texts:
        return []
    # Imported lazily: the engine is heavier and optional at import time.
    from app.nlp.engine import get_engine, target_terms

    batch = [t or "" for t in texts]
    kind_list = list(kinds) if kinds is not None else None
    targets = [target_terms(company)] * len(batch) if company is not None else None
    results = get_engine().score(batch, kind_list, targets)
    if len(results) != len(batch):
        raise RuntimeError(f"sentiment engine returned {len(results)} results for {len(batch)} texts")
    return [enrich(text, result, company) for text, result in zip(batch, results, strict=True)]


def analyze_text(text: str, kind: str | None = None, company: CompanyRef | None = None) -> TextAnalysis:
    """Convenience wrapper for a single text."""
    return analyze_texts([text], [kind] if kind else None, company)[0]
