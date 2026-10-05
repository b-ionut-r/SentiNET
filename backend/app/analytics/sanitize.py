"""Defensive normalization of structured intel before it is analysed.

Upstream providers (yfinance, SEC, GDELT) occasionally hand back NaN/inf
numbers or timezone-naive datetimes. One such value must neither crash the
analysis nor silently poison a score, so before anything is computed:

* a non-finite float becomes the field's numeric default (e.g. `0.0` for
  `InsiderView.sell_value`) or `None` ("not available");
* non-finite entries of numeric lists / series are dropped;
* a naive datetime is read as UTC (these are day-level stamps such as analyst
  action dates, where the time zone cannot shift the meaning).

Text signals are not touched here: `prepare` treats naive signal timestamps as
undated rather than guessing.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from app.analytics.inputs import AnalysisInputs

M = TypeVar("M", bound=BaseModel)


def clean_model(model: M) -> M:
    """A copy of `model` with non-finite floats and naive datetimes fixed (itself when clean)."""
    updates: dict[str, Any] = {}
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        cleaned = _clean(value, field.default)
        if cleaned is not value:
            updates[name] = cleaned
    return model.model_copy(update=updates) if updates else model


def _numeric_default(default: Any) -> float | None:
    if isinstance(default, (int, float)) and not isinstance(default, bool) and math.isfinite(default):
        return float(default)
    return None


def _clean(value: Any, default: Any = None) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _numeric_default(default)
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    if isinstance(value, BaseModel):
        return clean_model(value)
    if isinstance(value, list):
        out: list[Any] = []
        changed = False
        for item in value:
            cleaned = _clean(item)
            if cleaned is None and item is not None:  # a non-finite list entry: drop it
                changed = True
                continue
            changed |= cleaned is not item
            out.append(cleaned)
        return out if changed else value
    return value


def clean_series(series: list[tuple[date, float]] | None) -> list[tuple[date, float]] | None:
    """Daily (date, value) pairs with missing / non-finite values dropped."""
    if series is None:
        return None
    out = []
    for day, value in series:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append((day, v))
    return out


def _opt(model: M | None) -> M | None:
    return clean_model(model) if model is not None else None


def sanitize_inputs(inputs: AnalysisInputs) -> AnalysisInputs:
    """`inputs` with every structured-intel model normalized (see module docstring)."""
    now = inputs.now if inputs.now.tzinfo is not None else inputs.now.replace(tzinfo=UTC)
    return replace(
        inputs,
        now=now,
        profile=_opt(inputs.profile),
        quote=_opt(inputs.quote),
        technicals=_opt(inputs.technicals),
        analysts=_opt(inputs.analysts),
        insiders=_opt(inputs.insiders),
        earnings=_opt(inputs.earnings),
        filings=[clean_model(f) for f in inputs.filings],
        calendar_catalysts=[clean_model(c) for c in inputs.calendar_catalysts],
        tone=_opt(inputs.tone),
        wiki_views=clean_series(inputs.wiki_views),
        previous=_opt(inputs.previous),
    )
