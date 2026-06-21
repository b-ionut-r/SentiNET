"""Trending keyword extraction over the scored signals."""
from __future__ import annotations

from app.schemas import Signal, TrendingKeyword
from app.utils.text import extract_keywords


def compute_trending(signals: list[Signal], ticker: str, top_n: int = 12) -> list[TrendingKeyword]:
    texts = [s.text for s in signals]
    pairs = extract_keywords(texts, top_n=top_n, ticker=ticker)
    return [TrendingKeyword(keyword=k, count=c) for k, c in pairs if c > 1]
