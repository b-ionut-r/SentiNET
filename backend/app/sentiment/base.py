"""The SentimentEngine strategy interface.

Every engine maps a batch of texts to scores in [-1, 1]. Keeping this a small
Protocol lets us swap VADER <-> FinBERT <-> hybrid behind one seam.
"""
from __future__ import annotations

from typing import Protocol

from app.schemas import SentimentLabel

# Threshold around 0 that we treat as "neutral".
NEUTRAL_BAND = 0.05


def label_for(score: float) -> SentimentLabel:
    if score > NEUTRAL_BAND:
        return "bullish"
    if score < -NEUTRAL_BAND:
        return "bearish"
    return "neutral"


class SentimentEngine(Protocol):
    name: str

    def score_batch(self, texts: list[str]) -> list[float]:
        """Return a compound score in [-1, 1] for each input text."""
        ...
