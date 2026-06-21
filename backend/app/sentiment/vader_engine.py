"""VADER-based sentiment, tuned with a small finance lexicon.

VADER is pure-Python, instant to load and strong on the social-media register
(caps, negation, intensifiers, emoji). We extend its lexicon with finance terms
so phrases like "beat", "guidance cut" or "to the moon" score correctly.
"""
from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Domain terms VADER doesn't know, mapped to roughly [-4, 4] valence.
FINANCE_LEXICON: dict[str, float] = {
    "beat": 2.2, "beats": 2.2, "outperform": 2.5, "outperforms": 2.5,
    "upgrade": 2.6, "upgraded": 2.6, "bullish": 3.0, "rally": 2.4,
    "rallies": 2.4, "soar": 3.2, "soars": 3.2, "surge": 3.0, "surges": 3.0,
    "jumps": 2.2, "gains": 1.8, "record": 1.6, "breakout": 2.2,
    "moon": 2.8, "mooning": 3.0, "buyback": 1.8, "dividend": 1.2,
    "long": 1.0, "calls": 1.2, "tendies": 2.5, "squeeze": 1.5,
    "miss": -2.2, "misses": -2.2, "missed": -2.2, "downgrade": -2.6,
    "downgraded": -2.6, "bearish": -3.0, "plunge": -3.2, "plunges": -3.2,
    "plummet": -3.2, "crash": -3.4, "crashes": -3.4, "selloff": -2.6,
    "sell-off": -2.6, "tank": -2.8, "tanks": -2.8, "tanking": -2.8,
    "dump": -2.4, "dumps": -2.4, "dumping": -2.4, "puts": -1.4,
    "short": -1.2, "bankruptcy": -3.5, "bankrupt": -3.5, "fraud": -3.4,
    "lawsuit": -1.8, "investigation": -1.6, "halt": -1.5, "halted": -1.5,
    "guidance": -0.4, "warning": -1.8, "warns": -1.8, "cut": -1.4,
    "cuts": -1.4, "layoffs": -2.0, "loss": -1.6, "losses": -1.6,
    "bagholder": -2.5, "rugpull": -3.2, "overvalued": -1.8, "bubble": -1.6,
}


class VaderEngine:
    name = "vader"

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()
        self._analyzer.lexicon.update(FINANCE_LEXICON)

    def score_batch(self, texts: list[str]) -> list[float]:
        scores = []
        for text in texts:
            if not text:
                scores.append(0.0)
                continue
            scores.append(self._analyzer.polarity_scores(text)["compound"])
        return scores
