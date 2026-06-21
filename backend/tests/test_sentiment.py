"""VADER engine + labeling sanity checks (offline)."""
from app.sentiment.base import label_for
from app.sentiment.vader_engine import VaderEngine


def test_bullish_text_scores_positive():
    engine = VaderEngine()
    [score] = engine.score_batch(["Earnings beat expectations, stock soars to record high!"])
    assert score > 0.2
    assert label_for(score) == "bullish"


def test_bearish_text_scores_negative():
    engine = VaderEngine()
    [score] = engine.score_batch(["Massive miss, guidance cut, shares crash and plunge"])
    assert score < -0.2
    assert label_for(score) == "bearish"


def test_neutral_text_is_neutral():
    engine = VaderEngine()
    [score] = engine.score_batch(["The company will report results next week."])
    assert label_for(score) == "neutral"


def test_finance_lexicon_applies():
    engine = VaderEngine()
    [score] = engine.score_batch(["downgrade"])
    assert score < 0
