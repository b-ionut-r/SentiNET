"""FinBERT integration tests.

FinBERT's weights download from Hugging Face at first use, which isn't possible
in a no-egress environment. So we test the two things that matter and are
verifiable offline:
  1. The score-mapping logic (positive/negative/neutral -> [-1, 1]) via a fake
     transformers pipeline injected in place of the real one.
  2. The factory's graceful fallback to VADER when FinBERT can't be loaded.
"""
from app.sentiment.factory import get_engine, reset_engine
from app.sentiment.finbert_engine import FinBertEngine


def test_finbert_score_mapping(monkeypatch):
    # Bypass __init__ (which requires transformers/torch) — we only test the
    # score-mapping logic here, with a fake pipeline standing in for the model.
    engine = FinBertEngine.__new__(FinBertEngine)
    engine._pipe = None

    def fake_pipeline(texts):
        # Mimic transformers sentiment-analysis output shape.
        out = []
        for t in texts:
            if "beat" in t:
                out.append({"label": "positive", "score": 0.92})
            elif "miss" in t:
                out.append({"label": "negative", "score": 0.88})
            else:
                out.append({"label": "neutral", "score": 0.99})
        return out

    engine._pipe = fake_pipeline  # bypass real model load
    scores = engine.score_batch(["earnings beat", "big miss", "no change"])
    assert scores[0] > 0.9      # positive -> +conf
    assert scores[1] < -0.8     # negative -> -conf
    assert scores[2] == 0.0     # neutral -> 0


def test_factory_falls_back_to_vader(monkeypatch):
    reset_engine()
    monkeypatch.setattr("app.config.settings.sentiment_engine", "finbert")
    # Force FinBERT construction to fail so the factory must fall back.
    import app.sentiment.finbert_engine as fb

    def boom(self):
        raise RuntimeError("torch not installed")

    monkeypatch.setattr(fb.FinBertEngine, "__init__", boom)
    engine = get_engine()
    assert engine.name == "vader"
    reset_engine()
