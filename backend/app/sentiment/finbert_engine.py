"""Optional FinBERT engine — lazy-loaded, never imported at startup.

This keeps the heavy torch + ~440MB model out of the default path. It is only
constructed when SENTIMENT_ENGINE=finbert, and the model itself loads on first
use. Requires `transformers` and `torch` to be installed separately.
"""
from __future__ import annotations


class FinBertEngine:
    name = "finbert"

    def __init__(self) -> None:
        # Verify the heavy deps are importable up front so the engine factory
        # can fall back to VADER if they're missing. The model weights still
        # load lazily on first score_batch (no download at construction).
        import importlib.util

        for pkg in ("transformers", "torch"):
            if importlib.util.find_spec(pkg) is None:
                raise RuntimeError(f"FinBERT requires '{pkg}' to be installed")
        self._pipe = None  # model loaded lazily on first score_batch

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline  # imported lazily on purpose

        self._pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=256,
        )

    def score_batch(self, texts: list[str]) -> list[float]:
        self._ensure_loaded()
        assert self._pipe is not None
        out = self._pipe([t[:512] or " " for t in texts])
        scores = []
        for item in out:
            label = item["label"].lower()
            conf = float(item["score"])
            if label == "positive":
                scores.append(conf)
            elif label == "negative":
                scores.append(-conf)
            else:
                scores.append(0.0)
        return scores
