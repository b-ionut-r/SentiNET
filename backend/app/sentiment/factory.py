"""Pick a sentiment engine from config, with a safe fallback to VADER."""
from __future__ import annotations

import logging

from app.config import settings
from app.sentiment.base import SentimentEngine
from app.sentiment.vader_engine import VaderEngine

logger = logging.getLogger(__name__)

_engine: SentimentEngine | None = None


def get_engine() -> SentimentEngine:
    global _engine
    if _engine is not None:
        return _engine

    choice = settings.sentiment_engine.lower().strip()
    if choice == "finbert":
        try:
            from app.sentiment.finbert_engine import FinBertEngine

            _engine = FinBertEngine()
            logger.info("Using FinBERT sentiment engine")
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("FinBERT unavailable (%s); falling back to VADER", exc)
            _engine = VaderEngine()
    else:
        _engine = VaderEngine()
        logger.info("Using VADER sentiment engine")
    return _engine


def reset_engine() -> None:
    """Test helper to clear the cached singleton."""
    global _engine
    _engine = None
