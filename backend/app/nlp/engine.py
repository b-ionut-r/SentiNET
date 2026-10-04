"""Sentiment engines and the process-wide engine registry.

The default **Sentinel** engine scores market text with finance evidence
(`app.nlp.rules`: lexicon, analyst/earnings/guidance rules, direction x metric
composition, negation/contrast/hedge scopes) blended with a finance-corrected
VADER, weighted by register: headlines lean on finance evidence, social posts
give VADER (caps, emphasis, emoji) more say.

    engine = get_engine()
    [a] = engine.score(["Nvidia price target raised to $250 from $220 at MS"], ["news"])
    a.score, a.label, a.confidence, a.drivers   # 0.6, "bullish", 0.8, [("price target raised ...", 0.6)]

Scores are in [-1, 1]; ``label_for`` maps them to labels with ``NEUTRAL_BAND``.
Weak or self-cancelling evidence is pulled into the neutral band on purpose:
most market text is neutral, and a single soft word should not flip a label.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.nlp import lexicon as lx
from app.nlp import rules
from app.nlp.rules import Evidence, extract
from app.nlp.types import Label, SentimentEngine, TextAnalysis

logger = logging.getLogger(__name__)

NEUTRAL_BAND = 0.05
"""|score| <= NEUTRAL_BAND is labelled "neutral"."""

MAX_DRIVERS = 5
SOCIAL_KINDS = frozenset({"social"})


def label_for(score: float) -> Label:
    """Label a score in [-1, 1] using the shared neutral band."""
    if score > NEUTRAL_BAND:
        return "bullish"
    if score < -NEUTRAL_BAND:
        return "bearish"
    return "neutral"


@dataclass(frozen=True)
class SentinelParams:
    """Blend/calibration knobs (tuned on the Twitter-financial *train* split only)."""

    scale: float = 1.2  # raw evidence -> tanh(raw / scale)
    news_fin_weight: float = 0.9  # news/analysis/filing: finance evidence 0.9 + VADER 0.1
    social_fin_weight: float = 0.55  # social: finance evidence 0.55 + VADER 0.45 (caps, emoji, slang)
    news_deadzone: float = 0.30  # blended |score| below this is pulled to 0 (soft threshold)
    social_deadzone: float = 0.15
    vader_solo: float = 0.75  # VADER's weight is scaled by this when no finance evidence exists


def _finance_vader() -> SentimentIntensityAnalyzer:
    """VADER with words removed whose sense differs in finance ("gross margin", "shares", "crude")."""
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon = {k: v for k, v in analyzer.lexicon.items() if k not in lx.VADER_NEUTRALIZE}
    return analyzer


class VaderEngine:
    """Plain VADER (benchmark baseline). Standard +-0.05 compound thresholds."""

    name = "vader"

    def __init__(self) -> None:
        self._vader = SentimentIntensityAnalyzer()

    def score(self, texts: list[str], kinds: Optional[list[str]] = None) -> list[TextAnalysis]:
        out = []
        for text in texts:
            c = self._vader.polarity_scores(text or "")["compound"] if text else 0.0
            out.append(TextAnalysis(score=round(c, 4), label=label_for(c), confidence=round(abs(c), 3)))
        return out


class SentinelEngine:
    """Finance lexicon + rules + VADER, register-aware. Deterministic and fast."""

    name = "sentinel"

    def __init__(self, params: SentinelParams | None = None) -> None:
        self.params = params or SentinelParams()
        self._vader = _finance_vader()

    # ------------------------------------------------------------------ API
    def score(self, texts: list[str], kinds: Optional[list[str]] = None) -> list[TextAnalysis]:
        kinds = list(kinds or [])
        return [self.analyze(t, kinds[i] if i < len(kinds) else None) for i, t in enumerate(texts)]

    def analyze(self, text: str, kind: Optional[str] = "news") -> TextAnalysis:
        """Score one text. ``kind`` "social" switches to the social register."""
        if not text or not text.strip():
            return TextAnalysis(score=0.0, label="neutral", confidence=0.0)
        social = (kind or "news") in SOCIAL_KINDS
        ev = extract(text, social=social)
        return self._from_evidence(ev, social)

    def evidence(self, text: str, kind: Optional[str] = "news") -> Evidence:
        """Raw evidence (for debugging / the eval script's error analysis)."""
        return extract(text, social=(kind or "news") in SOCIAL_KINDS)

    # ------------------------------------------------------------ internals
    def _from_evidence(self, ev: Evidence, social: bool) -> TextAnalysis:
        p = self.params
        values = [h.value for h in ev.hits]
        raw = sum(values)
        mass = sum(abs(v) for v in values)
        fin = math.tanh(raw / p.scale)
        vader = self._vader.polarity_scores(ev.text)["compound"] if ev.text else 0.0
        w_fin = p.social_fin_weight if social else p.news_fin_weight
        w_vader = 1.0 - w_fin
        if not values:
            w_vader *= p.vader_solo
        dampen = (rules.QUESTION_FACTOR if ev.question else 1.0) * (rules.LISTICLE_FACTOR if ev.listicle else 1.0)
        blended = w_fin * fin + w_vader * vader * dampen
        deadzone = p.social_deadzone if social else p.news_deadzone
        mag = max(0.0, abs(blended) - deadzone) / (1.0 - deadzone)
        score = max(-1.0, min(1.0, math.copysign(mag, blended)))
        label = label_for(score)
        confidence = self._confidence(ev, raw, mass, fin, vader, blended, label, deadzone)
        drivers = self._drivers(ev, w_fin, w_vader, vader if not values else 0.0)
        return TextAnalysis(score=round(score, 4), label=label, confidence=round(confidence, 3), drivers=drivers)

    def _confidence(self, ev: Evidence, raw: float, mass: float, fin: float, vader: float, blended: float,
                    label: Label, deadzone: float) -> float:
        n_words = sum(1 for t in ev.tokens if t.kind == "w")
        shape = 1.0
        if ev.question:
            shape *= 0.75
        if ev.listicle:
            shape *= 0.8
        if ev.text_hedge or ev.hedges:
            shape *= 0.85
        if n_words < 3:
            shape *= 0.7
        elif n_words > 80:
            shape *= 0.85
        if label == "neutral":
            # certain when there is simply no evidence; less so when it is weak or cancels out
            near = min(1.0, abs(blended) / (deadzone + NEUTRAL_BAND))
            conf = 0.3 + 0.35 * (1.0 - near)
            if mass > 0.6 and abs(raw) < 0.5 * mass:
                conf *= 0.8  # mixed signals ("EPS beats, misses on revenue")
            return max(0.05, min(0.95, conf * (0.85 + 0.15 * shape)))
        strength = 1.0 - math.exp(-abs(raw) / 0.9) if mass else 0.25 * abs(vader)
        consistency = abs(raw) / mass if mass else 0.5
        agree = 1.0
        if abs(vader) >= 0.2 and fin and math.copysign(1, vader) != math.copysign(1, fin):
            agree = 0.75
        elif abs(vader) >= 0.2 and fin:
            agree = 1.08
        conf = (0.3 + 0.7 * strength) * (0.55 + 0.45 * consistency) * agree * shape
        return max(0.05, min(0.98, conf))

    def _drivers(self, ev: Evidence, w_fin: float, w_vader: float, vader_only: float) -> list[tuple[str, float]]:
        scale = self.params.scale
        merged: dict[str, float] = {}
        for h in ev.hits:
            term = " ".join(h.term.split())
            if not term:
                continue
            merged[term] = merged.get(term, 0.0) + w_fin * math.tanh(h.value / scale)
        if not merged and abs(vader_only) >= 0.05:
            # no finance evidence: explain VADER's call with the words it reacted to
            lex = self._vader.lexicon
            for t in ev.tokens:
                v = lex.get(t.text)
                if v is not None and abs(v) >= 1.0:
                    merged[t.text] = merged.get(t.text, 0.0) + w_vader * math.tanh(v / 4.0)
        ranked = sorted(merged.items(), key=lambda kv: -abs(kv[1]))
        return [(term, round(impact, 3)) for term, impact in ranked if abs(impact) >= 0.01][:MAX_DRIVERS]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_engine: Optional[SentimentEngine] = None
_lock = threading.Lock()


def _build(name: str) -> SentimentEngine:
    name = (name or "sentinel").strip().lower().replace("_", "-")
    if name == "vader":
        return VaderEngine()
    if name in ("finbert", "finbert-local"):
        from app.nlp.transformer import FinBertEngine

        return FinBertEngine(SentinelEngine())
    if name in ("finbert-api", "finbertapi", "hf"):
        from app.config import settings
        from app.nlp.transformer import FinBertApiEngine

        return FinBertApiEngine(SentinelEngine(), token=settings.hf_token)
    if name != "sentinel":
        logger.warning("unknown sentiment engine %r; using sentinel", name)
    return SentinelEngine()


def get_engine() -> SentimentEngine:
    """Process-wide engine chosen by ``settings.sentiment_engine`` (falls back to Sentinel)."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from app.config import settings

                try:
                    _engine = _build(settings.sentiment_engine)
                except Exception as exc:  # noqa: BLE001 - any optional-engine failure degrades to sentinel
                    logger.warning("sentiment engine %r unavailable (%s); using sentinel",
                                   settings.sentiment_engine, exc)
                    _engine = SentinelEngine()
    return _engine


def reset_engine() -> None:
    """Drop the cached engine (tests, or after changing settings)."""
    global _engine
    with _lock:
        _engine = None
