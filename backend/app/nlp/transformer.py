"""Optional FinBERT engines, ensembled with Sentinel.

* ``finbert``     - local ``transformers`` pipeline (ProsusAI/finbert), loaded lazily on
  first use; needs ``transformers`` + ``torch`` installed (not in the default deps).
* ``finbert-api`` - Hugging Face Inference API (needs ``HF_TOKEN``); batched,
  cached, with a circuit breaker so a rate-limited or cold API never stalls an analysis.

Both blend ~0.6 transformer + 0.4 Sentinel and keep Sentinel's drivers (FinBERT
has no token-level explanation). Any failure degrades to plain Sentinel results.
"""
from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional
from collections.abc import Sequence

import httpx

from app.nlp.engine import SentinelEngine, label_for
from app.nlp.types import TextAnalysis

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{FINBERT_MODEL}"
TRANSFORMER_WEIGHT = 0.6
MAX_CHARS = 512  # FinBERT sees <= 512 tokens; headlines/posts fit, long bodies are truncated

Probs = tuple[float, float, float]  # (positive, negative, neutral)


def _probs_from(items: Any) -> Optional[Probs]:
    """[{"label": "positive", "score": 0.9}, ...] -> (pos, neg, neu); None if malformed."""
    if not isinstance(items, list):
        return None
    found: dict[str, float] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("label"), str):
            try:
                found[item["label"].lower()] = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                return None
    if not {"positive", "negative", "neutral"} & found.keys():
        return None
    return found.get("positive", 0.0), found.get("negative", 0.0), found.get("neutral", 0.0)


class _EnsembleEngine:
    """Shared blending: transformer probabilities + Sentinel score/drivers."""

    name = "finbert"

    def __init__(self, sentinel: SentinelEngine, weight: float = TRANSFORMER_WEIGHT) -> None:
        self._sentinel = sentinel
        self._weight = weight

    def score(self, texts: list[str], kinds: Optional[list[str]] = None) -> list[TextAnalysis]:
        base = self._sentinel.score(texts, kinds)
        try:
            probs = self._predict([(t or "")[:MAX_CHARS] for t in texts])
        except Exception as exc:  # noqa: BLE001 - optional engine must never break scoring
            logger.warning("%s unavailable (%s); using sentinel scores", self.name, exc)
            return base
        return [self._blend(b, p) for b, p in zip(base, probs, strict=True)]

    def _predict(self, texts: list[str]) -> list[Optional[Probs]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _blend(self, s: TextAnalysis, p: Optional[Probs]) -> TextAnalysis:
        if p is None:
            return s
        pos, neg, neu = p
        t_score = pos - neg
        score = max(-1.0, min(1.0, self._weight * t_score + (1.0 - self._weight) * s.score))
        label = label_for(score)
        t_label = "neutral" if neu >= max(pos, neg) else ("bullish" if pos > neg else "bearish")
        conf = self._weight * max(pos, neg, neu) + (1.0 - self._weight) * s.confidence
        if t_label != s.label and "neutral" not in (t_label, s.label):
            conf *= 0.7  # the two models flatly disagree
        return TextAnalysis(score=round(score, 4), label=label, confidence=round(min(0.99, conf), 3),
                            drivers=s.drivers)


class FinBertEngine(_EnsembleEngine):
    """Local FinBERT via ``transformers`` (lazy: the ~440 MB model loads on first score)."""

    name = "finbert"

    def __init__(self, sentinel: SentinelEngine, batch_size: int = 32) -> None:
        for pkg in ("transformers", "torch"):
            if importlib.util.find_spec(pkg) is None:
                raise RuntimeError(f"finbert engine needs '{pkg}' (pip install transformers torch)")
        super().__init__(sentinel)
        self._batch = batch_size
        self._pipe: Any = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._pipe is None and not self._failed:
                try:
                    from transformers import pipeline  # heavy import, deferred on purpose

                    self._pipe = pipeline("text-classification", model=FINBERT_MODEL, top_k=None,
                                          truncation=True, max_length=256)
                except Exception as exc:  # noqa: BLE001
                    self._failed = True
                    logger.warning("could not load %s (%s); finbert disabled", FINBERT_MODEL, exc)
        return self._pipe

    def _predict(self, texts: list[str]) -> list[Optional[Probs]]:
        pipe = self._load()
        if pipe is None:
            return [None] * len(texts)
        out: list[Optional[Probs]] = []
        for i in range(0, len(texts), self._batch):
            chunk = [t or " " for t in texts[i: i + self._batch]]
            out.extend(_probs_from(r) for r in pipe(chunk))
        return out


class FinBertApiEngine(_EnsembleEngine):
    """FinBERT through the Hugging Face Inference API (free tier needs ``HF_TOKEN``)."""

    name = "finbert-api"

    def __init__(self, sentinel: SentinelEngine, token: str, *, batch_size: int = 16, timeout: float = 20.0,
                 cooldown: float = 300.0, cache_size: int = 4096) -> None:
        if not token:
            raise RuntimeError("finbert-api engine needs HF_TOKEN")
        super().__init__(sentinel)
        self._token = token
        self._batch = batch_size
        self._timeout = timeout
        self._cooldown = cooldown
        self._cache: OrderedDict[str, Probs] = OrderedDict()
        self._cache_size = cache_size
        self._disabled_until = 0.0
        self._lock = threading.Lock()
        self._client: Optional[httpx.Client] = None

    def _request(self, batch: Sequence[str]) -> Any:
        """POST one batch; returns the decoded JSON (separate so tests can stub the network)."""
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        resp = self._client.post(HF_API_URL, json={"inputs": list(batch)},
                                 headers={"Authorization": f"Bearer {self._token}"})
        resp.raise_for_status()
        return resp.json()

    def _predict(self, texts: list[str]) -> list[Optional[Probs]]:
        out: list[Optional[Probs]] = [None] * len(texts)
        todo: list[int] = []
        with self._lock:
            for i, t in enumerate(texts):
                hit = self._cache.get(t)
                if hit is not None:
                    self._cache.move_to_end(t)
                    out[i] = hit
                elif t.strip():
                    todo.append(i)
        if not todo or time.monotonic() < self._disabled_until:
            return out
        for start in range(0, len(todo), self._batch):
            idx = todo[start: start + self._batch]
            try:
                data = self._request([texts[i] for i in idx])
            except (httpx.HTTPError, ValueError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                self._disabled_until = time.monotonic() + (self._cooldown if status in (401, 403, 429) else 30.0)
                logger.warning("finbert-api request failed (%s); falling back to sentinel", exc)
                break
            rows = data if isinstance(data, list) else []
            if len(idx) == 1 and rows and isinstance(rows[0], dict):
                rows = [rows]  # single input may come back un-nested
            with self._lock:
                for i, row in zip(idx, rows, strict=False):  # a short/malformed reply leaves the rest None
                    probs = _probs_from(row)
                    out[i] = probs
                    if probs is not None:
                        self._cache[texts[i]] = probs
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return out
