"""Optional FinBERT engines, ensembled with Sentinel.

* ``finbert``     - local ``transformers`` pipeline (ProsusAI/finbert); needs ``transformers`` +
  ``torch`` installed (not in the default deps). The ~440 MB model loads in a background
  thread on first use, so scoring never blocks on it: until it is ready, texts get Sentinel.
* ``finbert-api`` - Hugging Face Inference API (needs ``HF_TOKEN``); batched, cached, with a
  circuit breaker and a total time budget per call, so a cold or rate-limited API never
  stalls an analysis.

Both blend ~0.6 transformer + 0.4 Sentinel and keep Sentinel's drivers (FinBERT has no
token-level explanation). Any failure degrades to plain Sentinel results, and ``name``
says so ("sentinel (finbert-api unavailable)"), so reports never claim a model that did
not run.
"""
from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Optional

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
    """Shared blending: transformer probabilities + Sentinel score/drivers.

    ``name`` reports what actually scored the last batch: the transformer's name when it scored
    every text, ``"<name> (k/n; rest sentinel)"`` when it scored some, and
    ``"sentinel (<name> <reason>)"`` when it scored none."""

    engine_name = "finbert"

    def __init__(self, sentinel: SentinelEngine, weight: float = TRANSFORMER_WEIGHT) -> None:
        self._sentinel = sentinel
        self._weight = weight
        self.name = self.engine_name
        self.last_coverage: Optional[float] = None  # share of the last batch the transformer scored
        self._why = "unavailable"  # reason shown when the transformer scored nothing

    def score(self, texts: list[str], kinds: Optional[list[str]] = None) -> list[TextAnalysis]:
        base = self._sentinel.score(texts, kinds)
        try:
            probs = self._predict([(t or "")[:MAX_CHARS] for t in texts])
        except Exception as exc:  # noqa: BLE001 - optional engine must never break scoring
            logger.warning("%s unavailable (%s); using sentinel scores", self.engine_name, exc)
            self._why = "unavailable"
            probs = [None] * len(texts)
        self._note_coverage(texts, probs)
        return [self._blend(b, p) for b, p in zip(base, probs, strict=True)]

    def _note_coverage(self, texts: list[str], probs: list[Optional[Probs]]) -> None:
        scorable = [p for t, p in zip(texts, probs, strict=True) if (t or "").strip()]
        if not scorable:
            return
        k, n = sum(p is not None for p in scorable), len(scorable)
        self.last_coverage = k / n
        if k == n:
            self.name = self.engine_name
        elif k:
            self.name = f"{self.engine_name} ({k}/{n}; rest sentinel)"
        else:
            self.name = f"sentinel ({self.engine_name} {self._why})"

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
    """Local FinBERT via ``transformers``; the model loads in the background on first use."""

    engine_name = "finbert"

    def __init__(self, sentinel: SentinelEngine, batch_size: int = 32, budget: float = 10.0) -> None:
        for pkg in ("transformers", "torch"):
            if importlib.util.find_spec(pkg) is None:
                raise RuntimeError(f"finbert engine needs '{pkg}' (pip install transformers torch)")
        super().__init__(sentinel)
        self._batch = batch_size
        self._budget = budget  # seconds of inference per score() call; the rest gets Sentinel
        self._pipe: Any = None
        self._failed = False
        self._loader: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _load_model(self) -> None:
        try:
            from transformers import pipeline  # heavy import, deferred on purpose

            pipe = pipeline("text-classification", model=FINBERT_MODEL, top_k=None, truncation=True,
                            max_length=256)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not load %s (%s); finbert disabled", FINBERT_MODEL, exc)
            with self._lock:
                self._failed = True
            return
        with self._lock:
            self._pipe = pipe

    def _load(self) -> Any:
        """The pipeline if ready, else None; the first call starts loading it in the background."""
        with self._lock:
            if self._pipe is None and not self._failed and self._loader is None:
                self._loader = threading.Thread(target=self._load_model, name="finbert-load", daemon=True)
                self._loader.start()
            return self._pipe

    def _predict(self, texts: list[str]) -> list[Optional[Probs]]:
        out: list[Optional[Probs]] = [None] * len(texts)
        pipe = self._load()
        if pipe is None:
            self._why = "failed to load" if self._failed else "loading"
            return out
        deadline = time.monotonic() + self._budget
        self._why = "returned no usable scores"  # unless the budget runs out below
        for i in range(0, len(texts), self._batch):
            if time.monotonic() > deadline:
                self._why = "over time budget"
                logger.warning("finbert over its %.0fs budget after %d/%d texts; rest scored by sentinel",
                               self._budget, i, len(texts))
                break
            chunk = [t or " " for t in texts[i: i + self._batch]]
            for k, row in enumerate(pipe(chunk)):
                out[i + k] = _probs_from(row)
        return out


class FinBertApiEngine(_EnsembleEngine):
    """FinBERT through the Hugging Face Inference API (free tier needs ``HF_TOKEN``)."""

    engine_name = "finbert-api"

    def __init__(self, sentinel: SentinelEngine, token: str, *, batch_size: int = 16, timeout: float = 20.0,
                 budget: float = 8.0, cooldown: float = 300.0, cache_size: int = 4096) -> None:
        if not token:
            raise RuntimeError("finbert-api engine needs HF_TOKEN")
        super().__init__(sentinel)
        self._token = token
        self._batch = batch_size
        self._timeout = timeout  # per request
        self._budget = budget  # all requests of one score() call together
        self._cooldown = cooldown
        self._cache: OrderedDict[str, Probs] = OrderedDict()
        self._cache_size = cache_size
        self._disabled_until = 0.0
        self._lock = threading.Lock()
        self._client: Optional[httpx.Client] = None

    def _request(self, batch: Sequence[str], timeout: float) -> Any:
        """POST one batch; returns the decoded JSON (separate so tests can stub the network)."""
        if self._client is None:
            self._client = httpx.Client()
        resp = self._client.post(HF_API_URL, json={"inputs": list(batch), "parameters": {"top_k": 3}},
                                 headers={"Authorization": f"Bearer {self._token}"}, timeout=timeout)
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
        if not todo:
            return out
        now = time.monotonic()
        if now < self._disabled_until:
            self._why = "unavailable: cooling down"
            return out
        deadline = now + self._budget
        self._why = "returned no usable scores"  # unless a failure below says why
        for start in range(0, len(todo), self._batch):
            remaining = deadline - time.monotonic()
            if remaining < 0.5:
                self._why = "over time budget"
                self._disabled_until = time.monotonic() + 30.0  # a slow (cold) API: give it a moment
                logger.warning("finbert-api over its %.0fs budget after %d/%d texts; rest scored by sentinel",
                               self._budget, start, len(todo))
                break
            idx = todo[start: start + self._batch]
            try:
                data = self._request([texts[i] for i in idx], timeout=min(self._timeout, remaining))
            except (httpx.HTTPError, ValueError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                self._disabled_until = time.monotonic() + (self._cooldown if status in (401, 403, 429) else 30.0)
                self._why = f"unavailable: HTTP {status}" if status else "unavailable"
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
