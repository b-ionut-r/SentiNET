"""FinBERT ensembles (network stubbed) and the engine registry."""
from __future__ import annotations

import importlib.util
import logging
import threading

import httpx
import pytest

from app.config import settings
from app.nlp import engine as engine_mod
from app.nlp import transformer
from app.nlp.engine import SentinelEngine, VaderEngine, get_engine, label_for, reset_engine
from app.nlp.transformer import FinBertApiEngine, FinBertEngine, _probs_from

POS = [{"label": "positive", "score": 0.9}, {"label": "neutral", "score": 0.07}, {"label": "negative", "score": 0.03}]
NEG = [{"label": "negative", "score": 0.8}, {"label": "neutral", "score": 0.15}, {"label": "positive", "score": 0.05}]
NEU = [{"label": "neutral", "score": 0.9}, {"label": "positive", "score": 0.06}, {"label": "negative", "score": 0.04}]


class StubApi(FinBertApiEngine):
    """FinBERT-API engine whose HTTP call is replaced by canned replies."""

    def __init__(self, replies: list[object], **kw: object) -> None:
        super().__init__(SentinelEngine(), token="hf_test", **kw)  # type: ignore[arg-type]
        self.replies = list(replies)
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self.delay = 0.0
        self.clock = [0.0]

    def _request(self, batch, timeout):  # type: ignore[override]
        self.calls.append(list(batch))
        self.timeouts.append(timeout)
        if self.delay:
            self.clock[0] += self.delay  # each request "takes" this long on the fake clock
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", transformer.HF_API_URL)
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(status, request=req))


# --------------------------------------------------------------------------- #
# Parsing + blending
# --------------------------------------------------------------------------- #
def test_probs_from_parses_and_rejects_malformed() -> None:
    assert _probs_from(POS) == (0.9, 0.03, 0.07)
    assert _probs_from([{"label": "Positive", "score": "0.5"}]) == (0.5, 0.0, 0.0)  # top-1 only, any case
    assert _probs_from({"error": "loading"}) is None
    assert _probs_from([{"label": "joy", "score": 1.0}]) is None
    assert _probs_from([{"label": "positive", "score": "x"}]) is None


def test_api_engine_blends_and_keeps_sentinel_drivers() -> None:
    texts = ["Nvidia price target raised to $250 from $220", "Shares slump 9% after guidance cut"]
    eng = StubApi([[POS, NEG]])
    base = SentinelEngine().score(texts)
    out = eng.score(texts)
    assert [a.label for a in out] == ["bullish", "bearish"]
    for a, b, p in zip(out, base, ((0.9, 0.03), (0.05, 0.8)), strict=True):
        expected = 0.6 * (p[0] - p[1]) + 0.4 * b.score
        assert a.score == pytest.approx(expected, abs=1e-3)
        assert a.label == label_for(a.score)
        assert a.drivers == b.drivers  # FinBERT has no token-level explanation
        assert 0.0 <= a.confidence <= 1.0
    assert eng.calls == [texts]


def test_api_flat_disagreement_lowers_confidence() -> None:
    text = "Apple beats estimates and raises guidance"
    agree = StubApi([[POS]]).score([text])[0]
    clash = StubApi([[NEG]]).score([text])[0]
    assert clash.confidence < agree.confidence


def test_api_caches_and_skips_blank_texts() -> None:
    eng = StubApi([[POS], [NEU]])
    eng.score(["Stocks rally", "   "])
    eng.score(["Stocks rally", "Fed holds rates"])
    assert eng.calls == [["Stocks rally"], ["Fed holds rates"]]


def test_api_single_input_may_come_back_unnested() -> None:
    a = StubApi([POS]).score(["Record quarter for Acme"])[0]
    assert a.label == "bullish"


def test_api_failure_falls_back_to_sentinel_and_opens_breaker() -> None:
    texts = ["Acme shares tumble 12% on weak outlook"]
    eng = StubApi([_http_error(429)], cooldown=300.0)
    base = SentinelEngine().score(texts)
    first = eng.score(texts)
    assert [(a.score, a.label) for a in first] == [(b.score, b.label) for b in base]
    second = eng.score(texts)  # breaker open: no further request
    assert [a.score for a in second] == [b.score for b in base]
    assert len(eng.calls) == 1


def test_api_name_reports_what_actually_scored() -> None:
    eng = StubApi([[POS], _http_error(503)])
    assert eng.name == "finbert-api"
    eng.score(["Stocks rally"])
    assert eng.name == "finbert-api" and eng.last_coverage == 1.0
    eng.score(["Fed holds rates"])  # the API failed: Sentinel scored it, and the name must say so
    assert eng.name.startswith("sentinel (finbert-api unavailable") and eng.last_coverage == 0.0
    eng.score(["Stocks rally", "Fed holds rates"])  # breaker open; one text still cached
    assert eng.name == "finbert-api (1/2; rest sentinel)"


def test_api_total_time_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    texts = [f"Acme headline {i}" for i in range(5)]
    eng = StubApi([[POS], [POS], [POS], [POS], [POS]], batch_size=1, budget=8.0, timeout=20.0)
    eng.delay = 3.0  # every request takes 3 s
    monkeypatch.setattr(transformer.time, "monotonic", lambda: eng.clock[0])
    out = eng.score(texts)
    assert len(eng.calls) == 3  # 0 s, 3 s, 6 s; at 9 s the 8 s budget is spent
    assert eng.timeouts[0] == 8.0 and eng.timeouts[1] == pytest.approx(5.0)  # never past the deadline
    assert eng.name == "finbert-api (3/5; rest sentinel)"
    base = SentinelEngine().score(texts)
    assert [a.score for a in out[3:]] == [b.score for b in base[3:]]


def test_api_malformed_reply_falls_back_per_item() -> None:
    eng = StubApi([[{"error": "Model is loading"}, POS]])
    base = SentinelEngine().score(["Fed holds rates", "Acme beats estimates"])
    out = eng.score(["Fed holds rates", "Acme beats estimates"])
    assert out[0].score == base[0].score  # unusable row: plain Sentinel
    assert out[1].score != base[1].score


def test_api_engine_needs_a_token() -> None:
    with pytest.raises(RuntimeError):
        FinBertApiEngine(SentinelEngine(), token="")


def test_local_finbert_requires_optional_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(transformer.importlib.util, "find_spec",
                        lambda name, *a: None if name in ("transformers", "torch") else real(name, *a))
    with pytest.raises(RuntimeError, match="transformers"):
        FinBertEngine(SentinelEngine())


def test_local_finbert_degrades_when_model_cannot_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transformer.importlib.util, "find_spec", lambda name, *a: object())
    eng = FinBertEngine(SentinelEngine())
    monkeypatch.setattr(eng, "_load", lambda: None)
    out = eng.score(["Acme beats estimates"])
    assert out[0].score == SentinelEngine().score(["Acme beats estimates"])[0].score
    assert eng.name == "sentinel (finbert loading)"


def test_local_finbert_loads_in_background_without_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transformer.importlib.util, "find_spec", lambda name, *a: object())
    eng = FinBertEngine(SentinelEngine())
    release = threading.Event()

    def slow_load() -> None:
        release.wait(5)
        eng._pipe = lambda chunk: [POS for _ in chunk]

    monkeypatch.setattr(eng, "_load_model", slow_load)
    first = eng.score(["Acme beats estimates"])  # model still loading: Sentinel answers at once
    assert first[0].score == SentinelEngine().score(["Acme beats estimates"])[0].score
    assert eng.name == "sentinel (finbert loading)"
    release.set()
    assert eng._loader is not None
    eng._loader.join(5)
    eng.score(["Acme beats estimates"])
    assert eng.name == "finbert"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fresh_registry():
    reset_engine()
    yield
    reset_engine()


@pytest.mark.parametrize(("name", "cls"), [("sentinel", SentinelEngine), ("vader", VaderEngine),
                                           ("SENTINEL", SentinelEngine), ("nonsense", SentinelEngine)])
def test_get_engine_by_setting(monkeypatch: pytest.MonkeyPatch, fresh_registry: None, name: str, cls: type) -> None:
    monkeypatch.setattr(settings, "sentiment_engine", name)
    eng = get_engine()
    assert isinstance(eng, cls)
    assert get_engine() is eng  # process-wide singleton


def test_get_engine_falls_back_when_optional_engine_unavailable(monkeypatch: pytest.MonkeyPatch,
                                                                fresh_registry: None,
                                                                caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(settings, "sentiment_engine", "finbert-api")
    monkeypatch.setattr(settings, "hf_token", "")
    with caplog.at_level(logging.WARNING, logger=engine_mod.__name__):
        eng = get_engine()
    assert isinstance(eng, SentinelEngine)
    assert "unavailable" in caplog.text


def test_get_engine_builds_api_ensemble_with_token(monkeypatch: pytest.MonkeyPatch, fresh_registry: None) -> None:
    monkeypatch.setattr(settings, "sentiment_engine", "finbert-api")
    monkeypatch.setattr(settings, "hf_token", "hf_test")
    eng = get_engine()
    assert isinstance(eng, FinBertApiEngine) and eng.name == "finbert-api"


def test_reset_engine_rebuilds(monkeypatch: pytest.MonkeyPatch, fresh_registry: None) -> None:
    monkeypatch.setattr(settings, "sentiment_engine", "vader")
    assert isinstance(get_engine(), VaderEngine)
    monkeypatch.setattr(settings, "sentiment_engine", "sentinel")
    assert isinstance(get_engine(), VaderEngine)  # cached until reset
    reset_engine()
    assert isinstance(get_engine(), SentinelEngine)
