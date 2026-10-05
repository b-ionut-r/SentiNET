"""Benchmark harness (offline): metric math, dataset parsing, and the committed results summary."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from scripts import eval_engine as ev


def test_metrics_math() -> None:
    gold = ["bullish", "bullish", "bearish", "neutral", "neutral"]
    pred = ["bullish", "neutral", "bearish", "neutral", "bullish"]
    m = ev.metrics(gold, pred)
    assert m["accuracy"] == 0.6
    pc = m["per_class"]
    assert pc["bullish"]["precision"] == 0.5 and pc["bullish"]["recall"] == 0.5
    assert pc["bearish"]["f1"] == 1.0
    assert m["macro_f1"] == pytest.approx((0.5 + 1.0 + 0.5) / 3, abs=1e-4)
    assert m["confusion"]["bullish"] == {"bullish": 1, "bearish": 0, "neutral": 1}


def test_directional_metrics() -> None:
    d = ev.directional_metrics(["bullish", "bullish", "bearish", "bearish"],
                               ["bullish", "neutral", "bullish", "bearish"])
    assert d["coverage"] == 0.75
    assert d["accuracy_covered"] == pytest.approx(2 / 3, abs=1e-4)
    assert d["flipped"] == 0.25
    assert d["balanced_accuracy_covered"] == pytest.approx((0.5 + 1.0) / 2)


def test_fiqa_labels_follow_the_preregistered_threshold() -> None:
    assert ev._fiqa_label(0.1) == "bullish"
    assert ev._fiqa_label(0.09) == "neutral"
    assert ev._fiqa_label(-0.1) == "neutral"
    assert ev._fiqa_label(-0.11) == "bearish"


def test_loaders_parse_cached_files(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "CACHE", tmp_path)
    (tmp_path / "twitter_valid.csv").write_text('text,label\n"Stocks rally",1\nFed holds,2\n', encoding="utf-8")
    assert ev.load_twitter("valid") == [ev.Example("Stocks rally", "bullish"), ev.Example("Fed holds", "neutral")]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", "Sales rose .@positive\nCEO named .@neutral\n")
    (tmp_path / "FinancialPhraseBank-v1.0.zip").write_bytes(buf.getvalue())
    assert [e.gold for e in ev.load_phrasebank()] == ["bullish", "neutral"]
    rows = [{"sentence": "A up", "type": "post", "score": 0.5}, {"sentence": "B", "type": "headline", "score": -0.4},
            {"sentence": "A up", "type": "post", "score": -0.6}]  # conflicting aspects -> dropped
    (tmp_path / "fiqa_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    assert ev.load_fiqa("post") == []
    assert ev.load_fiqa("headline") == [ev.Example("B", "bearish", "news")]
    payload = {"fetched": "2026-10-04", "messages": [{"id": 1, "body": "x", "tag": "bullish"},
                                                     {"id": 2, "body": "y", "tag": "bearish"},
                                                     {"id": 3, "body": "z", "tag": None}]}
    (tmp_path / "stocktwits_sample.json").write_text(json.dumps(payload), encoding="utf-8")
    odd, fetched = ev.load_stocktwits("odd")
    assert odd == [ev.Example("x", "bullish", "social")] and fetched == "2026-10-04"


def test_committed_results_show_sentinel_beating_vader() -> None:
    data = json.loads(ev.RESULTS_PATH.read_text(encoding="utf-8"))
    held = data["summary"]
    for name in ("twitter_valid", "phrasebank_allagree", "fiqa_headlines", "fiqa_posts"):
        s, v = held[name]["sentinel"], held[name]["vader"]
        assert s["macro_f1"] > v["macro_f1"] and s["accuracy"] > v["accuracy"], name
        assert s["polar_accuracy_committed"] > v["polar_accuracy_committed"], name
    assert held["twitter_valid"]["sentinel"]["macro_f1"] >= 0.75
    assert held["stocktwits_odd"]["sentinel"]["accuracy_covered"] > held["stocktwits_odd"]["vader"]["accuracy_covered"]
    assert data["throughput_texts_per_s"]["sentinel"] >= 2000
    assert "twitter_train" not in data["held_out"]
